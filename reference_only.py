"""Training-free "reference_only" control on the SynergyAmodal UNet.

Follows sd-webui-controlnet (scripts/hook.py) and diffusers' community StableDiffusionReferencePipeline
(examples/community/stable_diffusion_reference.py), attention part only (reference_attn, no AdaIN):

  1. At every denoising step the reference latent is noised to the current timestep and run through the
     same UNet ("write"). Every self-attention layer keeps its input tokens in a bank.
  2. The real pass ("read") attends over its own tokens plus the banked reference tokens.
  3. With classifier-free guidance, style_fidelity mixes in plain self-attention for the unconditional rows.

They patch BasicTransformerBlock.forward; here the same thing is an attention processor on every attn1,
which for SD2's LayerNorm blocks computes exactly the same.

Inpainting UNets also need the extra input channels for the reference pass. webui fills them as if
nothing were masked (mask 0, masked image = the reference). The SynergyAmodal analogue: the whole
reference object is visible, so mask = 1 - reference mask, and both image latents are the reference latent.
"""
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0

from trainer_deocclusion_pseudo import InpaintingTrainer


class ReferenceAttnControl:
    def __init__(self, style_fidelity=0.5, attention_auto_machine_weight=1.0):
        self.style_fidelity = style_fidelity
        self.attention_auto_machine_weight = attention_auto_machine_weight
        self.mode = None  # 'write', 'read' or None (plain UNet)
        self.uc_mask = None  # rows of the batch that are unconditional, with classifier-free guidance


class ReferenceAttnProcessor:
    def __init__(self, control, attn_weight):
        self.control = control
        self.attn_weight = attn_weight
        self.bank = []
        self.processor = AttnProcessor2_0()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        control = self.control

        if control.mode == 'write':
            self.bank.append(hidden_states.detach().clone())

        elif control.mode == 'read':
            bank, self.bank = self.bank, []
            if bank and control.attention_auto_machine_weight > self.attn_weight:
                attn_output_uc = self.processor(attn, hidden_states, encoder_hidden_states=torch.cat([hidden_states] + bank, dim=1))
                attn_output_c = attn_output_uc.clone()
                if control.uc_mask is not None and control.style_fidelity > 0:
                    attn_output_c[control.uc_mask] = self.processor(attn, hidden_states[control.uc_mask])
                return control.style_fidelity * attn_output_c + (1.0 - control.style_fidelity) * attn_output_uc

        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


def add_reference_attn(unet, control):
    """Puts a ReferenceAttnProcessor on every self-attention layer. As in the reference implementations,
    the layers are ordered from widest to narrowest and layer i gets attn_weight i / n, so a control weight
    below 1 drops the narrow (deep) layers first."""
    names = [name for name in unet.attn_processors if name.endswith('attn1.processor')]
    width = lambda name: unet.get_submodule(name[:-len('.processor')]).to_q.in_features
    names = sorted(names, key=lambda name: -width(name))

    processors = dict(unet.attn_processors)
    for i, name in enumerate(names):
        processors[name] = ReferenceAttnProcessor(control, attn_weight=float(i) / float(len(names)))
    unet.set_attn_processor(processors)


class ReferenceOnlyInpainter(InpaintingTrainer):
    """The SynergyAmodal UNet (EMA weights, as in app.py) with reference_only attention. Nothing is trained."""

    def __init__(self, opt, style_fidelity=0.5, attention_auto_machine_weight=1.0):
        super().__init__(opt)

        checkpoint = torch.load(opt.base_ckpt, map_location='cpu', mmap=True, weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        self.model_ema.load_ema_params(self.unet)
        del self.model_ema
        self.unet.requires_grad_(False).eval()

        self.control = ReferenceAttnControl(style_fidelity, attention_auto_machine_weight)
        add_reference_attn(self.unet, self.control)

    @torch.no_grad()
    def inference_ddim(self, noise, image_condition, text_condition, reference, reference_mask, num_inference_steps=25, cfg_image=1, cfg_text=1, strength=1, initial_latent=None):
        B = noise.shape[0]
        assert self.num_train_timesteps % num_inference_steps == 0
        self.scheduler.config.steps_offset = self.num_train_timesteps // num_inference_steps - 1
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        negative_text_condition = self.encode_text(['']).repeat(B, 1, 1)

        # The reference at the generation size, like webui resizes it, and its "nothing hidden" condition.
        size = (noise.shape[-2] * 8, noise.shape[-1] * 8)
        reference = F.interpolate(reference, size=size, mode='bilinear')
        reference_mask = (F.interpolate(reference_mask, size=size, mode='bilinear') > 0.5).to(reference.dtype)
        reference_latent = self.latent_scale_fn(self.vae.encode(reference * 2 - 1))
        mask = F.interpolate(1 - reference_mask, size=noise.shape[-2:])
        reference_condition = torch.cat([mask, reference_latent, reference_latent], dim=1)

        use_cfg = not (cfg_image == 1 and cfg_text == 1)
        if use_cfg:
            # Rows: [prompt, no prompt, no prompt and no image condition]; the last two are unconditional.
            self.control.uc_mask = torch.arange(3 * B, device=self.device) >= B
            reference_condition = reference_condition.repeat(3, 1, 1, 1)
            text_condition_all = torch.cat([text_condition, negative_text_condition, negative_text_condition], dim=0)

        begin_index = int(len(timesteps) * (1 - strength))

        if initial_latent is None or begin_index == 0:
            latents_noisy = noise
        else:
            latents_noisy = self.scheduler.add_noise(initial_latent.repeat(B, 1, 1, 1), noise, timesteps[begin_index][None].repeat(B))

        for t in timesteps[begin_index:]:
            # Fresh noise on the reference at every step, as in both reference implementations.
            reference_xt = self.scheduler.add_noise(reference_latent, torch.randn_like(reference_latent), t[None].repeat(B))

            if not use_cfg:
                self.control.mode = 'write'
                self.unet(torch.cat([reference_xt, reference_condition], dim=1), t, encoder_hidden_states=text_condition)
                self.control.mode = 'read'
                noise_pred = self.unet(torch.cat([latents_noisy, image_condition], dim=1), t, encoder_hidden_states=text_condition).sample
            else:
                self.control.mode = 'write'
                self.unet(torch.cat([reference_xt.repeat(3, 1, 1, 1), reference_condition], dim=1), t, encoder_hidden_states=text_condition_all)
                self.control.mode = 'read'
                noise_pred, noise_pred1, noise_pred2 = self.unet(
                    torch.cat([
                        torch.cat([latents_noisy, image_condition], dim=1),
                        torch.cat([latents_noisy, image_condition], dim=1),
                        torch.cat([latents_noisy, torch.zeros_like(image_condition)], dim=1),
                    ], dim=0),
                    t,
                    encoder_hidden_states=text_condition_all,
                ).sample.chunk(3, 0)

                noise_pred = noise_pred2 + (noise_pred1 - noise_pred2) * cfg_image + (noise_pred - noise_pred1) * cfg_text

            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy).prev_sample.to(self.device)

        self.control.mode = None
        self.control.uc_mask = None

        return latents_noisy
