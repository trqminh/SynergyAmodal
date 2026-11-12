"""reference_only attention, trained: reference_only.py with LoRA on the self-attention layers.

Same mechanism as reference_only.py: the reference goes through the same UNet ("write"), every self-attention
layer keeps its input tokens, and the real pass ("read") attends over its own tokens plus the reference ones.
What changes, so it can be trained like a ReferenceNet (Animate Anyone) with one shared UNet:

  - The SynergyAmodal UNet stays frozen. LoRA on to_q, to_k, to_v and to_out of every attn1 is the only
    trainable part, and it is active in both passes.
  - The reference is clean (timestep 0), as in ReferenceNet, so at inference it is written once and the
    bank is reused at every denoising step.
  - The reference is dropped 10% of the time, like the IP-Adapter: a dropped row attends to itself only.
    At inference that gives the unconditional rows for classifier-free guidance, in place of style_fidelity.
"""
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.training_utils import compute_snr
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import save_image

from trainer_deocclusion_pseudo import InpaintingTrainer
from utils import import_str


class LoRALinear(nn.Module):
    """base(x) + up(down(x)), initialized like diffusers' LoRALinearLayer, so it starts as the base layer."""

    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x))


class ReferenceAttnControl:
    def __init__(self):
        self.mode = None  # 'write', 'read' or None (plain UNet)
        self.drop = None  # rows of the read batch that do not see the reference
        self.processors = []

    def clear(self):
        for processor in self.processors:
            processor.bank = None


class ReferenceAttnProcessor:
    """reference_only.py's processor without the detach (gradients go back into the reference pass)
    and with a per-row drop instead of style_fidelity."""

    def __init__(self, control):
        self.control = control
        self.bank = None
        self.processor = AttnProcessor2_0()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        control = self.control

        if control.mode == 'write':
            self.bank = hidden_states

        elif control.mode == 'read':
            # The reference is written once per batch; with classifier-free guidance the read batch is a repeat of it.
            bank = self.bank.repeat(hidden_states.shape[0] // self.bank.shape[0], 1, 1)
            output = self.processor(attn, hidden_states, encoder_hidden_states=torch.cat([hidden_states, bank], dim=1))
            if control.drop is not None and control.drop.any():
                output = torch.where(control.drop[:, None, None], self.processor(attn, hidden_states), output)
            return output

        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


def add_reference_attn(unet, control, rank):
    """Puts a ReferenceAttnProcessor and LoRA on every self-attention layer. Returns the LoRA layers."""
    processors = dict(unet.attn_processors)
    lora_layers = nn.ModuleList()
    for name in processors:
        if not name.endswith('attn1.processor'):
            continue
        processors[name] = ReferenceAttnProcessor(control)
        control.processors.append(processors[name])

        attn = unet.get_submodule(name[:-len('.processor')])
        attn.to_q = LoRALinear(attn.to_q, rank)
        attn.to_k = LoRALinear(attn.to_k, rank)
        attn.to_v = LoRALinear(attn.to_v, rank)
        attn.to_out[0] = LoRALinear(attn.to_out[0], rank)
        lora_layers.extend([attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]])
    unet.set_attn_processor(processors)

    return lora_layers


class ReferenceOnlyTrainer(InpaintingTrainer):
    def __init__(self, opt):
        super().__init__(opt)

        # Lock the SynergyAmodal UNet, using its EMA weights like app.py does.
        checkpoint = torch.load(opt.base_ckpt, map_location='cpu', mmap=True, weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        self.model_ema.load_ema_params(self.unet)
        del self.model_ema
        self.unet.requires_grad_(False).eval()

        self.control = ReferenceAttnControl()
        lora_layers = add_reference_attn(self.unet, self.control, opt.lora_rank)
        self.lora_params = [p for layer in lora_layers for p in [layer.down.weight, layer.up.weight]]

    def configure_optimizers(self):
        return torch.optim.AdamW(self.lora_params, lr=self.opt.training.learning_rate, weight_decay=self.opt.training.weight_decay)

    def reference_input(self, reference, reference_mask, size):
        """The reference latent and its inpainting channels, as in reference_only.py: the whole reference object
        is visible, so mask = 1 - reference mask, and both image latents are the reference latent."""
        reference = F.interpolate(reference, size=size, mode='bilinear')
        reference_mask = (F.interpolate(reference_mask, size=size, mode='bilinear') > 0.5).to(reference.dtype)
        reference_latent = self.latent_scale_fn(self.vae.encode(reference * 2 - 1))
        mask = F.interpolate(1 - reference_mask, size=reference_latent.shape[-2:])
        return torch.cat([reference_latent, mask, reference_latent, reference_latent], dim=1)

    def write_reference(self, reference_input, text_condition):
        t = torch.zeros(reference_input.shape[0], dtype=torch.long, device=self.device)
        self.control.mode = 'write'
        self.unet(reference_input, t, encoder_hidden_states=text_condition)
        self.control.mode = 'read'

    def training_step(self, batch, _):
        image = batch['image']
        modal_mask = batch['modal_mask']
        amodal_image = batch['amodal_image']

        modal_image = image * modal_mask + 1 * (1 - modal_mask)

        B = image.shape[0]

        # Same inputs, noise and loss as the UNet was trained with in InpaintingTrainer.
        with torch.no_grad():
            captions = [('' if random.random() < 0.5 else caption) for caption in batch['caption']]
            text_condition = self.encode_text(captions)

            all = torch.cat([image, modal_image, amodal_image], dim=0) * 2 - 1
            all_latents = self.latent_scale_fn(self.vae.encode(all))
            image_latent, modal_image_latent, input_latents = all_latents.split([B, B, B], dim=0)

            t = torch.randint(0, self.num_train_timesteps, (B,), dtype=torch.long, device=self.device)

            noise = torch.randn_like(input_latents) + 0.1 * torch.randn(B, input_latents.shape[1], 1, 1, device=self.device)
            input_latents_noisy = self.scheduler.add_noise(input_latents, noise, t)

            mask = F.interpolate(1 - modal_mask, size=(64, 64))
            image_condition = torch.cat([mask, modal_image_latent, image_latent], dim=1)
            image_condition = image_condition * (torch.rand(B, 1, 1, 1, device=self.device) > 0.1)

            reference_input = self.reference_input(batch['reference'], batch['reference_mask'], image.shape[-2:])

        self.write_reference(reference_input, text_condition)
        self.control.drop = torch.rand(B, device=self.device) < self.opt.training.p_drop_reference
        noise_pred = self.unet(
            torch.cat([input_latents_noisy, image_condition], dim=1), t, encoder_hidden_states=text_condition, return_dict=False
        )[0]
        self.control.mode = None
        self.control.drop = None
        self.control.clear()

        snr = compute_snr(self.scheduler, t)
        weights = torch.stack([snr, 5.0 * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr

        loss = (F.mse_loss(noise_pred, noise, reduction='none').mean([-1, -2, -3]) * weights).mean()

        self.log('losses/total_loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, _):
        if not self.trainer.is_global_zero:
            return

        output = self.inference(batch, reference=batch['reference'], reference_mask=batch['reference_mask'])[0]

        rows = [
            batch['image'],
            batch['modal_mask'].repeat(1, 3, 1, 1),
            F.interpolate(batch['reference'], size=batch['image'].shape[-2:], mode='bilinear'),
            batch['amodal_image'],
            output,
        ]
        save_dir = Path(self.trainer.default_root_dir) / 'samples'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_image(
            torch.stack(rows, dim=1).flatten(0, 1).float(),
            save_dir / f'step_{self.global_step:06d}.png',
            nrow=len(rows),
        )

    @torch.no_grad()
    def inference_ddim(self, noise, image_condition, text_condition, reference, reference_mask, num_inference_steps=25, cfg_image=1, cfg_text=1, strength=1, initial_latent=None):
        B = noise.shape[0]
        assert self.num_train_timesteps % num_inference_steps == 0
        self.scheduler.config.steps_offset = self.num_train_timesteps // num_inference_steps - 1
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        negative_text_condition = self.encode_text(['']).repeat(B, 1, 1)

        # The reference is clean, so its tokens are the same at every step: write them once.
        reference_input = self.reference_input(reference, reference_mask, (noise.shape[-2] * 8, noise.shape[-1] * 8))
        self.write_reference(reference_input, text_condition)

        use_cfg = not (cfg_image == 1 and cfg_text == 1)
        if use_cfg:
            # As in IPAdapterTrainer: rows [prompt + reference, no prompt, no prompt and no image condition],
            # the last two without the reference, so cfg_text guides text and reference together.
            self.control.drop = torch.arange(3 * B, device=self.device) >= B

        begin_index = int(len(timesteps) * (1 - strength))

        if initial_latent is None or begin_index == 0:
            latents_noisy = noise
        else:
            latents_noisy = self.scheduler.add_noise(initial_latent.repeat(B, 1, 1, 1), noise, timesteps[begin_index][None].repeat(B))

        for t in timesteps[begin_index:]:
            if not use_cfg:
                noise_pred = self.unet(torch.cat([latents_noisy, image_condition], dim=1), t, encoder_hidden_states=text_condition).sample
            else:
                noise_pred, noise_pred1, noise_pred2 = self.unet(
                    torch.cat([
                        torch.cat([latents_noisy, image_condition], dim=1),
                        torch.cat([latents_noisy, image_condition], dim=1),
                        torch.cat([latents_noisy, torch.zeros_like(image_condition)], dim=1),
                    ], dim=0),
                    t,
                    encoder_hidden_states=torch.cat([text_condition, negative_text_condition, negative_text_condition], dim=0),
                ).sample.chunk(3, 0)

                noise_pred = noise_pred2 + (noise_pred1 - noise_pred2) * cfg_image + (noise_pred - noise_pred1) * cfg_text

            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy).prev_sample.to(self.device)

        self.control.mode = None
        self.control.drop = None
        self.control.clear()

        return latents_noisy


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reference_only.yaml", help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    train_dataset = import_str(opt.dataset.module)(**opt.dataset.args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.training.batch_size, num_workers=opt.training.num_workers, shuffle=True)

    val_dataset = import_str(opt.val_dataset.module)(**opt.val_dataset.args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.training.batch_size, num_workers=2)

    system = ReferenceOnlyTrainer(opt)

    trainer = Trainer(
        callbacks=[ModelCheckpoint(every_n_train_steps=opt.training.save_every_n_steps)],
        default_root_dir=os.path.join(opt.output_dir, opt.experiment_name),
        max_steps=opt.training.max_steps,
        log_every_n_steps=10,
        accumulate_grad_batches=opt.training.accumulate_grad_batches,
        precision=opt.training.precision,
        accelerator='auto',
        devices=opt.training.devices,
        benchmark=True,
        check_val_every_n_epoch=None,
        val_check_interval=opt.training.sample_every_n_steps,
        limit_val_batches=1,
    )

    trainer.fit(
        model=system,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=opt.training.resume_from_checkpoint,
    )
