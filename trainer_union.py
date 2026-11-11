"""One model for all three hints: box and scribble through one ControlNet-Union, the reference through IP-Adapter.

The SynergyAmodal UNet stays locked. The ControlNet-Union and the IP-Adapter modules are trained together, with
Uni-ControlNet's hint dropout so that any subset of the hints works at inference.
"""
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_snr
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import save_image
from transformers import CLIPVisionModelWithProjection

from controlnet_union import ControlNetUnionModel
from ip_adapter import ImageProjModel, add_ip_adapter, set_scale
from trainer_deocclusion_pseudo import InpaintingTrainer
from utils import import_str

# The control types of the ControlNet-Union, in this order.
SPATIAL_HINTS = ('box', 'scribble')


class UnionTrainer(InpaintingTrainer):
    def __init__(self, opt):
        super().__init__(opt)

        # Lock the SynergyAmodal UNet, using its EMA weights like app.py does.
        checkpoint = torch.load(opt.base_ckpt, map_location='cpu', mmap=True, weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        self.model_ema.load_ema_params(self.unet)
        del self.model_ema
        self.unet.requires_grad_(False).eval()

        # InpaintingTrainer widens conv_in to 13 channels but leaves the config at 9.
        self.unet.register_to_config(in_channels=self.unet.conv_in.weight.shape[1])

        # Box and scribble: one ControlNet-Union, copied from the UNet like a plain ControlNet.
        self.controlnet = ControlNetUnionModel.from_unet(self.unet)
        assert self.controlnet.config.num_control_type == len(SPATIAL_HINTS)

        # Optionally start from one of our single-hint ControlNets. The union modules are new either way.
        if opt.get('init_controlnet_ckpt'):
            state_dict = torch.load(opt.init_controlnet_ckpt, map_location='cpu', weights_only=False)['state_dict']
            state_dict = {k[len('controlnet.'):]: v for k, v in state_dict.items() if k.startswith('controlnet.')}
            missing, unexpected = self.controlnet.load_state_dict(state_dict, strict=False)
            assert not unexpected, unexpected
            print(f"Loaded {opt.init_controlnet_ckpt}, new modules: {sorted({k.split('.')[0] for k in missing})}")

        # Reference: IP-Adapter on the UNet's cross-attention, as in trainer_ip_adapter.py.
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(opt.image_encoder_path)
        self.image_encoder.requires_grad_(False).eval()

        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=opt.num_tokens,
        )
        self.adapter_modules = add_ip_adapter(self.unet, num_tokens=opt.num_tokens)

    def configure_optimizers(self):
        # Each part keeps the learning rate it was trained with on its own.
        training = self.opt.training
        return torch.optim.AdamW([
            {'params': self.controlnet.parameters(), 'lr': training.controlnet_learning_rate},
            {'params': list(self.image_proj_model.parameters()) + list(self.adapter_modules.parameters()), 'lr': training.ip_adapter_learning_rate},
        ], weight_decay=training.weight_decay)

    @torch.no_grad()
    def encode_reference(self, clip_image):
        return self.image_encoder(clip_image).image_embeds

    def sample_hints_to_keep(self, B):
        """Uni-ControlNet's dropout: keep all hints with p_keep_all, drop all with p_drop_all, and otherwise drop
        each one with p_drop_each. Returns a (B, 3) mask for box, scribble and reference."""
        training = self.opt.training
        keep = torch.rand(B, 3, device=self.device) > training.p_drop_each
        r = torch.rand(B, 1, device=self.device)
        keep = torch.where(r < training.p_keep_all, True, keep)
        keep = torch.where((r >= training.p_keep_all) & (r < training.p_keep_all + training.p_drop_all), False, keep)
        return keep

    def predict_noise(self, sample, timestep, text_condition, ip_tokens, hints, control_type):
        # The ControlNet only sees the text, like IP-Adapter's CNAttnProcessor, which strips the image tokens
        # before the ControlNet's cross-attention.
        down_residuals, mid_residual = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=text_condition,
            controlnet_cond=hints,
            control_type=control_type,
            control_type_idx=list(range(len(SPATIAL_HINTS))),
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=torch.cat([text_condition, ip_tokens], dim=1),
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
        )[0]

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

            # A dropped box or scribble is a black image with its control type off. A dropped reference is a zero
            # image embedding, as in IP-Adapter.
            keep = self.sample_hints_to_keep(B)
            hints = [batch[name] * keep[:, i, None, None, None] for i, name in enumerate(SPATIAL_HINTS)]
            control_type = keep[:, :len(SPATIAL_HINTS)].float()
            image_embeds = self.encode_reference(batch['clip_image']) * keep[:, 2:]

        ip_tokens = self.image_proj_model(image_embeds)

        noise_pred = self.predict_noise(
            torch.cat([input_latents_noisy, image_condition], dim=1), t, text_condition, ip_tokens, hints, control_type
        )

        snr = compute_snr(self.scheduler, t)
        weights = torch.stack([snr, 5.0 * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr

        loss = (F.mse_loss(noise_pred, noise, reduction='none').mean([-1, -2, -3]) * weights).mean()

        self.log('losses/total_loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, _):
        if not self.trainer.is_global_zero:
            return

        # One row per hint on its own, then all three.
        rows = [
            batch['image'],
            batch['modal_mask'].repeat(1, 3, 1, 1),
            batch['box'],
            batch['scribble'],
            F.interpolate(batch['reference'], size=batch['image'].shape[-2:], mode='bilinear'),
            batch['amodal_image'],
            self.inference(batch, box=batch['box'])[0],
            self.inference(batch, scribble=batch['scribble'])[0],
            self.inference(batch, clip_image=batch['clip_image'])[0],
            self.inference(batch, box=batch['box'], scribble=batch['scribble'], clip_image=batch['clip_image'])[0],
        ]
        save_dir = Path(self.trainer.default_root_dir) / 'samples'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_image(
            torch.stack(rows, dim=1).flatten(0, 1).float(),
            save_dir / f'step_{self.global_step:06d}.png',
            nrow=len(rows),
        )

    @torch.no_grad()
    def inference_ddim(self, noise, image_condition, text_condition, box=None, scribble=None, clip_image=None, scale=1.0, num_inference_steps=25, cfg_image=1, cfg_text=1, strength=1, initial_latent=None):
        """box, scribble: (B, 3, H, W) in [0, 1]. clip_image: CLIP-processed reference. Leave a hint as None to not
        give it; it is then dropped the same way as in training."""
        B = noise.shape[0]
        assert self.num_train_timesteps % num_inference_steps == 0
        self.scheduler.config.steps_offset = self.num_train_timesteps // num_inference_steps - 1
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        negative_text_condition = self.encode_text(['']).repeat(B, 1, 1)

        H, W = noise.shape[-2] * 8, noise.shape[-1] * 8
        given = {'box': box, 'scribble': scribble}
        hints = [given[name] if given[name] is not None else torch.zeros(B, 3, H, W, device=self.device, dtype=noise.dtype) for name in SPATIAL_HINTS]
        control_type = torch.tensor([[float(given[name] is not None) for name in SPATIAL_HINTS]], device=self.device).repeat(B, 1)

        if clip_image is not None:
            image_embeds = self.encode_reference(clip_image)
        else:
            image_embeds = torch.zeros(B, self.image_encoder.config.projection_dim, device=self.device, dtype=noise.dtype)
        ip_tokens = self.image_proj_model(image_embeds)
        negative_ip_tokens = self.image_proj_model(torch.zeros_like(image_embeds))

        set_scale(self.unet, scale)

        begin_index = int(len(timesteps) * (1 - strength))

        if initial_latent is None or begin_index == 0:
            latents_noisy = noise
        else:
            latents_noisy = self.scheduler.add_noise(initial_latent.repeat(B, 1, 1, 1), noise, timesteps[begin_index][None].repeat(B))

        for t in timesteps[begin_index:]:
            if cfg_image == 1 and cfg_text == 1:
                noise_pred = self.predict_noise(torch.cat([latents_noisy, image_condition], dim=1), t, text_condition, ip_tokens, hints, control_type)
            else:
                # As before: the box and scribble stay on for the unconditional passes (ControlNet), and the
                # reference goes with the text (IP-Adapter), so cfg_text guides text and reference together.
                noise_pred, noise_pred1, noise_pred2 = self.predict_noise(
                    torch.cat([
                        torch.cat([latents_noisy, image_condition], dim=1),
                        torch.cat([latents_noisy, image_condition], dim=1),
                        torch.cat([latents_noisy, torch.zeros_like(image_condition)], dim=1),
                    ], dim=0),
                    t,
                    torch.cat([text_condition, negative_text_condition, negative_text_condition], dim=0),
                    torch.cat([ip_tokens, negative_ip_tokens, negative_ip_tokens], dim=0),
                    [hint.repeat(3, 1, 1, 1) for hint in hints],
                    control_type.repeat(3, 1),
                ).chunk(3, 0)

                noise_pred = noise_pred2 + (noise_pred1 - noise_pred2) * cfg_image + (noise_pred - noise_pred1) * cfg_text

            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy).prev_sample.to(self.device)

        set_scale(self.unet, 1.0)

        return latents_noisy


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/union_hints.yaml", help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    train_dataset = import_str(opt.dataset.module)(**opt.dataset.args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.training.batch_size, num_workers=opt.training.num_workers, shuffle=True)

    val_dataset = import_str(opt.val_dataset.module)(**opt.val_dataset.args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.training.batch_size, num_workers=2)

    system = UnionTrainer(opt)

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
