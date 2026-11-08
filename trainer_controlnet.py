import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import ControlNetModel
from diffusers.training_utils import compute_snr
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import save_image

from trainer_deocclusion_pseudo import InpaintingTrainer
from utils import import_str


class ControlNetTrainer(InpaintingTrainer):
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

        # The trainable copy: same encoder and mid block weights, plus the hint
        # encoder and the zero convolutions.
        self.controlnet = ControlNetModel.from_unet(self.unet)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.controlnet.parameters(), lr=self.opt.training.learning_rate)

    def predict_noise(self, sample, timestep, text_condition, hint):
        down_residuals, mid_residual = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=text_condition,
            controlnet_cond=hint,
            return_dict=False,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=text_condition,
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

        noise_pred = self.predict_noise(
            torch.cat([input_latents_noisy, image_condition], dim=1), t, text_condition, batch['hint']
        )

        snr = compute_snr(self.scheduler, t)
        weights = torch.stack([snr, 5.0 * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr

        loss = (F.mse_loss(noise_pred, noise, reduction='none').mean([-1, -2, -3]) * weights).mean()

        self.log('losses/total_loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, _):
        # Same idea as ControlNet's ImageLogger: save samples to look at during training.
        if not self.trainer.is_global_zero:
            return

        output = self.inference(batch, hint=batch['hint'])[0]

        rows = [
            batch['image'],
            batch['modal_mask'].repeat(1, 3, 1, 1),
            batch['hint'],
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
    def inference_ddim(self, noise, image_condition, text_condition, hint, num_inference_steps=25, cfg_image=1, cfg_text=1, strength=1, initial_latent=None):
        B = noise.shape[0]
        assert self.num_train_timesteps % num_inference_steps == 0
        self.scheduler.config.steps_offset = self.num_train_timesteps // num_inference_steps - 1
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps

        negative_text_condition = self.encode_text(['']).repeat(B, 1, 1)

        begin_index = int(len(timesteps) * (1 - strength))

        if initial_latent is None or begin_index == 0:
            latents_noisy = noise
        else:
            latents_noisy = self.scheduler.add_noise(initial_latent.repeat(B, 1, 1, 1), noise, timesteps[begin_index][None].repeat(B))

        for t in timesteps[begin_index:]:
            if cfg_image == 1 and cfg_text == 1:
                noise_pred = self.predict_noise(torch.cat([latents_noisy, image_condition], dim=1), t, text_condition, hint)
            else:
                # As in ControlNet, the hint stays on for the unconditional passes.
                noise_pred, noise_pred1, noise_pred2 = self.predict_noise(
                    torch.cat([
                        torch.cat([latents_noisy, image_condition], dim=1),
                        torch.cat([latents_noisy, image_condition], dim=1),
                        torch.cat([latents_noisy, torch.zeros_like(image_condition)], dim=1),
                    ], dim=0),
                    t,
                    torch.cat([text_condition, negative_text_condition, negative_text_condition], dim=0),
                    hint.repeat(3, 1, 1, 1),
                ).chunk(3, 0)

                noise_pred = noise_pred2 + (noise_pred1 - noise_pred2) * cfg_image + (noise_pred - noise_pred1) * cfg_text

            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy).prev_sample.to(self.device)

        return latents_noisy


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlnet_scribble.yaml", help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    train_dataset = import_str(opt.dataset.module)(**opt.dataset.args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.training.batch_size, num_workers=opt.training.num_workers, shuffle=True)

    val_dataset = import_str(opt.val_dataset.module)(**opt.val_dataset.args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.training.batch_size, num_workers=2)

    system = ControlNetTrainer(opt)

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
