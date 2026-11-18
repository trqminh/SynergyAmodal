"""Union model with reliability gates: box, scribble and reference in one model, each weighted per pixel.

- Box and scribble: the union model's ControlNet-Union (trainer_union.py), with a SpatialHintGate per hint.
- Reference: trained reference_only attention (trainer_reference_only.py) in place of the IP-Adapter, with a
  per-token gate in every self-attention layer.

The ControlNet does not read the reference tokens (like the IP-Adapter tokens before), and the reference pass gets no
ControlNet residuals. The model starts from the union model's ControlNet and the reference-only LoRA; the gates
start almost open, so the untrained model is close to the two put together.
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

from gated_union import GatedControlNetUnionModel, GatedReferenceAttnControl, GatedReferenceAttnProcessor, add_reference_gates, set_reliability
from trainer_reference_only import ReferenceOnlyTrainer
from trainer_union import SPATIAL_HINTS, UnionTrainer
from utils import import_str


class GatedUnionTrainer(ReferenceOnlyTrainer):
    control_cls = GatedReferenceAttnControl
    processor_cls = GatedReferenceAttnProcessor

    # Same hint dropout as the union model, over box, scribble and reference.
    sample_hints_to_keep = UnionTrainer.sample_hints_to_keep

    def __init__(self, opt):
        # The frozen SynergyAmodal UNet with reference attention and its LoRA.
        super().__init__(opt)
        self.reference_gates = add_reference_gates(self.unet)

        # Box and scribble. The UNet's attention layers now have LoRA, so the ControlNet takes only the UNet's config;
        # its weights come from the union model.
        self.unet.register_to_config(in_channels=self.unet.conv_in.weight.shape[1])
        self.controlnet = GatedControlNetUnionModel.from_unet(self.unet, load_weights_from_unet=False)
        assert self.controlnet.config.num_control_type == len(SPATIAL_HINTS)

        state_dict = torch.load(opt.init_union_ckpt, map_location='cpu', mmap=True, weights_only=False)['state_dict']
        state_dict = {k[len('controlnet.'):]: v for k, v in state_dict.items() if k.startswith('controlnet.')}
        missing, unexpected = self.controlnet.load_state_dict(state_dict, strict=False)
        assert not unexpected and all(k.startswith('hint_gates.') for k in missing), (missing, unexpected)

        state_dict = torch.load(opt.init_reference_ckpt, map_location='cpu', mmap=True, weights_only=False)['state_dict']
        lora = {k: v for k, v in state_dict.items() if '.attn1.' in k and k.endswith(('.down.weight', '.up.weight'))}
        assert len(lora) == len(self.lora_params), (len(lora), len(self.lora_params))
        self.load_state_dict(lora, strict=False)

    def open_gates(self):
        """Sets every gate to 1, so the model is the union ControlNet and the reference attention combined, ungated."""
        with torch.no_grad():
            for gate in self.controlnet.hint_gates:
                gate.conv_out.weight.zero_()
                gate.conv_out.bias.fill_(30.0)
            for gate in self.reference_gates:
                gate.weight.zero_()
                gate.bias.fill_(30.0)

    def configure_optimizers(self):
        # The ControlNet keeps the union model's learning rate; LoRA and the new gates the reference-only one.
        training = self.opt.training
        controlnet_params = [p for n, p in self.controlnet.named_parameters() if not n.startswith('hint_gates.')]
        gate_params = list(self.controlnet.hint_gates.parameters()) + list(self.reference_gates.parameters())
        return torch.optim.AdamW([
            {'params': controlnet_params, 'lr': training.controlnet_learning_rate},
            {'params': self.lora_params + gate_params, 'lr': training.learning_rate},
        ], weight_decay=training.weight_decay)

    def predict_noise(self, sample, timestep, text_condition, hints, control_type, validity):
        down_residuals, mid_residual = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=text_condition,
            controlnet_cond=hints,
            control_type=control_type,
            control_type_idx=list(range(len(SPATIAL_HINTS))),
            validity=validity,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=text_condition,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
        )[0]

    def end_read(self):
        self.control.mode = None
        self.control.drop = None
        self.control.gates = None
        self.control.clear()

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

            # A dropped box or scribble is a black image with its control type off, as in the union model.
            # A dropped reference attends to itself only.
            keep = self.sample_hints_to_keep(B)
            hints = [batch[name] * keep[:, i, None, None, None] for i, name in enumerate(SPATIAL_HINTS)]
            control_type = keep[:, :len(SPATIAL_HINTS)].float()

            reference_input = self.reference_input(batch['reference'], batch['reference_mask'], image.shape[-2:])

        self.write_reference(reference_input, text_condition)
        self.control.drop = ~keep[:, len(SPATIAL_HINTS)]
        self.control.gates = []
        noise_pred = self.predict_noise(
            torch.cat([input_latents_noisy, image_condition], dim=1), t, text_condition, hints, control_type, batch['valid_canvas']
        )
        reference_gates = torch.cat([gate.mean(dim=(1, 2)) for gate in self.control.gates])
        self.end_read()

        snr = compute_snr(self.scheduler, t)
        weights = torch.stack([snr, 5.0 * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr

        loss = (F.mse_loss(noise_pred, noise, reduction='none').mean([-1, -2, -3]) * weights).mean()

        # With perturbed hints (HintUnionPerturbedDataset): supervise the box and scribble gates toward 0 where the hint
        # is wrong and 1 where it is right, if gate_loss_weight > 0.
        if 'box_wrong' in batch:
            gate_loss, gate_logs = self.spatial_gate_loss(batch, keep)
            gate_loss_weight = self.opt.training.get('gate_loss_weight', 0.0)
            self.log('losses/gate_loss', gate_loss, sync_dist=True)
            if gate_loss_weight > 0:
                loss = loss + gate_loss_weight * gate_loss
        else:
            gate_logs = {}

        self.log('losses/total_loss', loss, prog_bar=True, sync_dist=True)

        # Mean gate of each hint, over the samples that have it. Not synced: a rank may have no sample with a hint and
        # skip its log, and a synced log is a collective that the other ranks would wait on forever.
        with torch.no_grad():
            for i, name in enumerate(SPATIAL_HINTS):
                if keep[:, i].any():
                    self.log(f'gates/{name}', self.controlnet.gates[i][keep[:, i]].mean())
            reference_gates = reference_gates.view(-1, B).mean(0)
            if keep[:, -1].any():
                self.log('gates/reference', reference_gates[keep[:, -1]].mean())
            if 'reference_wrong' in batch:
                wrong = batch['reference_wrong'] > 0.5
                for suffix, which in (('wrong', wrong), ('right', ~wrong)):
                    if (keep[:, -1] & which).any():
                        self.log(f'gates/reference_{suffix}', reference_gates[keep[:, -1] & which].mean())
            for key, value in gate_logs.items():
                self.log(key, value)

        return loss

    def spatial_gate_loss(self, batch, keep):
        """Binary cross-entropy of each box and scribble gate against 1 - <hint>_wrong, at the gate's resolution (a
        latent cell is wrong if any of its pixels is). Only over the samples that have the hint, where the canvas is
        fully valid (there the gate is the plain sigmoid). Wrong and right cells weigh half each, since wrong cells are
        few. Returns the loss averaged over the hints, and the mean gate on wrong and right cells for logging."""
        losses, logs = [], {}
        for i, name in enumerate(SPATIAL_HINTS):
            gate = self.controlnet.gates[i].float()
            size = gate.shape[-2:]
            wrong = F.adaptive_max_pool2d(batch[f'{name}_wrong'].float(), size) > 0.5
            supervised = (F.interpolate(batch['valid_canvas'].float(), size=size, mode='area') > 0.999) & keep[:, i, None, None, None]

            gate = gate.clamp(1e-4, 1 - 1e-4)
            terms = []
            for suffix, cells, bce in (('wrong', supervised & wrong, -(1 - gate).log()), ('right', supervised & ~wrong, -gate.log())):
                if cells.any():
                    terms.append(bce[cells].mean())
                    logs[f'gates/{name}_{suffix}'] = gate[cells].mean().detach()
            if terms:
                losses.append(sum(terms) / len(terms))
        loss = sum(losses) / len(losses) if losses else self.controlnet.gates[0].sum() * 0
        return loss, logs

    def validation_step(self, batch, _):
        if not self.trainer.is_global_zero:
            return

        reference = {'reference': batch['reference'], 'reference_mask': batch['reference_mask']}

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
            self.inference(batch, **reference)[0],
            self.inference(batch, box=batch['box'], scribble=batch['scribble'], **reference)[0],
        ]
        save_dir = Path(self.trainer.default_root_dir) / 'samples'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_image(
            torch.stack(rows, dim=1).flatten(0, 1).float(),
            save_dir / f'step_{self.global_step:06d}.png',
            nrow=len(rows),
        )

    @torch.no_grad()
    def inference_ddim(self, noise, image_condition, text_condition, box=None, scribble=None, reference=None, reference_mask=None, validity=None, reference_scale=1.0, num_inference_steps=25, cfg_image=1, cfg_text=1, strength=1, initial_latent=None, gate_beta=0.0, gates_off=False):
        """box, scribble: (B, 3, H, W) in [0, 1]. reference, reference_mask: (B, 3, h, w) and (B, 1, h, w).
        validity: (B, 1, H, W), where box and scribble are known; None means everywhere.
        gate_beta: offset on every reliability gate logit (> 0 follows the hints more, < 0 less). gates_off: every gate
        is 1, so every hint is followed as given.
        Leave a hint as None to not give it; it is then dropped the same way as in training."""
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
        if validity is None:
            validity = torch.ones(B, 1, H, W, device=self.device, dtype=noise.dtype)

        # The reference is clean, so its tokens are the same at every step: write them once. Without a reference the
        # UNet runs plainly, which is what a dropped reference does in training.
        use_cfg = not (cfg_image == 1 and cfg_text == 1)
        if reference is not None:
            self.write_reference(self.reference_input(reference, reference_mask, (H, W)), text_condition)
            self.control.scale = reference_scale
            if use_cfg:
                # As in the union model: box and scribble stay on for the unconditional rows, and the reference goes
                # with the text, so cfg_text guides text and reference together.
                self.control.drop = torch.arange(3 * B, device=self.device) >= B

        set_reliability(self.controlnet, self.control, gate_beta, gates_off)

        begin_index = int(len(timesteps) * (1 - strength))

        if initial_latent is None or begin_index == 0:
            latents_noisy = noise
        else:
            latents_noisy = self.scheduler.add_noise(initial_latent.repeat(B, 1, 1, 1), noise, timesteps[begin_index][None].repeat(B))

        for t in timesteps[begin_index:]:
            if not use_cfg:
                noise_pred = self.predict_noise(torch.cat([latents_noisy, image_condition], dim=1), t, text_condition, hints, control_type, validity)
            else:
                noise_pred, noise_pred1, noise_pred2 = self.predict_noise(
                    torch.cat([
                        torch.cat([latents_noisy, image_condition], dim=1),
                        torch.cat([latents_noisy, image_condition], dim=1),
                        torch.cat([latents_noisy, torch.zeros_like(image_condition)], dim=1),
                    ], dim=0),
                    t,
                    torch.cat([text_condition, negative_text_condition, negative_text_condition], dim=0),
                    [hint.repeat(3, 1, 1, 1) for hint in hints],
                    control_type.repeat(3, 1),
                    validity.repeat(3, 1, 1, 1),
                ).chunk(3, 0)

                noise_pred = noise_pred2 + (noise_pred1 - noise_pred2) * cfg_image + (noise_pred - noise_pred1) * cfg_text

            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy).prev_sample.to(self.device)

        self.end_read()
        self.control.scale = 1.0
        set_reliability(self.controlnet, self.control)

        return latents_noisy


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/union_gated.yaml", help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    train_dataset = import_str(opt.dataset.module)(**opt.dataset.args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.training.batch_size, num_workers=opt.training.num_workers, shuffle=True)

    val_dataset = import_str(opt.val_dataset.module)(**opt.val_dataset.args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.training.batch_size, num_workers=2)

    system = GatedUnionTrainer(opt)

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
