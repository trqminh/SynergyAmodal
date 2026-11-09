"""Test a hint ControlNet checkpoint on a held-out manifest.

Every record runs four times with the same noise:
  base     the SynergyAmodal UNet alone, no ControlNet
  blank    the ControlNet with an all-black hint
  hint     the ControlNet with the record's own hint
  swapped  the ControlNet with the next record's hint, to see if the output follows the hint

The amodal mask comes from the mask decoder, as in app_controlnet.py. Metrics, per condition:
  amodal_iou    predicted vs. ground-truth amodal mask
  hidden_iou    predicted vs. ground-truth hidden part (amodal minus visible)
  in_box        fraction of the predicted hidden part inside the bounding box of the given hint
                (the record's own hint for base, blank and hint; the next record's for swapped)
  box_fill      fraction of that box covered by the predicted amodal mask

Usage:
  python eval_controlnet.py --config configs/controlnet_box.yaml [--ckpt PATH] [--device cuda:1] [--num 200]
"""
import argparse
import glob
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

CONDITIONS = ['base', 'blank', 'hint', 'swapped']


def hint_box(hint):
    """Bounding box [x0, y0, x1, y1] of a hint map (H, W), or None if it is empty."""
    ys, xs = torch.nonzero(hint > 0.5, as_tuple=True)
    if len(ys) == 0:
        return None
    return [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]


def box_mask(box, shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    if box is not None:
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True
    return mask


def iou(a, b):
    union = (a | b).sum().item()
    return (a & b).sum().item() / union if union else 1.0


def draw_box(image, box, color=(1.0, 0.0, 0.0), width=3):
    """Red outline of a box on a (3, H, W) image."""
    image = image.clone()
    if box is None:
        return image
    x0, y0, x1, y1 = box
    c = torch.tensor(color, dtype=image.dtype)[:, None, None]
    image[:, y0:y0 + width, x0:x1] = c
    image[:, max(y0, y1 - width):y1, x0:x1] = c
    image[:, y0:y1, x0:x0 + width] = c
    image[:, y0:y1, max(x0, x1 - width):x1] = c
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/controlnet_box.yaml')
    parser.add_argument('--ckpt', help='defaults to the newest checkpoint of the experiment')
    parser.add_argument('--manifest', help='defaults to the val_dataset manifest of the config')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num', type=int, default=200, help='records to test, from the start of the manifest')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--cfg_text', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grid_records', type=int, default=16, help='records shown in the comparison grids')
    parser.add_argument('--out', help='defaults to <experiment>/eval/<checkpoint name>')
    args = parser.parse_args()

    # The config paths are relative to this folder, as in training.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from app_controlnet import load_mask_decoder
    from trainer_controlnet import ControlNetTrainer
    from trainer_deocclusion_pseudo import InpaintingTrainer
    from utils import import_str

    opt = OmegaConf.load(args.config)
    experiment = Path(opt.output_dir) / opt.experiment_name
    ckpt = args.ckpt or max(glob.glob(str(experiment / 'lightning_logs/version_*/checkpoints/*.ckpt')), key=os.path.getmtime)
    out = Path(args.out or experiment / 'eval' / Path(ckpt).stem)
    (out / 'grids').mkdir(parents=True, exist_ok=True)
    print(f'Checkpoint: {ckpt}\nOutput: {out}', flush=True)

    dataset_args = dict(opt.val_dataset.args)
    if args.manifest:
        dataset_args['manifest'] = args.manifest
    dataset = import_str(opt.val_dataset.module)(**dataset_args)
    num = min(args.num, len(dataset))

    device = args.device
    system = ControlNetTrainer.load_from_checkpoint(ckpt, map_location='cpu', opt=opt).eval()
    mask_decoder = load_mask_decoder(system.vae, os.path.join(os.path.dirname(opt.base_ckpt), 'vae.ckpt')).bfloat16().to(device)
    system = system.bfloat16().to(device)

    controlnet_ddim = system.inference_ddim
    unet_only_ddim = partial(InpaintingTrainer.inference_ddim, system)

    def decode_mask(latent, *args):
        return mask_decoder.decode(latent).sample.sigmoid().round(),

    def run(batch, condition, hint):
        # Same noise for every condition of a batch.
        torch.manual_seed(args.seed + batch['index'][0])
        if condition == 'base':
            system.inference_ddim = lambda *a, hint=None, **k: unet_only_ddim(*a, **k)
        else:
            system.inference_ddim = controlnet_ddim
        rgb, mask = system.inference(batch, hint=hint, postprocess_latent=decode_mask,
                                     num_inference_steps=args.steps, cfg_text=args.cfg_text)
        return rgb.float().cpu(), mask.float().cpu()[:, 0] > 0.5

    rows = []
    grid = []
    for start in range(0, num, args.batch_size):
        indices = list(range(start, min(start + args.batch_size, num)))
        samples = [dataset[i] for i in indices]
        batch = {k: torch.stack([s[k] for s in samples]).to(device) for k in ('image', 'modal_mask')}
        batch['caption'] = [s['caption'] for s in samples]
        batch['index'] = indices
        own = torch.stack([s['hint'] for s in samples])
        # The next record's hint, wrapping within the manifest so the last batch has one too.
        swapped = torch.stack([dataset[(i + 1) % len(dataset)]['hint'] for i in indices])
        hints = {'base': own, 'blank': torch.zeros_like(own), 'hint': own, 'swapped': swapped}

        outputs = {c: run(batch, c, hints[c].to(device)) for c in CONDITIONS}

        for j, (i, s) in enumerate(zip(indices, samples)):
            visible = s['modal_mask'][0] > 0.5
            amodal = s['amodal_mask'][0] > 0.5
            hidden = amodal & ~visible
            row = {'index': i, 'caption': s['caption']}
            for c in CONDITIONS:
                pred = outputs[c][1][j]
                pred_hidden = pred & ~visible
                box = hint_box(hints[c if c != 'blank' else 'hint'][j][0])
                in_box = box_mask(box, pred.shape)
                row[c] = {
                    'amodal_iou': iou(pred, amodal),
                    'hidden_iou': iou(pred_hidden, hidden),
                    'in_box': (pred_hidden & in_box).sum().item() / max(1, pred_hidden.sum().item()),
                    'box_fill': (pred & in_box).sum().item() / max(1, in_box.sum().item()),
                }
            rows.append(row)

            if len(grid) < args.grid_records:
                own_box = hint_box(own[j][0])
                swapped_box = hint_box(swapped[j][0])
                grid.append([
                    draw_box(s['image'], own_box),
                    s['amodal_image'],
                    outputs['base'][0][j],
                    outputs['blank'][0][j],
                    draw_box(outputs['hint'][0][j], own_box),
                    draw_box(outputs['swapped'][0][j], swapped_box, color=(0.0, 0.4, 1.0)),
                ])
        print(f'{len(rows)}/{num}', flush=True)

    # Columns: input with its box (red), ground truth, base UNet, blank hint, own box (red), swapped box (blue).
    for k in range(0, len(grid), 8):
        chunk = grid[k:k + 8]
        save_image(torch.stack([image for record in chunk for image in record]), out / 'grids' / f'grid_{k // 8:02d}.png', nrow=6, padding=4, pad_value=0.5)

    summary = {c: {m: float(np.mean([r[c][m] for r in rows])) for m in rows[0][c]} for c in CONDITIONS}
    result = {'checkpoint': ckpt, 'manifest': dataset_args['manifest'], 'records': len(rows), 'steps': args.steps,
              'cfg_text': args.cfg_text, 'seed': args.seed, 'summary': summary}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    with open(out / 'per_record.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    print(f'\n{"condition":<10}' + ''.join(f'{m:>12}' for m in summary['base']))
    for c in CONDITIONS:
        print(f'{c:<10}' + ''.join(f'{v:>12.3f}' for v in summary[c].values()))


if __name__ == '__main__':
    main()
