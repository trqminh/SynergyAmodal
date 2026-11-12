"""Run SynergyAmodal, with or without a hint ControlNet, on a held-out manifest and save every prediction.

Methods:
  synergy     the SynergyAmodal UNet alone (EMA weights from the config's base_ckpt); the hint is not used
  controlnet  the same UNet plus a trained hint ControlNet, given each record's own hint
  ip_adapter  the same UNet plus a trained reference IP-Adapter, given each record's own reference image
  reference_only  the same UNet with training-free reference_only attention (reference_only.py), own reference image
  union       the same UNet plus the trained union model (trainer_union.py), given box, scribble and reference together

Writes <out>/preds/<sample_id>_rgb.png, <out>/preds/<sample_id>_mask.png and <out>/run.json.
The mask comes from the mask decoder, as in app_controlnet.py. Records whose predictions already
exist are skipped, so an interrupted run resumes with the same noise.
Metrics are computed by scripts/collect_heldout_eval.py in the parent workspace, not here.

Usage:
  python eval_heldout.py --method synergy --config configs/controlnet_box.yaml --out OUT
  python eval_heldout.py --method controlnet --config configs/controlnet_box.yaml --ckpt CKPT --out OUT
  python eval_heldout.py --method ip_adapter --config configs/ip_adapter_reference.yaml --ckpt CKPT \
      --root DATASET_ROOT --manifest manifests/test.provisional.jsonl --out OUT
  python eval_heldout.py --method union --config configs/union_hints.yaml --ckpt CKPT \
      --root DATASET_ROOT --manifest manifests/test.provisional.jsonl --layer 0 --out OUT
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['synergy', 'controlnet', 'ip_adapter', 'reference_only', 'union'], required=True)
    parser.add_argument('--config', required=True, help='a ControlNet or IP-Adapter config; gives base_ckpt and the dataset root')
    parser.add_argument('--ckpt', help='ControlNet or IP-Adapter checkpoint, for --method controlnet / ip_adapter')
    parser.add_argument('--root', help="dataset root, defaults to the config's")
    parser.add_argument('--manifest', default='manifests/locked_holdout.provisional.jsonl', help='relative to the dataset root')
    parser.add_argument('--layer', type=int, help='keep only the records of this build layer (manifests/layers.json), '
                        'e.g. 0 for the first 2000 records of the uCO3D test set')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--cfg_text', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.method in ('controlnet', 'ip_adapter', 'union') and not args.ckpt:
        parser.error(f'--ckpt is required for --method {args.method}')

    # Resolve user paths before moving to this folder, where the config's own paths are relative.
    out = Path(args.out).resolve()
    config = Path(args.config).resolve()
    ckpt = str(Path(args.ckpt).resolve()) if args.ckpt else None
    root = str(Path(args.root).resolve()) if args.root else None
    command = ' '.join([sys.executable] + sys.argv)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from app_controlnet import load_mask_decoder
    from trainer_controlnet import ControlNetTrainer
    from trainer_deocclusion_pseudo import InpaintingTrainer
    from trainer_ip_adapter import IPAdapterTrainer
    from trainer_union import UnionTrainer
    from reference_only import ReferenceOnlyInpainter
    from utils import import_str

    opt = OmegaConf.load(config)
    dataset_args = dict(opt.dataset.args)
    dataset_args['manifest'] = args.manifest
    if root:
        dataset_args['root'] = root
    # The reference loader's train/dev split only applies to its training manifest.
    dataset_args.pop('split', None)
    dataset = import_str(opt.dataset.module)(**dataset_args)
    if args.layer is not None:
        # Same manifest order, so batches (and their noise) match earlier runs made on that layer alone.
        layers = json.loads((Path(dataset_args['root']) / 'manifests/layers.json').read_text())['layers']
        keep = set(next(l for l in layers if l['layer'] == args.layer)['sample_ids'])
        dataset.records = [r for r in dataset.records if r['identity']['sample_id'] in keep]
        assert len(dataset.records) == len(keep), f'{len(keep) - len(dataset.records)} layer records missing from the manifest'
    sample_ids = [r['identity']['sample_id'] for r in dataset.records]

    preds = out / 'preds'
    preds.mkdir(parents=True, exist_ok=True)

    def done(i):
        return all((preds / f'{sample_ids[i]}_{k}.png').exists() for k in ('rgb', 'mask'))

    batches = [list(range(s, min(s + args.batch_size, len(dataset)))) for s in range(0, len(dataset), args.batch_size)]
    todo = [b for b in batches if not all(done(i) for i in b)]
    print(f'Output: {out}\nRecords: {len(dataset)}, batches left: {len(todo)}/{len(batches)}', flush=True)

    device = args.device
    if args.method == 'controlnet':
        system = ControlNetTrainer.load_from_checkpoint(ckpt, map_location='cpu', opt=opt).eval()
    elif args.method == 'ip_adapter':
        system = IPAdapterTrainer.load_from_checkpoint(ckpt, map_location='cpu', opt=opt).eval()
    elif args.method == 'reference_only':
        system = ReferenceOnlyInpainter(opt).eval()
    elif args.method == 'union':
        system = UnionTrainer.load_from_checkpoint(ckpt, map_location='cpu', opt=opt).eval()
    else:
        # The constructor loads the SynergyAmodal EMA UNet from base_ckpt; the new ControlNet is never used.
        system = ControlNetTrainer(opt).eval()
        unet_only = partial(InpaintingTrainer.inference_ddim, system)
        system.inference_ddim = lambda *a, hint=None, **k: unet_only(*a, **k)
    mask_decoder = load_mask_decoder(system.vae, os.path.join(os.path.dirname(opt.base_ckpt), 'vae.ckpt')).bfloat16().to(device)
    system = system.bfloat16().to(device)

    def decode_mask(latent, *_):
        return mask_decoder.decode(latent).sample.sigmoid().round(),

    started = datetime.now(timezone.utc).isoformat()
    for n, indices in enumerate(todo):
        samples = [dataset[i] for i in indices]
        batch = {k: torch.stack([s[k] for s in samples]).to(device) for k in ('image', 'modal_mask')}
        batch['caption'] = [s['caption'] for s in samples]
        if args.method == 'ip_adapter':
            condition = {'clip_image': torch.stack([s['clip_image'] for s in samples]).to(device)}
        elif args.method == 'reference_only':
            condition = {k: torch.stack([s[k] for s in samples]).to(device) for k in ('reference', 'reference_mask')}
        elif args.method == 'union':
            condition = {k: torch.stack([s[k] for s in samples]).to(device) for k in ('box', 'scribble', 'clip_image')}
        elif args.method == 'controlnet':
            condition = {'hint': torch.stack([s['hint'] for s in samples]).to(device)}
        else:
            condition = {}
        # Noise depends only on the batch start, so every method sees the same noise per batch.
        torch.manual_seed(args.seed + indices[0])
        rgb, mask = system.inference(batch, **condition, postprocess_latent=decode_mask,
                                     num_inference_steps=args.steps, cfg_text=args.cfg_text)
        rgb, mask = rgb.float().cpu(), mask.float().cpu()[:, 0] > 0.5
        for j, i in enumerate(indices):
            save_image(rgb[j], preds / f'{sample_ids[i]}_rgb.png')
            Image.fromarray(mask[j].numpy().astype(np.uint8) * 255).save(preds / f'{sample_ids[i]}_mask.png')
        print(f'batch {n + 1}/{len(todo)}', flush=True)

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip())
    run = {
        'method': args.method,
        'checkpoint': ckpt,
        'base_ckpt': str(Path(opt.base_ckpt).resolve()),
        'config': str(config),
        'dataset_root': str(Path(dataset_args['root']).resolve()),
        'manifest': args.manifest,
        'layer': args.layer,
        'records': len(dataset),
        'mask_source': 'mask_decoder',
        'steps': args.steps,
        'cfg_text': args.cfg_text,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'uses_text': True,
        'uses_hint': args.method != 'synergy',
        'git_commit': commit,
        'git_dirty': dirty,
        'command': command,
        'started': started,
        'finished': datetime.now(timezone.utc).isoformat(),
    }
    (out / 'run.json').write_text(json.dumps(run, indent=2))
    print('done', flush=True)


if __name__ == '__main__':
    main()
