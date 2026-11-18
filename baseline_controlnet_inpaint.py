"""Training-free baseline: amodal completion with an off-the-shelf SD1.5 ControlNet-inpaint.

The inpaint mask comes from the box hint: box minus the visible mask (only model inputs are used,
not the occluder mask). Variants:
  inpaint           control_v11p_sd15_inpaint, text prompt only
  inpaint_scribble  + control_v11p_sd15_scribble on the scribble hint (Multi-ControlNet)
The amodal mask is SAM (box + visible points) on the completed image, unioned with the visible mask.

  python baseline_controlnet_inpaint.py --sample ../assets/<sample_dir> --out OUT
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CACHE = Path(__file__).resolve().parent / '../all_data/outputs/synergy_amodal/cache/huggingface'
SD15 = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
CN_INPAINT = 'lllyasviel/control_v11p_sd15_inpaint'
CN_SCRIBBLE = 'lllyasviel/control_v11p_sd15_scribble'
SAM = 'facebook/sam-vit-huge'
NEGATIVE = 'blurry, low quality, distorted, extra objects'


def load_sample(d):
    d = Path(d)
    rec = json.loads((d / 'record.json').read_text())
    inp = rec['inputs']
    gray = lambda f: np.array(Image.open(d / f).convert('L')) > 127
    s = dict(
        id=rec['identity']['sample_id'],
        text=inp['text'],
        observed=Image.open(d / inp['observed_rgb']).convert('RGB'),
        visible=gray(inp['visible_mask']),
        box=np.load(d / inp['box'])['values'] > 0,
        scribble=np.load(d / inp['scribble'])['values'] > 0,
    )
    sup = rec['supervision']
    s['amodal_gt'] = gray(sup['amodal_mask'])
    s['hidden_gt'] = gray(sup['hidden_mask'])
    return s


def to_img(mask):
    return Image.fromarray(mask.astype(np.uint8) * 255).convert('RGB')


def dilate(mask, r=1):
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return torch.nn.functional.max_pool2d(t, 2 * r + 1, stride=1, padding=r)[0, 0].numpy() > 0


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def build_pipe(scribble, device):
    from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
    kw = dict(cache_dir=CACHE, torch_dtype=torch.float16, variant='fp16')
    cns = [ControlNetModel.from_pretrained(CN_INPAINT, **kw)]
    if scribble:
        cns.append(ControlNetModel.from_pretrained(CN_SCRIBBLE, **kw))
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        SD15, controlnet=cns if scribble else cns[0], safety_checker=None, requires_safety_checker=False, **kw)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


def inpaint_condition(image, mask):
    # control_v11p_sd15_inpaint convention: masked pixels set to -1.
    x = np.array(image).astype(np.float32) / 255.0
    x[mask] = -1.0
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


@torch.no_grad()
def segment(sam, proc, image, s, device):
    # Prompt: box covering visible + hint box, positive points on the visible region.
    box = bbox(s['visible'] | s['box'])
    ys, xs = np.nonzero(s['visible'])
    idx = np.random.default_rng(0).choice(len(xs), size=min(4, len(xs)), replace=False)
    points = [[[[float(xs[i]), float(ys[i])] for i in idx]]]
    inputs = proc(image, input_boxes=[[box]], input_points=points, input_labels=[[[1] * len(idx)]],
                  return_tensors='pt').to(device)
    out = sam(**inputs, multimask_output=False)
    m = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0]
    return m[0, 0].numpy().astype(bool) | s['visible']


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sample', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--scribble-scales', type=float, nargs='+', default=[0.0, 1.0],
                   help='scribble ControlNet weight; 0 is the text-only inpaint variant')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    s = load_sample(args.sample)
    out = Path(args.out) / s['id']
    out.mkdir(parents=True, exist_ok=True)

    mask = s['box'] & ~s['visible']
    mask_img = to_img(mask)
    scribble_img = to_img(dilate(s['scribble']))
    mask_img.save(out / 'inpaint_mask.png')
    scribble_img.save(out / 'scribble_control.png')

    from transformers import SamModel, SamProcessor
    sam = SamModel.from_pretrained(SAM, cache_dir=CACHE).to(args.device).eval()
    proc = SamProcessor.from_pretrained(SAM, cache_dir=CACHE)

    # One Multi-ControlNet pipe; scribble weight 0 gives the text-only inpaint variant.
    pipe = build_pipe(True, args.device)
    cond = inpaint_condition(s['observed'], mask)
    metrics = {}
    for scale in args.scribble_scales:
        for seed in args.seeds:
            v = f'scribble{scale:g}_seed{seed}'
            result = pipe(
                prompt=s['text'], negative_prompt=NEGATIVE,
                image=s['observed'], mask_image=mask_img,
                control_image=[cond, scribble_img], controlnet_conditioning_scale=[1.0, scale],
                num_inference_steps=args.steps, guidance_scale=7.5, strength=1.0,
                generator=torch.Generator(args.device).manual_seed(seed),
            ).images[0]
            # Keep observed pixels outside the inpaint mask exactly.
            comp = np.where(mask[..., None], np.array(result), np.array(s['observed']))
            comp_img = Image.fromarray(comp)
            amodal = segment(sam, proc, comp_img, s, args.device)
            obj = np.where(amodal[..., None], comp, 255).astype(np.uint8)

            comp_img.save(out / f'{v}_completed.png')
            to_img(amodal).save(out / f'{v}_amodal_mask.png')
            Image.fromarray(obj).save(out / f'{v}_object.png')
            metrics[v] = dict(amodal_iou=iou(amodal, s['amodal_gt']),
                              hidden_iou=iou(amodal & ~s['visible'], s['hidden_gt']))
            print(v, metrics[v], flush=True)

    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
