import argparse
import base64
import copy
import glob
import io
import os

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import to_tensor

from utils import crop_and_resize, put_back

SIZE = 512
# Refinement crops from this canvas: the same layout as the 512 one, twice the size.
FULL_SIZE = 1024

# A plain canvas, shown 1:1 over the image, so a canvas pixel is a hint pixel.
SKETCH_SCRIPT = """
<script>
function sketchRedraw(c) {
    const g = c.getContext('2d');
    g.clearRect(0, 0, c.width, c.height);
    g.strokeStyle = '#fff';
    g.lineWidth = 2;
    g.lineCap = g.lineJoin = 'round';
    for (const stroke of c.strokes || []) {
        g.beginPath();
        g.moveTo(...stroke[0]);
        for (const point of stroke) g.lineTo(...point);
        g.stroke();
    }
}
function sketchPoint(e) {
    const c = e.target, r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height];
}
document.addEventListener('pointerdown', e => {
    if (e.target.id !== 'sketch') return;
    e.target.setPointerCapture(e.pointerId);
    e.target.strokes = [...(e.target.strokes || []), [sketchPoint(e)]];
    e.target.drawing = true;
    sketchRedraw(e.target);
});
document.addEventListener('pointermove', e => {
    if (e.target.id !== 'sketch' || !e.target.drawing) return;
    e.target.strokes.at(-1).push(sketchPoint(e));
    sketchRedraw(e.target);
});
document.addEventListener('pointerup', e => {
    if (e.target.id === 'sketch') e.target.drawing = false;
});
</script>
"""
UNDO_JS = "() => { const c = document.getElementById('sketch'); (c.strokes || []).pop(); sketchRedraw(c); }"
CLEAR_JS = "() => { const c = document.getElementById('sketch'); c.strokes = []; sketchRedraw(c); }"
# Sends the canvas to Python in place of the hidden sketch textbox.
READ_SKETCH_JS = "(state, sketch, caption, cfg_text, strength, seed) => [state, document.getElementById('sketch').toDataURL(), caption, cfg_text, strength, seed]"

system = None
mask_decoder = None
zim = None
examples = None
device = None


# Same algorithm as thin() in amodal_data/operations/scribble.py, without scipy.
def thinning_luts():
    # Neighbour bits: N=1, NE=2, E=4, SE=8, S=16, SW=32, W=64, NW=128.
    first, second = np.zeros(256, bool), np.zeros(256, bool)
    for code in range(256):
        p = [(code >> i) & 1 for i in range(8)]
        count = sum(p)
        transitions = sum(p[i] == 0 and p[(i + 1) % 8] == 1 for i in range(8))
        if 2 <= count <= 6 and transitions == 1:
            n, e, s, w = p[0], p[2], p[4], p[6]
            first[code] = n * e * s == 0 and e * s * w == 0
            second[code] = n * e * w == 0 and n * s * w == 0
    return first, second


THIN_FIRST, THIN_SECOND = thinning_luts()
THIN_WEIGHTS = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], np.float32)


def thin(mask):
    out = mask.astype(bool).copy()
    while True:
        changed = False
        for lut in (THIN_FIRST, THIN_SECOND):
            codes = cv2.filter2D(out.astype(np.float32), -1, THIN_WEIGHTS, borderType=cv2.BORDER_CONSTANT).astype(np.int32)
            remove = out & lut[codes]
            if remove.any():
                out[remove] = False
                changed = True
        if not changed:
            return out


def scribble_to_lines(ink, visible, valid):
    # Same as normalize_user_scribble: one pixel wide, and nothing on the visible
    # object, because training hints only cover the occluded part.
    return thin(ink) & valid & ~visible


def letterbox(image, size=SIZE):
    # Same as amodal_data/source.py: the full image, centered on a black canvas.
    W, H = image.size
    scale = min(size / W, size / H)
    nw, nh = max(1, round(W * scale)), max(1, round(H * scale))
    left, top = (size - nw) // 2, (size - nh) // 2

    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[top:top + nh, left:left + nw] = np.array(image.convert("RGB").resize((nw, nh), Image.Resampling.BICUBIC))
    valid = np.zeros((size, size), bool)
    valid[top:top + nh, left:left + nw] = True
    return rgb, valid


def resize_mask(mask, size):
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)) > 0


def crop_to_valid(image, valid):
    rows, cols = np.where(valid.any(1))[0], np.where(valid.any(0))[0]
    return image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def to_uint8(image):
    return (image.float().permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def to_hint(lines):
    # White lines on black in [0, 1], repeated to 3 channels, like the training hint.
    return to_tensor(lines.astype(np.float32))[None].repeat(1, 3, 1, 1).to(device)


def highlight(state):
    if state["visible"] is None:
        return state["rgb"]
    return np.where(state["visible"][..., None], state["rgb"], (state["rgb"] * 0.3).astype(np.uint8))


def blank_canvas(state):
    # A new canvas element starts with no strokes.
    buffer = io.BytesIO()
    background = np.zeros((SIZE, SIZE, 3), np.uint8) if state is None else highlight(state)
    Image.fromarray(background).save(buffer, format="PNG")
    image = base64.b64encode(buffer.getvalue()).decode()
    return (
        f'<div style="position:relative;width:{SIZE}px;height:{SIZE}px">'
        f'<img src="data:image/png;base64,{image}" draggable="false" style="position:absolute;inset:0;width:{SIZE}px;height:{SIZE}px;max-width:none;margin:0">'
        f'<canvas id="sketch" width="{SIZE}" height="{SIZE}" style="position:absolute;inset:0;width:{SIZE}px;height:{SIZE}px;cursor:crosshair;touch-action:none"></canvas>'
        "</div>"
    )


def load_image(image):
    rgb, valid = letterbox(image)
    full_rgb, full_valid = letterbox(image, FULL_SIZE)
    state = {
        "rgb": rgb,
        "valid": valid,
        "visible": None,
        "full_rgb": full_rgb,
        "full_valid": full_valid,
        "full_visible": None,
        "points": [],
        "labels": [],
    }
    return state, highlight(state), blank_canvas(state), None


def load_example(index):
    sample = examples[int(index)]
    record = examples.records[int(index)]
    rgb = to_uint8(sample["image"])
    valid = examples.load_mask(record["inputs"]["valid_canvas"])[0].numpy() > 0
    visible = sample["modal_mask"][0].numpy() > 0
    state = {
        "rgb": rgb,
        "valid": valid,
        "visible": visible,
        # The dataset only stores 512, so refining a dev example upsamples it.
        "full_rgb": np.array(Image.fromarray(rgb).resize((FULL_SIZE, FULL_SIZE), Image.Resampling.BICUBIC)),
        "full_valid": resize_mask(valid, FULL_SIZE),
        "full_visible": resize_mask(visible, FULL_SIZE),
        "points": [],
        "labels": [],
    }
    training_hint = (sample["hint"][0].numpy() * 255).astype(np.uint8)
    return state, highlight(state), blank_canvas(state), training_hint, sample["caption"]


def select_object(state, point_type, evt: gr.SelectData):
    if state is None:
        raise gr.Error("Upload an image first.")
    if not state["points"]:
        zim.set_image(state["full_rgb"])
    # Clicks are on the 512 view, ZIM sees the full canvas.
    state["points"].append([evt.index[0] * FULL_SIZE // SIZE, evt.index[1] * FULL_SIZE // SIZE])
    state["labels"].append(1 if point_type == "Object" else 0)
    masks, _, _ = zim.predict(
        point_coords=np.array(state["points"]),
        point_labels=np.array(state["labels"]),
        multimask_output=False,
    )
    state["full_visible"] = (masks[0] > 0.5) & state["full_valid"]
    state["visible"] = resize_mask(state["full_visible"], SIZE) & state["valid"]
    return state, highlight(state), blank_canvas(state)


def clear_selection(state):
    if state is None:
        raise gr.Error("Upload an image first.")
    state.update(visible=None, full_visible=None, points=[], labels=[])
    return state, highlight(state), blank_canvas(state)


def decode_mask(latent, *args):
    return mask_decoder.decode(latent).sample.sigmoid().round(),


@torch.no_grad()
def generate(state, sketch, caption, cfg_text, strength, seed):
    if state is None or state["visible"] is None:
        raise gr.Error("Upload an image and select the object first.")

    # Only where the pen touched counts. Thinning takes care of the stroke width.
    ink = np.array(Image.open(io.BytesIO(base64.b64decode(sketch.split(",", 1)[1]))).convert("RGBA"))[..., 3] > 0

    lines = scribble_to_lines(ink, state["visible"], state["valid"])
    if ink.any() and not lines.any():
        raise gr.Error("Draw in the occluded part, outside the highlighted object.")
    # An empty canvas sends an all-black hint through the ControlNet.
    sketched = lines.any()

    torch.manual_seed(int(seed))

    # Global pass, on the same canvas as training.
    batch = {
        "image": to_tensor(state["rgb"])[None].to(device),
        "modal_mask": to_tensor(state["visible"].astype(np.float32))[None].to(device),
        "caption": [caption],
    }
    rgb, mask = system.inference(batch, hint=to_hint(lines), postprocess_latent=decode_mask, cfg_text=cfg_text)
    rgba = torch.cat([rgb, mask], dim=1).float().cpu()
    rgba = F.interpolate(rgba, size=(FULL_SIZE, FULL_SIZE), mode="bilinear", align_corners=False)[0]
    rgb, mask = rgba.split([3, 1], dim=0)
    mask = (mask > 0.5).float()
    if mask.sum() == 0:
        raise gr.Error("The mask decoder found no object. Try another seed.")
    rgb = rgb * mask + 1 * (1 - mask)
    global_rgba = torch.cat([rgb, mask], dim=0)

    # Refinement, as in app.py: crop around the predicted object and denoise the crop.
    # The hint is cropped with it. Resizing blurs a line, so it is thinned again.
    full_rgb = to_tensor(state["full_rgb"])
    full_visible = to_tensor(state["full_visible"].astype(np.float32))
    full_lines = to_tensor(resize_mask(lines, FULL_SIZE).astype(np.float32))
    crop, bbox = crop_and_resize(torch.cat([full_rgb, full_visible, rgb, full_lines], dim=0), mask[0], min_size=256, padding_value=0)
    crop_rgb, crop_visible, crop_output, crop_ink = crop.split([3, 1, 3, 1], dim=0)
    crop_visible = (crop_visible > 0.5).float()
    crop_lines = scribble_to_lines(crop_ink[0].numpy() > 0, crop_visible[0].numpy() > 0, np.ones((SIZE, SIZE), bool))

    if crop_lines.any() or not sketched:
        batch = {
            "image": crop_rgb[None].to(device),
            "modal_mask": crop_visible[None].to(device),
            "caption": [caption],
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            initial_latent = system.latent_scale_fn(system.vae.encode(crop_output[None].to(device) * 2 - 1)).float()
        rgb, mask = system.inference(
            batch,
            hint=to_hint(crop_lines),
            postprocess_latent=decode_mask,
            cfg_text=cfg_text,
            strength=strength,
            initial_latent=initial_latent,
        )
        rgba = put_back(torch.cat([rgb, mask], dim=1)[0].float().cpu(), FULL_SIZE, FULL_SIZE, bbox)
        rgba[3] = (rgba[3] > 0.5).float()
    else:
        gr.Warning("No sketch inside the predicted object, so refinement was skipped.")
        rgba = global_rgba

    valid = state["full_valid"]
    return (
        crop_to_valid(to_uint8(rgba), valid),
        crop_to_valid(to_uint8(global_rgba[:3]), valid),
        crop_to_valid(to_uint8(rgba[3:]).squeeze(-1), valid),
        lines.astype(np.uint8) * 255,
        crop_lines.astype(np.uint8) * 255,
    )


def build_app():
    with gr.Blocks(title="SynergyAmodal ControlNet", head=SKETCH_SCRIPT) as app:
        gr.Markdown(
            "# SynergyAmodal ControlNet\n"
            "**1.** Upload an image or load a dev example. **2.** Click the visible part of the object. "
            "**3.** Sketch the hidden part, then generate. With no sketch, the ControlNet gets a blank hint. "
            "Selecting the object again clears the sketch."
        )
        state = gr.State()
        sketch = gr.Textbox(visible=False)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload image", sources=["upload", "clipboard"], height=SIZE)
                example_index = gr.Slider(0, len(examples) - 1, value=0, step=1, label="Dev example")
                example_button = gr.Button("Load dev example")
            with gr.Column():
                selection = gr.Image(label="Click the visible object", interactive=False, height=SIZE, width=SIZE)
                point_type = gr.Radio(["Object", "Not object"], value="Object", label="Click marks")
                clear_button = gr.Button("Clear selection")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Sketch the hidden part (optional)**")
                drawing = gr.HTML(blank_canvas(None))
                with gr.Row():
                    undo_button = gr.Button("Undo stroke")
                    clear_sketch_button = gr.Button("Clear sketch")
                caption = gr.Textbox(label="Caption of the whole object")
                cfg_text = gr.Slider(1, 10, value=1, label="Text CFG")
                strength = gr.Slider(0.1, 1, value=1, label="Refinement noise strength")
                seed = gr.Number(value=0, precision=0, label="Seed")
                generate_button = gr.Button("Generate", variant="primary")
            with gr.Column():
                output = gr.Image(label="Refined output (RGBA)", interactive=False, height=SIZE, width=SIZE, image_mode="RGBA", format="png")
                with gr.Row():
                    global_output = gr.Image(label="Global pass", interactive=False, height=256, width=256, format="png")
                    mask = gr.Image(label="Mask", interactive=False, height=256, width=256, format="png")
                with gr.Row():
                    hint = gr.Image(label="Your hint, as the model sees it", interactive=False, height=256, width=256, format="png")
                    crop_hint = gr.Image(label="Hint for the refinement crop", interactive=False, height=256, width=256, format="png")
                training_hint = gr.Image(label="Training hint of the dev example", interactive=False, height=256, width=256, format="png")

        image_input.upload(load_image, [image_input], [state, selection, drawing, training_hint])
        example_button.click(load_example, [example_index], [state, selection, drawing, training_hint, caption])
        selection.select(select_object, [state, point_type], [state, selection, drawing])
        clear_button.click(clear_selection, [state], [state, selection, drawing])
        undo_button.click(None, js=UNDO_JS)
        clear_sketch_button.click(None, js=CLEAR_JS)
        generate_button.click(
            generate,
            [state, sketch, caption, cfg_text, strength, seed],
            [output, global_output, mask, hint, crop_hint],
            js=READ_SKETCH_JS,
        )

    return app


def load_mask_decoder(vae, path):
    # The same model as RGBADecoderTrainer: the SD VAE with a one-channel decoder output,
    # using the EMA weights like app.py does.
    vae = copy.deepcopy(vae)
    vae.decoder.conv_out = nn.Conv2d(vae.decoder.conv_out.in_channels, 1, kernel_size=3, padding=1)
    state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    vae.load_state_dict({k[len("vae."):]: v for k, v in state_dict.items() if k.startswith("vae.")})
    ema = "model_ema.models.0."
    vae.decoder.load_state_dict({k[len(ema):]: v for k, v in state_dict.items() if k.startswith(ema)})
    del vae.encoder
    return vae.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlnet_scribble.yaml")
    parser.add_argument("--ckpt", help="defaults to the newest checkpoint from training")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    # The config paths are relative to this folder, as in training.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from datasets.hint_scribble import HintScribbleDataset
    from trainer_controlnet import ControlNetTrainer
    from zim import ZimPredictor, zim_model_registry

    opt = OmegaConf.load(args.config)
    ckpt = args.ckpt or max(
        glob.glob(os.path.join(opt.output_dir, opt.experiment_name, "lightning_logs/version_*/checkpoints/*.ckpt")),
        key=os.path.getmtime,
    )
    print(f"Loading {ckpt}")

    device = args.device
    # bf16 weights, because inference already runs under bf16 autocast and this halves the memory.
    system = ControlNetTrainer.load_from_checkpoint(ckpt, map_location="cpu", opt=opt).eval()
    mask_decoder = load_mask_decoder(system.vae, os.path.join(os.path.dirname(opt.base_ckpt), "vae.ckpt")).bfloat16().to(device)
    system = system.bfloat16().to(device)

    # The text encoder is fast enough on the CPU, which saves about 700 MB on the GPU.
    system.text_encoder.float().cpu()

    def encode_text_on_cpu(text):
        tokens = system.tokenizer(text, padding="max_length", max_length=system.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        return system.text_encoder(tokens.input_ids)[0].to(device)

    system.encode_text = encode_text_on_cpu

    # ZIM runs on the CPU, so selecting an object uses no GPU memory.
    zim = ZimPredictor(zim_model_registry["vit_l"](checkpoint=os.path.join(os.path.dirname(opt.base_ckpt), "zim_ckpts/zim_vit_l_2092")).eval())

    examples = HintScribbleDataset(opt.val_dataset.args.root, opt.val_dataset.args.manifest)

    build_app().launch(server_name=args.host, server_port=args.port)
