import argparse
import base64
import copy
import glob
import io
import json
import os
import types

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

SCRIBBLE, BOX, REFERENCE = "Scribble", "Box", "Reference"
# Separate: one model per hint type. Union: one model for any subset of the three hints (trainer_union.py).
SEPARATE, UNION = "Separate models", "Union"
SYNERGY_EXAMPLES, UCO3D_EXAMPLES = "SynergyAmodal dev (scribble, box)", "uCO3D dev (scribble, box, reference)"

# A plain canvas, shown 1:1 over the image, so a canvas pixel is a hint pixel.
# The canvas keeps both a scribble and a box; only the one for the current hint type is shown and drawn on.
SKETCH_SCRIPT = """
<script>
window.hintMode = 'Scribble';
function drawStrokes(c, g) {
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
function sketchRedraw(c) {
    const g = c.getContext('2d');
    g.clearRect(0, 0, c.width, c.height);
    if (window.hintMode === 'Box') {
        if (!c.box) return;
        const [x0, y0, x1, y1] = c.box;
        g.fillStyle = 'rgba(255, 255, 255, 0.3)';
        g.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
        g.strokeStyle = '#fff';
        g.lineWidth = 2;
        g.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    } else {
        drawStrokes(c, g);
    }
}
function sketchPoint(e) {
    const c = e.target, r = c.getBoundingClientRect();
    return [(e.clientX - r.left) * c.width / r.width, (e.clientY - r.top) * c.height / r.height];
}
// The strokes alone, on a transparent canvas, so a box drawn in the other mode never becomes ink.
function sketchInk(c) {
    const ink = document.createElement('canvas');
    ink.width = c.width;
    ink.height = c.height;
    drawStrokes(c, ink.getContext('2d'));
    return ink.toDataURL();
}
document.addEventListener('pointerdown', e => {
    if (e.target.id !== 'sketch') return;
    e.target.setPointerCapture(e.pointerId);
    const p = sketchPoint(e);
    if (window.hintMode === 'Box') e.target.box = [...p, ...p];
    else e.target.strokes = [...(e.target.strokes || []), [p]];
    e.target.drawing = true;
    sketchRedraw(e.target);
});
document.addEventListener('pointermove', e => {
    if (e.target.id !== 'sketch' || !e.target.drawing) return;
    const p = sketchPoint(e);
    if (window.hintMode === 'Box') e.target.box = [e.target.box[0], e.target.box[1], ...p];
    else e.target.strokes.at(-1).push(p);
    sketchRedraw(e.target);
});
document.addEventListener('pointerup', e => {
    if (e.target.id === 'sketch') e.target.drawing = false;
});
</script>
"""
UNDO_JS = """() => {
    const c = document.getElementById('sketch');
    if (window.hintMode === 'Box') c.box = null; else (c.strokes || []).pop();
    sketchRedraw(c);
}"""
CLEAR_JS = """() => {
    const c = document.getElementById('sketch');
    if (window.hintMode === 'Box') c.box = null; else c.strokes = [];
    sketchRedraw(c);
}"""
SET_MODE_JS = "(mode) => { window.hintMode = mode; const c = document.getElementById('sketch'); if (c) sketchRedraw(c); }"
# Sends the strokes and the box to Python in place of the hidden sketch textbox. Gradio passes the outputs
# after the inputs, so the inputs are named one by one and exactly those are returned.
READ_SKETCH_JS = """(state, sketch, model, hints, reference, caption, cfg_text, strength, seed) => {
    const c = document.getElementById('sketch');
    const sketchJson = JSON.stringify({ink: sketchInk(c), box: c.box || null});
    return [state, sketchJson, model, hints, reference, caption, cfg_text, strength, seed];
}"""

system = None
# Every model runs on the same frozen UNet. Keyed by None (the plain UNet), a hint type, or UNION.
models = {}
image_processor = None
mask_decoder = None
zim = None
# Keyed by SYNERGY_EXAMPLES and UCO3D_EXAMPLES.
example_sets = {}
# The box dataset's dev records, keyed by sample ID without the s_/b_ prefix.
box_examples = None
box_example_index = {}
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


def box_to_map(box, valid):
    # Same as box_map in amodal_data/operations/box.py: the box filled, clipped to the canvas,
    # and empty off the valid canvas. Unlike a scribble, it may cover the visible object.
    h, w = valid.shape
    xs, ys = sorted([box[0], box[2]]), sorted([box[1], box[3]])
    x0, x1 = max(0, round(xs[0])), min(w, round(xs[1]))
    y0, y1 = max(0, round(ys[0])), min(h, round(ys[1]))
    out = np.zeros((h, w), bool)
    out[y0:max(y0, y1), x0:max(x0, x1)] = True
    return out & valid


def tight_box_map(mask):
    # The filled box around a mask, to make a resized box a clean rectangle again.
    out = np.zeros_like(mask, dtype=bool)
    if mask.any():
        ys, xs = np.nonzero(mask)
        out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return out


def use_model(kind):
    # Routes inference through the model for this hint type, the union model, or the plain UNet for None,
    # using that model's trainer's inference_ddim and predict_noise on the shared system.
    model = models[kind]
    system.inference_ddim = types.MethodType(model["cls"].inference_ddim, system)
    if hasattr(model["cls"], "predict_noise"):
        system.predict_noise = types.MethodType(model["cls"].predict_noise, system)
    for name in ("controlnet", "image_proj_model"):
        if name in model:
            setattr(system, name, model[name])
    # IP-Adapter's processors read the image tokens off the end of the text tokens, so each model brings its own.
    # set_attn_processor pops from the dict it gets, hence the copy.
    system.unet.set_attn_processor(dict(model["processors"]))


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


def hint_view(box, scribble, drawn):
    # The box in gray and the scribble in white, for the hints the model got.
    view = np.zeros(box.shape, np.uint8)
    if BOX in drawn:
        view[box] = 100
    if SCRIBBLE in drawn:
        view[scribble] = 255
    return view


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
        "training_hints": {},
    }
    return state, highlight(state), blank_canvas(state), None


def show_training_hint(state, hint_type):
    if state is None:
        return None
    return state["training_hints"].get(hint_type)


def load_example(example_set, index, hint_type):
    examples = example_sets[example_set]
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
        "training_hints": {},
    }
    reference = None
    if example_set == UCO3D_EXAMPLES:
        # Every uCO3D record has all three hints.
        state["training_hints"][SCRIBBLE] = (sample["scribble"][0].numpy() * 255).astype(np.uint8)
        state["training_hints"][BOX] = (sample["box"][0].numpy() * 255).astype(np.uint8)
        reference = Image.open(examples.root / record["inputs"]["reference"]).convert("RGB")
    else:
        # The box dataset has the same observation for this object, so both training hints can be shown.
        state["training_hints"][SCRIBBLE] = (sample["hint"][0].numpy() * 255).astype(np.uint8)
        box_index = box_example_index.get(record["identity"]["sample_id"].split("_", 1)[1])
        if box_index is not None:
            state["training_hints"][BOX] = (box_examples[box_index]["hint"][0].numpy() * 255).astype(np.uint8)
    return state, highlight(state), blank_canvas(state), show_training_hint(state, hint_type), sample["caption"], reference


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
def generate(state, sketch, model, hints, reference, caption, cfg_text, strength, seed):
    if state is None or state["visible"] is None:
        raise gr.Error("Upload an image and select the object first.")

    hints = set(hints)
    sketch = json.loads(sketch)
    box = np.zeros_like(state["valid"]) if sketch["box"] is None else box_to_map(sketch["box"], state["valid"])
    # Only where the pen touched counts. Thinning takes care of the stroke width.
    ink = np.array(Image.open(io.BytesIO(base64.b64decode(sketch["ink"].split(",", 1)[1]))).convert("RGBA"))[..., 3] > 0
    scribble = scribble_to_lines(ink, state["visible"], state["valid"])
    if SCRIBBLE in hints and ink.any() and not scribble.any():
        raise gr.Error("Draw in the occluded part, outside the highlighted object.")

    clip_image = None
    if REFERENCE in hints:
        if reference is None:
            raise gr.Error("Upload a reference image, or load a uCO3D example.")
        clip_image = image_processor(images=reference.convert("RGB"), return_tensors="pt").pixel_values.to(device)

    if model == UNION:
        # The union model was trained with any subset of the hints, so an empty box or scribble is left out.
        for name, hint_map in ((BOX, box), (SCRIBBLE, scribble)):
            if name in hints and not hint_map.any():
                hints.discard(name)
                gr.Info(f"No {name.lower()} drawn, so the union model ran without one.")
        kind = UNION
    else:
        if len(hints) > 1:
            raise gr.Error("The separate models take one hint at a time. Pick the union model to combine hints.")
        kind = next(iter(hints), None)
        # The box ControlNet never saw an empty hint and reads one as "nothing is hidden",
        # so with no box the plain SynergyAmodal UNet runs instead.
        # An empty scribble canvas sends an all-black hint through the scribble ControlNet.
        if kind == BOX and not box.any():
            kind = None
            gr.Info("No box drawn, so the plain SynergyAmodal UNet was used.")
    use_model(kind)

    # The box and scribble the model gets, which the refinement crop has to keep.
    drawn = [name for name, hint_map in ((BOX, box), (SCRIBBLE, scribble)) if name in hints and kind in (name, UNION) and hint_map.any()]

    def hint_inputs(box, scribble):
        if kind == UNION:
            return {
                "box": to_hint(box) if BOX in hints else None,
                "scribble": to_hint(scribble) if SCRIBBLE in hints else None,
                "clip_image": clip_image,
            }
        if kind == SCRIBBLE:
            return {"hint": to_hint(scribble)}
        if kind == BOX:
            return {"hint": to_hint(box)}
        if kind == REFERENCE:
            return {"clip_image": clip_image}
        return {}

    torch.manual_seed(int(seed))

    # Global pass, on the same canvas as training.
    batch = {
        "image": to_tensor(state["rgb"])[None].to(device),
        "modal_mask": to_tensor(state["visible"].astype(np.float32))[None].to(device),
        "caption": [caption],
    }
    rgb, mask = system.inference(batch, postprocess_latent=decode_mask, cfg_text=cfg_text, **hint_inputs(box, scribble))
    rgba = torch.cat([rgb, mask], dim=1).float().cpu()
    rgba = F.interpolate(rgba, size=(FULL_SIZE, FULL_SIZE), mode="bilinear", align_corners=False)[0]
    rgb, mask = rgba.split([3, 1], dim=0)
    mask = (mask > 0.5).float()
    if mask.sum() == 0:
        raise gr.Error("The mask decoder found no object. Try another seed.")
    rgb = rgb * mask + 1 * (1 - mask)
    global_rgba = torch.cat([rgb, mask], dim=0)

    # Refinement, as in app.py: crop around the predicted object and denoise the crop.
    # The box and scribble are cropped with it. Resizing blurs a line, so it is thinned again,
    # and a box is made a clean rectangle again. The reference is the same for the crop.
    full_rgb = to_tensor(state["full_rgb"])
    full_visible = to_tensor(state["full_visible"].astype(np.float32))
    full_box = to_tensor(resize_mask(box, FULL_SIZE).astype(np.float32))
    full_scribble = to_tensor(resize_mask(scribble, FULL_SIZE).astype(np.float32))
    crop, bbox = crop_and_resize(torch.cat([full_rgb, full_visible, rgb, full_box, full_scribble], dim=0), mask[0], min_size=256, padding_value=0)
    crop_rgb, crop_visible, crop_output, crop_box, crop_scribble = crop.split([3, 1, 3, 1, 1], dim=0)
    crop_visible = (crop_visible > 0.5).float()
    crop_box = tight_box_map(crop_box[0].numpy() > 0.5)
    crop_scribble = scribble_to_lines(crop_scribble[0].numpy() > 0, crop_visible[0].numpy() > 0, np.ones((SIZE, SIZE), bool))

    lost = [name for name in drawn if not {BOX: crop_box, SCRIBBLE: crop_scribble}[name].any()]
    if not lost:
        batch = {
            "image": crop_rgb[None].to(device),
            "modal_mask": crop_visible[None].to(device),
            "caption": [caption],
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            initial_latent = system.latent_scale_fn(system.vae.encode(crop_output[None].to(device) * 2 - 1)).float()
        rgb, mask = system.inference(
            batch,
            postprocess_latent=decode_mask,
            cfg_text=cfg_text,
            strength=strength,
            initial_latent=initial_latent,
            **hint_inputs(crop_box, crop_scribble),
        )
        rgba = put_back(torch.cat([rgb, mask], dim=1)[0].float().cpu(), FULL_SIZE, FULL_SIZE, bbox)
        rgba[3] = (rgba[3] > 0.5).float()
    else:
        gr.Warning(f"No {' or '.join(name.lower() for name in lost)} inside the refinement crop, so refinement was skipped.")
        rgba = global_rgba

    valid = state["full_valid"]
    return (
        crop_to_valid(to_uint8(global_rgba), valid),
        crop_to_valid(to_uint8(rgba), valid),
        crop_to_valid(to_uint8(global_rgba[3:]).squeeze(-1), valid),
        hint_view(box, scribble, drawn),
        hint_view(crop_box, crop_scribble, drawn),
    )


def build_app():
    with gr.Blocks(title="SynergyAmodal ControlNet", head=SKETCH_SCRIPT) as app:
        gr.Markdown(
            "# SynergyAmodal ControlNet\n"
            "**1.** Upload an image or load a dev example. **2.** Click the visible part of the object. "
            "**3.** Pick the model and the hints to use, draw them, then generate. "
            "*Scribble*: sketch lines of the hidden part; with no sketch, the scribble ControlNet gets a blank hint. "
            "*Box*: drag a box around the hidden part; with no box, the plain SynergyAmodal UNet runs. "
            "*Reference*: another view of the object, ideally cropped on white like the uCO3D examples. "
            "The separate models take one hint at a time; the union model takes any of them together "
            "and leaves out an empty box or scribble. "
            "The canvas keeps one scribble and one box, and selecting the object again clears both."
        )
        state = gr.State()
        sketch = gr.Textbox(visible=False)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload image", sources=["upload", "clipboard"], height=SIZE)
                example_set = gr.Radio(list(example_sets), value=SYNERGY_EXAMPLES, label="Dev examples")
                example_index = gr.Slider(0, len(example_sets[SYNERGY_EXAMPLES]) - 1, value=0, step=1, label="Dev example")
                example_button = gr.Button("Load dev example")
            with gr.Column():
                selection = gr.Image(label="Click the visible object", interactive=False, height=SIZE, width=SIZE)
                point_type = gr.Radio(["Object", "Not object"], value="Object", label="Click marks")
                clear_button = gr.Button("Clear selection")

        with gr.Row():
            with gr.Column():
                model = gr.Radio([SEPARATE] + ([UNION] if UNION in models else []), value=SEPARATE, label="Model")
                hints = gr.CheckboxGroup([SCRIBBLE, BOX, REFERENCE], value=[SCRIBBLE], label="Hints to use")
                hint_type = gr.Radio([SCRIBBLE, BOX], value=SCRIBBLE, label="Draw on the canvas")
                gr.Markdown("**Draw the hint for the hidden part (optional)**")
                drawing = gr.HTML(blank_canvas(None))
                with gr.Row():
                    undo_button = gr.Button("Undo stroke / remove box")
                    clear_sketch_button = gr.Button("Clear hint")
                reference = gr.Image(type="pil", label="Reference image", sources=["upload", "clipboard"], height=256, width=256)
                caption = gr.Textbox(label="Caption of the whole object")
                cfg_text = gr.Slider(1, 10, value=1, label="Text CFG")
                strength = gr.Slider(0.1, 1, value=1, label="Refinement noise strength")
                seed = gr.Number(value=0, precision=0, label="Seed")
                generate_button = gr.Button("Generate", variant="primary")
            with gr.Column():
                output = gr.Image(label="Global pass (RGBA)", interactive=False, height=SIZE, width=SIZE, image_mode="RGBA", format="png")
                with gr.Row():
                    refined_output = gr.Image(label="Refined (RGBA)", interactive=False, height=256, width=256, image_mode="RGBA", format="png")
                    mask = gr.Image(label="Mask", interactive=False, height=256, width=256, format="png")
                with gr.Row():
                    hint = gr.Image(label="Your hint, as the model sees it", interactive=False, height=256, width=256, format="png")
                    crop_hint = gr.Image(label="Hint for the refinement crop", interactive=False, height=256, width=256, format="png")
                training_hint = gr.Image(label="Training hint of the dev example", interactive=False, height=256, width=256, format="png")

        image_input.upload(load_image, [image_input], [state, selection, drawing, training_hint])
        example_set.change(lambda name: gr.update(maximum=len(example_sets[name]) - 1, value=0), [example_set], [example_index])
        example_button.click(load_example, [example_set, example_index, hint_type], [state, selection, drawing, training_hint, caption, reference])
        hint_type.change(None, [hint_type], None, js=SET_MODE_JS)
        hint_type.change(show_training_hint, [state, hint_type], [training_hint])
        selection.select(select_object, [state, point_type], [state, selection, drawing])
        clear_button.click(clear_selection, [state], [state, selection, drawing])
        undo_button.click(None, js=UNDO_JS)
        clear_sketch_button.click(None, js=CLEAR_JS)
        generate_button.click(
            generate,
            [state, sketch, model, hints, reference, caption, cfg_text, strength, seed],
            [output, refined_output, mask, hint, crop_hint],
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


def load_ip_adapter(unet, state_dict, num_tokens, clip_embeddings_dim):
    # Puts new IP-Adapter processors on the UNet and loads them and the image projection from a checkpoint of
    # trainer_ip_adapter.py or trainer_union.py. Returns the image projection and the processors.
    add_ip_adapter(unet, num_tokens=num_tokens)
    processors = {k[len("unet."):]: v for k, v in state_dict.items() if k.startswith("unet.") and ".processor." in k}
    missing, unexpected = unet.load_state_dict(processors, strict=False)
    assert not unexpected and not any(".processor." in k for k in missing), (unexpected, missing)

    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=clip_embeddings_dim,
        clip_extra_context_tokens=num_tokens,
    )
    image_proj_model.load_state_dict({k[len("image_proj_model."):]: v for k, v in state_dict.items() if k.startswith("image_proj_model.")})
    return image_proj_model.eval(), dict(unet.attn_processors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlnet_scribble.yaml", help="scribble ControlNet config")
    parser.add_argument("--ckpt", help="scribble checkpoint; defaults to the newest from training")
    parser.add_argument("--box_config", default="configs/controlnet_box.yaml")
    parser.add_argument("--box_ckpt", help="box checkpoint; defaults to the newest from training")
    parser.add_argument("--ref_config", default="configs/ip_adapter_reference.yaml")
    parser.add_argument("--ref_ckpt", help="reference IP-Adapter checkpoint; defaults to the newest from training")
    parser.add_argument("--union_config", default="configs/union_hints.yaml")
    parser.add_argument("--union_ckpt", help="union checkpoint; defaults to the newest from training, if there is one")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    # The config paths are relative to this folder, as in training.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    from controlnet_union import ControlNetUnionModel
    from datasets.hint_scribble import HintScribbleDataset
    from datasets.hint_union import HintUnionDataset
    from ip_adapter import ImageProjModel, add_ip_adapter
    from trainer_controlnet import ControlNetTrainer
    from trainer_deocclusion_pseudo import InpaintingTrainer
    from trainer_ip_adapter import IPAdapterTrainer
    from trainer_union import UnionTrainer
    from zim import ZimPredictor, zim_model_registry

    def newest_checkpoint(opt):
        ckpts = glob.glob(os.path.join(opt.output_dir, opt.experiment_name, "lightning_logs/version_*/checkpoints/*.ckpt"))
        return max(ckpts, key=os.path.getmtime) if ckpts else None

    def load_state_dict(path):
        print(f"Loading {path}")
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)["state_dict"]

    def controlnet_weights(state_dict):
        return {k[len("controlnet."):]: v for k, v in state_dict.items() if k.startswith("controlnet.")}

    opt = OmegaConf.load(args.config)
    box_opt = OmegaConf.load(args.box_config)
    ref_opt = OmegaConf.load(args.ref_config)
    union_opt = OmegaConf.load(args.union_config)
    ckpt = args.ckpt or newest_checkpoint(opt)
    box_ckpt = args.box_ckpt or newest_checkpoint(box_opt)
    ref_ckpt = args.ref_ckpt or newest_checkpoint(ref_opt)
    union_ckpt = args.union_ckpt or newest_checkpoint(union_opt)
    print(f"Loading {ckpt}")

    device = args.device
    # bf16 weights, because inference already runs under bf16 autocast and this halves the memory.
    system = ControlNetTrainer.load_from_checkpoint(ckpt, map_location="cpu", opt=opt).eval()
    default_processors = dict(system.unet.attn_processors)

    # All the models were trained on the same frozen UNet, so for the others only their own weights are loaded.
    # The box ControlNet goes into a copy of the scribble one.
    box_controlnet = copy.deepcopy(system.controlnet)
    box_controlnet.load_state_dict(controlnet_weights(load_state_dict(box_ckpt)))

    # One image encoder for both IP-Adapters, as in training.
    system.image_encoder = CLIPVisionModelWithProjection.from_pretrained(ref_opt.image_encoder_path).eval()
    system.encode_reference = types.MethodType(IPAdapterTrainer.encode_reference, system)
    image_processor = CLIPImageProcessor()
    clip_embeddings_dim = system.image_encoder.config.projection_dim

    ref_image_proj_model, ref_processors = load_ip_adapter(system.unet, load_state_dict(ref_ckpt), ref_opt.num_tokens, clip_embeddings_dim)
    # Back to the UNet's own processors: from_unet below loads the UNet's weights strictly.
    system.unet.set_attn_processor(dict(default_processors))

    models = {
        None: {"cls": InpaintingTrainer, "processors": default_processors},
        SCRIBBLE: {"cls": ControlNetTrainer, "controlnet": system.controlnet, "processors": default_processors},
        BOX: {"cls": ControlNetTrainer, "controlnet": box_controlnet, "processors": default_processors},
        REFERENCE: {"cls": IPAdapterTrainer, "image_proj_model": ref_image_proj_model, "processors": ref_processors},
    }

    if union_ckpt is not None:
        union_state_dict = load_state_dict(union_ckpt)
        union_controlnet = ControlNetUnionModel.from_unet(system.unet)
        union_controlnet.load_state_dict(controlnet_weights(union_state_dict))
        union_image_proj_model, union_processors = load_ip_adapter(system.unet, union_state_dict, union_opt.num_tokens, clip_embeddings_dim)
        models[UNION] = {"cls": UnionTrainer, "controlnet": union_controlnet, "image_proj_model": union_image_proj_model, "processors": union_processors}
        del union_state_dict
    else:
        print("No union checkpoint, so the union model is not available.")

    system.unet.set_attn_processor(dict(default_processors))

    # Only what is on the system now moves with it below, so the rest is moved here.
    for model in models.values():
        for name in ("controlnet", "image_proj_model"):
            if name in model:
                model[name].eval().bfloat16().to(device)
        for processor in model["processors"].values():
            if isinstance(processor, nn.Module):
                processor.bfloat16().to(device)

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

    example_sets = {
        SYNERGY_EXAMPLES: HintScribbleDataset(opt.val_dataset.args.root, opt.val_dataset.args.manifest),
        # The same held-out uCO3D sequences as the IP-Adapter and union sample grids.
        UCO3D_EXAMPLES: HintUnionDataset(**ref_opt.val_dataset.args),
    }
    box_examples = HintScribbleDataset(box_opt.val_dataset.args.root, box_opt.val_dataset.args.manifest)
    box_example_index = {r["identity"]["sample_id"].split("_", 1)[1]: i for i, r in enumerate(box_examples.records)}

    build_app().launch(server_name=args.host, server_port=args.port)
