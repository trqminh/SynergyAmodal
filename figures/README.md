# SynergyAmodal training and inference

[Two-page editable Draw.io figure](synergy_amodal.drawio)

| Page | Editable | Preview | Vector |
| --- | --- | --- | --- |
| Training | [Draw.io](training.drawio) | [PNG](training.png) | [SVG](training.svg) |
| Inference | [Draw.io](inference.drawio) | [PNG](inference.png) | [SVG](inference.svg) |

The figures follow `../../pixart-alpha-amodal/figures`: white background,
pastel green VAE blocks, pastel blue model blocks, rounded gray enclosures,
snowflakes for frozen/fixed weights, flames for trainable weights, and orange
loss/update arrows. Labels, blocks, and paths are editable. Existing reference
photos and schematic noise tiles are embedded; the photos are illustrations,
not SynergyAmodal predictions. No training or inference was run to make these
figures. Source hashes are recorded in `provenance.json`.

Both pages show the actual inverted condition mask beside the visible mask:
black on the visible object and white elsewhere. Its pixels are the complement
of the embedded visible mask, sampled to 64 × 64 using the training code's
nearest-neighbor convention, and embedded as a crisp vector tile.

## Training

Panel A follows `../trainer_deocclusion_pseudo.py`:

- The backbone is `stabilityai/stable-diffusion-2-inpainting`. Its pretrained
  9-channel input convolution gains four zero-initialized scene-image channels.
  The full U-Net is trained; the VAE and CLIP text encoder are frozen.
- Input order is `[noisy target, inverted visible mask, visible-object latent,
  scene-image latent]`, with `4 + 1 + 4 + 4 = 13` channels. The spatial mask is
  resized directly, not VAE encoded. White in the illustrated visible mask
  identifies the visible object; the U-Net receives its complement.
- All RGB inputs are normalized to `[-1, 1]` and use sampled VAE posteriors
  scaled by `0.18215`. The visible-object RGB has a white background.
- Only the complete target latent receives diffusion noise. Noise includes
  the implemented `0.1` per-channel offset. Full-target epsilon MSE is weighted
  by `min(SNR, 5) / SNR`.
- Actual training code independently empties captions with probability 0.5
  and zeros the entire image condition with probability 0.1. The configured
  `p_drop_condition` value is not used for these decisions in `training_step`.

Panel B follows `../trainer_deocclusion_pseudo_decoder.py`: separately train
the VAE decoder to predict a one-channel amodal mask using BCE-with-logits
plus soft Dice loss (smoothing 0.1). The output convolution is zero-initialized.
The encoder and post-quantization convolution remain frozen. No diffusion
U-Net is used in this training path. Both trainers maintain separate EMA
shadows with beta 0.999; `update_average` is called at the beginning of each
training step, not specifically after each optimizer update.

## Inference

The main panel follows `InpaintingTrainer.inference` / `inference_ddim`;
the top and bottom strips include the full `../app.py` workflow:

- ZIM obtains a visible mask from clicks. The app can erode it, then resizes
  and pads RGB and mask to 512 square for the global pass.
- Image conditions are encoded once per pass and reused throughout 25 DDIM
  steps. Image/text guidance uses three branches when either guidance scale
  differs from 1; otherwise the code makes one conditional prediction.
- The final latent is divided by `0.18215`. The original RGB VAE and learned
  mask VAE decode that same latent separately. Mask logits use sigmoid and
  rounding; the mask supplies the alpha channel.
- The global prediction is restored to the input-image coordinate frame.
  Its amodal mask defines a padded square crop of the original scene,
  visible mask, and coarse completion. The crop is resized to 512 square.
- A second pass uses the same models and caption with re-encoded crop
  conditions. For local strength below 1, the encoded coarse RGB crop is
  noised at the starting timestep. At strength 1 (the app default), it is
  ignored and sampling begins from fresh Gaussian noise.
- The refined RGBA crop is placed back into the image frame, with alpha
  thresholded at 0.5. The app loads EMA weights for both trained components.
  Snowflakes on inference denote fixed weights/no optimization.

## Rebuild

From the parent workspace:

```bash
python SynergyAmodal/figures/build_figures.py
python SynergyAmodal/figures/export_figures.py
```

The builder uses Pillow to read the mask pixels and extracts existing embedded
assets from the reference Draw.io file. The exporter uses draw.io Desktop and Xvfb,
with `DRAWIO_TOOLS_ROOT` defaulting to `/tmp/hytrol-drawio-tools`; expected
executables are `app/opt/drawio/drawio` and `xserver/usr/bin/xvfb-run` under
that root. PNG and SVG exports come directly from the editable Draw.io files.
