# Research plan: multi-hint amodal completion (CVPR)

## Contributions

1. **Setting:** amodal completion guided by any combination of hints (box, scribble, reference image, text).
2. **Dataset:** the training set and the benchmark.
3. **Model:** one model that takes any subset of the hints.

The current model (ControlNet-Union for box and scribble, plus IP-Adapter or reference attention for the reference) is a
solid baseline. As a contribution, though, it reads as "ControlNet++ plus a ReferenceNet". The model's novelty should
come from what is specific to this setting, not from swapping in newer modules.

## What makes this setting different

- **Hints are partial and uncertain.**
  - The box is a loose bound.
  - The scribble covers about 5% of the hidden region and has a tolerance.
  - The reference comes from another view (view gap up to about 75°) and has a validity mask.
- **The visible region is a hard constraint; hints are soft.** Hints can be wrong, or disagree with each other or with
  what is visible.
- **The hints differ in kind.**
  - Box and scribble give geometry in the target view.
  - The reference gives appearance from another view.
  - Text gives meaning.

Generic controllable generation (ControlNet and its successors) treats a hint as a dense, trusted input. That gap is
the opening for contribution 3.

**Story:** hints as partial evidence. The model trusts the visible region fully and treats each hint according to how
reliable it is, everywhere in the image.

## Status (2026-11-12)

- [x] Trained reference attention (`trainer_reference_only.py`): the same UNet runs on the reference, with LoRA on its
      self-attention.
  - At step 18k of 50k it beats the IP-Adapter (step 50k) on the handsaw example, with `cfg_text=3` and the same
    noise.
  - Its outputs have the saw handle with a grip hole from the reference in all 4 samples.
  - The IP-Adapter gives a knife handle, no handle, a thin loop, or a dark blue stub.
- [ ] Held-out eval of reference-only vs IP-Adapter, both with `--cfg_text 3`.

## Plan

### Step 1: show the problem (about a week)

Find out how the current union model fails before designing anything new. Build the method around the failures that
actually occur.

- [ ] Write a hint perturbation tool for the held-out set:
  - [ ] Box: looser boxes, shifted boxes, boxes that cut off part of the object.
  - [ ] Scribble: shifted strokes, strokes partly removed, false strokes added.
  - [ ] Reference: a large view gap, a reference of a different object (to test conflicts).
  - [ ] Conflicts: hints that contradict the visible region or each other.
- [ ] Run the union model on clean and perturbed hints.
- [ ] Measure:
  - Over-obedience: does it copy wrong hints literally?
  - Drift: does it change or contradict the visible region?
  - Under-use: does it ignore correct hints?
- [ ] Make the motivation figure from the clearest failures.

**Decision point:** if the model already handles perturbed hints well, change the story before building step 2.

### Step 2: one model contribution, reliability-aware hint fusion

- [ ] **Hint branches:**
  - Geometry: ControlNet-Union for box and scribble.
  - Appearance: reference attention in place of the IP-Adapter.
  - The ControlNet does not read the reference bank, and the reference pass gets no ControlNet residuals.
- [ ] **Spatial reliability gate:** a lightweight per-pixel weighting of each hint.
  - Inputs: `valid_canvas`, `reference_validity`, the scribble tolerance, and the noisy features.
  - The gate maps should be easy to read, so they can be a figure.
- [ ] **Robust training:** hint perturbation and conflicting hints during training, so hints are treated as soft
      evidence and the visible region as hard.
- [ ] **Hint dropout:** Uni-ControlNet dropout over all hints. A dropped reference uses `control.drop` for that sample.
- [ ] **Initialization:** the ControlNet from the union checkpoint, the LoRA from the reference-only checkpoint, then a
      short joint fine-tune.
  - Learning rates: 1e-5 for the ControlNet, 1e-4 for the LoRA.
  - Memory: about 30 GB at batch size 4.
- [ ] Optional first check: combine the two trained checkpoints at inference without training, to see whether they
      get in each other's way.

Not now (higher risk, harder to explain): a full mixture of experts, and "every hint as a reference" (SD's
self-attention has no positional encoding, so sparse scribble tokens are hard to place). Combining hints at the
noise-prediction level with per-hint, spatial guidance weights stays possible as an inference-time option and as a
user control.

### Step 3: grow the benchmark (cheap, strengthens contributions 1 and 2)

- [ ] **Hint-adherence metrics:**
  - The amodal mask stays inside the box.
  - Edges match the scribble within its tolerance.
  - The visible region is preserved.
  - Appearance matches the reference (DINO similarity in the completed region).
- [ ] **Noisy-hint split:** perturbed hints at several noise levels.
- [ ] **Conflicting-hint split:** hints that disagree with the visible region or with each other.

### Step 4: close the synthetic-to-real gap

The scribbles are synthetic HED lines and the boxes are computed from masks. Reviewers will raise this first.

- [ ] Collect a small test set with real human hints (100–200 examples): drawn boxes, drawn scribbles, and
      references people pick themselves.
- [ ] Show the model holds up on it.

### Step 5: experiments

- [ ] **Baselines:**
  - SynergyAmodal without hints.
  - The single-hint ControlNets (box, scribble).
  - The IP-Adapter.
  - Training-free reference-only.
  - Trained reference-only.
  - The plain union model.
- [ ] **Ablations:** without the gate, without perturbation training, without the validity inputs, and with the
      IP-Adapter in place of reference attention.
- [ ] **Robustness curves:** quality and adherence as hint noise increases.
- [ ] **Qualitative results:** gate maps, any-subset hint grids, results on human hints.
- [ ] Use the same settings for every method (seed, steps, `cfg_text`).

## If the deadline is close

Do steps 1, 3 and 5 with the plain union model (reference attention in place of the IP-Adapter). Leave the gate and
perturbation training as future work.
