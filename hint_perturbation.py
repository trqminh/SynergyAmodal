"""Makes the box, scribble and reference hints of a record wrong in controlled ways, to test how a model handles hints
that disagree with the truth, with each other, or with the visible region.

A perturbation is a spec, '<hint>:<mode>[:<level>]'. Sizes are in pixels at 512.

  box:loose:F         each side moves out by 0.5-1 x F of the box's width (or height): a careless box
  box:shift:F         the box moves by F x its size in a random direction
  box:cut:F           each side moves in by up to F x the size, one side by all of it: part of the occluded region is outside
  scribble:shift:PX   every line moves PX pixels in one random direction: a misplaced drawing
  scribble:warp:PX    a smooth random warp of up to PX pixels: a wobbly hand
  scribble:drop:F     about a fraction F of the line pixels erased in 24 px disks: strokes left out
  scribble:add:F      false strokes, up to F x the record's line pixels, from another object's scribble of the same
                      category (else super category), scaled into the box
  reference:swap        the reference of another object of the same category (another video), else of the same super
                        category (about 1 in 5 records of the test set)
  reference:swap_other  the reference of an object of another super category

Several specs together (one per hint) make hints that disagree with each other.
The view gap of the reference is not perturbed: the test set already spans 60-120 degrees (audit.reference.view_gap_deg),
so results are split by it instead.

Each spec draws from its own random generator, seeded by (seed, sample_id, spec), so a perturbation is the same alone and
in a combination, and in every run. Donors (for add and swap) come from another source_group, so never from the same
object.

For a box or scribble, the record also gets where the perturbed hint is wrong, as two maps:
  false    hint pixels that are wrong: a scribble line farther than the tolerance (3 px) from every true line, or box
           area outside the true box
  missing  true hint pixels the perturbed hint lost
"""
import hashlib
import json
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

HINTS = ('box', 'scribble', 'reference')
MODES = {
    'box': ('loose', 'shift', 'cut'),
    'scribble': ('shift', 'warp', 'drop', 'add'),
    'reference': ('swap', 'swap_other'),
}
DROP_RADIUS = 24
SCRIBBLE_TOLERANCE = 3


def parse_spec(spec):
    hint, mode, *level = spec.split(':')
    assert hint in HINTS and mode in MODES[hint], f'unknown perturbation {spec}'
    if hint == 'reference':
        assert not level, f'{spec}: a reference swap takes no level'
        return hint, mode, None
    assert len(level) == 1, f'{spec}: needs a level, e.g. {hint}:{mode}:0.25'
    return hint, mode, float(level[0])


def rng_for(seed, sample_id, spec):
    return np.random.default_rng(int(hashlib.md5(f'{seed}/{sample_id}/{spec}'.encode()).hexdigest()[:16], 16))


# --- box ---------------------------------------------------------------------------------------------------------

def box_of(values):
    ys, xs = np.nonzero(values)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def draw_box(xyxy, shape):
    H, W = shape
    x0, y0, x1, y1 = np.round(xyxy).astype(int)
    x0, x1 = np.clip([x0, x1], 0, W)
    y0, y1 = np.clip([y0, y1], 0, H)
    out = np.zeros(shape, np.uint8)
    out[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = 1
    return out


def perturb_box(values, mode, level, rng):
    x0, y0, x1, y1 = box_of(values)
    size = np.array([x1 - x0, y1 - y0, x1 - x0, y1 - y0], float)
    xyxy = np.array([x0, y0, x1, y1], float)
    out_sign = np.array([-1, -1, 1, 1])

    if mode == 'loose':
        xyxy = xyxy + out_sign * rng.uniform(0.5, 1, 4) * level * size
    elif mode == 'shift':
        angle = rng.uniform(0, 2 * np.pi)
        xyxy = xyxy + np.tile([np.cos(angle), np.sin(angle)], 2) * level * size
    elif mode == 'cut':
        amount = rng.uniform(0, 1, 4)
        amount[rng.integers(4)] = 1
        # Never past the box's center, so something is left.
        xyxy = xyxy - out_sign * np.minimum(amount * level, 0.45) * size
    return draw_box(xyxy, values.shape), {'xyxy': [round(float(v), 1) for v in xyxy]}


# --- scribble ----------------------------------------------------------------------------------------------------

def move_lines(lines, dx, dy):
    """Moves every line pixel (y, x) to (y + dy[y, x], x + dx[y, x]) and redraws the lines between neighbors, so a
    stretched line stays connected. dx, dy: arrays of lines.shape, or scalars."""
    H, W = lines.shape
    ys, xs = np.nonzero(lines)
    dx = np.broadcast_to(dx, lines.shape)[ys, xs]
    dy = np.broadcast_to(dy, lines.shape)[ys, xs]
    moved = np.round(np.stack([xs + dx, ys + dy], 1)).astype(np.int32)

    index = -np.ones(lines.shape, int)
    index[ys, xs] = np.arange(len(ys))
    segments = [np.stack([moved, moved], 1)]  # every pixel, also the isolated ones
    for oy, ox in ((0, 1), (1, 0), (1, 1), (1, -1)):  # each neighbor pair once
        ny, nx = ys + oy, xs + ox
        inside = (ny < H) & (nx >= 0) & (nx < W)
        a = np.nonzero(inside)[0]
        b = index[ny[a], nx[a]]
        a, b = a[b >= 0], b[b >= 0]
        segments.append(np.stack([moved[a], moved[b]], 1))

    out = np.zeros(lines.shape, np.uint8)
    segments = np.concatenate(segments)
    cv2.polylines(out, list(segments[:, :, None]), False, 1, 1)
    return out


def smooth_field(shape, amplitude, rng, cells=8):
    """A smooth random displacement, largest |value| = amplitude, varying over about 1/cells of the image."""
    field = cv2.resize(rng.standard_normal((cells + 1, cells + 1)).astype(np.float32), shape[::-1], interpolation=cv2.INTER_CUBIC)
    return field / np.abs(field).max() * amplitude


def erase_disks(lines, fraction, rng, radius=DROP_RADIUS):
    """Erases disks around random line pixels until about `fraction` of the line pixels are gone."""
    out = lines.copy()
    total = lines.sum()
    while out.any() and out.sum() > (1 - fraction) * total:
        ys, xs = np.nonzero(out)
        i = rng.integers(len(ys))
        cv2.circle(out, (int(xs[i]), int(ys[i])), radius, 0, -1)
    return out


def perturb_scribble(lines, mode, level, rng, box=None, donor_lines=None):
    if mode == 'shift':
        angle = rng.uniform(0, 2 * np.pi)
        return move_lines(lines, level * np.cos(angle), level * np.sin(angle)), {'angle_deg': round(float(np.degrees(angle)), 1)}
    if mode == 'warp':
        return move_lines(lines, smooth_field(lines.shape, level, rng), smooth_field(lines.shape, level, rng)), {}
    if mode == 'drop':
        return erase_disks(lines, level, rng), {}
    if mode == 'add':
        # The donor's lines, scaled and moved to fill the box, then thinned out to the budget.
        x0, y0, x1, y1 = box_of(box)
        dx0, dy0, dx1, dy1 = box_of(donor_lines)
        scale = min((x1 - x0) / (dx1 - dx0), (y1 - y0) / (dy1 - dy0))
        ys, xs = np.mgrid[:lines.shape[0], :lines.shape[1]]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        dcx, dcy = (dx0 + dx1) / 2, (dy0 + dy1) / 2
        added = move_lines(donor_lines, cx + (xs - dcx) * scale - xs, cy + (ys - dcy) * scale - ys) * box
        budget = level * lines.sum()
        if added.sum() > budget:
            added = erase_disks(added, 1 - budget / added.sum(), rng)
        return np.maximum(lines, added), {'scale': round(float(scale), 3)}


def distance_to(lines):
    """Distance in pixels from every pixel to the nearest line pixel."""
    if not lines.any():
        return np.full(lines.shape, np.inf, np.float32)
    return cv2.distanceTransform((1 - lines).astype(np.uint8), cv2.DIST_L2, 5)


def scribble_errors(true, perturbed, tolerance=SCRIBBLE_TOLERANCE):
    return {
        'false': (perturbed.astype(bool) & (distance_to(true) > tolerance)).astype(np.uint8),
        'missing': (true.astype(bool) & (distance_to(perturbed) > tolerance)).astype(np.uint8),
    }


def box_errors(true, perturbed):
    return {
        'false': (perturbed.astype(bool) & ~true.astype(bool)).astype(np.uint8),
        'missing': (true.astype(bool) & ~perturbed.astype(bool)).astype(np.uint8),
    }


# --- records -----------------------------------------------------------------------------------------------------

class HintPerturber:
    """Perturbs the hints of a HintUnionDataset (or HintReferenceDataset) item.

    dataset: gives the root, the loaders and the CLIP image processor.
    donor_records: where donors are drawn from, by default the dataset's records. Give the whole manifest when the
        dataset is a subset, so small categories still have another object.
    """

    def __init__(self, dataset, specs, seed=0, donor_records=None):
        self.dataset = dataset
        self.specs = [(spec, *parse_spec(spec)) for spec in specs]
        hints = [hint for _, hint, _, _ in self.specs]
        assert len(hints) == len(set(hints)), f'one perturbation per hint: {specs}'
        self.seed = seed

        self.donors = donor_records if donor_records is not None else dataset.records
        self.by_category = defaultdict(list)
        self.by_super_category = defaultdict(list)
        for i, record in enumerate(self.donors):
            source = record['audit']['source']
            self.by_category[source['category']].append(i)
            self.by_super_category[source['super_category']].append(i)

    def pick_donor(self, record, rng, same_category):
        """Another object (another source_group), of the same category if there is one, else of the same super
        category. With same_category False, an object of another super category."""
        group = record['identity']['source_group']
        source = record['audit']['source']
        if same_category:
            for level, pool in (('category', self.by_category[source['category']]),
                                ('super_category', self.by_super_category[source['super_category']])):
                pool = [i for i in pool if self.donors[i]['identity']['source_group'] != group]
                if pool:
                    return self.donors[pool[rng.integers(len(pool))]], level
            raise ValueError(f'no donor for {record["identity"]["sample_id"]}')
        while True:
            donor = self.donors[rng.integers(len(self.donors))]
            if donor['audit']['source']['super_category'] != source['super_category']:
                return donor, 'other_super_category'

    def load_lines(self, record):
        return np.load(self.dataset.root / record['inputs']['scribble'])['values'].astype(np.uint8)

    def __call__(self, index, item, specs=None, seed=None):
        """Perturbs item (dataset[index]) in place. Adds item['perturbation'] (what was done, per spec) and
        item['errors'] ({hint: {'false': map, 'missing': map}}, for box and scribble).
        specs, seed: override the perturber's own, e.g. to draw new ones for every training item."""
        record = self.dataset.records[index]
        sample_id = record['identity']['sample_id']
        specs = self.specs if specs is None else [(spec, *parse_spec(spec)) for spec in specs]
        seed = self.seed if seed is None else seed
        item['perturbation'] = {}
        item['errors'] = {}
        true_box = item['box'][0].numpy().astype(np.uint8) if 'box' in item else None

        for spec, hint, mode, level in specs:
            rng = rng_for(seed, sample_id, spec)

            if hint == 'reference':
                donor, match = self.pick_donor(record, rng, same_category=(mode == 'swap'))
                inputs = donor['inputs']
                reference = Image.open(self.dataset.root / inputs['reference']).convert('RGB')
                item['reference'] = to_tensor(reference)
                item['reference_mask'] = self.dataset.load_mask(inputs['reference_validity'])
                item['clip_image'] = self.dataset.clip_image_processor(images=reference, return_tensors='pt').pixel_values[0]
                item['perturbation'][spec] = {'donor': donor['identity']['sample_id'], 'match': match,
                                              'donor_category': donor['audit']['source']['category']}
                continue

            true = item[hint][0].numpy().astype(np.uint8)
            if hint == 'box':
                perturbed, info = perturb_box(true, mode, level, rng)
                item['errors'][hint] = box_errors(true, perturbed)
            else:
                if mode == 'add':
                    donor, match = self.pick_donor(record, rng, same_category=True)
                    perturbed, info = perturb_scribble(true, mode, level, rng, box=true_box,
                                                       donor_lines=self.load_lines(donor))
                    info.update(donor=donor['identity']['sample_id'], match=match)
                else:
                    perturbed, info = perturb_scribble(true, mode, level, rng)
                item['errors'][hint] = scribble_errors(true, perturbed)

            info['pixels'] = [int(true.sum()), int(perturbed.sum())]
            info['false_pixels'] = int(item['errors'][hint]['false'].sum())
            info['missing_pixels'] = int(item['errors'][hint]['missing'].sum())
            item['perturbation'][spec] = info
            item[hint] = torch.from_numpy(perturbed).float()[None].repeat(3, 1, 1)

        return item

    def save(self, item, sample_id, folder):
        """Writes the perturbed box and scribble, their error maps, and what was done, to folder/<sample_id>_*."""
        for hint, errors in item['errors'].items():
            Image.fromarray(item[hint][0].numpy().astype(np.uint8) * 255).save(folder / f'{sample_id}_{hint}.png')
            for kind, error in errors.items():
                Image.fromarray(error * 255).save(folder / f'{sample_id}_{hint}_{kind}.png')
        (folder / f'{sample_id}.json').write_text(json.dumps(item['perturbation'], indent=1))
