from collections import defaultdict

import numpy as np
import torch

from datasets.hint_reference import HintReferenceDataset
from hint_perturbation import HINTS, MODES, HintPerturber


class HintUnionDataset(HintReferenceDataset):
    """Loader for hint_dataset_uco3d_ref_scribble_box_v1 with all three hints: box, scribble and reference.

    Every record has all three. Which ones the model sees is decided in the trainer.
    """

    def load_spatial(self, path):
        # Like the scribble and box datasets: 3-channel, white on black, in [0, 1].
        values = np.load(self.root / path)['values']
        return torch.from_numpy(values).float()[None].repeat(3, 1, 1)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        inputs = self.records[idx]['inputs']
        item['box'] = self.load_spatial(inputs['box'])
        item['scribble'] = self.load_spatial(inputs['scribble'])
        # Where the box and scribble are known. Outside it, a missing line means unknown, not "no edge".
        item['valid_canvas'] = self.load_mask(inputs['valid_canvas'])

        return item


class HintUnionPerturbedDataset(HintUnionDataset):
    """HintUnionDataset whose hints are made wrong at random (hint_perturbation.py), to train the reliability gates.
    The target stays the true object, so following a wrong hint costs loss.

    An item gets wrong hints with probability p_perturb. Each hint is then wrong with probability p_each, and at least
    one is. A wrong hint takes one of its modes, uniformly, with a level drawn uniformly from the mode's range:
        modes: {box: {loose: [lo, hi], ...}, scribble: {...}, reference: {swap: null, swap_other: null}}
    A hint left out of modes is never perturbed. New perturbations are drawn every time an item is loaded.

    Adds box_wrong and scribble_wrong ((1, H, W): 1 where the hint is wrong, its false and missing pixels; zeros for a
    right hint) and reference_wrong (0 or 1), for gate supervision and logging. Donors (swapped references, false
    strokes) come from this dataset's records, so a train item never gets a dev object.
    """

    def __init__(self, *args, p_perturb=0.5, p_each=0.5, modes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_perturb = p_perturb
        self.p_each = p_each
        self.modes = defaultdict(list)
        for hint, hint_modes in (modes or {}).items():
            for mode, level_range in hint_modes.items():
                assert mode in MODES[hint], f'unknown perturbation {hint}:{mode}'
                assert (level_range is None) == (hint == 'reference'), f'{hint}:{mode}: wrong level range {level_range}'
                self.modes[hint].append((mode, None if level_range is None else tuple(level_range)))
        self.hints = [hint for hint in HINTS if self.modes[hint]]
        self.perturber = HintPerturber(self, [])

    def sample_specs(self, rng):
        if not self.hints or rng.random() >= self.p_perturb:
            return []
        wrong = rng.random(len(self.hints)) < self.p_each
        if not wrong.any():
            wrong[rng.integers(len(self.hints))] = True
        specs = []
        for hint, is_wrong in zip(self.hints, wrong):
            if is_wrong:
                mode, level_range = self.modes[hint][rng.integers(len(self.modes[hint]))]
                specs.append(f'{hint}:{mode}' if level_range is None else f'{hint}:{mode}:{rng.uniform(*level_range):.4g}')
        return specs

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        # A fresh generator from OS entropy, so dataloader workers do not repeat each other's draws.
        rng = np.random.default_rng()
        specs = self.sample_specs(rng)
        self.perturber(idx, item, specs, seed=int(rng.integers(2 ** 62)))

        for hint in ('box', 'scribble'):
            errors = item['errors'].get(hint)
            wrong = torch.zeros_like(item[hint][:1]) if errors is None else torch.from_numpy(errors['false'] | errors['missing']).float()[None]
            item[f'{hint}_wrong'] = wrong
        item['reference_wrong'] = torch.tensor(float(any(spec.startswith('reference:') for spec in specs)))
        # Variable-size records of what was done; not collatable.
        del item['perturbation'], item['errors']

        return item
