import numpy as np
import torch

from datasets.hint_reference import HintReferenceDataset


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

        return item
