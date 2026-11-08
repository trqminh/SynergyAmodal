import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


class HintScribbleDataset(torch.utils.data.Dataset):
    def __init__(self, root, manifest):
        super().__init__()
        self.root = Path(root)

        with open(self.root / manifest) as f:
            self.records = [json.loads(line) for line in f]

        print(f"Dataset length: {len(self.records)}")

    def __len__(self):
        return len(self.records)

    def load_rgb(self, path):
        return to_tensor(Image.open(self.root / path).convert("RGB"))

    def load_mask(self, path):
        return (to_tensor(Image.open(self.root / path).convert("L")) > 0.5).float()

    def __getitem__(self, idx):
        inputs = self.records[idx]['inputs']
        supervision = self.records[idx]['supervision']

        # The lines are in channel 2. Like ControlNet's scribble model, the hint
        # is 3-channel, white lines on black, in [0, 1].
        lines = np.load(self.root / inputs['spatial'])['values'][2]
        hint = torch.from_numpy(lines).float()[None].repeat(3, 1, 1)

        return {
            'image': self.load_rgb(inputs['observed_rgb']),
            'modal_mask': self.load_mask(inputs['visible_mask']),
            'amodal_image': self.load_rgb(supervision['complete_rgb']),
            'amodal_mask': self.load_mask(supervision['amodal_mask']),
            'hint': hint,
            'caption': inputs['text'],
        }
