import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from transformers import CLIPImageProcessor


def is_dev_group(group, dev_every):
    """Puts about 1 in dev_every source groups in dev. A hash, so the split does not depend on manifest order."""
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % dev_every == 0


class HintReferenceDataset(torch.utils.data.Dataset):
    """Loader for hint_dataset_uco3d_ref_scribble_box_v1, reference hint only.

    The reference is a 224x224 crop of the same object from another view, already on a white background,
    so it goes to the CLIP image processor as is, like the images in IP-Adapter's tutorial_train.py.

    The dataset has no dev manifest yet. With split='train' or 'dev', the records are split by
    source_group (one uCO3D sequence) so that no object is in both.
    """

    def __init__(self, root, manifest, split=None, dev_every=50):
        super().__init__()
        self.root = Path(root)

        with open(self.root / manifest) as f:
            self.records = [json.loads(line) for line in f]

        if split is not None:
            assert split in ('train', 'dev')
            self.records = [r for r in self.records if is_dev_group(r['identity']['source_group'], dev_every) == (split == 'dev')]

        self.clip_image_processor = CLIPImageProcessor()

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

        reference = Image.open(self.root / inputs['reference']).convert("RGB")

        return {
            'image': self.load_rgb(inputs['observed_rgb']),
            'modal_mask': self.load_mask(inputs['visible_mask']),
            'amodal_image': self.load_rgb(supervision['complete_rgb']),
            'amodal_mask': self.load_mask(supervision['amodal_mask']),
            'reference': to_tensor(reference),
            'reference_mask': self.load_mask(inputs['reference_validity']),
            'clip_image': self.clip_image_processor(images=reference, return_tensors="pt").pixel_values[0],
            'caption': inputs['text'],
        }
