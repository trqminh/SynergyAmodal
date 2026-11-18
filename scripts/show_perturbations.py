"""Draws records with clean and perturbed hints side by side, to check the perturbations by eye.

One row per record, one column per spec (the first is clean). Each panel is the observed image with the box outline
(blue), the scribble (red), the reference in the corner, and where the perturbed hint is wrong: false (yellow) and
missing (magenta).

Usage (from SynergyAmodal/):
  python scripts/show_perturbations.py --out perturbations.png --records 6 \
      --spec box:loose:0.5 --spec box:cut:0.5 --spec scribble:shift:16 --spec scribble:add:1 --spec reference:swap
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.hint_union import HintUnionDataset  # noqa: E402
from hint_perturbation import HintPerturber  # noqa: E402


def thick(mask, px=2):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


def panel(item, title):
    image = (item['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = np.ascontiguousarray((image * 0.6 + 255 * 0.4).astype(np.uint8))
    box = item['box'][0].numpy().astype(np.uint8)
    contours, _ = cv2.findContours(box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (40, 90, 255), 3)
    image[thick(item['scribble'][0].numpy(), 1)] = (230, 30, 30)
    for hint, errors in item.get('errors', {}).items():
        image[thick(errors['missing'], 2) & ~thick(item[hint][0].numpy(), 1)] = (220, 0, 220)
        image[thick(errors['false'], 1) & (item[hint][0].numpy() > 0)] = (250, 210, 0)

    reference = (item['reference'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    reference = cv2.resize(np.ascontiguousarray(reference), (128, 128))
    image[4:132, -132:-4] = reference
    cv2.rectangle(image, (image.shape[1] - 133, 3), (image.shape[1] - 3, 133), (0, 0, 0), 1)
    cv2.putText(image, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../all_data/datasets/hint_dataset_uco3d_ref_scribble_box_v1_test')
    parser.add_argument('--manifest', default='manifests/test.provisional.jsonl')
    parser.add_argument('--spec', action='append', required=True)
    parser.add_argument('--records', type=int, default=6)
    parser.add_argument('--start', type=int, default=0, help='first record, then every len/records-th')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    dataset = HintUnionDataset(args.root, args.manifest)
    perturbers = [HintPerturber(dataset, [spec], seed=args.seed) for spec in args.spec]
    step = len(dataset) // args.records
    rows = []
    for index in range(args.start, args.start + step * args.records, step):
        row = [panel(dataset[index], 'clean')]
        for spec, perturber in zip(args.spec, perturbers):
            row.append(panel(perturber(index, dataset[index]), spec))
        rows.append(np.concatenate(row, 1))
    grid = np.concatenate(rows, 0)
    grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.out, grid[:, :, ::-1])
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
