# Medical Image Segmentation with U-Net

A complete, ready-to-run PyTorch implementation of **U-Net** for medical
image segmentation (tumors, organs, lesions, cells, etc.). Includes model
code, training/evaluation loops, inference script, data augmentation, and a
synthetic-data generator so you can run the full pipeline immediately, even
before plugging in a real dataset.

## Project structure

```
unet_project/
├── data/
│   ├── images/          # input images (grayscale or RGB)
│   └── masks/           # matching binary/label masks, same filenames
├── checkpoints/         # saved model weights + training history
├── outputs/             # predicted masks / overlays from inference
├── src/
│   ├── model.py          # U-Net architecture
│   ├── dataset.py         # Dataset class, augmentations, synthetic data generator
│   ├── utils.py           # Dice/BCE loss, Dice/IoU metrics, early stopping
│   ├── config.py          # all hyperparameters in one place
│   ├── train.py           # training loop with checkpointing
│   ├── evaluate.py        # compute Dice/IoU/precision/recall on a test set
│   └── inference.py       # run predictions on new images
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
cd unet_project
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

GPU is optional but strongly recommended for real datasets (CPU works fine
for the synthetic demo).

## 2. Get data

**Option A — try it instantly with synthetic data** (random blob shapes on
noisy backgrounds, just to validate the pipeline runs end-to-end):

```bash
cd src
python dataset.py
```

This fills `data/images/` and `data/masks/` with 60 synthetic image/mask
pairs.

**Option B — use a real medical dataset.** Drop your own images/masks into
`data/images/` and `data/masks/`, matched by filename (e.g.
`case_001.png` in both folders). Good public datasets to start with:

| Dataset | Task |
|---|---|
| [ISIC](https://www.isic-archive.com/) | Skin lesion segmentation |
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | Polyp segmentation (endoscopy) |
| [BraTS](https://www.med.upenn.edu/cbica/brats/) | Brain tumor segmentation (MRI) |
| [PROMISE12](https://promise12.grand-challenge.org/) | Prostate MRI segmentation |
| [LUNA16](https://luna16.grand-challenge.org/) | Lung nodule segmentation (CT) |

Masks should be single-channel images where any non-zero pixel = foreground.
For grayscale scans (CT/MRI/X-ray) keep `N_CHANNELS = 1` in `config.py`; for
RGB images (dermoscopy/endoscopy) set `N_CHANNELS = 3`.

## 3. Train

```bash
cd src
python train.py
```

Or override settings from the command line:

```bash
python train.py --epochs 100 --batch_size 16 --lr 5e-5 --img_size 256
```

Training prints per-epoch Dice/IoU/loss, saves the **best** checkpoint (by
validation Dice) to `checkpoints/best_model.pth`, and writes a
`checkpoints/history.json` with the full training curve for plotting.

Key defaults (edit in `src/config.py`):
- Loss: combined **Dice + BCE** (robust to class imbalance, common for
  medical segmentation where the foreground region is small)
- Optimizer: Adam, `lr=1e-4`, with `ReduceLROnPlateau` scheduling
- Early stopping: stops if val loss doesn't improve for 10 epochs
- Augmentation: flips, rotation, shift/scale/rotate, brightness/contrast,
  Gaussian noise, elastic deformation (via `albumentations`)

## 4. Evaluate

```bash
python evaluate.py --checkpoint ../checkpoints/best_model.pth
```

Reports Dice coefficient, IoU (Jaccard), pixel accuracy, precision, and
recall over the dataset.

## 5. Run inference on new images

```bash
python inference.py --checkpoint ../checkpoints/best_model.pth \
                     --input ../data/images \
                     --output ../outputs \
                     --overlay
```

For each input image this saves:
- `<name>_mask.png` — binary predicted mask
- `<name>_prob.png` — grayscale probability heatmap
- `<name>_overlay.png` — original image with the mask overlaid in red (if `--overlay` is passed)

You can also point `--input` at a single image file instead of a directory.

## Architecture notes

The model (`src/model.py`) is a standard U-Net: a contracting encoder path
(4 downsampling stages, doubling channels each time) and a symmetric
expanding decoder path with skip connections concatenating encoder features
at each resolution, ending in a 1×1 conv to produce per-pixel logits.

- `base_c=64` matches the original paper; lower it (e.g. `32`) if you're
  memory constrained.
- `bilinear=True` upsampling is lighter than transposed convolutions and
  usually trains just as well.
- Output is binary (`n_classes=1`) by default — extend `n_classes` and swap
  the loss for `CrossEntropyLoss`/multi-class Dice for multi-organ
  segmentation.

## Tips for real datasets

- **Class imbalance** (small lesion vs. large background) is why Dice+BCE
  loss is used instead of plain BCE or accuracy.
- **Small datasets**: rely heavily on augmentation (already configured) and
  consider 5-fold cross-validation instead of a single train/val split.
- **3D volumes** (CT/MRI): this implementation is 2D, working slice-by-slice.
  For volumetric segmentation, extend `Conv2d`→`Conv3d` throughout `model.py`.
- **Multi-class** (e.g. multiple organs): set `N_CLASSES = num_organs`,
  switch to `CrossEntropyLoss` + multi-class soft Dice, and use `argmax`
  instead of thresholding at inference time.
