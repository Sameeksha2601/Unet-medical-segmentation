"""
Sorts the Kaggle "LGG MRI Segmentation" dataset (per-patient folders, each
containing paired <name>.tif image + <name>_mask.tif mask files) into the
flat data/images and data/masks folders this project expects, with matching
filenames in both.

Usage (run from anywhere, edit the two paths below or pass as arguments):
    python sort_lgg_dataset.py "C:\\Users\\Samee\\Downloads\\archive (1)\\kaggle_3m" "C:\\Users\\Samee\\unet_project\\unet_project\\data"

By default it also SKIPS slices with an all-black (empty) mask, since a large
fraction of LGG slices have no tumor at all — keeping only slices that
contain some tumor gives a smaller, faster-to-train, more useful dataset.
Pass --keep_empty to include them too.
"""

import argparse
import os
import shutil

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Sort LGG MRI dataset into images/masks folders")
    p.add_argument("source_dir", help=r"path to extracted kaggle_3m folder")
    p.add_argument("dest_data_dir", help=r"path to project's data folder (containing images/ and masks/)")
    p.add_argument("--keep_empty", action="store_true", help="also copy slices with no tumor (empty mask)")
    p.add_argument("--limit", type=int, default=None, help="max number of pairs to copy (for a quick test)")
    return p.parse_args()


def main():
    args = parse_args()
    images_out = os.path.join(args.dest_data_dir, "images")
    masks_out = os.path.join(args.dest_data_dir, "masks")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    copied = 0
    skipped_empty = 0

    for root, _dirs, files in os.walk(args.source_dir):
        mask_files = sorted(f for f in files if f.lower().endswith("_mask.tif"))
        for mask_name in mask_files:
            image_name = mask_name.replace("_mask.tif", ".tif")
            mask_path = os.path.join(root, mask_name)
            image_path = os.path.join(root, image_name)

            if not os.path.exists(image_path):
                continue  # no matching image, skip

            if not args.keep_empty:
                mask_arr = np.array(Image.open(mask_path).convert("L"))
                if mask_arr.max() == 0:
                    skipped_empty += 1
                    continue  # empty mask, skip

            # matching filenames in both output folders
            out_name = image_name  # e.g. TCGA_CS_4941_19960909_11.tif
            shutil.copy2(image_path, os.path.join(images_out, out_name))
            shutil.copy2(mask_path, os.path.join(masks_out, out_name))
            copied += 1

            if args.limit and copied >= args.limit:
                print(f"Reached limit of {args.limit} pairs.")
                print(f"Copied: {copied} | Skipped (empty masks): {skipped_empty}")
                return

    print(f"Done.\nCopied: {copied} pairs\nSkipped (empty masks): {skipped_empty}")
    print(f"Images -> {images_out}\nMasks  -> {masks_out}")


if __name__ == "__main__":
    main()
