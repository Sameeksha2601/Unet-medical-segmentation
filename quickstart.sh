#!/bin/bash
# Quickstart: generates synthetic data, trains for a few epochs, evaluates,
# and runs inference — so you can see the full pipeline work end-to-end.
set -e

echo "== Installing dependencies =="
pip install -r requirements.txt

cd src

echo "== Generating synthetic demo dataset =="
python dataset.py

echo "== Training (10 quick epochs) =="
python train.py --epochs 10

echo "== Evaluating =="
python evaluate.py

echo "== Running inference on a few sample images =="
python inference.py --overlay

echo ""
echo "Done! Check ../outputs for predicted masks and ../checkpoints for the trained model."
