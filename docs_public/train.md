# Sentinel Training Pipeline

Sentinel includes a local-first training/eval/export workflow under `sentinel/ml/`.

This repository provides generic research and simulation tooling only. It does not implement weapon targeting or weapons employment logic.

## Dataset Config

Use a JSON or YAML file with at least:

```yaml
name: quick-motion-v1
train: /data/datasets/quick-motion/images/train
val: /data/datasets/quick-motion/images/val
test: /data/datasets/quick-motion/images/test
names:
  0: person
  1: hand
  2: dog
tag: quick-motion-v1
metadata:
  source: webcam+capture-card
```

## Commands

Train:

```bash
sentinel train \
  --dataset-config datasets/quick_motion.yaml \
  --model yolov8n.pt \
  --alias quick-motion-v1 \
  --profile fast_motion \
  --epochs 25 \
  --image-size 960 \
  --seed 13 \
  --deterministic
```

Evaluate:

```bash
sentinel eval \
  --model quick-motion-v1 \
  --dataset-config datasets/quick_motion.yaml \
  --seed 13 \
  --deterministic
```

Export:

```bash
sentinel export \
  --model quick-motion-v1 \
  --format onnx \
  --optimize
```

Benchmark:

```bash
sentinel benchmark --profile fast_motion --frames 240
```

## Reproducibility

Each run writes metadata artifacts including:

- seed and deterministic flags
- resolved dataset config and dataset hash
- git SHA when available
- key metrics and run timestamp

Model manifests are stored as JSON (or YAML if desired) and can be looked up by alias through `artifacts/ml/registry.json`.

## Commit Hygiene

- Training/eval/export outputs default to `artifacts/ml/...`.
- This repo ignores `artifacts/` and `runs/` so generated experiment files are not committed by default.
- Keep dataset files and model weights external to the repository unless intentionally versioned.
