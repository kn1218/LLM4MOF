# mof2zeo - Geometry Prediction for MOFs

Deep learning model that predicts seven geometric descriptors (SA, CV, density, VF, Di, Df, Dif) from MOF building block combinations (topology + node + edge).

## Overview

mof2zeo is a PyTorch Lightning model following the MOF-NET architecture. It embeds three categorical
identifiers, combines them through an interaction layer and predicts each descriptor with its own
output head. The identifiers are:
- **Topology**: Network topology (e.g., pcu, sql, etc.)
- **Node**: Metal cluster / SBU
- **Edge**: Organic linker

Predicted descriptors are used to rank candidates in discovery mode before structures are assembled and relaxed.

## Structure

```
mof2zeo/
├── __init__.py           # Package init, exports version and root path
├── model.py              # MOFNET model class (PyTorch Lightning)
├── dataset.py            # CSVDataset and MOFGenDataset classes
├── train.py              # Training script
├── config.yaml           # Model hyperparameters (latent_dim, hid_dim, etc.)
├── ckpt/                 # Trained model checkpoint (Git LFS)
│   └── epoch=487-step=1039440.ckpt
└── data/                 # Vocabularies and normalization statistics
    ├── topology.txt      # 952 topologies
    ├── node.txt          # 518 nodes
    ├── edge.txt          # 156 linkers
    ├── feature_name.txt  # the seven descriptors, in output order
    ├── mean.csv          # per-descriptor mean used for normalization
    └── std.csv           # per-descriptor standard deviation
```

The training split is not shipped. The held-out split the model is evaluated on is
`source_data/figureS6_mof2zeo_validation_split.csv`.

## Usage (via filter_candidate.py)

The primary usage is through `filter_candidate.py`:

```bash
python core/filter_candidate.py \
  --constraints agent2_output.json \
  --output ranked_candidates.json \
  --top_n 10
```

### How It Works

1. **Input**: Constraints from Agent 2 (topology, node, edge requirements)
2. **Generate Combinations**: Create all valid topology+node+edge combinations
3. **Predict Geometry**: Run each combination through MOFNET to predict:
   - SA (accessible surface area)
   - CV (unit-cell volume)
   - density (framework density)
   - VF (void fraction)
   - Di (largest included sphere diameter)
   - Df (largest free sphere diameter)
   - Dif (diffusion-limiting sphere diameter)
4. **Rank**: Sort by target property match and output top N

## Model Architecture

Based on config.yaml:
- Latent dimension: 128
- Hidden dimensions: 64 → 32
- Output: 7 geometric descriptors

## Files

| File | Description |
|------|-------------|
| `model.py` | MOFNET class (PyTorch Lightning module) for descriptor prediction |
| `dataset.py` | CSVDataset and MOFGenDataset (PyTorch datasets) |
| `train.py` | Training script for model training |
| `config.yaml` | Hyperparameters (latent_dim, hid_dim, learning_rate, etc.) |

## Dependencies

- torch
- pytorch-lightning
- pandas
- numpy
- scikit-learn
- pyyaml

## Installation

The package is installed as part of LLM4MOF:

```bash
pip install -e .
```

The checkpoint ships via Git LFS:

```bash
git lfs pull
```

## Loading the model

`core/filter_candidate.py` loads the model. It reads `config.yaml`, the normalization statistics in
`data/mean.csv` and `data/std.csv`, and the checkpoint in `ckpt/`; the paths are set in the
`MOF2ZEO_*` constants of `config.py`.
