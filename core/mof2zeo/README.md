# mof2zeo - Geometry Prediction for MOFs

Deep learning model that predicts geometric properties (Di, Df, SA, VF, density, CV, Dif) from MOF building block combinations (topology + node + edge).

## Overview

mof2zeo is a PyTorch Lightning-based graph neural network that predicts MOF geometric properties from:
- **Topology**: Network topology (e.g., pcu, sql, etc.)
- **Node**: Metal cluster / SBU
- **Edge**: Organic linker

Predicted properties are used to rank MOF candidates when no experimental data is available in databases.

## Structure

```
mof2zeo/
├── __init__.py           # Package init, exports version and root path
├── model.py              # MOFNET model class (PyTorch Lightning)
├── dataset.py            # CSVDataset and Scaler classes
├── train.py               # Training script
├── config.yaml           # Model hyperparameters (latent_dim, hid_dim, etc.)
├── ckpt/                 # Trained model checkpoint
│   └── epoch=478-step=213634.ckpt
├── data/                  # Training/validation data
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   ├── topology.txt
│   ├── node.txt
│   ├── edge.txt
│   └── feature_name.txt
└── scaler/               # Feature scalers for inverse transform
    ├── mean_all.csv
    └── std_all.csv
```

## Usage (via filter_candidate.py)

The primary usage is through `filter_candidate.py`:

```bash
python core/filter_candidate.py \
  --constraints agent2_output.json \
  --output test_result_agent3.json \
  --top_n 10
```

### How It Works

1. **Input**: Constraints from Agent 2 (topology, node, edge requirements)
2. **Generate Combinations**: Create all valid topology+node+edge combinations
3. **Predict Geometry**: Run each combination through MOFNET to predict:
   - Di (pore diameter)
   - Df (framework density)
   - SA (surface area)
   - VF (void fraction)
   - density
   - CV (pore volume)
   - Dif (diffusivity)
4. **Rank**: Sort by target property match and output top N

## Model Architecture

Based on config.yaml:
- Latent dimension: 128
- Hidden dimensions: 64 → 32
- Output: 7 geometric properties

## Files

| File | Description |
|------|-------------|
| `model.py` | MOFNET class - Graph neural network for MOF property prediction |
| `dataset.py` | CSVDataset (PyTorch Dataset) and Scaler for data preprocessing |
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

The package is installed as part of LLM2POR:

```bash
pip install -e .
```

The checkpoint is automatically downloaded via Git LFS:

```bash
git lfs pull
```

## Direct Model Usage

```python
import yaml
from mof2zeo.model import MOFNET
from mof2zeo.dataset import Scaler

# Load config
with open("core/mof2zeo/config.yaml") as f:
    config = yaml.safe_load(f)

# Load scaler
scaler = Scaler.load("core/mof2zeo/scaler/")

# Load model
model = MOFNET.load_from_checkpoint(
    "core/mof2zeo/ckpt/epoch=478-step=213634.ckpt",
    config=config,
    scaler=scaler,
)

# Predict
predictions = model.predict(topology, node, edge)
```