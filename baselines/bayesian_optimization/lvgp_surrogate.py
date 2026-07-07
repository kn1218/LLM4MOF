"""
lvgp_surrogate.py — Latent Variable Gaussian Process surrogate for MOF H₂ uptake.

Input : (topology, node, edge) categorical
      → 2-D latent vector per variable  → R^6 combined input
      → ExactGP with ARD RBF kernel
      → posterior mean μ(x), std σ(x)

Training: maximize marginal log likelihood (Adam).
          85:15 train/val split for metric logging (same convention as MOFEncoder surrogate).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import gpytorch
from scipy.stats import norm as scipy_norm

# ── vocabulary (same indices as MOFEncoder surrogate) ──────────────────────
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJ_ROOT, "core", "mof2zeo"))
from dataset import TOPO_DICT, NODE_DICT, EDGE_DICT

N_TOPOS    = len(TOPO_DICT)   # 952
N_NODES    = len(NODE_DICT)
N_EDGES    = len(EDGE_DICT)
LATENT_DIM = 2                # 2-D latent per variable (paper convention)


# ── Scaler ─────────────────────────────────────────────────────────────────
class _Scaler:
    """Standardise targets (zero mean, unit std)."""

    def fit(self, y: np.ndarray) -> "_Scaler":
        self.mean_ = float(y.mean())
        self.std_  = float(y.std()) + 1e-8
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean_) / self.std_

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y * self.std_ + self.mean_

    def state(self) -> dict:
        return {"mean": self.mean_, "std": self.std_}

    def load(self, d: dict) -> None:
        self.mean_ = d["mean"]
        self.std_  = d["std"]


# ── LVGP model ─────────────────────────────────────────────────────────────
class LVGPModel(gpytorch.models.ExactGP):
    """
    Latent Variable GP.
    train_x : (N, 3) LongTensor   [topo_id, node_id, edge_id]
    train_y : (N,)   FloatTensor  (standardised)
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=LATENT_DIM * 3)  # 6-D ARD
        )
        self.topo_emb = nn.Embedding(N_TOPOS, LATENT_DIM)
        self.node_emb = nn.Embedding(N_NODES, LATENT_DIM)
        self.edge_emb = nn.Embedding(N_EDGES, LATENT_DIM)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3) long → (N, 6) float latent."""
        return torch.cat([
            self.topo_emb(x[:, 0]),
            self.node_emb(x[:, 1]),
            self.edge_emb(x[:, 2]),
        ], dim=1)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        z = self.encode(x)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(z),
            self.covar_module(z),
        )


# ── helpers ────────────────────────────────────────────────────────────────
def _make_tensors(data_list: list, device: str) -> tuple:
    valid = [d for d in data_list
             if d["topo"] in TOPO_DICT
             and d["node"] in NODE_DICT
             and d["edge"] in EDGE_DICT]
    if not valid:
        return None, None
    x = torch.tensor(
        [[TOPO_DICT[d["topo"]], NODE_DICT[d["node"]], EDGE_DICT[d["edge"]]] for d in valid],
        dtype=torch.long, device=device,
    )
    y = torch.tensor([d["target"] for d in valid], dtype=torch.float32, device=device)
    return x, y


# ── train ──────────────────────────────────────────────────────────────────
def train_lvgp(
    data_list: list,
    val_ratio: float = 0.15,
    n_steps: int = 300,
    lr: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
) -> tuple:
    """
    Train LVGP on data_list (list of {topo, node, edge, target} dicts).
    Returns (model, likelihood, scaler, val_metrics_dict).
    """
    torch.manual_seed(seed)
    x_all, y_all = _make_tensors(data_list, device)
    if x_all is None or len(x_all) < 5:
        raise RuntimeError(f"Too few valid training samples ({len(data_list)} given)")

    n       = len(x_all)
    n_train = max(4, int(n * (1 - val_ratio)))
    perm    = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr_idx, va_idx = perm[:n_train], perm[n_train:]

    x_tr, y_tr = x_all[tr_idx], y_all[tr_idx]
    x_va, y_va = x_all[va_idx], y_all[va_idx]

    scaler   = _Scaler().fit(y_tr.cpu().numpy())
    y_tr_sc  = torch.tensor(
        scaler.transform(y_tr.cpu().numpy()), dtype=torch.float32, device=device
    )

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model      = LVGPModel(x_tr, y_tr_sc, likelihood).to(device)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll       = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = -mll(model(x_tr), y_tr_sc)
        loss.backward()
        optimizer.step()

    # ── val metrics ──────────────────────────────────────────────────────
    metrics = {"n_train": n_train, "n_val": int(len(va_idx))}
    if len(va_idx) > 1:
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred      = likelihood(model(x_va))
            mean_orig = scaler.inverse(pred.mean.cpu().numpy())
        y_va_np = y_va.cpu().numpy()
        ss_res  = float(np.sum((y_va_np - mean_orig) ** 2))
        ss_tot  = float(np.sum((y_va_np - y_va_np.mean()) ** 2))
        metrics["val_r2"]  = round(1 - ss_res / ss_tot if ss_tot > 0 else 0.0, 4)
        metrics["val_mae"] = round(float(np.mean(np.abs(y_va_np - mean_orig))), 4)

    return model, likelihood, scaler, metrics


# ── predict ────────────────────────────────────────────────────────────────
def predict_with_uncertainty(
    model: LVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    scaler: _Scaler,
    keys: list,
    device: str = "cpu",
) -> tuple:
    """
    keys : list of 'topo+node+edge' strings.
    Returns (mean_orig, std_orig, valid_indices).
    Invalid keys are silently skipped.
    """
    records, valid_idx = [], []
    for i, key in enumerate(keys):
        parts = key.split("+")
        if len(parts) != 3:
            continue
        t, n, e = parts
        if t in TOPO_DICT and n in NODE_DICT and e in EDGE_DICT:
            records.append([TOPO_DICT[t], NODE_DICT[n], EDGE_DICT[e]])
            valid_idx.append(i)

    if not records:
        return np.array([]), np.array([]), []

    x = torch.tensor(records, dtype=torch.long, device=device)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred    = likelihood(model(x))
        mean_sc = pred.mean.cpu().numpy()
        std_sc  = pred.variance.sqrt().cpu().numpy()

    mean_orig = scaler.inverse(mean_sc)
    std_orig  = std_sc * scaler.std_
    return mean_orig, std_orig, valid_idx


# ── EI ─────────────────────────────────────────────────────────────────────
def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    f_best: float,
) -> np.ndarray:
    """
    EI(x) = (μ − f*)·Φ(Z) + σ·φ(Z),   Z = (μ − f*) / σ
    Returns 0 where σ ≈ 0.
    """
    ei   = np.zeros_like(mean)
    mask = std > 1e-9
    Z    = (mean[mask] - f_best) / std[mask]
    ei[mask] = (
        (mean[mask] - f_best) * scipy_norm.cdf(Z) + std[mask] * scipy_norm.pdf(Z)
    )
    return np.maximum(ei, 0.0)


# ── checkpoint ─────────────────────────────────────────────────────────────
def save_checkpoint(
    model: LVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    scaler: _Scaler,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state":      model.state_dict(),
        "likelihood_state": likelihood.state_dict(),
        "scaler":           scaler.state(),
        "train_x":          model.train_inputs[0].cpu(),
        "train_y":          model.train_targets.cpu(),
    }, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple:
    ckpt       = torch.load(path, map_location=device, weights_only=False)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model      = LVGPModel(
        ckpt["train_x"].to(device),
        ckpt["train_y"].to(device),
        likelihood,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    likelihood.load_state_dict(ckpt["likelihood_state"])
    scaler = _Scaler()
    scaler.load(ckpt["scaler"])
    model.eval()
    likelihood.eval()
    return model, likelihood, scaler
