# Bayesian Optimization Baseline for MOF H₂ Uptake

## Overview

This experiment implements a Bayesian Optimization (BO) baseline for iterative MOF discovery,
following the LVGP-based Bayesian optimization of Comlek et al. (2023), building on the mixed-variable framework of Iyer et al. (2023), adapted for single-objective optimization.

**References:**
> Comlek, Y., et al. "Rapid design of top-performing metal-organic frameworks with qualitative representations of building blocks." *npj Computational Materials* (2023).
>
> Iyer, A., et al. "Data-centric mixed-variable Bayesian optimization for materials design." *arXiv:2302.09184* (2023).

---

## Problem Formulation

**Goal:** Maximize H₂ gravimetric uptake (g/L) at 77 K, 5 bar.

**Design variables:** Each MOF is represented as a triple of categorical variables:

$$\mathbf{x} = (\text{topology},\ \text{node},\ \text{edge})$$

- Topology: 952 levels (PORMAKE topology vocabulary)
- Node: ~800 levels (PORMAKE building block nodes)
- Edge: ~300 levels (PORMAKE building block edges)

**Feasibility:** Not all (topology, node, edge) combinations are valid PORMAKE structures.
Valid candidates are enumerated using the PORMAKE dictionary per topology.

**Oracle:** PORMAKE structure generation → LAMMPS geometry optimization → RASPA3 GCMC simulation (77 K, 5 bar).

---

## Surrogate Model: Latent Variable Gaussian Process (LVGP)

### Motivation

Standard GP cannot handle categorical inputs directly. LVGP (Zhang et al., 2020) maps each
categorical variable to a continuous latent space, enabling GP regression over discrete inputs.

### Latent Variable Mapping

Each level of each categorical variable is assigned a 2D latent vector:

$$\phi: x_i \mapsto \mathbf{z}_i \in \mathbb{R}^2$$

For a MOF $\mathbf{x} = (\text{topo}, \text{node}, \text{edge})$, the GP input is:

$$\mathbf{z}(\mathbf{x}) = \left[\mathbf{z}_\text{topo},\ \mathbf{z}_\text{node},\ \mathbf{z}_\text{edge}\right] \in \mathbb{R}^6$$

The latent vectors are **jointly learned** with the GP hyperparameters by maximizing the log marginal likelihood.

### GP Model

$$f(\mathbf{x}) \sim \mathcal{GP}\left(\mu,\ k(\mathbf{z}(\mathbf{x}),\ \mathbf{z}(\mathbf{x}'))\right)$$

Kernel: RBF (squared exponential) on the latent space:

$$k(\mathbf{z}, \mathbf{z}') = \sigma_f^2 \exp\!\left(-\frac{1}{2} \sum_{d=1}^{6} \frac{(z_d - z_d')^2}{\ell_d^2}\right)$$

where $\sigma_f^2$ is the signal variance and $\ell_d$ are per-dimension length scales (ARD).

### Posterior Prediction

Given training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$, the GP posterior at a new point $\mathbf{x}^*$ is:

$$p(f(\mathbf{x}^*) \mid \mathcal{D}) = \mathcal{N}\!\left(\mu(\mathbf{x}^*),\ \sigma^2(\mathbf{x}^*)\right)$$

$$\mu(\mathbf{x}^*) = \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}$$

$$\sigma^2(\mathbf{x}^*) = k_{**} - \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$$

where $K_{ij} = k(\mathbf{z}(\mathbf{x}_i), \mathbf{z}(\mathbf{x}_j))$, $\mathbf{k}_* = [k(\mathbf{z}(\mathbf{x}^*), \mathbf{z}(\mathbf{x}_i))]_i$, and $\sigma_n^2$ is observation noise.

---

## Acquisition Function: Expected Improvement (EI)

Since this is a **single-objective** problem (unlike the multi-objective EMI in the original paper),
we use standard Expected Improvement:

$$\text{EI}(\mathbf{x}) = \mathbb{E}\left[\max\!\left(f(\mathbf{x}) - f^*,\ 0\right)\right]$$

Given the GP posterior $f(\mathbf{x}) \sim \mathcal{N}(\mu(\mathbf{x}),\ \sigma^2(\mathbf{x}))$, this has a closed form:

$$\text{EI}(\mathbf{x}) = \underbrace{(\mu(\mathbf{x}) - f^*)\ \Phi(Z)}_{\text{exploitation}} + \underbrace{\sigma(\mathbf{x})\ \phi(Z)}_{\text{exploration}}$$

$$Z = \frac{\mu(\mathbf{x}) - f^*}{\sigma(\mathbf{x})}$$

where $f^* = \max_i y_i$ is the current best observed value, $\Phi$ is the standard normal CDF, and $\phi$ is the standard normal PDF.

**Comparison with UCB:**

$$\text{UCB}(\mathbf{x}) = \mu(\mathbf{x}) + \kappa\, \sigma(\mathbf{x})$$

EI and UCB both exploit $\mu$ and explore via $\sigma$, but EI automatically balances the trade-off
through $f^*$ without a manual hyperparameter $\kappa$.

---

## Iterative BO Protocol

```
iter 1:  40 random MOFs per rep (same as GA baseline, seed = rep index)

iter k (k = 2 ... 10):
  1. Train LVGP on accumulated data D_{k-1}
  2. Generate 952 × 3 = 2,856 candidate MOFs via PORMAKE worker
     (top-3 per topology, same as GA)
  3. Compute EI(x) for all candidates using LVGP posterior
  4. Select top-80 by EI → submit to HPC (LAMMPS + RASPA3)
  5. Collect simulation results → take top-40 by actual uptake
  6. D_k = D_{k-1} ∪ {top-40}
```

**5 independent replicates** (rep 1–5), each starting from the same 40 random MOFs as the GA baseline.

---

## Implementation

- **GP framework:** GPyTorch (ExactGP)
- **Latent embeddings:** `torch.nn.Embedding(n_levels, 2)` per variable, jointly optimized with GP hyperparameters via Adam (marginal likelihood)
- **Candidate generation:** Reuses GA PORMAKE worker infrastructure (`ga_hpc_worker.py`)
- **HPC simulation:** Same pipeline as GA baseline (PORMAKE → LAMMPS → RASPA3 on an HPC cluster)

---

## Differences from Iyer et al. (2023)

| Aspect | Iyer et al. | This work |
|--------|------------|-----------|
| Objective | CO₂ working capacity + selectivity (2 objectives) | H₂ uptake at 77 K, 5 bar (1 objective) |
| Acquisition | EMI (Expected Maximin Improvement) | EI (Expected Improvement) |
| Topology | fof (fixed) | 952 topologies |
| Design space | 47,740 MOFs (fixed pool) | Open-ended PORMAKE space |
| Batch size | varies | 80 candidates/iter → top-40 added |
