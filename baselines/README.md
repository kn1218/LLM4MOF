# Baselines

The genetic-algorithm (GA) and Bayesian-optimization (BO) baselines compared against the LLM4MOF framework
in the paper (Figure 5: H₂ uptake, 77 K / 5 bar). Both follow the **same protocol** as the framework for a fair
comparison — 5 independent replicates, each starting from the same 40 random MOFs, adding the top-40 candidates
per iteration over 10 iterations.

## `genetic_algorithm/`
- `run_ga_mof2zeo.py` — GA over PORMAKE (topology, node, edge): crossover and 1-component mutation from the
  observed MOFs, candidates scored by the mof2zeo surrogate, top-40 selected per iteration (`--n-runs 5`).
- `candidate_pool.csv` — the enumerated PORMAKE candidate space (building-block combinations + target) the GA
  searches over.
- `results/ga_rep{1..5}.csv` (+ `_iter1`) — per-iteration additions used in Figure 5.

## `bayesian_optimization/`
- `bo_iterative_run.py`, `lvgp_surrogate.py` — iterative LVGP-BO: a Latent-Variable Gaussian Process surrogate
  (GPyTorch) over the categorical design space, Expected-Improvement acquisition, top-80 → top-40 per iteration.
- `METHOD.md` — full method and the reference framework (Iyer et al., 2023) it adapts.
- `results/bo_rep{1..5}.csv` (+ `_iter1`) — per-iteration additions used in Figure 5.

## Notes on re-use
These are the scripts **as used**. Cluster-specific paths have been replaced with placeholders
(`<HPC_WORK>`, `<PROJECT_ROOT>`, `<HOME>`, `<env>`) — set them for your environment. The BO orchestration imports
the GA HPC worker modules (`ga_hpc_worker`, `ga_topo`) and submits simulations to a scheduler, so it is provided
for transparency of the comparison rather than as a turnkey pipeline. The surrogate steps read the mof2zeo model
(`../core/mof2zeo/`); no API keys are required.
