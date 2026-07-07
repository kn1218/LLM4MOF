# Baselines

The genetic-algorithm (GA) and Bayesian-optimization (BO) baselines the LLM4MOF framework is compared against in
the paper (Figure 5: H₂ uptake, 77 K / 5 bar). Both run over the PORMAKE design space using the **same live
simulation backend as the framework's discovery mode** (PORMAKE assembly → force-field relaxation → RASPA3 GCMC),
and share the same evaluation budget: 5 independent replicates, each starting from 40 random MOFs plus 9
surrogate-guided iterations that retain up to 40 successful evaluations each (≈ 400 evaluations per replicate).

## `genetic_algorithm/`

GA over PORMAKE (topology, node, edge) following **SI Note S6**: per-topology evolution with a population of 200
over 20 generations (mutation probability 0.2); each generation submits its top-80 candidates for live simulation,
of which up to 40 successful evaluations are retained.

- `results/ga_rep{1..5}.csv` (+ `_iter1`) — the per-iteration additions plotted in Figure 5
  (columns: `filename, uptake, surrogate_pred_g_L, surrogate_rank, iteration`).
- **Orchestration code** (`ga_topo`, `ga_sim_worker`, and the per-topology driver) runs on the HPC scheduler and
  will be added here; it shares the live simulation backend with the framework's discovery mode. The authoritative
  method description is SI Note S6.

> Note: an earlier version of this folder contained a prototype pool-lookup GA script that did **not** produce the
> Figure 5 results and did not match Note S6; it has been removed to avoid confusion. The result files above are
> from the live GA described in Note S6.

## `bayesian_optimization/`

Iterative LVGP-BO following **SI Note S7**: a Latent-Variable Gaussian Process surrogate (GPyTorch) over the
categorical design space with Expected-Improvement acquisition; each iteration scores candidates by EI and submits
the top-80 for live simulation, retaining up to 40. See `METHOD.md` for the full method and references.

- `bo_iterative_run.py`, `lvgp_surrogate.py` — the BO driver and surrogate (as used).
- `results/bo_rep{1..5}.csv` (+ `_iter1`) — the per-iteration additions plotted in Figure 5.

## Notes on re-use

Cluster-specific paths are replaced with placeholders (`<HPC_WORK>`, `<PROJECT_ROOT>`, `<HOME>`, `<env>`). The BO
driver imports the GA orchestration modules (`ga_topo`) and submits to a scheduler, so it becomes runnable once
those modules are added; it is provided for transparency of the comparison. No API keys are required.
