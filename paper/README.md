# Paper figures & data

Publication figures for the accompanying preprint (arXiv:2606.29459) and a map from each figure to the
experiment logs that produced it.

## Figures (`figures/`)

| File | Content | Backing data |
|------|---------|--------------|
| `Figure1.{png,svg}` | Framework schematic (method overview) | — (schematic) |
| `Figure2.{png,pdf}` | Database-mode reasoning-beam comparison | all `database_mode/*` runs |
| `Figure3/<task>.{png,pdf}` | Search-space design surfaces, the descriptor-space panels of Figures 3, S3 and S4 (one panel per task) | one replicate per task (below) |
| `Figure4.{png,pdf}` | Live-discovery composite (H₂, 77 K & 160 K / 5 bar) | `live_simulation/H2_{77K,160K}_5bar/*` |
| `Figure5.{png,pdf,svg}` | Live performance + operating cost (H₂, 77 K / 5 bar); Figure 6 of the current version | `live_simulation/H2_77K_5bar/*` |

File names follow the numbering of the first preprint version. Figures 5 and 7 of the current
version are not provided as image files; the values behind them are in
[`source_data/`](../source_data/).

**Figure 3 panels** — the replicate each was drawn from:

| Panel | Backing run |
|-------|-------------|
| `H2_volumetric_5bar`   | `database_mode/H2_volumetric_5bar/replicate_4` |
| `H2_volumetric_100bar` | `database_mode/H2_volumetric_100bar/replicate_3` |
| `H2_gravimetric_5bar`  | `database_mode/H2_gravimetric_5bar/replicate_5` |
| `H2_gravimetric_100bar`| `database_mode/H2_gravimetric_100bar/replicate_5` |
| `methane`              | `database_mode/methane/replicate_1` |
| `CO2`                  | `database_mode/CO2/replicate_5` |
| `Xe_Kr`                | `database_mode/Xe_Kr/replicate_4` |
| `bandgap_high`         | `database_mode/bandgap_high/replicate_1` |
| `bandgap_low`          | `database_mode/bandgap_low/replicate_5` |

## Experimental data

The values behind every figure and table, and the agent reasoning traces, are in
[`source_data/`](../source_data/). The full closed-loop experiment logs are not included in this
repository (~1.6 GB).
They are **available from the authors upon reasonable request**, and will be deposited in a public archive
with a DOI upon publication. Structure:

```
experiments/
├── database_mode/     45 runs — 9 tasks × 5 replicates
└── live_simulation/   20 runs — H2_77K_5bar, H2_160K_5bar, C2H6_C2H4, SF6, 5 replicates each
```

Each `replicate_N/` holds the 10-iteration record: `raw_user_input.txt`, `conversation_history.json`,
`memory_ledger.json`, `usage_log.json`, and `iteration_1..10/` with `agent1_output.json`, `agent2_output.json`,
`beam_data.csv` (the full scored candidate pool — enables independent recomputation of every beam / design
surface / percentile), `feedback_selected.txt`, `sensitivity_report.csv`; live runs add `batch_manifest.json`
and `hpc_results/batch_results.json` (RASPA GCMC results).

**Live-simulation scope:** the four discovery campaigns of the paper — H₂ at 77 K and at 160 K (5 bar),
and C₂H₆/C₂H₄ selectivity and SF₆ uptake (298 K, 1 bar).
