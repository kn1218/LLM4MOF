# Paper figures & data

Publication figures for the accompanying preprint (arXiv:2606.29459) and a map from each figure to the
experiment logs that produced it.

## Figures (`figures/`)

| File | Content | Backing data |
|------|---------|--------------|
| `Figure2.{png,pdf}` | Database-mode reasoning-beam comparison | all `database_mode/*` runs |
| `Figure3/<task>.{png,pdf}` | Search-space design surfaces (one panel per task) | one replicate per task (below) |
| `Figure4.{png,pdf}` | Live-discovery composite (H₂, 77 K & 160 K / 5 bar) | `live_simulation/H2_{77K,160K}_5bar/*` |
| `Figure5.{png,pdf,svg}` | Live performance + operating cost (H₂, 77 K / 5 bar) | `live_simulation/H2_77K_5bar/*` |

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

## Experimental data (archived on Zenodo)

The full closed-loop experiment logs behind these figures are archived on Zenodo (DOI: _pending_), not in this
repository (they are large). Structure:

```
experiments/
├── database_mode/     45 runs — 9 tasks × 5 replicates
└── live_simulation/   10 runs — H2_160K_5bar + H2_77K_5bar, 5 replicates each
```

Each `replicate_N/` holds the 10-iteration record: `raw_user_input.txt`, `conversation_history.json`,
`memory_ledger.json`, `usage_log.json`, and `iteration_1..10/` with `agent1_output.json`, `agent2_output.json`,
`beam_data.csv` (the full scored candidate pool — enables independent recomputation of every beam / design
surface / percentile), `feedback_selected.txt`, `sensitivity_report.csv`; live runs add `batch_manifest.json`
and `hpc_results/batch_results.json` (RASPA GCMC results).

**Live-simulation scope:** only the two live conditions used in the paper are included — H₂ 160 K / 5 bar and
H₂ 77 K / 5 bar.
