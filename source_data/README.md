# Source data for the paper: "Interpretable Inverse Design of Metal-Organic Frameworks with Large Language Model Agents"

This folder holds the data behind every figure and table of the paper, together
with the agent reasoning traces. The framework itself, the agent prompts, the
building-block library, the reference property tables used in database mode and
the force fields are elsewhere in this repository and are not duplicated here.

`MANIFEST.csv` lists every file with the display item it supports.

## Database mode (Figures 2 and 3, Supplementary Figures S3, S4, S5, S7)

`figure2_database_mode_beam_values.csv` carries the per-iteration values of all
four beams for nine design tasks in five independent replicates, which is what
the trajectory panels plot. `figure3_beam_descriptor_medians.csv` carries, for
each iteration and beam, the median target and the median of every pore
descriptor together with the number of matched candidates, which are the points
the descriptor-space panels plot. `figureS7_memory_ledger_ablation.csv` carries
the same per-iteration hypothesis-beam medians for the two configurations that
differ only in whether the memory ledger is active.

Volumetric uptakes are given in g/L and gravimetric uptakes in mol/kg, matching
the figures; the `unit` column states this per row.

## Discovery mode (Figures 4 and 5)

One row per successfully simulated structure, with the beam that proposed it,
the replicate, the iteration, the simulated property and the pore geometry
computed for that structure: hydrogen at 77 K and at 160 K, ethane/ethylene
selectivity at 298 K, and SF6 capture at 298 K, five replicates each.

For the two new chemistries two iteration columns are provided. `iteration`
counts productive iterations, which is what the figures plot, because a
campaign retries an iteration that returns no results and the raw numbering
therefore differs between replicates. `iteration_raw` is the number on disk,
kept so a trajectory can be audited.

`figure4_design_space_top100.csv` carries the individual structures behind the
design-space panel of Figure 4 and the metal counts beside it: the top hundred
hypothesis-beam structures for each of four database-mode conditions and two
discovery-mode conditions.

`figure6_token_usage_and_cost.csv` carries the operating cost of the closed
loop per iteration and replicate: model calls, cached and non-cached input
tokens, output tokens, and the inference cost recomputed from the published
price list rather than copied.

## Comparison with other approaches (Figures 6 and 7)

The genetic-algorithm and Bayesian-optimization baselines, five replicates
each, run in the same design space under the same evaluator and the same
evaluation budget as the closed loop. For the generative comparison, the
structures generated at three training-set sizes, 360, 1,000 and 18,463
property-labeled structures, with the 360-label case given per training
replicate and pooled. Those structures were evaluated with the generative
model's own simulation pipeline, which differs from the pipeline used for the
closed loop, so the two halves of Figure 7a are shown side by side and are not
ranked against each other.

## Language-model benchmark (Table 1, Table S7, Figures S8 and S9)

First-attempt scores for six language models across seven tasks and five
replicates, the per-task grid, the iterations needed to reach the database top
10%, and the replicate spread used for the stability control.

## Enrichment of top-tier candidates (Table S2)

`tableS2_hit_rates.csv` is the table itself. For each task it carries the
database top-1% threshold and, for the hypothesis beam and the random baseline,
how many of the candidates that beam proposed cleared the threshold, how many it
proposed in total, the resulting hit rate and the ratio between the two.

Both beams supply up to ten candidates per iteration under the same sampling
rule, pooled over five replicates and ten iterations, so the baseline reaches
five hundred while the hypothesis beam supplies fewer wherever its constraints
admitted fewer than ten. Every row can be rebuilt from
`figure2_database_mode_beam_values.csv`, which holds those same candidates one
per row.


## Agent reasoning traces

`agent_reasoning_traces_database_mode.jsonl` holds one record per iteration of
every database-mode run: the hypothesis the first agent produced and the
constraints the second agent derived from it. Together with the beam values in
the same deposit, this makes each design decision in the paper inspectable.

## Two things that are deliberately not duplicated here

The random-baseline beam draws from the full reference table, which the loop
records again at every iteration. Those rows are the property tables already
published with the code, so the deposit carries the sampled per-iteration
values that the figures use rather than about a gigabyte of repeated table.

## Geometry surrogate (Figure S6)

`figureS6_mof2zeo_validation_split.csv` is the held-out split the surrogate is
evaluated on, 84,228 assemblies with their reference Zeo++ descriptors. Running
the released checkpoint in the code repository over this file reproduces the
parity plot. The training split is not deposited; it is an order of magnitude
larger and nothing in the paper depends on it.

## Structures

`winner_structures/` holds the nine frameworks the paper names, as CIFs: the
four database-mode winners and the global best of Supplementary Table S1, and
the best structure of each of the four discovery campaigns. They are assembled
from the released building-block library, in the form the identifier denotes,
before the structural relaxation described in Supplementary Note S12. The hMOF
and QMOF winners are entries in public databases and are cited by their own
identifiers rather than copied here.

Every other structure in the deposit is given as its assembly identifier,
topology+node+edge, and can be rebuilt the same way.

## Two things to know before counting

The baseline files record the iterations of the search loop and begin at
iteration 2. Each replicate opens with a round of forty randomly drawn
structures, which is not part of the search and is not listed. Adding it back
gives the totals in Supplementary Note S14: 327 to 368 for the genetic
algorithm and 212 to 265 for the Bayesian optimizer.

Figures 3c and S3c place the hypothesis beam in a derived electronic space.
`figure3c_electronic_space_trajectory.csv` carries the trajectory itself and
`figure3c_metal_axis_order.csv` the ordering the metal axis uses. The background
distribution is computed from the reference index in the code repository.

## Column conventions

One name per thing, across every file: `structure` for the assembly identifier,
`replicate` for the run, `iteration` for the plotted step and `iteration_raw`
for the number on disk, `backend` for the language model, `beam` for the
diagnostic beam, and the measured quantity named with its unit, `uptake_g_L`.

The baseline files carry the evaluated uptake of every structure.

The wide backend tables use the short task keys as column headers. They map to
the readable names in the same order they appear in
`figure2_database_mode_beam_values.csv`, which carries both.

## Conventions

Beams are named as in the paper: Beam 1 is the full hypothesis, Beam 2 is
chemistry only, Beam 3 is metal only, and Beam 4 is the random baseline.
Structure identifiers follow the assembly convention topology+node+edge, so a
structure can be rebuilt from the building-block dictionaries in the code
repository.

## Reuse

Released under CC BY 4.0. If you use these data, please cite the paper and this
deposit.
