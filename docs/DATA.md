# Data Availability

LLM4MOF evaluates LLM-proposed designs against three precomputed MOF databases and one
geometry-prediction model checkpoint. These files total ~200 MB and **ship in this repository via
Git LFS** for immediate use after cloning (`git lfs pull`).

> **Archival note:** these files — and the closed-loop experiment logs behind the paper's figures —
> will be deposited in a public archive with a permanent DOI **upon publication**. During peer review
> they are **available from the authors upon reasonable request**; if the GitHub LFS bandwidth quota is
> exhausted, request them from the authors.

## Contents

| File | Size | Role |
|------|------|------|
| `data/hMOF/hmof_index.json` | 66 MB | hMOF gas-adsorption database (CH₄, CO₂, Xe/Kr, H₂ uptakes) |
| `data/qmof_index_v2.json` | 26 MB | QMOF electronic-structure index (band gaps) |
| `data/qmof.csv` | 22 MB | QMOF property table |
| `core/mof2zeo/ckpt/epoch=487-step=1039440.ckpt` | 79 MB | mof2zeo geometry-prediction model checkpoint |
| `data/pormake_bb_dictionary_v7.json` | 1.6 MB | PORMAKE building-block dictionary (nodes + edges) |
| `data/pormake_topo_dictionary_v3.json` | 0.5 MB | PORMAKE topology dictionary |
| `data/total_characteristics_h2_{100bar,5bar}_77K[_gperL,_mol_kg].csv` | ~7 MB (6 files) | PORMAKE-assembled H₂ markschemes (2 pressures × 3 units) |
| `data/unified_vocabulary.json` | 0.03 MB | Canonical functional-group vocabulary |

Approximate database scale: PORMAKE building-block design space ~14k single-node assemblies,
hMOF ~51k structures, QMOF ~20k structures.

## Integrity (SHA-256)

```
0afbc062d088fc2b4a47cfa123458bc30f13130368dc36e9c02ec740741a8055  data/hMOF/hmof_index.json
420083c16db58e8775f2060e201588a7e942d4a2adca43b0c9d981b5a4022c4f  data/qmof.csv
ff8383e983752d238389494884e7d2761c62b1f540df2f58afc76af36bbf2da9  data/qmof_index_v2.json
3bc85271bdc5f531908703349ef512bf0cfec60dfd208a489a05b1fa2b041023  data/pormake_bb_dictionary_v7.json
107fc20aa45af565672259c69263fde2385c8be106bb0da1ac1dc0febac075c7  data/pormake_topo_dictionary_v3.json
2b96c8a42a627f6923542c08d295437c5b122bd45206a750fc26bd00b4a013c4  data/total_characteristics_h2_100bar_77K.csv
d0ea1ad252b222f6168a2eb3f32f570f0bf7fc87a0a8a1f065cb7ba3de9c7417  data/total_characteristics_h2_100bar_77K_gperL.csv
86ebd17109e8b6d9642f8693ea78b5f4df411e825144a26d3f46bb58699d5f7b  data/total_characteristics_h2_100bar_77K_mol_kg.csv
5a7a7edf63ff2af9fbe636a19e2d8a0e24ec72327275350813547dda943b8f31  data/total_characteristics_h2_5bar_77K.csv
e19609f25eb3827837412bdbe7c43a52a2667743b91e309ce9af2b6b24d0c18f  data/total_characteristics_h2_5bar_77K_gperL.csv
419c5d1b41003dd94b802eceed5ee2280216d7d263dc4418a98b2dc5a38dd2db  data/total_characteristics_h2_5bar_77K_mol_kg.csv
d923e38f9b9800c61b89a0b0fd49bd53f0711fe3e5146ddaed91efec02d65854  data/unified_vocabulary.json
4469eff79bafb243a7921de8ea60ea8b0a8f9f781628910f3d28e131d1b6c9ab  core/mof2zeo/ckpt/epoch=487-step=1039440.ckpt
```

Verify after download with `sha256sum -c` (or `Get-FileHash -Algorithm SHA256` on Windows).

## Source databases (third-party provenance)

- **hMOF** — derived from the hypothetical MOF database (Wilmer et al., *Nat. Chem.* 2012) and the
  authors' published reduced subset. Gas-uptake values are precomputed GCMC results.
- **QMOF** — derived from the Quantum MOF database (Rosen et al., *Matter* 2021); DFT-computed
  band gaps.
- **PORMAKE markschemes** — H₂ uptakes for single-node/single-edge assemblies generated with
  PORMAKE (Lee et al.) and evaluated under the conditions in `data/DATABASE_CONDITIONS.md`.

Users redistributing these data should also cite the original database publications above.

## Fetching the data

**From the repository (Git LFS):**
```bash
git lfs install
git lfs pull
```

**If the GitHub LFS quota is exhausted:** request the files from the authors (a public archive with a
permanent DOI will be linked here upon publication).

## Planned archival (maintainer steps, on publication)

1. Bundle the 13 files above (preserving relative paths) into a single archive, e.g.
   `llm4mof_data_v1.zip`.
2. Create a new deposit at <https://zenodo.org> → upload the archive → set the title
   (e.g. *"LLM4MOF evaluation databases and mof2zeo checkpoint"*), authors, and an open license
   (CC-BY-4.0 is typical for data).
3. Reserve/publish to mint the DOI, then add it to the **Archival note** above and to the paper's
   Data Availability statement.
4. (Optional) Link the GitHub release to Zenodo so future tagged releases are auto-archived.
