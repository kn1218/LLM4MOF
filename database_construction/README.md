# Database construction

The pipelines that build and **LLM-enrich** the evaluation databases and the unified functional-group
vocabulary that the framework screens against. The **final products** ship in [`../data/`](../data/); this folder
is the *provenance* — how those files were constructed — provided for transparency and reproducibility.

## Pipelines

| Folder | Builds | Shipped product (`../data/`) |
|--------|--------|------------------------------|
| `1_PORMAKE_pipeline/` | PORMAKE building-block metadata (XYZ → SMILES → SMARTS → LLM tags → BB dictionary) | `pormake_bb_dictionary_v7.json` |
| `2_QMOF_pipeline/` | QMOF metal/linker enrichment + band-gap index | `qmof_index_v2.json`, `qmof.csv` |
| `3_hMOF_pipeline/` | hMOF metal-node + LLM tag enrichment + index | `hMOF/hmof_index.json` |
| `4_vocabulary/` | Unified functional-group vocabulary (consolidates the per-database tags) | `unified_vocabulary.json` |
| `5_shared/` | Shared SMARTS library used across pipelines | — |
| `6_CoRE-MOF_pipeline/` | CoRE-MOF enrichment — **additional framework tooling; not used in the paper** | (not shipped) |

Each pipeline's `source_samples/` and `output_samples/` hold small before/after examples so the
transformations are legible without the full multi-GB source databases.

## Method summary

See [`SI_METHODOLOGY.md`](SI_METHODOLOGY.md) for the full write-up (per-database steps, the LLM-enrichment
prompts and provider used, cross-validation, and known limitations).

## Running

The LLM-enrichment steps call OpenAI / Google Gemini and read the key from the environment (`OPENAI_API_KEY`,
`GEMINI_API_KEY`) via `.env` — the same convention as the main repo (copy `../.env.example` to `.env`). No keys
are hardcoded. The non-LLM steps (parsing, SMARTS, indexing) need no keys. Scripts are numbered in execution
order within each folder; paths resolve relative to each script.

> These scripts are provided as the **as-used construction code**. They reproduce the shipped databases from the
> raw third-party sources (PORMAKE, QMOF, hMOF — cited in [`../docs/DATA.md`](../docs/DATA.md)); the raw source
> databases themselves are not redistributed here.
