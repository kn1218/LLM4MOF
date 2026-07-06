"""LLM-based Layer 2 enrichment for MOF records (QMOF, CoRE-MOF, hMOF).

Fills readable_name, design_hints, and llm_additions using Gemini (primary)
or OpenAI (fallback). Supports batch processing with checkpoint/resume and
rate limiting. Auto-detects database from record keys.

Usage:
    # QMOF (default)
    python -m qmof.qmof_pipeline.llm_enricher --provider openai --batch-size 100
    # CoRE-MOF
    python -m qmof.qmof_pipeline.llm_enricher --provider openai --input-dir CoRE-MOF/coremof_enriched_v2
    # hMOF
    python -m qmof.qmof_pipeline.llm_enricher --provider openai --input-dir hMOF/hmof_enriched_v2
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


def _safe_print(text: str) -> None:
    """Print with fallback for non-UTF-8 terminals (e.g. cp949 on Korean Windows)."""
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")  # "gemini" or "openai"
RATE_LIMIT_DELAY = float(os.environ.get("LLM_RATE_LIMIT", "0.3"))  # seconds

BASE_DIR = Path(__file__).parent.parent.parent  # database_construction
DEFAULT_INPUT_DIR = BASE_DIR / "qmof" / "qmof_enriched_v2"

# ---------------------------------------------------------------------------
# Database detection
# ---------------------------------------------------------------------------

DB_ID_KEYS = {"qmof_id": "QMOF", "coremof_id": "CoRE-MOF", "hmof_id": "hMOF"}


def _detect_database(record: dict[str, Any]) -> str:
    """Detect database from record keys. Returns 'QMOF', 'CoRE-MOF', or 'hMOF'."""
    for key, db in DB_ID_KEYS.items():
        if key in record:
            return db
    return record.get("database", "QMOF")


def _get_record_id(record: dict[str, Any]) -> str:
    """Extract the record ID regardless of database."""
    for key in DB_ID_KEYS:
        if key in record:
            return record[key]
    return "unknown"


def _get_checkpoint_path(input_dir: Path) -> Path:
    """Derive per-database checkpoint path from input directory."""
    # Determine which database based on input dir name
    dir_name = input_dir.name.lower()
    if "coremof" in dir_name:
        return BASE_DIR / "CoRE-MOF" / "_audit" / "llm_enrichment_checkpoint.json"
    elif "hmof" in dir_name:
        return BASE_DIR / "hMOF" / "_audit" / "llm_enrichment_checkpoint.json"
    else:
        return BASE_DIR / "qmof" / "_audit" / "llm_enrichment_checkpoint.json"


# ---------------------------------------------------------------------------
# System prompts (database-specific)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are a reticular chemistry expert specializing in Metal-Organic Frameworks (MOFs).
{db_context}

Given the MOF's metadata, provide:
1. readable_name: A concise human-friendly name for this MOF (2-10 words). \
Use IUPAC-style naming where possible. Include the metal and linker type. \
Examples: "copper paddlewheel BDC framework", "zirconium UiO-66 analog", \
"zinc imidazolate ZIF".
2. design_hints: 1-3 sentences about potential applications and performance. \
Reference specific properties from the data. Be specific to THIS MOF.
3. llm_additions: A list of functional groups or structural motifs that \
rule-based SMARTS detection might have missed. Only include groups you are \
confident about. Common misses: paddlewheel, UiO-type, MIL-type, ZIF-type, \
HKUST-type, MOF-5-type, porphyrin-based, pillared-layer, interpenetrated, \
breathing, flexible.

Respond in JSON format only:
{{"readable_name": "...", "design_hints": "...", "llm_additions": ["motif1", "motif2"]}}

Rules:
- If llm_additions has nothing new beyond rule_based groups, return [].
- Do NOT repeat groups already in rule_based.
- readable_name should be lowercase except proper nouns/acronyms (UiO, ZIF, MIL, HKUST, MOF-5).
- Keep it factual. No speculation beyond what the data supports."""

SYSTEM_PROMPTS = {
    "QMOF": _SYSTEM_PROMPT_BASE.format(
        db_context="You are analyzing a MOF record from the QMOF database — a real experimentally "
        "characterized MOF with DFT-computed electronic properties (bandgap). "
        "Reference pore size (LCD/PLD), bandgap, topology, and metal chemistry in your design hints."
    ),
    "CoRE-MOF": _SYSTEM_PROMPT_BASE.format(
        db_context="You are analyzing a MOF record from the CoRE-MOF database — a real experimentally "
        "synthesized MOF with stability data (thermal, water, solvent stability) and structural "
        "characterization. Reference stability properties, open metal sites, surface area, "
        "pore geometry, and topology in your design hints."
    ),
    "hMOF": _SYSTEM_PROMPT_BASE.format(
        db_context="You are analyzing a MOF record from the hMOF database — a HYPOTHETICAL "
        "(computationally generated) MOF from the Snurr group. It has NOT been synthesized. "
        "It includes simulated gas adsorption data (H2, CH4, CO2, Xe, Kr). "
        "Reference gas uptake values, selectivity, pore geometry, and topology in your "
        "design hints. Note synthesizability considerations where relevant."
    ),
}


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(record: dict[str, Any]) -> str:
    """Build user prompt from a v2 record dict. Auto-detects database."""
    db = _detect_database(record)
    rec_id = _get_record_id(record)
    l1 = record.get("layer1_facts", {})
    l2 = record.get("layer2_semantics", {})
    mn = l1.get("metal_node", {})

    parts: list[str] = [
        f"{db} ID: {rec_id}",
        f"Database: {db}",
        f"Formula: {l1.get('formula', 'N/A')}",
    ]

    # Metals
    metals = mn.get("metals", [])
    if metals:
        parts.append(f"Metals: {', '.join(metals)}")
    if mn.get("nuclearity"):
        parts.append(f"Nuclearity: {mn['nuclearity']}")
    if mn.get("geometry"):
        parts.append(f"Node geometry: {mn['geometry']}")
    if mn.get("sbu_type"):
        parts.append(f"SBU type: {mn['sbu_type']}")

    # Topology
    if l1.get("topology"):
        parts.append(f"Topology: {l1['topology']}")

    # Pore & crystal (common to all DBs)
    if l1.get("lcd") is not None:
        parts.append(f"LCD: {l1['lcd']:.2f} Å")
    if l1.get("pld") is not None:
        parts.append(f"PLD: {l1['pld']:.2f} Å")
    if l1.get("density") is not None:
        parts.append(f"Density: {l1['density']:.3f} g/cm³")
    if l1.get("volume") is not None:
        parts.append(f"Volume: {l1['volume']:.1f} ų")
    if l1.get("surface_area_m2g") is not None:
        parts.append(f"Surface area: {l1['surface_area_m2g']:.1f} m²/g")
    if l1.get("void_fraction") is not None:
        parts.append(f"Void fraction: {l1['void_fraction']:.3f}")

    # QMOF-specific: Bandgap
    if db == "QMOF":
        for bg_key in ("bandgap_hse06", "bandgap_hse06_10hf", "bandgap_hle17", "bandgap_pbe"):
            val = l1.get(bg_key)
            if val is not None:
                parts.append(f"Bandgap ({bg_key.split('_', 1)[1]}): {val:.2f} eV")
                break

    # CoRE-MOF-specific: Stability
    if db == "CoRE-MOF":
        stab = l1.get("stability", {})
        if stab.get("thermal_stability_C") is not None:
            parts.append(f"Thermal stability: {stab['thermal_stability_C']:.0f} °C")
        if stab.get("water_stability") is not None:
            parts.append(f"Water stability score: {stab['water_stability']:.2f}")
        if stab.get("solvent_stability") is not None:
            parts.append(f"Solvent stability score: {stab['solvent_stability']:.2f}")
        if mn.get("has_open_metal_sites") is not None:
            parts.append(f"Open metal sites: {mn['has_open_metal_sites']}")
        if l1.get("catenation") is not None:
            parts.append(f"Catenation: {l1['catenation']}")

    # hMOF-specific: Gas adsorption
    if db == "hMOF":
        parts.append("Synthesized: No (hypothetical)")
        gas = l1.get("gas_adsorption", {})
        gas_labels = {
            "h2_uptake_2bar_77K": "H₂ uptake (2 bar, 77K)",
            "h2_uptake_100bar_77K": "H₂ uptake (100 bar, 77K)",
            "ch4_uptake_35bar_298K": "CH₄ uptake (35 bar, 298K)",
            "co2_uptake_2_5bar_298K": "CO₂ uptake (2.5 bar, 298K)",
            "xe_loading_1bar_273K": "Xe loading (1 bar, 273K)",
            "kr_loading_1bar_273K": "Kr loading (1 bar, 273K)",
            "xekr_selectivity_1bar": "Xe/Kr selectivity (1 bar)",
        }
        for field, label in gas_labels.items():
            val = gas.get(field)
            if val is not None:
                parts.append(f"{label}: {val:.4g}")

    # SMILES (first 3 each)
    nodes = l1.get("smiles_nodes", [])[:3]
    linkers = l1.get("smiles_linkers", [])[:3]
    if nodes:
        parts.append(f"Node SMILES: {'; '.join(str(s) for s in nodes)}")
    if linkers:
        parts.append(f"Linker SMILES: {'; '.join(str(s) for s in linkers)}")

    # Rule-based FGs already detected
    fg = l2.get("functional_groups", {})
    rb = fg.get("rule_based", [])
    if rb:
        parts.append(f"Rule-based FGs already detected: {rb}")

    # Core scaffolds & heterocycles from linker enrichment
    le_list = l2.get("linker_enrichment", [])
    all_scaffolds: list[str] = []
    all_hetero: list[str] = []
    for le in le_list:
        all_scaffolds.extend(le.get("core_scaffold", []))
        all_hetero.extend(le.get("heterocycles", []))
    if all_scaffolds:
        parts.append(f"Core scaffolds: {sorted(set(all_scaffolds))}")
    if all_hetero:
        parts.append(f"Heterocycles: {sorted(set(all_hetero))}")

    # Coordinating groups
    cg = mn.get("coordinating_groups", [])
    if cg:
        parts.append(f"Coordinating groups: {cg}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Provider-specific enrichment
# ---------------------------------------------------------------------------

_EMPTY_RESULT: dict[str, Any] = {"readable_name": None, "design_hints": None, "llm_additions": []}


def _validate_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a parsed LLM response."""
    return {
        "readable_name": str(raw["readable_name"]) if raw.get("readable_name") else None,
        "design_hints": str(raw["design_hints"]) if raw.get("design_hints") else None,
        "llm_additions": list(raw.get("llm_additions") or []),
    }


def enrich_single_gemini(
    record: dict[str, Any],
    model_name: str = "gemini-2.0-flash",
    max_retries: int = 3,
) -> dict[str, Any]:
    """Enrich a single record via Google Gemini API."""
    import google.generativeai as genai  # type: ignore[import-untyped]

    db = _detect_database(record)
    system_prompt = SYSTEM_PROMPTS.get(db, SYSTEM_PROMPTS["QMOF"])

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 400},
    )

    user_prompt = _build_user_prompt(record)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(user_prompt)
            text = response.text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            return _validate_result(result)
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return dict(_EMPTY_RESULT)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            _safe_print(f"  ERROR (gemini) after {max_retries} retries: {e}")
            return dict(_EMPTY_RESULT)
    return dict(_EMPTY_RESULT)  # unreachable but satisfies type checker


def enrich_single_openai(
    record: dict[str, Any],
    client: Any = None,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> dict[str, Any]:
    """Enrich a single record via OpenAI API."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    db = _detect_database(record)
    system_prompt = SYSTEM_PROMPTS.get(db, SYSTEM_PROMPTS["QMOF"])
    user_prompt = _build_user_prompt(record)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or ""
            result = json.loads(text)
            return _validate_result(result)
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return dict(_EMPTY_RESULT)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            _safe_print(f"  ERROR (openai) after {max_retries} retries: {e}")
            return dict(_EMPTY_RESULT)
    return dict(_EMPTY_RESULT)  # unreachable but satisfies type checker


def enrich_single(record: dict[str, Any], provider: Optional[str] = None) -> dict[str, Any]:
    """Dispatch to gemini or openai based on provider setting."""
    prov = (provider or LLM_PROVIDER).lower()
    if prov == "openai":
        return enrich_single_openai(record)
    return enrich_single_gemini(record)


# ---------------------------------------------------------------------------
# Apply enrichment to record
# ---------------------------------------------------------------------------

def apply_llm_enrichment(record: dict[str, Any], enrichment: dict[str, Any]) -> dict[str, Any]:
    """Merge LLM enrichment fields into a QMOF v2 record dict (returns copy)."""
    result = json.loads(json.dumps(record))  # deep copy

    l2 = result.setdefault("layer2_semantics", {})
    l2["readable_name"] = enrichment["readable_name"]
    l2["design_hints"] = enrichment["design_hints"]

    # Merge llm_additions without duplicating rule_based
    fg = l2.setdefault("functional_groups", {})
    existing = set(fg.get("rule_based", []))
    new_additions = [g for g in enrichment["llm_additions"] if g not in existing]
    fg["llm_additions"] = new_additions

    # Update provenance
    l2["source"] = "smarts_rules+llm"
    prov = result.setdefault("provenance", {})
    prov["layer2_method"] = "smarts_rules+llm"

    return result


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoint or return empty state."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "completed_ids": [],
        "errors": [],
        "stats": {"enriched": 0, "skipped": 0, "errors": 0},
    }


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Persist checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_llm_enrichment(
    input_dir: Path = DEFAULT_INPUT_DIR,
    batch_size: int = 100,
    skip_existing: bool = True,
    checkpoint_file: Optional[Path] = None,
    provider: Optional[str] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict[str, int]:
    """Process all JSON files in input_dir with LLM enrichment.

    Args:
        input_dir: Directory containing QMOF v2 JSON files.
        batch_size: Save checkpoint & print progress every N records.
        skip_existing: Skip records that already have readable_name set.
        checkpoint_file: Path for checkpoint JSON (default: AUDIT_DIR).
        provider: "gemini" or "openai" (default: env LLM_PROVIDER).
        dry_run: Print prompts without making API calls.
        limit: Process only first N records (for testing).

    Returns:
        Summary stats dict.
    """
    ckpt_path = checkpoint_file or _get_checkpoint_path(Path(input_dir))
    checkpoint = _load_checkpoint(ckpt_path)
    completed_set = set(checkpoint["completed_ids"])

    input_dir = Path(input_dir)
    json_files = sorted(
        [f for f in input_dir.glob("*.json") if not f.name.startswith("_")],
        key=lambda f: f.stem,
    )
    if limit:
        json_files = json_files[:limit]

    total = len(json_files)
    prov_str = (provider or LLM_PROVIDER).lower()
    _safe_print(f"LLM enrichment: {total} MOFs, provider={prov_str}, dry_run={dry_run}")
    _safe_print(f"  Checkpoint: {ckpt_path}  (resuming {len(completed_set)} done)")

    enriched = checkpoint["stats"]["enriched"]
    skipped = checkpoint["stats"]["skipped"]
    errors = checkpoint["stats"]["errors"]

    for i, filepath in enumerate(json_files):
        rec_id = filepath.stem

        # Skip if already done in previous run
        if rec_id in completed_set:
            skipped += 1
            continue

        data = json.loads(filepath.read_text(encoding="utf-8"))

        # Skip if record already has readable_name
        if skip_existing and data.get("layer2_semantics", {}).get("readable_name"):
            skipped += 1
            completed_set.add(rec_id)
            checkpoint["completed_ids"].append(rec_id)
            continue

        if dry_run:
            prompt = _build_user_prompt(data)
            db = _detect_database(data)
            sys_prompt = SYSTEM_PROMPTS.get(db, SYSTEM_PROMPTS["QMOF"])
            sep = "=" * 60
            lines = [
                f"\n{sep}",
                f"[DRY-RUN] {rec_id} (database={db})",
                sep,
                "--- SYSTEM PROMPT (first 200 chars) ---",
                sys_prompt[:200] + "...",
                "--- USER PROMPT ---",
                prompt,
                f"{sep}\n",
            ]
            _safe_print("\n".join(lines))
            enriched += 1
            completed_set.add(rec_id)
            checkpoint["completed_ids"].append(rec_id)
        else:
            # Rate limit
            time.sleep(RATE_LIMIT_DELAY)

            enrichment = enrich_single(data, provider=provider)

            if enrichment["readable_name"]:
                updated = apply_llm_enrichment(data, enrichment)
                filepath.write_text(
                    json.dumps(updated, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                enriched += 1
                completed_set.add(rec_id)
                checkpoint["completed_ids"].append(rec_id)
            else:
                errors += 1
                checkpoint["errors"].append(rec_id)

        # Checkpoint every batch_size
        checkpoint["stats"] = {"enriched": enriched, "skipped": skipped, "errors": errors}
        if (i + 1) % batch_size == 0 or (i + 1) == total:
            if not dry_run:
                _save_checkpoint(ckpt_path, checkpoint)
            _safe_print(f"  [{i+1}/{total}] enriched={enriched} skipped={skipped} errors={errors}")

    _safe_print(f"\n=== LLM Enrichment {'(DRY-RUN) ' if dry_run else ''}Complete ===")
    _safe_print(f"  Total:    {total}")
    _safe_print(f"  Enriched: {enriched}")
    _safe_print(f"  Skipped:  {skipped}")
    _safe_print(f"  Errors:   {errors}")

    return {"total": total, "enriched": enriched, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM enrichment for MOF v2 records — QMOF/CoRE-MOF/hMOF (Gemini/OpenAI)"
    )
    parser.add_argument("--provider", default=None, help="gemini or openai")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without API calls")
    parser.add_argument("--limit", type=int, default=None, help="Process only N records")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory override")
    args = parser.parse_args()

    run_llm_enrichment(
        input_dir=Path(args.input_dir) if args.input_dir else DEFAULT_INPUT_DIR,
        batch_size=args.batch_size,
        provider=args.provider,
        dry_run=args.dry_run,
        limit=args.limit,
    )
