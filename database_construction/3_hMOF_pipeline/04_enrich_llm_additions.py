#!/usr/bin/env python3
"""Targeted llm_additions enrichment for hMOF and PORMAKE.

ONLY updates the llm_additions field in each JSON. Preserves existing
readable_name and design_hints. Uses GPT-4o-mini via OpenAI API.

Checkpoint/resume supported — safe to interrupt and restart.

Usage:
    python enrich_llm_additions.py --db hmof              # hMOF (51K records)
    python enrich_llm_additions.py --db hmof --limit 100  # test on 100
    python enrich_llm_additions.py --db pormake           # PORMAKE (867 records)
    python enrich_llm_additions.py --db hmof --dry-run    # report only
"""

import json
import os
import sys
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load API keys
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

import openai

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = Path(__file__).parent / "enrich_llm_additions.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent

DB_CONFIG = {
    "hmof": {
        "input_dir": BASE_DIR / "hMOF" / "hmof_enriched_v2",
        "checkpoint": BASE_DIR / "SI_data_provenance" / "llm_additions_hmof_checkpoint.json",
        "id_field": "hmof_id",
    },
    "pormake": {
        "input_dir": BASE_DIR / "PORMAKE" / "bb_metadata_v8",
        "checkpoint": BASE_DIR / "SI_data_provenance" / "llm_additions_pormake_checkpoint.json",
        "id_field": "bb_id",
    },
    "coremof": {
        "input_dir": BASE_DIR / "CoRE-MOF" / "coremof_enriched_v2",
        "checkpoint": BASE_DIR / "SI_data_provenance" / "llm_additions_coremof_checkpoint.json",
        "id_field": "coremof_id",
    },
}

# ---------------------------------------------------------------------------
# Prompts — ask ONLY for llm_additions
# ---------------------------------------------------------------------------

HMOF_PROMPT = """\
You are a reticular chemistry expert. Given a MOF's metadata, identify functional groups \
or structural motifs that SMARTS pattern matching might have missed.

This is a HYPOTHETICAL MOF from the hMOF database (computationally generated, not synthesized).

Return ONLY a JSON object with one field:
{{"llm_additions": ["motif1", "motif2"]}}

Rules:
- Only include groups you are confident about based on the SMILES and metal node data.
- Do NOT repeat groups already in rule_based.
- If nothing new beyond rule_based, return {{"llm_additions": []}}.
- Common SMARTS blind spots: paddlewheel, UiO-type, MIL-type, MOF-5-type, HKUST-type, \
pillared-layer, interpenetrated, breathing, flexible, rod-shaped SBU.
"""

PORMAKE_PROMPT = """\
You are a reticular chemistry expert. Given a MOF building block's metadata, identify \
functional groups or chemical features that SMARTS pattern matching might have missed.

This is a PORMAKE building block (a molecular fragment used to assemble MOFs).

Return ONLY a JSON object with one field:
{{"llm_additions": ["group1", "group2"]}}

Rules:
- Only include groups you are confident about based on the SMILES.
- Do NOT repeat groups already in rule_based.
- If nothing new beyond rule_based, return {{"llm_additions": []}}.
- Common SMARTS blind spots: porphyrin, paddlewheel, IRMOF-type, UiO-type, catechol, \
salicylate, squarate, metallocycle, crown ether, calixarene, BODIPY, perylene, \
corrole, salen, bipyridyl, terpyridyl, phenanthroline.
"""

COREMOF_PROMPT = """\
You are a reticular chemistry expert. Given a MOF's metadata, identify functional groups \
or structural motifs that SMARTS pattern matching might have missed.

This is an experimentally synthesized MOF from the CoRE-MOF database with stability data.

Return ONLY a JSON object with one field:
{{"llm_additions": ["motif1", "motif2"]}}

Rules:
- Only include groups you are confident about based on the SMILES and metal node data.
- Do NOT repeat groups already in rule_based.
- If nothing new beyond rule_based, return {{"llm_additions": []}}.
- Common SMARTS blind spots: paddlewheel, UiO-type, MIL-type, ZIF-type, HKUST-type, \
MOF-5-type, pillared-layer, interpenetrated, breathing, flexible, rod-shaped SBU.
"""

SYSTEM_PROMPTS = {
    "hmof": HMOF_PROMPT,
    "pormake": PORMAKE_PROMPT,
    "coremof": COREMOF_PROMPT,
}


def build_user_prompt(data: dict, db: str) -> str:
    """Build the user prompt from a record's metadata."""
    parts = []

    if db == "hmof":
        l1 = data.get("layer1_facts", {})
        l2 = data.get("layer2_semantics", {})
        fg = l2.get("functional_groups", {})
        mn = l1.get("metal_node", {})
        gas = l1.get("gas_adsorption", {})

        parts.append(f"hMOF ID: {data.get('hmof_id')}")
        parts.append(f"Metals: {mn.get('metals', [])}")
        parts.append(f"Nuclearity: {mn.get('nuclearity')}")
        parts.append(f"Geometry: {mn.get('geometry')}")
        parts.append(f"SBU Type: {mn.get('sbu_type')}")
        parts.append(f"Oxidation States: {mn.get('oxidation_states')}")
        parts.append(f"Coordinating Groups: {mn.get('coordinating_groups', [])}")
        parts.append(f"Topology: {l1.get('topology')}")
        parts.append(f"SMILES Nodes: {l1.get('smiles_nodes', [])}")
        parts.append(f"SMILES Linkers: {l1.get('smiles_linkers', [])}")
        parts.append(f"Rule-based FGs: {fg.get('rule_based', [])}")
        parts.append(f"Surface Area: {l1.get('surface_area_m2g')} m2/g")

    elif db == "pormake":
        l1 = data.get("layer1_facts", {})
        l2 = data.get("layer2_semantics", {})
        fg = l2.get("functional_groups", {})

        parts.append(f"BB ID: {data.get('bb_id')}")
        parts.append(f"Type: {data.get('bb_type')}")
        parts.append(f"Formula: {l1.get('formula')}")
        parts.append(f"SMILES: {l1.get('smiles')}")
        parts.append(f"Metals: {l1.get('metals', [])}")
        parts.append(f"Connectivity: {l1.get('connection_points', {}).get('count')}")
        parts.append(f"Rule-based FGs: {fg.get('rule_based', [])}")

    elif db == "coremof":
        l1 = data.get("layer1_facts", {})
        l2 = data.get("layer2_semantics", {})
        fg = l2.get("functional_groups", {})
        mn = l1.get("metal_node", {})
        stab = l1.get("stability", {})

        parts.append(f"CoRE-MOF ID: {data.get('coremof_id')}")
        parts.append(f"Metals: {mn.get('metals', [])}")
        parts.append(f"Nuclearity: {mn.get('nuclearity')}")
        parts.append(f"Geometry: {mn.get('geometry')}")
        parts.append(f"SBU Type: {mn.get('sbu_type')}")
        parts.append(f"Oxidation States: {mn.get('oxidation_states')}")
        parts.append(f"Coordinating Groups: {mn.get('coordinating_groups', [])}")
        parts.append(f"Topology: {l1.get('topology')}")
        parts.append(f"SMILES Nodes: {l1.get('smiles_nodes', [])}")
        parts.append(f"SMILES Linkers: {l1.get('smiles_linkers', [])}")
        parts.append(f"Rule-based FGs: {fg.get('rule_based', [])}")
        parts.append(f"Surface Area: {l1.get('surface_area_m2g')} m2/g")
        parts.append(f"Thermal Stability: {stab.get('thermal_stability_C')} C")
        parts.append(f"Synthesized: {l1.get('synthesized')}")

    return "\n".join(parts)


def call_llm(system_prompt: str, user_prompt: str, client: openai.OpenAI) -> list:
    """Call GPT-4o-mini for llm_additions only. Returns list of strings."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
        additions = raw.get("llm_additions", [])
        if isinstance(additions, list):
            return [str(a) for a in additions]
        return []
    except Exception as e:
        log.warning(f"  LLM call failed: {e}")
        return []


def load_checkpoint(path: Path) -> set:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("completed_ids", []))
    return set()


def save_checkpoint(path: Path, completed: set, stats: dict):
    data = {
        "completed_ids": sorted(completed),
        "stats": stats,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Targeted llm_additions enrichment")
    parser.add_argument("--db", required=True, choices=["hmof", "pormake", "coremof"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and start fresh")
    args = parser.parse_args()

    config = DB_CONFIG[args.db]
    input_dir = config["input_dir"]
    ckpt_path = config["checkpoint"]
    id_field = config["id_field"]
    system_prompt = SYSTEM_PROMPTS[args.db]

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    log.info("=" * 60)
    log.info(f"LLM Additions Enrichment: {args.db.upper()}")
    log.info("=" * 60)
    log.info(f"  Input:      {input_dir}")
    log.info(f"  Checkpoint: {ckpt_path}")
    log.info(f"  Mode:       {'DRY RUN' if args.dry_run else 'APPLY'}")
    log.info(f"  Limit:      {args.limit or 'all'}")
    log.info(f"  Batch size: {args.batch_size}")
    log.info(f"  Model:      gpt-4o-mini, T=0.3")
    log.info(f"  Timestamp:  {datetime.now(timezone.utc).isoformat()}")
    log.info("")

    # Load checkpoint
    if args.reset and ckpt_path.exists():
        ckpt_path.unlink()
        log.info("  Checkpoint cleared.")
    completed = load_checkpoint(ckpt_path)
    log.info(f"  Checkpoint: {len(completed)} already completed")

    # Collect files
    files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix == ".json" and not f.name.startswith("_")
    ])
    if args.limit:
        files = files[:args.limit]

    total = len(files)
    enriched = 0
    skipped_ckpt = 0
    skipped_has = 0
    empty_result = 0
    nonempty_result = 0
    errors = 0
    t0 = time.time()

    for i, fpath in enumerate(files):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            rec_id = data.get(id_field, fpath.stem)

            # Skip if already in checkpoint
            if rec_id in completed:
                skipped_ckpt += 1
                continue

            # Skip if llm_additions already non-empty
            fg = data.get("layer2_semantics", {}).get("functional_groups", {})
            existing = fg.get("llm_additions", [])
            if existing and len(existing) > 0:
                skipped_has += 1
                completed.add(rec_id)
                continue

            if args.dry_run:
                enriched += 1
                completed.add(rec_id)
                continue

            # Build prompt and call LLM
            user_prompt = build_user_prompt(data, args.db)
            additions = call_llm(system_prompt, user_prompt, client)

            # Filter out duplicates with rule_based
            rule_based = set(fg.get("rule_based", []))
            new_additions = [a for a in additions if a.lower() not in {r.lower() for r in rule_based}]

            # Update the JSON
            fg["llm_additions"] = new_additions

            # Update source tag
            l2 = data.get("layer2_semantics", {})
            source = l2.get("source", "")
            if "+llm_additions" not in source:
                l2["source"] = source + "+llm_additions"

            # Write back
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if new_additions:
                nonempty_result += 1
                log.info(f"  {rec_id}: {new_additions}")
            else:
                empty_result += 1

            enriched += 1
            completed.add(rec_id)

        except Exception as e:
            errors += 1
            log.error(f"  ERROR [{fpath.name}]: {e}")

        # Checkpoint + progress
        done = i + 1
        if done % args.batch_size == 0 or done == total:
            elapsed = time.time() - t0
            rate = (enriched + skipped_ckpt + skipped_has) / elapsed if elapsed > 0 else 0
            log.info(
                f"  [{done:>6}/{total}] {elapsed:.1f}s, {rate:.1f} rec/s | "
                f"enriched={enriched} nonempty={nonempty_result} empty={empty_result} "
                f"skip_ckpt={skipped_ckpt} skip_has={skipped_has} err={errors}"
            )
            if not args.dry_run:
                save_checkpoint(ckpt_path, completed, {
                    "enriched": enriched, "nonempty": nonempty_result,
                    "empty": empty_result, "errors": errors,
                })

    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total files:     {total}")
    log.info(f"  LLM calls made:  {enriched}")
    log.info(f"  Non-empty result:{nonempty_result}")
    log.info(f"  Empty result:    {empty_result}")
    log.info(f"  Skipped (ckpt):  {skipped_ckpt}")
    log.info(f"  Skipped (has):   {skipped_has}")
    log.info(f"  Errors:          {errors}")
    log.info(f"  Time:            {elapsed:.1f}s")
    if enriched > 0:
        cost_est = enriched * 0.00018  # rough GPT-4o-mini estimate
        log.info(f"  Est. cost:       ~${cost_est:.2f}")

    if args.dry_run:
        log.info("\n  DRY RUN -- no files modified, no LLM calls made.")


if __name__ == "__main__":
    main()
