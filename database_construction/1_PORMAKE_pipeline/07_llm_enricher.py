"""Phase 6: LLM-based Layer 2 enrichment using OpenAI API.

Fills readable_name, design_hints, and llm_additions for all BBs.
Uses batch processing with GPT-4o-mini for cost efficiency.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .config import OUTPUT_DIR


SYSTEM_PROMPT = """You are a reticular chemistry expert specializing in Metal-Organic Frameworks (MOFs).
You are analyzing a PORMAKE building block — a molecular fragment used to assemble MOF crystal structures.

Given the building block's metadata, provide:
1. readable_name: A concise human-friendly name (e.g., "terphenyl dicarboxylate linker", "zinc paddlewheel SBU", "iron porphyrin node"). 2-8 words. Use standard chemistry nomenclature.
2. design_hints: 1-2 sentences about how this BB would perform in a MOF. Mention relevant properties like rigidity, pore geometry, gas adsorption potential, catalytic activity, or stability. Be specific to THIS molecule.
3. llm_additions: A list of functional groups or chemical features that SMARTS pattern matching might have missed. Only include groups you are confident about based on the SMILES. Common misses: porphyrin, paddlewheel, IRMOF-type, UiO-type, MIL-type, catechol, salicylate, squarate, metallocycle, crown ether, calixarene, BODIPY, perylene, corrole, salen, bipyridyl, terpyridyl, phenanthroline.

Respond in JSON format only:
{"readable_name": "...", "design_hints": "...", "llm_additions": ["group1", "group2"]}

Rules:
- If llm_additions has nothing new beyond what rule_based already detected, return an empty list [].
- Do NOT repeat groups already in rule_based.
- readable_name should be lowercase except proper nouns/acronyms.
- For metal nodes, mention the metal and cluster type (paddlewheel, octahedral, etc.) in readable_name.
- For edges/linkers, mention the core scaffold and key functional groups in readable_name."""


def _build_user_prompt(data: dict) -> str:
    """Build the user prompt from a BB JSON."""
    l1 = data["layer1_facts"]
    l2 = data["layer2_semantics"]

    parts = [
        f"BB ID: {data['bb_id']}",
        f"Type: {data['bb_type']}",
        f"Formula: {l1['formula']}",
        f"MW: {l1['molecular_weight']:.1f} Da",
        f"SMILES: {l1['smiles']}",
        f"Connection points: {l1['connection_points']['count']}",
        f"Connection chemistry: {l1['connection_points']['connection_chemistry']}",
        f"Metals: {l1['metals']}",
        f"Has metal: {l1['has_metal']}",
        f"Rings: {l1['num_rings']} (sizes: {l1['bond_graph']['ring_sizes']})",
        f"Is rigid: {l1['is_rigid']}",
        f"Is planar: {l1['is_planar']}",
        f"Rule-based FGs already detected: {l2['functional_groups']['rule_based']}",
        f"Core scaffold: {l2['core_scaffold']}",
    ]

    if l1.get("nuclearity") is not None:
        parts.append(f"Nuclearity (metal count): {l1['nuclearity']}")
    if l1.get("metal_coordination"):
        for mc in l1["metal_coordination"]:
            ligands = [f"{a['element']}({a['bond_type']})" for a in mc["bonded_atoms"]]
            parts.append(
                f"Metal {mc['element']} coordination: {mc['coordination_number']} "
                f"[{', '.join(ligands)}]"
            )
    if l1.get("length_angstroms") is not None:
        parts.append(f"Length: {l1['length_angstroms']:.1f} Å")

    return "\n".join(parts)


def enrich_single(
    client: OpenAI,
    data: dict,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> dict:
    """Enrich a single BB with LLM-generated metadata.

    Returns dict with readable_name, design_hints, llm_additions.
    """
    user_prompt = _build_user_prompt(data)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = json.loads(content)

            # Validate structure
            return {
                "readable_name": str(result.get("readable_name", "")),
                "design_hints": str(result.get("design_hints", "")),
                "llm_additions": list(result.get("llm_additions", [])),
            }
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {
                "readable_name": None,
                "design_hints": None,
                "llm_additions": [],
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            print(f"  ERROR after {max_retries} retries: {e}")
            return {
                "readable_name": None,
                "design_hints": None,
                "llm_additions": [],
            }


def apply_enrichment(data: dict, enrichment: dict) -> dict:
    """Apply LLM enrichment to a BB JSON dict (returns new dict)."""
    result = json.loads(json.dumps(data))  # deep copy

    l2 = result["layer2_semantics"]
    l2["readable_name"] = enrichment["readable_name"]
    l2["design_hints"] = enrichment["design_hints"]

    # Add llm_additions (don't duplicate rule_based)
    existing = set(l2["functional_groups"]["rule_based"])
    new_additions = [g for g in enrichment["llm_additions"] if g not in existing]
    l2["functional_groups"]["llm_additions"] = new_additions

    # Update source tag
    l2["source"] = "rule-based+llm"

    # Update provenance
    result["provenance"]["layer2_method"] = "smarts_rules+llm"

    return result


def run_llm_enrichment(
    output_dir: Path = OUTPUT_DIR,
    model: str = "gpt-4o-mini",
    batch_size: int = 50,
    skip_existing: bool = True,
) -> dict:
    """Run LLM enrichment on all BB JSONs.

    Args:
        output_dir: Directory with BB JSON files.
        model: OpenAI model to use.
        batch_size: Print progress every N BBs.
        skip_existing: If True, skip BBs that already have readable_name set.

    Returns:
        Summary dict.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)
    output_dir = Path(output_dir)

    # Collect all BB JSON files
    json_files = sorted(
        [f for f in output_dir.glob("*.json") if not f.name.startswith("_")],
        key=lambda f: f.stem,
    )

    total = len(json_files)
    enriched = 0
    skipped = 0
    errors = 0

    print(f"LLM enrichment: {total} BBs, model={model}")

    for i, filepath in enumerate(json_files):
        bb_id = filepath.stem

        # Load current JSON
        data = json.loads(filepath.read_text(encoding="utf-8"))

        # Skip if already enriched
        if skip_existing and data["layer2_semantics"].get("readable_name"):
            skipped += 1
            continue

        # Call LLM
        enrichment = enrich_single(client, data, model=model)

        if enrichment["readable_name"]:
            # Apply and write back
            updated = apply_enrichment(data, enrichment)
            filepath.write_text(
                json.dumps(updated, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            enriched += 1
        else:
            errors += 1

        # Progress
        if (i + 1) % batch_size == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] enriched={enriched} skipped={skipped} errors={errors}")

    print(f"\n=== LLM Enrichment Complete ===")
    print(f"  Total:    {total}")
    print(f"  Enriched: {enriched}")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {errors}")

    return {"total": total, "enriched": enriched, "skipped": skipped, "errors": errors}


if __name__ == "__main__":
    run_llm_enrichment()
