#!/usr/bin/env python3
"""
Phase 1: Vocabulary Frequency Audit & Vocabulary Construction
============================================================
Scans three data sources for functional-group vocabulary:
  1. BB dictionary v3.2  (pormake_bb_dictionary_v3.2.json)
  2. V7 edge JSONs       (BuildingBlock_meta_data_v7_20260305/E*_metadata.json)
  3. QMOF global JSONs    (qmof_global_jsons_v2/qmof-*_analysis.json)

Produces:
  - freq_bb_dictionary.json   — tag frequencies from BB dictionary
  - freq_v7_edges.json        — tag frequencies from V7 edge metadata
  - freq_qmof_globals.json    — tag frequencies from QMOF global JSONs
  - cross_reference_report.md — unified cross-reference table
  - synonyms_detected.json    — candidate synonym pairs

All outputs go to logs/phase1_vocab_audit_YYYYMMDD_HHMMSS/
"""

import json
import os
import sys
import glob
from collections import Counter, defaultdict
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

BB_DICT_PATH   = os.path.join(DATA_DIR, "pormake_bb_dictionary_v3.2.json")
V7_EDGE_DIR    = os.path.join(DATA_DIR, "BuildingBlock_meta_data_v7_20260305")
QMOF_JSON_DIR  = os.path.join(DATA_DIR, "qmof_global_jsons_v2")

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR   = os.path.join(BASE_DIR, "logs", f"phase1_vocab_audit_{TIMESTAMP}")


# ── helpers ────────────────────────────────────────────────────────────────

def canon(s: str) -> str:
    """Lowercase, strip, hyphens→underscores, spaces→underscores."""
    return s.lower().strip().replace("-", "_").replace(" ", "_")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved {path}")


def sorted_counter(ctr: Counter) -> dict:
    """Return counter as dict sorted by descending frequency."""
    return dict(ctr.most_common())


# ── Source 1: BB dictionary v3.2 ──────────────────────────────────────────

def scan_bb_dictionary(path: str):
    """
    Extract every tag from BB dictionary entries.
    Fields scanned:
      - functional_groups (flat list)
      - connection_chemistry / ligand_chemistry
      - metals
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tag_counter      = Counter()
    tag_by_field     = defaultdict(Counter)  # field_name → Counter
    entry_count      = {"Node": 0, "Edge": 0, "Other": 0}

    for item in data:
        bb_type = item.get("Type", "Other")
        entry_count[bb_type] = entry_count.get(bb_type, 0) + 1

        # functional_groups (flat array)
        for g in item.get("functional_groups", []):
            tag_counter[g] += 1
            tag_by_field["functional_groups"][g] += 1

        # connection_chemistry / ligand_chemistry
        chem = item.get("ligand_chemistry", item.get("connection_chemistry", []))
        if isinstance(chem, str):
            chem = [chem]
        if chem:
            for c in chem:
                tag_counter[c] += 1
                tag_by_field["connection_chemistry"][c] += 1

        # metals
        for m in item.get("metals", []):
            tag_counter[m] += 1
            tag_by_field["metals"][m] += 1

    print(f"[BB Dictionary] Scanned {len(data)} entries  "
          f"(Node={entry_count.get('Node',0)}, Edge={entry_count.get('Edge',0)})")
    print(f"  Unique tags: {len(tag_counter)}")

    return tag_counter, tag_by_field, entry_count, len(data)


# ── Source 2: V7 Edge JSONs ───────────────────────────────────────────────

def scan_v7_edges(directory: str):
    """
    Extract tags from V7 edge metadata files.
    Fields scanned:
      - functional_groups.linker_substituents
      - functional_groups.linker_backbone
      - core_scaffold
      - substituents
      - heterocycles
      - linker_type
      - metals
    """
    pattern = os.path.join(directory, "E*_metadata.json")
    files = sorted(glob.glob(pattern))

    tag_counter  = Counter()
    tag_by_field = defaultdict(Counter)

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            item = json.load(f)

        fg = item.get("functional_groups", {})
        for g in fg.get("linker_substituents", []):
            tag_counter[g] += 1
            tag_by_field["linker_substituents"][g] += 1
        for g in fg.get("linker_backbone", []):
            tag_counter[g] += 1
            tag_by_field["linker_backbone"][g] += 1

        for g in item.get("core_scaffold", []):
            tag_counter[g] += 1
            tag_by_field["core_scaffold"][g] += 1
        for g in item.get("substituents", []):
            tag_counter[g] += 1
            tag_by_field["substituents"][g] += 1
        for g in item.get("heterocycles", []):
            tag_counter[g] += 1
            tag_by_field["heterocycles"][g] += 1
        for g in item.get("linker_type", []):
            tag_counter[g] += 1
            tag_by_field["linker_type"][g] += 1
        for m in item.get("metals", []):
            tag_counter[m] += 1
            tag_by_field["metals"][m] += 1

    print(f"[V7 Edges] Scanned {len(files)} files")
    print(f"  Unique tags: {len(tag_counter)}")

    return tag_counter, tag_by_field, len(files)


# ── Source 3: QMOF Global JSONs ──────────────────────────────────────────

def scan_qmof_globals(directory: str):
    """
    Extract tags from QMOF global JSON files.
    Fields scanned (under metal_node.chemistry.functional_groups):
      - coordinating_groups
      - linker_substituents
      - linker_backbone
      - metal_terminal_ligands
    Also: metal_node.composition.metals
    """
    pattern = os.path.join(directory, "qmof-*_analysis.json")
    files = sorted(glob.glob(pattern))

    tag_counter  = Counter()
    tag_by_field = defaultdict(Counter)

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            item = json.load(f)

        metal_node = item.get("metal_node", {})

        # metals
        metals = metal_node.get("composition", {}).get("metals", [])
        for m in metals:
            tag_counter[m] += 1
            tag_by_field["metals"][m] += 1

        # functional groups
        fg = metal_node.get("chemistry", {}).get("functional_groups", {})
        for g in fg.get("coordinating_groups", []):
            tag_counter[g] += 1
            tag_by_field["coordinating_groups"][g] += 1
        for g in fg.get("linker_substituents", []):
            tag_counter[g] += 1
            tag_by_field["linker_substituents"][g] += 1
        for g in fg.get("linker_backbone", []):
            tag_counter[g] += 1
            tag_by_field["linker_backbone"][g] += 1
        for g in fg.get("metal_terminal_ligands", []):
            tag_counter[g] += 1
            tag_by_field["metal_terminal_ligands"][g] += 1

    print(f"[QMOF Globals] Scanned {len(files)} files")
    print(f"  Unique tags: {len(tag_counter)}")

    return tag_counter, tag_by_field, len(files)


# ── Cross-reference & Synonym Detection ──────────────────────────────────

def build_cross_reference(bb_tags, v7_tags, qmof_tags):
    """
    Build a unified cross-reference of all tags across all sources.
    Returns: list of dicts for the report table.
    """
    all_raw_tags = set(bb_tags.keys()) | set(v7_tags.keys()) | set(qmof_tags.keys())

    rows = []
    for tag in sorted(all_raw_tags, key=lambda t: t.lower()):
        canonical = canon(tag)
        rows.append({
            "tag_raw":   tag,
            "canonical": canonical,
            "bb_count":  bb_tags.get(tag, 0),
            "v7_count":  v7_tags.get(tag, 0),
            "qmof_count": qmof_tags.get(tag, 0),
            "total":     bb_tags.get(tag, 0) + v7_tags.get(tag, 0) + qmof_tags.get(tag, 0),
            "sources":   ", ".join(
                s for s, c in [("BB", bb_tags.get(tag, 0)),
                               ("V7", v7_tags.get(tag, 0)),
                               ("QMOF", qmof_tags.get(tag, 0))]
                if c > 0
            ),
        })

    return rows


def detect_synonyms(rows):
    """
    Detect candidate synonyms: tags that share the same canonical form
    OR are substring-similar.
    """
    # Group by canonical form
    canonical_groups = defaultdict(list)
    for r in rows:
        canonical_groups[r["canonical"]].append(r["tag_raw"])

    synonyms = []
    for canonical, group in canonical_groups.items():
        if len(group) > 1:
            synonyms.append({
                "canonical": canonical,
                "variants": sorted(group),
                "reason": "same_canonical_form",
            })

    # Substring similarity (e.g. "Aromatic" vs "Aromatic_Ring")
    all_raw = [r["tag_raw"] for r in rows]
    for i, a in enumerate(all_raw):
        for b in all_raw[i+1:]:
            ca, cb = canon(a), canon(b)
            if ca == cb:
                continue  # already caught above
            if (ca in cb or cb in ca) and abs(len(ca) - len(cb)) <= 6:
                synonyms.append({
                    "canonical_a": ca,
                    "canonical_b": cb,
                    "variants": [a, b],
                    "reason": "substring_overlap",
                })

    return synonyms


# ── Report Generation ─────────────────────────────────────────────────────

def generate_cross_reference_report(rows, bb_meta, v7_meta, qmof_meta):
    """Generate a Markdown cross-reference report."""

    lines = [
        "# Phase 1: Vocabulary Cross-Reference Report",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Source Summary",
        "",
        f"| Source | Entries/Files | Unique Tags |",
        f"|--------|--------------|-------------|",
        f"| BB Dictionary v3.2 | {bb_meta['entry_count']} entries | {bb_meta['unique_tags']} |",
        f"| V7 Edge Metadata | {v7_meta['file_count']} files | {v7_meta['unique_tags']} |",
        f"| QMOF Global JSONs | {qmof_meta['file_count']} files | {qmof_meta['unique_tags']} |",
        "",
        f"**Total unique raw tags across all sources:** {len(rows)}",
        "",
        "## Cross-Reference Table",
        "",
        "| Raw Tag | Canonical | BB Dict | V7 Edge | QMOF | Total | Sources |",
        "|---------|-----------|---------|---------|------|-------|---------|",
    ]

    for r in sorted(rows, key=lambda x: -x["total"]):
        lines.append(
            f"| {r['tag_raw']} | `{r['canonical']}` | {r['bb_count']} | "
            f"{r['v7_count']} | {r['qmof_count']} | {r['total']} | {r['sources']} |"
        )

    lines.append("")
    lines.append("## Tags Only in One Source")
    lines.append("")

    for source_label, check_key in [("BB Only", "bb_count"),
                                      ("V7 Only", "v7_count"),
                                      ("QMOF Only", "qmof_count")]:
        only_rows = [r for r in rows
                     if r[check_key] > 0
                     and sum(r[k] for k in ["bb_count", "v7_count", "qmof_count"]) == r[check_key]]
        if only_rows:
            lines.append(f"### {source_label}")
            for r in sorted(only_rows, key=lambda x: -x[check_key]):
                lines.append(f"- `{r['tag_raw']}` (count={r[check_key]})")
            lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 1: VOCABULARY FREQUENCY AUDIT")
    print("=" * 70)
    print(f"Timestamp: {TIMESTAMP}")
    print()

    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Log directory: {LOG_DIR}\n")

    # ── Scan Source 1: BB Dictionary ──
    bb_tags, bb_by_field, bb_entry_count, bb_total = scan_bb_dictionary(BB_DICT_PATH)
    save_json(sorted_counter(bb_tags), os.path.join(LOG_DIR, "freq_bb_dictionary.json"))
    save_json({k: sorted_counter(v) for k, v in bb_by_field.items()},
              os.path.join(LOG_DIR, "freq_bb_dictionary_by_field.json"))

    # ── Scan Source 2: V7 Edges ──
    v7_tags, v7_by_field, v7_file_count = scan_v7_edges(V7_EDGE_DIR)
    save_json(sorted_counter(v7_tags), os.path.join(LOG_DIR, "freq_v7_edges.json"))
    save_json({k: sorted_counter(v) for k, v in v7_by_field.items()},
              os.path.join(LOG_DIR, "freq_v7_edges_by_field.json"))

    # ── Scan Source 3: QMOF Globals ──
    qmof_tags, qmof_by_field, qmof_file_count = scan_qmof_globals(QMOF_JSON_DIR)
    save_json(sorted_counter(qmof_tags), os.path.join(LOG_DIR, "freq_qmof_globals.json"))
    save_json({k: sorted_counter(v) for k, v in qmof_by_field.items()},
              os.path.join(LOG_DIR, "freq_qmof_globals_by_field.json"))

    # ── Cross-Reference ──
    print("\n--- Building Cross-Reference ---")
    rows = build_cross_reference(bb_tags, v7_tags, qmof_tags)
    print(f"  Total unique raw tags: {len(rows)}")

    # ── Synonym Detection ──
    synonyms = detect_synonyms(rows)
    print(f"  Candidate synonym groups: {len(synonyms)}")
    save_json(synonyms, os.path.join(LOG_DIR, "synonyms_detected.json"))

    # ── Report ──
    bb_meta   = {"entry_count": bb_total, "unique_tags": len(bb_tags)}
    v7_meta   = {"file_count": v7_file_count, "unique_tags": len(v7_tags)}
    qmof_meta = {"file_count": qmof_file_count, "unique_tags": len(qmof_tags)}

    report_md = generate_cross_reference_report(rows, bb_meta, v7_meta, qmof_meta)
    report_path = os.path.join(LOG_DIR, "cross_reference_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  [OK] Saved {report_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"Outputs saved to: {LOG_DIR}")
    print(f"  - freq_bb_dictionary.json")
    print(f"  - freq_bb_dictionary_by_field.json")
    print(f"  - freq_v7_edges.json")
    print(f"  - freq_v7_edges_by_field.json")
    print(f"  - freq_qmof_globals.json")
    print(f"  - freq_qmof_globals_by_field.json")
    print(f"  - synonyms_detected.json")
    print(f"  - cross_reference_report.md")


if __name__ == "__main__":
    main()
