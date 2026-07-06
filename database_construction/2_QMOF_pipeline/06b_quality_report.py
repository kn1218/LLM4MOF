#!/usr/bin/env python3
"""
QMOF Enrichment Quality Report: v1 vs v2 Comparison

Compares enrichment quality metrics between v1 and v2 versions.
Generates a formatted comparison table and detailed JSON report.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
V1_DIR = BASE_DIR / "qmof" / "qmof_enriched_v1"
V2_DIR = BASE_DIR / "qmof" / "qmof_enriched_v2"
REPORT_PATH = BASE_DIR / "qmof" / "_audit" / "v1_vs_v2_quality_report.json"

# Ensure output directory exists
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_v1_files(sample_size: int = None) -> Tuple[List[Dict], int, int]:
    """Load v1 enriched JSON files. Returns (records, total_count, errors)."""
    records = []
    total_count = 0
    errors = 0

    v1_files = sorted([f for f in V1_DIR.glob("*.json") if f.name != "_pipeline_report.json"])
    
    if sample_size:
        v1_files = v1_files[:sample_size]

    for fpath in v1_files:
        total_count += 1
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                records.append(data)
        except Exception as e:
            errors += 1

    return records, total_count, errors


def load_v2_files(sample_size: int = None) -> Tuple[List[Dict], int, int]:
    """Load v2 JSON files. Returns (records, total_count, errors)."""
    records = []
    total_count = 0
    errors = 0

    v2_files = sorted(V2_DIR.glob("*.json"))
    
    if sample_size:
        v2_files = v2_files[:sample_size]

    for fpath in v2_files:
        total_count += 1
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                records.append(data)
        except Exception as e:
            errors += 1

    return records, total_count, errors


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def analyze_v1(records: List[Dict]) -> Dict[str, Any]:
    """Analyze v1 records."""
    stats = {
        "total_records": len(records),
        "records_with_fg_detected": 0,
        "fg_types": set(),
        "records_with_metal_node": 0,
        "records_with_validation": 0,
        "errors": 0,
    }

    for record in records:
        try:
            linker_types = record.get("enrichment", {}).get("structural", {}).get("linker_type", [])
            if linker_types:
                stats["records_with_fg_detected"] += 1
                stats["fg_types"].update(linker_types)

            metal_node = record.get("metal_node", {})
            if metal_node and len(metal_node) > 0:
                stats["records_with_metal_node"] += 1

            validation = record.get("enrichment", {}).get("smiles_validation", {})
            if validation:
                stats["records_with_validation"] += 1

        except Exception as e:
            stats["errors"] += 1

    stats["fg_types"] = list(stats["fg_types"])
    stats["unique_fg_types"] = len(stats["fg_types"])

    return stats


def analyze_v2(records: List[Dict]) -> Dict[str, Any]:
    """Analyze v2 records."""
    stats = {
        "total_records": len(records),
        "records_with_fg_detected": 0,
        "unique_fg_types": set(),
        "fg_per_record": [],
        "fg_frequency": Counter(),
        "records_with_metals": 0,
        "metal_frequency": Counter(),
        "records_with_validation": 0,
        "validation_status_dist": defaultdict(int),
        "records_with_valid_smiles": 0,
        "total_issues": 0,
        "issues_per_record": [],
        "issue_severity_dist": defaultdict(int),
        "errors": 0,
    }

    for record in records:
        try:
            all_fgs = set()
            linker_enrichment = record.get("layer2_semantics", {}).get("linker_enrichment", [])
            
            for linker in linker_enrichment:
                rule_based_fgs = linker.get("functional_groups", {}).get("rule_based", [])
                if rule_based_fgs:
                    stats["records_with_fg_detected"] += 1
                    all_fgs.update(rule_based_fgs)
                    stats["fg_frequency"].update(rule_based_fgs)

            if all_fgs:
                stats["fg_per_record"].append(len(all_fgs))
                stats["unique_fg_types"].update(all_fgs)

            metal_node = record.get("layer1_facts", {}).get("metal_node", {})
            if metal_node and metal_node.get("metals"):
                stats["records_with_metals"] += 1
                stats["metal_frequency"].update(metal_node.get("metals", []))

            smiles_nodes_valid = record.get("layer1_facts", {}).get("smiles_nodes_valid", [])
            smiles_linkers_valid = record.get("layer1_facts", {}).get("smiles_linkers_valid", [])
            if smiles_nodes_valid and smiles_linkers_valid:
                if all(smiles_nodes_valid) and all(smiles_linkers_valid):
                    stats["records_with_valid_smiles"] += 1

            validation_report = record.get("validation_report", {})
            if validation_report:
                stats["records_with_validation"] += 1
                status = validation_report.get("status", "unknown")
                stats["validation_status_dist"][status] += 1

                issue_log = validation_report.get("issue_log", [])
                if issue_log:
                    stats["total_issues"] += len(issue_log)
                    stats["issues_per_record"].append(len(issue_log))
                    for issue in issue_log:
                        severity = issue.get("severity", "unknown")
                        stats["issue_severity_dist"][severity] += 1

        except Exception as e:
            stats["errors"] += 1

    stats["unique_fg_types"] = list(stats["unique_fg_types"])
    stats["unique_fg_count"] = len(stats["unique_fg_types"])
    stats["avg_fg_per_record"] = (
        sum(stats["fg_per_record"]) / len(stats["fg_per_record"])
        if stats["fg_per_record"]
        else 0
    )
    stats["avg_issues_per_record"] = (
        sum(stats["issues_per_record"]) / len(stats["issues_per_record"])
        if stats["issues_per_record"]
        else 0
    )
    stats["top_20_fgs"] = dict(stats["fg_frequency"].most_common(20))
    stats["top_10_metals"] = dict(stats["metal_frequency"].most_common(10))

    del stats["fg_frequency"]
    del stats["metal_frequency"]
    del stats["fg_per_record"]
    del stats["issues_per_record"]
    stats["validation_status_dist"] = dict(stats["validation_status_dist"])
    stats["issue_severity_dist"] = dict(stats["issue_severity_dist"])

    return stats


# ============================================================================
# REPORTING
# ============================================================================
def format_number(n: int) -> str:
    """Format number with thousands separator."""
    return f"{n:,}"


def print_comparison_table(v1_stats: Dict, v2_stats: Dict) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("QMOF Enrichment: v1 vs v2 Quality Report")
    print("=" * 80)
    print(f"{'Metric':<40} {'v1':>15} {'v2':>15} {'Delta':>15}")
    print("-" * 80)

    v1_total = v1_stats["total_records"]
    v2_total = v2_stats["total_records"]
    delta = v2_total - v1_total
    print(
        f"{'Total records':<40} {format_number(v1_total):>15} "
        f"{format_number(v2_total):>15} {f'+{delta}' if delta >= 0 else delta:>15}"
    )

    v1_fg = v1_stats["records_with_fg_detected"]
    v2_fg = v2_stats["records_with_fg_detected"]
    del
