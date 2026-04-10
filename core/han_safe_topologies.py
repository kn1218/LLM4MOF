"""
HAN_SAFE_TOPOS — Canonical topology whitelist for live simulation.

The set is the intersection of:
  1. mof2zeo's 964-topology training vocabulary (core/mof2zeo/data/topology.txt)
  2. Single-node-type topologies in pormake's V3 dictionary

As of 2026-04-09, all 964 mof2zeo topologies are already single-node-type,
so HAN_SAFE_TOPOS == mof2zeo vocab.  We keep the intersection logic for safety
in case either vocabulary changes in a future update.
"""

import os
import json
from typing import FrozenSet

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _load_mof2zeo_vocab() -> set[str]:
    """Read the mof2zeo topology vocabulary file (one ID per line)."""
    with open(config.MOF2ZEO_TOPOLOGY_FILE, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _load_single_node_topologies() -> set[str]:
    """Return pormake topology IDs that have exactly one node type."""
    with open(config.TOPO_DICTIONARY_V3_PATH, "r", encoding="utf-8") as f:
        topo_data = json.load(f)
    return {
        t["ID"]
        for t in topo_data
        if len(set(t.get("node_connectivities", []))) == 1
        and len(t.get("node_connectivities", [])) >= 1
    }


def _compute_han_safe_topos() -> FrozenSet[str]:
    """Compute and freeze the intersection at import time."""
    mof2zeo_vocab = _load_mof2zeo_vocab()
    single_node_ids = _load_single_node_topologies()
    result = frozenset(mof2zeo_vocab & single_node_ids)
    print(f"[HAN_SAFE_TOPOS] {len(result)} eligible topologies "
          f"(mof2zeo={len(mof2zeo_vocab)}, single-node={len(single_node_ids)})")
    return result


# Computed once at import time
HAN_SAFE_TOPOS: FrozenSet[str] = _compute_han_safe_topos()


def is_han_safe(topology_id: str) -> bool:
    """Check if a topology is in the safe set."""
    return topology_id in HAN_SAFE_TOPOS


def filter_matchmaker_result(result: dict) -> dict:
    """
    Filter a matchmaker result dict, keeping only HAN_SAFE topologies.

    Args:
        result: dict with at least 'topology' key (list of topology IDs)

    Returns:
        New dict with filtered topology list; other keys unchanged.
    """
    if result.get("status") == "error":
        return result

    original_topos = result.get("topology", [])
    filtered_topos = [t for t in original_topos if t in HAN_SAFE_TOPOS]

    dropped = len(original_topos) - len(filtered_topos)
    if dropped > 0:
        print(f"[HAN_SAFE_TOPOS] Filtered {dropped}/{len(original_topos)} "
              f"topologies (kept {len(filtered_topos)})")

    return {**result, "topology": filtered_topos}
