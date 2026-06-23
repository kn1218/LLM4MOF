"""
SIM_SAFE_TOPOS — Canonical topology whitelist for live simulation.

The set is the intersection of:
  1. mof2zeo's 964-topology training vocabulary (core/mof2zeo/data/topology.txt)
  2. Single-node-type topologies in pormake's V3 dictionary

As of 2026-04-09, all 964 mof2zeo topologies are already single-node-type,
so SIM_SAFE_TOPOS == mof2zeo vocab.  We keep the intersection logic for safety
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


def _load_bblist_topologies() -> set[str]:
    """Canonical topology whitelist = the retrained mof2zeo vocab (core/mof2zeo/data/topology.txt). Fail-open:
    empty set means 'no extra restriction' (the intersection below is skipped)."""
    try:
        with open(config.BBLIST_TOPOLOGY_PATH, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception:
        return set()


def _compute_sim_safe_topos() -> FrozenSet[str]:
    """Compute and freeze the intersection at import time, excluding LAMMPS blacklist."""
    mof2zeo_vocab = _load_mof2zeo_vocab()
    single_node_ids = _load_single_node_topologies()
    blacklist = getattr(config, "LAMMPS_TOPOLOGY_BLACKLIST", set())
    result = (mof2zeo_vocab & single_node_ids) - blacklist
    bblist = _load_bblist_topologies()
    if bblist:  # restrict to the canonical whitelist when present
        result = result & bblist
    result = frozenset(result)
    print(f"[SIM_SAFE_TOPOS] {len(result)} eligible topologies "
          f"(mof2zeo={len(mof2zeo_vocab)}, single-node={len(single_node_ids)}, "
          f"bblist={len(bblist)}, blacklisted={len(blacklist)})")
    return result


# Computed once at import time
SIM_SAFE_TOPOS: FrozenSet[str] = _compute_sim_safe_topos()


def is_sim_safe(topology_id: str) -> bool:
    """Check if a topology is in the safe set."""
    return topology_id in SIM_SAFE_TOPOS


def filter_matchmaker_result(result: dict) -> dict:
    """
    Filter a matchmaker result dict, keeping only SIM_SAFE topologies.

    Args:
        result: dict with at least 'topology' key (list of topology IDs)

    Returns:
        New dict with filtered topology list; other keys unchanged.
    """
    if result.get("status") == "error":
        return result

    original_topos = result.get("topology", [])
    filtered_topos = [t for t in original_topos if t in SIM_SAFE_TOPOS]

    dropped = len(original_topos) - len(filtered_topos)
    if dropped > 0:
        print(f"[SIM_SAFE_TOPOS] Filtered {dropped}/{len(original_topos)} "
              f"topologies (kept {len(filtered_topos)})")

    return {**result, "topology": filtered_topos}
