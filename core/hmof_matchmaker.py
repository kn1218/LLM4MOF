"""
hMOF Matchmaker - Direct MOF-level filtering for the hMOF database.

Mirrors QMOFMatchmaker pattern but handles gas adsorption properties
(H2, CH4, CO2, Xe, Kr) and structural descriptors (SA, VF, PLD, LCD).

The hMOF database contains 51,163 hypothetical MOFs from the Snurr group
with GCMC-simulated gas adsorption data.
"""

import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional

import config
from core.constraint_utils import (
    parse_functional_groups, check_negative_tags, canon, get_approved_vocab,
    check_categorized_groups, check_linker_branches
)


class HMOFMatchmaker:
    """
    Direct MOF-level filtering for hMOF database.
    Unlike the standard Matchmaker (which assembles nodes/edges/topologies),
    HMOFMatchmaker filters whole hMOFs directly based on Agent 2 constraints.
    """

    # Available gas adsorption properties
    GAS_PROPERTIES = [
        "h2_uptake_2bar_77K",
        "h2_uptake_100bar_77K",
        "ch4_uptake_35bar_298K",
        "co2_uptake_2_5bar_298K",
        "xe_loading_1bar_273K",
        "kr_loading_1bar_273K",
        "xekr_selectivity_1bar",
    ]

    # Available structural properties
    STRUCTURAL_PROPERTIES = [
        "surface_area_m2g",
        "void_fraction",
        "density",
        "pld",
        "lcd",
    ]

    def __init__(self, index_path: Optional[str] = None):
        if index_path is None:
            index_path = getattr(config, "HMOF_INDEX_PATH", None)

        self.hmof_index: List[dict] = []
        if index_path and os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                self.hmof_index = json.load(f)
            print(f"[HMOFMatchmaker] Loaded {len(self.hmof_index)} hMOF entries")
        else:
            print(
                f"Warning: hMOF index not found at {index_path}. "
                "hMOF mode will not yield matches."
            )

    # ── Filter helpers ────────────────────────────────────────────────

    def _check_metals(self, item: dict, required_metals: List[str]) -> bool:
        if not required_metals:
            return True
        item_metals = [canon(m) for m in item.get("metals", [])]
        req_metals = [canon(m) for m in required_metals]
        if any(m == "any" for m in req_metals):
            return True
        return any(rm in item_metals for rm in req_metals)

    def _check_topology(self, item: dict, required_topologies: List[str]) -> bool:
        if not required_topologies:
            return True
        topo = item.get("topology")
        if topo is None:
            return False
        return canon(topo) in [canon(t) for t in required_topologies]

    def _check_and_tags(self, item: dict, and_tags: List[str]) -> bool:
        """All specified AND tags must be present."""
        if not and_tags:
            return True
        item_tags = [canon(t) for t in item.get("functional_groups", [])]
        return all(tag in item_tags for tag in and_tags)

    def _check_or_tags(self, item: dict, or_tags: List[str]) -> bool:
        """At least one of the OR tags must be present."""
        if not or_tags:
            return True
        item_tags = [canon(t) for t in item.get("functional_groups", [])]
        return any(tag in item_tags for tag in or_tags)

    def _check_range(
        self, item: dict, key: str, vmin: Optional[float], vmax: Optional[float]
    ) -> bool:
        """Check if item[key] falls within [vmin, vmax]. None means open bound."""
        val = item.get(key)
        if val is None:
            return True  # No data = don't filter out
        if vmin is not None and val < vmin:
            return False
        if vmax is not None and val > vmax:
            return False
        return True

    # ── Main match method ─────────────────────────────────────────────

    def match(
        self,
        specs: dict,
        tracker: Any = None,
        search_mode: str = "full",
        structural_filters: Optional[Dict[str, tuple]] = None,
    ) -> List[str]:
        """
        Filter hMOFs based on Agent 2 constraints.

        Args:
            specs: Agent 2 constraint dict with node_query, linker_query,
                   global_requirements
            tracker: Optional ProvenanceTracker
            search_mode: "full", "metal_only", or "linker_only"
            structural_filters: Optional dict of {property_name: (min, max)}
                e.g. {"surface_area_m2g": (1000, 5000), "pld": (5.0, 20.0)}

        Returns:
            List of matching hmof_ids
        """
        # Parse tags
        global_and_tags, linker_or_tags, neg_tags = parse_functional_groups(
            specs, approved_vocab=get_approved_vocab(), tracker=tracker
        )
        linker_branches = specs.get('linker_query', {}).get('linker_branches', [])

        node_query = specs.get("node_query", {})
        req_metals = node_query.get("metals_include", [])
        req_topologies = node_query.get("topology", [])
        if isinstance(req_topologies, str):
            req_topologies = [req_topologies] if req_topologies else []

        matched_ids = []

        for hmof in self.hmof_index:
            # 1. Negative tag check
            if search_mode in ["full", "linker_only"]:
                if not check_negative_tags(hmof, neg_tags):
                    continue

            # 2. Metal check
            if search_mode in ["full", "metal_only"]:
                if not self._check_metals(hmof, req_metals):
                    continue

            # 3. Topology check
            if search_mode in ["full", "metal_only"]:
                if not self._check_topology(hmof, req_topologies):
                    continue

            # 4a. AND tags
            if search_mode in ["full", "linker_only"]:
                if not self._check_and_tags(hmof, global_and_tags):
                    if tracker and search_mode == "full":
                        first_tag = global_and_tags[0] if global_and_tags else "Unknown"
                        tracker.record_first_fail(first_tag)
                    continue

            # 4b. OR tags
            if search_mode in ["full", "linker_only"]:
                if not self._check_or_tags(hmof, linker_or_tags):
                    if tracker and search_mode == "full":
                        first_tag = (
                            linker_or_tags[0] if linker_or_tags else "Unknown"
                        )
                        tracker.record_first_fail(first_tag)
                    continue

            # 4b.5: Branch matching (OR-of-ANDs)
            if search_mode in ["full", "linker_only"]:
                if linker_branches:
                    if not check_linker_branches(hmof, linker_branches):
                        continue

            # 4c. Categorized functional group check (OPTIONAL)
            if search_mode in ["full", "linker_only"]:
                linker_q = specs.get('linker_query', {})
                backbone_reqs = linker_q.get('backbone_requirements') or []
                substituent_reqs = linker_q.get('substituent_requirements') or []
                min_counts = linker_q.get('min_group_counts') or {}
                if backbone_reqs or substituent_reqs or min_counts:
                    if not check_categorized_groups(hmof, backbone_reqs, substituent_reqs, min_counts):
                        continue

            # 5. Structural property filters
            if structural_filters and search_mode == "full":
                skip = False
                for prop, (vmin, vmax) in structural_filters.items():
                    if not self._check_range(hmof, prop, vmin, vmax):
                        skip = True
                        break
                if skip:
                    continue

            # Record success
            if tracker and search_mode == "full":
                all_pos = global_and_tags + linker_or_tags
                for t in all_pos:
                    tracker.record(
                        t, "both", "exact_feature", hmof["hmof_id"], "none", None
                    )

            matched_ids.append(hmof["hmof_id"])

        return matched_ids

    # ── Data retrieval ────────────────────────────────────────────────

    def get_gas_data(self, hmof_ids: List[str]) -> pd.DataFrame:
        """Return gas adsorption data for matched hMOFs."""
        lookup = {h["hmof_id"]: h for h in self.hmof_index}
        rows = []
        for hid in hmof_ids:
            if hid in lookup:
                hmof = lookup[hid]
                row = {"hmof_id": hid}
                for prop in self.GAS_PROPERTIES:
                    row[prop] = hmof.get(prop)
                rows.append(row)
        return pd.DataFrame(rows)

    def get_property_data(
        self, hmof_ids: List[str], property_name: str
    ) -> pd.DataFrame:
        """Return a specific property for matched hMOFs."""
        lookup = {h["hmof_id"]: h for h in self.hmof_index}
        rows = []
        for hid in hmof_ids:
            if hid in lookup:
                rows.append(
                    {
                        "hmof_id": hid,
                        property_name: lookup[hid].get(property_name),
                    }
                )
        return pd.DataFrame(rows)

    def get_structural_data(self, hmof_ids: List[str]) -> pd.DataFrame:
        """Return structural properties for matched hMOFs."""
        lookup = {h["hmof_id"]: h for h in self.hmof_index}
        rows = []
        for hid in hmof_ids:
            if hid in lookup:
                hmof = lookup[hid]
                row = {"hmof_id": hid}
                for prop in self.STRUCTURAL_PROPERTIES:
                    row[prop] = hmof.get(prop)
                rows.append(row)
        return pd.DataFrame(rows)


# ── Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("hMOF MATCHMAKER TEST")
    print("=" * 60)

    mm = HMOFMatchmaker()

    if not mm.hmof_index:
        print("No hMOF data loaded. Exiting.")
        exit(1)

    # Test 1: Zn + Aromatic + Carboxyl
    specs = {
        "node_query": {"metals_include": ["Zn"]},
        "linker_query": {"functional_groups": ["Aromatic"]},
        "global_requirements": {
            "include_tags": ["Carboxyl"],
            "exclude_tags": [],
        },
    }
    ids = mm.match(specs)
    print(f"\nTest 1 - Zn + Aromatic + Carboxyl: {len(ids)} matches")

    if ids:
        gas = mm.get_gas_data(ids[:5])
        print(f"  Sample gas data:\n{gas.to_string(index=False)}")

    # Test 2: Cu + Nitrogen heterocycle, SA > 1000
    specs2 = {
        "node_query": {"metals_include": ["Cu"]},
        "linker_query": {"functional_groups": ["Nitrogen", "Heterocycle"]},
        "global_requirements": {"include_tags": [], "exclude_tags": ["Halogen"]},
    }
    ids2 = mm.match(
        specs2,
        structural_filters={"surface_area_m2g": (1000, None), "pld": (5.0, None)},
    )
    print(f"\nTest 2 - Cu + N-heterocycle + SA>1000 + PLD>5: {len(ids2)} matches")

    if ids2:
        struct = mm.get_structural_data(ids2[:5])
        print(f"  Sample structural data:\n{struct.to_string(index=False)}")

    # Test 3: Any metal, high H2 uptake filter
    specs3 = {
        "node_query": {"metals_include": ["Any"]},
        "linker_query": {"functional_groups": []},
        "global_requirements": {"include_tags": [], "exclude_tags": []},
    }
    ids3 = mm.match(specs3)
    print(f"\nTest 3 - Any metal (no chemistry filter): {len(ids3)} matches (should be ~all)")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
