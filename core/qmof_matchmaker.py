import json
import os
import pandas as pd
from typing import List, Dict, Tuple, Any

import config
from core.constraint_utils import parse_functional_groups, check_negative_tags, canon, get_approved_vocab

class QMOFMatchmaker:
    """
    Direct MOF-level filtering for QMOF database.
    Unlike standard Matchmaker (which assembles nodes/edges/topologies),
    QMOFMatchmaker filters whole QMOFs directly based on Agent 2 constraints.
    """
    def __init__(self, index_path=None):
        if index_path is None:
            index_path = getattr(config, 'QMOF_INDEX_PATH', None)
            
        self.qmof_index = []
        if index_path and os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                self.qmof_index = json.load(f)
        else:
            print(f"Warning: QMOF index not found at {index_path}. QMOF mode will not yield matches.")
            
    def _check_metals(self, item: dict, required_metals: List[str]) -> bool:
        if not required_metals:
            return True
        item_metals = [canon(m) for m in item.get("metals", [])]
        req_metals = [canon(m) for m in required_metals]
        
        # Check if ANY of the required metals are present (OR Logic)
        for rm in req_metals:
            if rm in item_metals:
                return True
                
        # If we get here, none of the required metals were found
        return False
        
    def _check_connectivity(self, item: dict, node_query: dict) -> bool:
        conns = node_query.get("connectivity", [])
        if not conns:
            return True
        item_conn = item.get("connectivity_points")
        if item_conn in conns:
            return True
        # also check str versions just in case
        if str(item_conn) in [str(c) for c in conns]:
            return True
        return False

    def _check_and_tags(self, item: dict, and_tags: List[str]) -> bool:
        """Check that ALL specified AND tags are present (SP-3.09 fix)."""
        if not and_tags:
            return True
        item_tags = [canon(t) for t in item.get("functional_groups", [])]
        return all(tag in item_tags for tag in and_tags)

    def _check_or_tags(self, item: dict, or_tags: List[str]) -> bool:
        """Check that at least one of the OR tags is present."""
        if not or_tags:
            return True
        item_tags = [canon(t) for t in item.get("functional_groups", [])]
        return any(tag in item_tags for tag in or_tags)

    def match(self, specs: dict, tracker: Any = None, search_mode: str = "full") -> List[str]:
        """
        Takes Agent 2 specs and returns a list of matching qmof_ids.
        Reuses constraint_utils logic for tags.
        search_mode: "full", "metal_only", or "linker_only"
        """
        # Parse tags — keep AND and OR semantics separate (SP-3.09 / TAG-001 fix)
        global_and_tags, linker_or_tags, neg_tags = parse_functional_groups(specs, approved_vocab=get_approved_vocab(), tracker=tracker)
        # DO NOT merge: global_and_tags require ALL present, linker_or_tags require ANY present
        all_pos_tags = global_and_tags + linker_or_tags  # Only for tracker recording
        
        node_query = specs.get("node_query", {})
        req_metals = node_query.get("metals_include", [])
        
        matched_ids = []
        
        for qmof in self.qmof_index:
            if search_mode in ["full", "linker_only"]:
                # 1. Negative tag check
                if not check_negative_tags(qmof, neg_tags):
                    continue
                
            if search_mode in ["full", "metal_only"]:
                # 2. Connectivity check
                if not self._check_connectivity(qmof, node_query):
                    continue
                    
                # 3. Metals check
                if not self._check_metals(qmof, req_metals):
                    continue
                
            if search_mode in ["full", "linker_only"]:
                # 4a. AND tags: ALL must be present (global requirements) — SP-3.09 fix
                if not self._check_and_tags(qmof, global_and_tags):
                    if tracker and search_mode == "full":
                        first_tag = global_and_tags[0] if global_and_tags else "Unknown"
                        tracker.record_first_fail(first_tag)
                    continue

                # 4b. OR tags: at least one must be present (linker functional groups)
                if not self._check_or_tags(qmof, linker_or_tags):
                    if tracker and search_mode == "full":
                        first_tag = linker_or_tags[0] if linker_or_tags else "Unknown"
                        tracker.record_first_fail(first_tag)
                    continue
                
            # Record success if tracker exists
            if tracker and search_mode == "full":
                for t in all_pos_tags:
                    tracker.record(t, "both", "exact_feature", qmof["qmof_id"], "none", None)
                
            matched_ids.append(qmof["qmof_id"])
            
        return matched_ids

    def get_bandgap_data(self, qmof_ids: List[str]) -> pd.DataFrame:
        """
        Fetches band gap data from the index for the given list of matched qmof_ids.
        Returns a DataFrame for display or analysis.
        """
        df_rows = []
        qmof_lookup = {q["qmof_id"]: q for q in self.qmof_index}
        
        for qid in qmof_ids:
            if qid in qmof_lookup:
                qmof = qmof_lookup[qid]
                df_rows.append({
                    "qmof_id": qid,
                    "bandgap": qmof.get("bandgap")
                })
                
        return pd.DataFrame(df_rows)
