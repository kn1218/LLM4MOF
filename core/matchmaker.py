# =============================================================================
# LLM4MOF Autonomous System - Matchmaker Engine
# =============================================================================
# Ported from the project's pilot prototype (Block 2)
# =============================================================================

import pandas as pd
import re
import os
import sys
import json

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import TOPO_DICTIONARY_V3_PATH


def _load_bblist(path):
    """Load a canonical building-block whitelist (one ID per line). Fail-open: returns None on any
    error so a missing list never breaks proposal (no extra filtering applied)."""
    try:
        with open(path, encoding="utf-8") as f:
            return frozenset(l.strip() for l in f if l.strip())
    except Exception:
        return None


# Canonical node/edge whitelist = the retrained mof2zeo vocab (core/mof2zeo/data, via config.BBLIST_*).
# The markscheme DBs are already
# pre-filtered to these; here we also restrict what the matchmaker can PROPOSE (live + markscheme).
_NODE_WHITELIST = _load_bblist(config.BBLIST_NODE_PATH)
_EDGE_WHITELIST = _load_bblist(config.BBLIST_EDGE_PATH)
from core.name_resolver import get_name_resolver
from core.constraint_utils import (
    canon, get_item_features, parse_functional_groups, check_global_requirements,
    check_negative_tags, get_approved_vocab, check_categorized_groups,
    check_linker_branches, strip_coordination_tags_from_branches
)


class Matchmaker:
    """
    The Matchmaker Engine connects Agent's constraints to the Database.
    It discovers valid topologies, nodes, and linkers based on the 
    specifications provided by Agent 2.
    """
    
    def __init__(self):
        """Load the building block and topology dictionaries."""
        print("--- LOADING MATCHMAKER DATABASES ---")
        
        try:
            # 1. Load Topology Dictionary V3 (JSON — mandatory)
            if os.path.exists(TOPO_DICTIONARY_V3_PATH):
                print(f"Loading V3 Topology Database from: {TOPO_DICTIONARY_V3_PATH}")
                with open(TOPO_DICTIONARY_V3_PATH, 'r', encoding='utf-8') as f:
                    self.topo_data = json.load(f)
                # Create lookup for O(1) access
                self.topo_lookup = {t['ID']: t for t in self.topo_data}
                print(f"Topologies Loaded (V3): {len(self.topo_data)} entries")
            else:
                raise FileNotFoundError(f"V3 Topology DB not found at {TOPO_DICTIONARY_V3_PATH}")
            
            # 2. Connect to NameResolver Singleton for BB Data
            # (Eliminates redundant JSON loading and ensures single-source-of-truth)
            print("Connecting to NameResolver Singleton...")
            resolver = get_name_resolver()
            self.bb_data = resolver.bb_data
            self.bb_lookup = resolver.bb_lookup
            print(f"Matchmaker connected. Access to {len(self.bb_data)} Building Blocks.")
                
        except Exception as e:
            print(f"CRITICAL ERROR LOADING MATCHMAKER DATA: {e}")
            raise
            
        # Load approved vocabulary from ontology
        self.approved_vocab = get_approved_vocab()

    def get_valid_topologies(self, available_cns: list, target_edge_cn: int = 2) -> tuple:
        """
        V3 Topology Matching with Single-Component Constraint.
        
        Finds topologies that:
        1. Are a SUBSET of available connectivities (we have the parts).
        2. Are STRICTLY SINGLE-COMPONENT (require only 1 node type and 1 edge type).
        3. Match the required edge connectivity (target_edge_cn).
        
        Args:
            available_cns: List of node connectivities we can provide (e.g., [12, 4])
            target_edge_cn: The linker connectivity we are restricted to (usually 2).
        
        Returns:
            (valid_codes_list, topology_by_cn_map)
        """
        valid_codes = []
        topology_by_cn = {}  # {node_cn: [topo_id, ...]}
        
        # Normalize input to set
        if isinstance(available_cns, int):
            available_set = {available_cns}
        elif isinstance(available_cns, list):
            available_set = set(available_cns)
        else:
            print(f"[Matchmaker] WARNING: Invalid connectivity type: {type(available_cns)}")
            return [], {}
        
        # Use JSON topology data
        if hasattr(self, 'topo_data') and self.topo_data is not None:
            for item in self.topo_data:
                # 1. Get Requirements from JSON
                # We use a set to ignore site indices (e.g., [3, 3] becomes {3})
                req_nodes = set(item.get('node_connectivities', []))
                req_edges = set(item.get('edge_connectivities', []))
                
                if not req_nodes:
                    continue  # Skip topologies with no node info
                
                # 2. The "Bag of Parts" Check (Robustness)
                # We still check this to ensure we actually have the specific node needed.
                if not req_nodes.issubset(available_set):
                    continue
                    
                # 3. The "Simplicity" Filter (Constraint)
                # STRICTLY enforce Single-Node and Single-Edge for now.
                if len(req_nodes) != 1:
                    continue  # Reject multi-node topologies (e.g., tbo needs 12+4)
                    
                if len(req_edges) != 1:
                    continue  # Reject multi-edge topologies
                
                # --- Edge Connectivity Check ---
                # Ensure the topology is compatible with our linker (e.g. 2-connected)
                # STRICT check: topology must require EXACTLY this edge connectivity
                if req_edges != {target_edge_cn}:
                    continue


                valid_codes.append(item['ID'])
                
                # Build Map
                node_cn = list(req_nodes)[0] # Safe because len=1 check passed
                if node_cn not in topology_by_cn:
                    topology_by_cn[node_cn] = []
                topology_by_cn[node_cn].append(item['ID'])
        
        return valid_codes, topology_by_cn
    
    @staticmethod
    def _check_abstract_features(item: dict, required_features: dict) -> bool:
        """
        Check if a building block's abstract_features match the required filter.
        
        Rules:
        - Only non-null required features are checked.
        - If the item's feature is None (unknown), it is NOT excluded (benefit of doubt).
        - If the item's feature is explicitly False and required is True (or vice versa), exclude.
        
        Args:
            item: BB dictionary item (Node or Edge)
            required_features: Dict of {feature_name: True/False/None}
            
        Returns:
            True if item passes all checks, False if any required feature mismatches.
        """
        if not required_features:
            return True
        
        item_af = item.get('abstract_features', {})
        if not item_af:
            return True  # No abstract_features data on item = don't exclude
        
        for feat_key, feat_val in required_features.items():
            if feat_val is None:
                continue  # null requirement = don't filter
            item_val = item_af.get(feat_key)
            if item_val is None:
                continue  # unknown in data = benefit of doubt, don't exclude
            if item_val != feat_val:
                return False  # explicit mismatch
        
        return True

    @staticmethod
    def _check_abstract_features_or(item: dict, required_features: dict) -> bool:
        """OR-gate version for linker abstract_features.

        Passes if the item matches AT LEAST ONE required feature.
        Used for linker_af to prevent AND-trap (e.g., HBD AND HBA → 0 linker matches).
        Falls back to True when required_features is empty or all values are None.
        """
        if not required_features:
            return True

        effective = [(k, v) for k, v in required_features.items() if v is not None]
        if not effective:
            return True  # All requirements were null = no filter

        item_af = item.get('abstract_features', {})
        if not item_af:
            return True  # No abstract_features data on item = benefit of doubt

        for feat_key, feat_val in effective:
            item_val = item_af.get(feat_key)
            if item_val is None:
                continue  # unknown = skip
            if item_val == feat_val:
                return True  # OR: any match is sufficient

        return False  # No feature matched

    def _search_nodes(self, specs: dict, req_cn: int) -> tuple:
        """Search nodes using V3 JSON logic."""
        node_candidates = []
        metal_query = specs['node_query']['metals_include']
        
        # Normalize Query
        target_metals = []
        if isinstance(metal_query, list): 
            target_metals = [m.strip().capitalize() for m in metal_query]
        elif isinstance(metal_query, str): 
            target_metals = [m.strip().capitalize() for m in metal_query.split(',')]
        
        is_any = any(m.lower() == "any" for m in target_metals)
        
        # Nuclearity Filter
        req_nuclearity = specs['node_query'].get('nuclearity')
        target_nuclearity = int(req_nuclearity) if (req_nuclearity is not None and str(req_nuclearity).isdigit() and int(req_nuclearity) > 0) else None



        # Prepare Negative Filters (Global "Avoid" Logic)
        vocab_set = getattr(self, 'approved_vocab', None)
        _, _, negative_tags = parse_functional_groups(specs, vocab_set)

        # --- Ligand Chemistry Prep (OR Logic) ---
        req_ligand_chem = specs['node_query'].get('ligand_chemistry', [])
        target_ligand_chem = set(l.lower() for l in req_ligand_chem)

        # Abstract Features Prep
        node_af = specs.get('node_query', {}).get('abstract_features', {})
        if not isinstance(node_af, dict): node_af = {}

        for item in self.bb_data:
            if item.get('Type', 'Node') != 'Node': continue

            # --- Canonical whitelist: propose only nodes in the mof2zeo vocab (core/mof2zeo/data/node.txt) ---
            if _NODE_WHITELIST is not None and item['ID'] not in _NODE_WHITELIST:
                continue

            # --- Connectivity ---
            raw_cn = item.get('connectivity')
            
            if isinstance(raw_cn, int): cns = {raw_cn}
            elif isinstance(raw_cn, list): cns = set(raw_cn)
            else: cns = set()
                
            if req_cn not in cns: continue
            
            # --- ROBUST NUCLEARITY ---
            if target_nuclearity:
                item_nuc = item.get('nuclearity', 0)
                if item_nuc != target_nuclearity: continue

            # --- Metals ---
            if not is_any:
                item_metals = item.get('metals', [])
                match = any(tm in item_metals for tm in target_metals)
                if not match: continue

            # --- Ligand Chemistry ---
            if target_ligand_chem:
                # Get item chemistry (try various V3/V2 fields)
                item_chem = item.get('ligand_chemistry', []) or item.get('connection_chemistry', [])
                if isinstance(item_chem, str): item_chem = [item_chem]
                
                # Check intersection
                item_chem_lower = set(c.lower() for c in item_chem)
                if not target_ligand_chem.intersection(item_chem_lower): continue
            
            # --- NEGATIVE FILTER (The "Bouncer") ---
            # If node has Forbidden Group, remove it immediately.
            if not check_negative_tags(item, negative_tags):
                continue

            # Note: Positive checks are deferred to Assembly Stage (Union Logic).

            # --- ABSTRACT FEATURES FILTER ---
            if not self._check_abstract_features(item, node_af):
                continue

            node_candidates.append(item['ID'])
            
        return node_candidates, target_metals

    def _search_linkers(self, specs: dict) -> tuple:
        """
        Search linkers using V3 JSON logic (Semantic Search).
        Returns: (candidates_list, diagnostics_dict)
        """
        linker_ads = []
        lspecs = specs['linker_query']
        
        # filters
        req_cn = int(lspecs.get('connectivity') or 2)
        
        # --- ROBUST LENGTH PARSING (Fix for NoneType Error) ---
        req_len_min = float(lspecs.get('length_min') or 0.0)
        req_len_max = float(lspecs.get('length_max') or 999.0)
        req_rigid = lspecs.get('is_rigid') 

        
        # Get Tags (Split Positive/Negative)
        vocab_set = getattr(self, 'approved_vocab', None)
        _, _, negative_tags = parse_functional_groups(specs, vocab_set)

        # Abstract Features Prep
        linker_af = specs.get('linker_query', {}).get('abstract_features', {})
        if not isinstance(linker_af, dict): linker_af = {}

        # Branch matching prep — strip coordination tags for PorMake edges
        raw_branches = specs.get('linker_query', {}).get('linker_branches', [])
        linker_branches = strip_coordination_tags_from_branches(raw_branches) if raw_branches else []

        # Diagnostics counters
        diag = {
            "total_linkers": 0, "cn_match": 0, "len_match": 0,
            "rigid_match": 0, "chem_match": 0, "af_match": 0, "final_match": 0
        }
        
        for item in self.bb_data:
            if item['Type'] != 'Edge': continue

            # --- Canonical whitelist: propose only edges in the mof2zeo vocab (core/mof2zeo/data/edge.txt) ---
            if _EDGE_WHITELIST is not None and item['ID'] not in _EDGE_WHITELIST:
                continue

            diag["total_linkers"] += 1
            
            # 1. Connectivity
            if item.get('connectivity') != req_cn: continue
            diag["cn_match"] += 1
            
            # 2. Length (Precise)
            length = item.get('length', 0.0)
            if not (req_len_min <= length <= req_len_max): continue
            diag["len_match"] += 1
            
            # 3. Rigidity (Optional)
            if req_rigid is not None:
                if item.get('is_rigid') != req_rigid: continue
            diag["rigid_match"] += 1

            if not check_negative_tags(item, negative_tags):
                continue

            # Note: Positive checks are deferred to Union Logic.
                

            diag["chem_match"] += 1

            # --- ABSTRACT FEATURES FILTER (OR logic: linker passes if ANY feature matches) ---
            if not self._check_abstract_features_or(item, linker_af):
                continue
            diag["af_match"] += 1

            # --- BRANCH MATCHING (OR-of-ANDs) ---
            if linker_branches:
                if not check_linker_branches(item, linker_branches):
                    continue

            linker_ads.append(item['ID'])
        
        diag["final_match"] = len(linker_ads)
        return linker_ads, diag


    def smart_matchmaker_single_node(self, specs: dict) -> dict:
        """
        Main orchestration function.
        Handles V3 JSON logic with Single-Component constraint.
        
        Architecture:
            Phase A: Topology Discovery (Global, using full CN list)
            Phase B: Node Search (Per-CN loop)
            Phase C: Linker Search (Independent)
            Phase D: Union Logic Assembly
        """
        candidates = {"topology": [], "node": [], "edge": [], "diagnostics": {}}
        
        # 1. PARSE CONNECTIVITY (Ensure List)
        # Agent 2 might return int (8), list ([8, 12]), or None (no constraint).
        # Normalize to list; if None/missing, default to common single-node CNs.
        raw_cn = specs['node_query'].get('connectivity')
        if raw_cn is None:
            target_connectivities = [3, 4, 6, 8, 12]  # default: all common single-node CNs
            print("[Matchmaker] WARNING: node connectivity unspecified, defaulting to [3,4,6,8,12]")
        elif isinstance(raw_cn, list):
            target_connectivities = [int(c) for c in raw_cn if c is not None]
            if not target_connectivities:
                target_connectivities = [3, 4, 6, 8, 12]
                print("[Matchmaker] WARNING: empty connectivity list, defaulting to [3,4,6,8,12]")
        else:
            target_connectivities = [int(raw_cn)]
            
        print(f"--- MATCHMAKER (V3=True) ---")
        print(f"Target Connectivities: {target_connectivities}")
        
        # ---------------------------------------------------------
        # PHASE A: TOPOLOGY DISCOVERY (Global Check)
        # Pass the FULL list (e.g., [12, 6]) to find matching single-node topologies.
        # This returns topos like ['fcu'] (for 12) and ['pcu'] (for 6) if valid.
        # ---------------------------------------------------------
        
        # Helper: Extract Linker Connectivity (Default 2)
        l_spec_cn = specs['linker_query'].get('connectivity', 2)
        try:
            target_edge_cn = int(l_spec_cn)
        except (ValueError, TypeError):
            target_edge_cn = 2
            
        candidates['topology'], candidates['topology_by_cn'] = self.get_valid_topologies(
            target_connectivities, target_edge_cn=target_edge_cn
        )
        print(f"Topologies Found: {len(candidates['topology'])}")
        
        if not candidates['topology']:
            print(f"Warning: No Single-Node topologies found for connectivities {target_connectivities} (Edge CN={target_edge_cn}).")
        
        # ---------------------------------------------------------
        # PHASE B: NODE SEARCH (Loop through each CN)
        # ---------------------------------------------------------
        all_nodes = []
        target_metals = None  # Track for error reporting
        
        for req_cn in target_connectivities:
            node_ids, target_metals = self._search_nodes(specs, req_cn)
            all_nodes.extend(node_ids)

        candidates['node'] = list(set(all_nodes))
        
        if not candidates['node']:
            # SP-3.08: Return structured error dict (not string) for consistent return type
            print(f"Warning: No nodes found for metals '{target_metals}' in {target_connectivities}.")
            return {
                'status': 'error',
                'reason': 'no_matching_nodes',
                'message': f"No nodes found for metals '{target_metals}' with connectivities {target_connectivities}.",
                'topology': candidates['topology'],
                'node': [],
                'edge': [],
                'diagnostics': candidates.get('diagnostics', {})
            }

        print(f"Nodes Found: {len(candidates['node'])}")

        # ---------------------------------------------------------
        # PHASE C: LINKER SEARCH (Independent of Node CN)
        # ---------------------------------------------------------
        linker_ids, diag = self._search_linkers(specs)
        candidates['diagnostics'] = diag

        # ---------------------------------------------------------
        # PHASE D: VIRTUAL ASSEMBLY & UNION LOGIC CHECK
        # ---------------------------------------------------------
        # We now have 'all_nodes' (Physics + Metals + Non-Banned)
        # And 'linker_ids' (Physics + Non-Banned)
        # We must filter them to ensure they form at least ONE valid pair
        # that satisfies the "Union Logic" (Global Requirements).
        
        vocab_set = getattr(self, 'approved_vocab', None)
        global_and_tags, linker_or_tags, _ = parse_functional_groups(specs, vocab_set)
        print(f"Applying Union Logic - AND tags: {global_and_tags}, OR tags: {linker_or_tags}")

        # When branches are present, skip OR-tag check in Union Logic
        # (branches already filtered linkers in _search_linkers)
        linker_branches_d = specs.get('linker_query', {}).get('linker_branches', [])
        use_branch_mode = bool(linker_branches_d)
        if use_branch_mode:
            print(f"Branch mode active ({len(linker_branches_d)} branches) - skipping OR-tags in Union Logic")

        if not global_and_tags and not linker_or_tags:
            # Shortcut: if no global requirements, all matches are valid
            candidates['edge'] = linker_ids
        else:
            valid_nodes = set()
            valid_edges = set()

            # Virtual Assembly Loop (N * M)
            # This validates that a component is 'useful' in at least one valid MOF.
            effective_or_tags = [] if use_branch_mode else linker_or_tags
            for n_id in all_nodes:
                for l_id in linker_ids:
                    if check_global_requirements(n_id, l_id, global_and_tags, self.bb_lookup,
                                                 linker_or_tags=effective_or_tags):
                        valid_nodes.add(n_id)
                        valid_edges.add(l_id)

            # Update lists with only VALID components
            # Note: We filter the originally found nodes to remove those that fail Union Logic
            candidates['node'] = list(valid_nodes)
            candidates['edge'] = list(valid_edges)

            print(f"Union Logic Pruning: Nodes {len(all_nodes)}->{len(candidates['node'])}, Linkers {len(linker_ids)}->{len(candidates['edge'])}")

        # ---------------------------------------------------------
        # PHASE E: OPTIONAL CATEGORIZED FUNCTIONAL GROUP FILTER
        # ---------------------------------------------------------
        # Only activates if Agent 2 specs contain backbone_requirements,
        # substituent_requirements, or min_group_counts in linker_query.
        # These refine linker selection by distinguishing backbone from
        # substituent chemistry and enforcing minimum group counts.
        linker_q = specs.get('linker_query', {})
        backbone_reqs = linker_q.get('backbone_requirements') or []
        substituent_reqs = linker_q.get('substituent_requirements') or []
        min_counts = linker_q.get('min_group_counts') or {}

        if backbone_reqs or substituent_reqs or min_counts:
            pre_count = len(candidates['edge'])
            filtered_edges = []
            for l_id in candidates['edge']:
                item = self.bb_lookup.get(l_id, {})
                if check_categorized_groups(item, backbone_reqs, substituent_reqs, min_counts):
                    filtered_edges.append(l_id)
            candidates['edge'] = filtered_edges
            print(f"Categorized Filter: Linkers {pre_count}->{len(candidates['edge'])}")

        print(f"Linkers Found: {len(candidates['edge'])}")
        print(f"   Diagnostics: {candidates['diagnostics']}")

        # Extract preferred_features (soft preference — not filtered, only used for ranking bonus)
        candidates['preferred_features'] = {
            'node': specs.get('node_query', {}).get('preferred_features') or {},
            'linker': specs.get('linker_query', {}).get('preferred_features') or {},
        }
        if any(candidates['preferred_features'].values()):
            print(f"   Preferred features: {candidates['preferred_features']}")

        return candidates


# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_matchmaker():
    """Test the matchmaker with V3 prompt structure."""
    
    print("\n" + "="*60)
    print("MATCHMAKER MODULE TEST (V3)")
    print("="*60 + "\n")
    
    # Sample Agent 2 output (V3 Structure)
    test_specs = {
        "node_query": {
            "metals_include": ["Zr"],
            "connectivity": [12],
            "nuclearity": 6,
            "abstract_features": {}
        },
        "linker_query": {
            "connectivity": 2,
            "length_min": 6.0,
            "length_max": 12.0,
            "is_rigid": True,
            "functional_groups": ["Oxygen"],
            "abstract_features": {}
        },
        "geometry_filter": {
            "target_Di_min": 12.0, "target_Di_max": 20.0,
            "target_Df_min": 7.0, "target_Df_max": 10.0
        }
    }
    
    matcher = Matchmaker()
    results = matcher.smart_matchmaker_single_node(test_specs)
    
    if results.get('status') == 'error':
        print(results['message'])
    else:
        print("\n--- RESULTS ---")
        print(f"Topologies: {len(results['topology'])}")
        print(f"Nodes: {len(results['node'])}")
        print(f"Linkers: {len(results['edge'])}")
        print(f"Diagnostics: {results.get('diagnostics')}")
        
        if len(results['node']) > 0 and len(results['edge']) > 0:
            print("[PASS] VALIDATION PASSED")
        else:
            print("[FAIL] VALIDATION FAILED - No candidates found")

    # --- TEST 2: With Abstract Features ---
    print("\n--- TEST 2: Abstract Features Filter ---")
    test_specs_af = {
        "node_query": {
            "metals_include": ["Zr"],
            "connectivity": [12],
            "nuclearity": 6,
            "abstract_features": {"has_open_metal_site": True}
        },
        "linker_query": {
            "connectivity": 2,
            "length_min": 6.0,
            "length_max": 12.0,
            "is_rigid": True,
            "functional_groups": ["Oxygen"],
            "abstract_features": {"is_conjugated": True}
        },
        "geometry_filter": {
            "target_Di_min": 12.0, "target_Di_max": 20.0,
            "target_Df_min": 7.0, "target_Df_max": 10.0
        }
    }
    results_af = matcher.smart_matchmaker_single_node(test_specs_af)
    
    if results_af.get('status') == 'error':
        print(results_af['message'])
    else:
        print(f"Nodes (with abstract_features): {len(results_af['node'])}")
        print(f"Linkers (with abstract_features): {len(results_af['edge'])}")
        
        # Verify: abstract_features should REDUCE candidates vs Test 1
        if len(results_af['node']) <= len(results['node']) and len(results_af['edge']) <= len(results['edge']):
            print("[PASS] Abstract features reduced candidate set (as expected)")
        else:
            print("[WARN] Abstract features did NOT reduce candidates (unexpected)")

    # --- TEST 3: Linker Branches ---
    print("\n--- TEST 3: Linker Branches (OR-of-ANDs) ---")
    test_specs_branches = {
        "node_query": {
            "metals_include": ["Zr"],
            "connectivity": [12],
            "nuclearity": 6,
            "abstract_features": {}
        },
        "linker_query": {
            "connectivity": 2,
            "length_min": 6.0,
            "length_max": 12.0,
            "is_rigid": True,
            "functional_groups": [],
            "abstract_features": {},
            "linker_branches": [
                {"description": "aromatic carboxylate", "required_tags": ["Benzene", "Carboxyl"]},
                {"description": "pyridine-based", "required_tags": ["Pyridine"]}
            ]
        },
        "geometry_filter": {
            "target_Di_min": 12.0, "target_Di_max": 20.0,
            "target_Df_min": 7.0, "target_Df_max": 10.0
        }
    }
    results_br = matcher.smart_matchmaker_single_node(test_specs_branches)
    
    if results_br.get('status') == 'error':
        print(results_br['message'])
    else:
        print(f"Nodes (with branches): {len(results_br['node'])}")
        print(f"Linkers (with branches): {len(results_br['edge'])}")
        
        # Branches should filter linkers compared to no-filter baseline
        if len(results_br['edge']) <= len(results['edge']):
            print("[PASS] Branches reduced or maintained linker count (as expected)")
        else:
            print("[WARN] Branches did NOT reduce linkers (unexpected)")

if __name__ == "__main__":
    test_matchmaker()
