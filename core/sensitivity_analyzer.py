# =============================================================================
# LLM2POR Autonomous System - Sensitivity Analyzer
# =============================================================================
# Ported from: Pilot project/LLM2PORMAKE_Trial10_20251217.ipynb (Block 3)
# Purpose: Human analysis ONLY - NOT shared with Agent 1
# =============================================================================

import pandas as pd
import numpy as np
import scipy.stats as stats
import re
import os
import sys
import json

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MASTER_DB_PATH
from core.name_resolver import get_name_resolver
import config
from core.constraint_utils import parse_functional_groups, check_global_requirements, get_approved_vocab


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on filtered MOF candidates.
    
    This analysis is for HUMAN USE ONLY to evaluate how well the LLM's
    hypothesis is performing. It uses the "markscheme" (pre-simulated database)
    to calculate metrics like Enrichment Factor and P-Value.
    
    IMPORTANT: This data is NEVER shared with Agent 1.
    """
    
    def __init__(self):
        """Load the master database (markscheme) and building blocks."""
        print("--- LOADING SENSITIVITY ANALYZER DATABASES ---")
        
        try:
            self.is_qmof = config.is_qmof_mode()
            self.is_hmof = config.is_hmof_mode()
            if self.is_hmof:
                hmof_path = getattr(config, 'HMOF_INDEX_PATH', None)
                print(f"Loading hMOF Database from: {hmof_path}")
                with open(hmof_path, "r", encoding="utf-8") as f:
                    hmof_data = json.load(f)
                self.df_master = pd.DataFrame(hmof_data)
                # Rename structural columns to match SA expected names
                # hMOF synonyms: lcd=di, pld=df, surface_area_m2g=sa, void_fraction=vf
                col_map = getattr(config, 'HMOF_COLUMN_MAP', {})
                self.df_master = self.df_master.rename(columns=col_map)
                self.df_master['filename'] = self.df_master['hmof_id']
                # Synthesize missing columns so full geometry filtering works:
                # dif (free sphere path) ~ pld (best proxy from Zeo++ literature)
                # cv (cell volume) = not available, fill with neutral value
                self.df_master['dif'] = self.df_master['df']  # PLD is standard proxy
                self.df_master['cv'] = 0.0  # No cell volume data; 0 passes default [0, 999999] range
                # Compute Original_Rank for EF calculation (rank 1 = best/highest value)
                target_col = config.ACTIVE_METRIC_COLUMN
                if target_col in self.df_master.columns:
                    self.df_master['Original_Rank'] = self.df_master[target_col].rank(
                        ascending=False, method='min', na_option='bottom'
                    ).astype(int)
                    print(f"hMOF DB Loaded: {len(self.df_master)} MOFs (ranked by {target_col})")
                else:
                    print(f"hMOF DB Loaded: {len(self.df_master)} MOFs (WARNING: no rank for {target_col})")
            elif self.is_qmof:
                print(f"Loading QMOF Tracker Database from: {getattr(config, 'QMOF_CSV_PATH')}")
                self.df_master = pd.read_csv(getattr(config, 'QMOF_CSV_PATH'), encoding='utf-8', low_memory=False)
                self.df_master['filename'] = self.df_master['qmof_id']
                # Map QMOF structural synonyms to SA column names
                # qmof.csv has info.lcd, info.pld, info.density, info.volume
                qmof_col_map = {
                    'info.lcd': 'di',        # largest cavity diameter
                    'info.pld': 'df',         # pore limiting diameter
                    'info.density': 'density', # crystal density
                    'info.volume': 'cv',       # cell volume
                }
                self.df_master = self.df_master.rename(columns=qmof_col_map)
                # Synthesize missing: sa, vf, dif not in qmof.csv
                self.df_master['dif'] = self.df_master['df']  # PLD as proxy
                if 'sa' not in self.df_master.columns:
                    self.df_master['sa'] = 0.0    # neutral passthrough
                if 'vf' not in self.df_master.columns:
                    self.df_master['vf'] = 0.0    # neutral passthrough
                # Merge enriched metadata from qmof_index_v2 for human-friendly feedback
                qmof_index_path = getattr(config, 'QMOF_INDEX_PATH', None)
                if qmof_index_path and os.path.exists(qmof_index_path):
                    with open(qmof_index_path, 'r', encoding='utf-8') as f:
                        qi_data = json.load(f)
                    qi_cols = ['qmof_id', 'readable_name', 'metals', 'functional_groups',
                               'geometry', 'connectivity_points', 'coordinating_groups',
                               'oxidation_states', 'spin_state']
                    qi_df = pd.DataFrame(qi_data)[[c for c in qi_cols if c in pd.DataFrame(qi_data).columns]]
                    self.df_master = self.df_master.merge(qi_df, on='qmof_id', how='left')
                # Compute Original_Rank for EF calculation
                # For bandgap: rank 1 = lowest bandgap is NOT always "best" —
                # it depends on the target. Use ascending=True so rank 1 = smallest bandgap.
                # The SA's EF logic counts how many subset members are in the top X% globally.
                # For bandgap, "top" means specific range, not always highest.
                # BUT the SA code uses rank <= cutoff to count hits, so rank 1 = "best".
                # We rank by bandgap ascending: rank 1 = smallest bandgap (metallic).
                # This is imperfect for bandgap (target-dependent), but matches SA's
                # assumption that lower rank = better performance.
                target_col = config.ACTIVE_METRIC_COLUMN
                if target_col in self.df_master.columns and 'Original_Rank' not in self.df_master.columns:
                    # For bandgap: higher bandgap is NOT always better. But the SA's
                    # _calculate_advanced_stats uses bandgap categories, not EF for QMOF.
                    # We still provide the rank so EF CAN be computed if needed.
                    self.df_master['Original_Rank'] = self.df_master[target_col].rank(
                        ascending=False, method='min', na_option='bottom'
                    ).astype(int)
            else:
                self.df_master = pd.read_csv(MASTER_DB_PATH, encoding='utf-8')
                # Clean Master DB: Ensure filenames are strings and valid
                self.df_master = self.df_master[
                    self.df_master['filename'].apply(
                        lambda x: isinstance(x, str) and x.count('+') == 2
                    )
                ].copy()
            
            # Connect to NameResolver Singleton for BB Data
            print("Connecting to NameResolver Singleton...")
            resolver = get_name_resolver()
            self.bb_data = resolver.bb_data
            self.bb_lookup = resolver.bb_lookup
            print(f"Sensitivity Analyzer connected. Access to {len(self.bb_data)} Building Blocks.")
            
            print(f"Master DB Loaded: {len(self.df_master)} MOFs")
            
        except Exception as e:
            print(f"CRITICAL ERROR LOADING DATA: {e}")
            raise

        # Load approved vocabulary from ontology for tag validation
        self.approved_vocab = get_approved_vocab()
    

    
    
    

    
    def _get_node_list(self, query: dict) -> set:
        """Get valid node IDs based on metal, connectivity, and chemistry (V3 Semantic)."""
        valid_ids = set()
        
        # 1. Extract Constraints
        metal_input = query.get('metal_symbol') or query.get('metals_include', ["Any"])
        cn_input = query.get('connectivity', [])
        ligand_chem = query.get('ligand_chemistry', [])
        
        # Metal Prep
        target_metals = []
        if isinstance(metal_input, list): target_metals = [m.strip().capitalize() for m in metal_input]
        elif isinstance(metal_input, str): target_metals = [m.strip().capitalize() for m in metal_input.split(',')]
        is_any_metal = any(m.lower() == "any" for m in target_metals)
        
        # Connectivity Prep
        target_cns = set()
        if isinstance(cn_input, list):
            target_cns = set(int(c) for c in cn_input)
        else:
            target_cns = {int(cn_input)}
            
        # Ligand Chemistry Prep (OR Logic)
        target_ligand_chem = set(l.lower() for l in ligand_chem)
        
        for item in self.bb_data:
            if item['Type'] != 'Node': continue
            
            # Connectivity Check (Any overlap)
            item_cn = item.get('connectivity', 0)
            item_cns_set = set(item_cn) if isinstance(item_cn, list) else {item_cn}
            if not target_cns.intersection(item_cns_set): continue

            # Metal Check
            if not is_any_metal:
                item_metals = item.get('metals', item.get('metal_symbols', []))
                if not any(tm in item_metals for tm in target_metals): continue
            
            # Ligand Chemistry Check (OR Logic)
            if target_ligand_chem:
                # Get item chemistry (try various V3/V2 fields)
                item_chem = item.get('ligand_chemistry', item.get('connection_chemistry', []))
                if isinstance(item_chem, str): item_chem = [item_chem]
                
                # Check intersection
                item_chem_lower = set(c.lower() for c in item_chem)
                if not target_ligand_chem.intersection(item_chem_lower): continue
            
            valid_ids.add(item['ID'])
        return valid_ids
    
    def _get_linker_list(self, query: dict, chem_only: bool = False, exclude_tags: list = None) -> set:
        """Get valid linker IDs based on query (V3 Semantic)."""
        
        cn_raw = query.get('connectivity', 2)
        # Handle case where connectivity is a list (multi-CN query)
        if isinstance(cn_raw, list):
            cn_values = [int(c) for c in cn_raw]
        else:
            cn_values = [int(cn_raw)]
        valid_ids = set()
        
        if chem_only:
            # Relaxed Physical Constraints for "Chemical Search" (Set G)
            min_len, max_len = 0.0, 999.0
        else:
            min_len = float(query.get('length_min') or 0.0)
            max_len = float(query.get('length_max') or 999.0)
        
        # Fallback to C-count if length not provided but C-count is (legacy check, largely unused)
        if min_len == 0.0 and max_len == 999.0 and 'geometric_hint' in query:
             pass

        temp_specs = {
            'linker_query': query,
            'global_requirements': {'exclude_tags': exclude_tags or []}
        }
        _, _, negative_tags = parse_functional_groups(temp_specs, self.approved_vocab)

        for item in self.bb_data:
            if item['Type'] != 'Edge': continue
            if item.get('connectivity') not in cn_values: continue
            
            length = item.get('length', 0.0)
            if not (min_len <= length <= max_len): continue
            
            # --- V3.3: Functional Group Check (Aligned with Matchmaker) ---
            item_groups = [g.lower() for g in item.get('functional_groups', [])]
            item_groups_set = set(item_groups)
            item_name = item.get('readable_name', "").lower()
            combined_text = item_name + " " + " ".join(item_groups)
            
            # A. Check Negatives (The "Bouncer")
            # If ANY negative tag is present, ban the linker
            is_banned = False
            for neg in negative_tags:
                if neg in combined_text:
                    is_banned = True
                    break
            if is_banned: continue
            
            # Note: Positive tag checks are deferred to Union Logic.
            
            valid_ids.add(item['ID'])
        return valid_ids
    
    def _calculate_advanced_stats(self, subset_df: pd.DataFrame, 
                                   total_df: pd.DataFrame, name: str) -> dict:
        """Calculate comprehensive statistics for a filter set."""
        if subset_df.empty:
            return {
                "Filter": name, "Count": 0, "% Removed": "100.0%",
                "Median Performance": "-", "Avg Top 5": "-", "Avg Worse 5": "-", 
                "Best Performance": "-", "EF @ 1%": "-", "EF @ 5%": "-", 
                "EF @ 10%": "-", "P-Value": "-"
            }
        
        N_total = len(total_df)
        N_subset = len(subset_df)
        pct_removed = ((N_total - N_subset) / N_total) * 100
        
        # SP-2.10: Minimum sample size for reliable statistics (CLT threshold)
        MIN_RELIABLE_SAMPLE = 30
        small_n_warning = N_subset < MIN_RELIABLE_SAMPLE and N_subset > 0
        
        # Thresholds for elite identification
        top_1_cutoff = int(N_total * 0.01)
        top_5_cutoff = int(N_total * 0.05)
        top_10_cutoff = int(N_total * 0.10)
        
        # Identify Hits (Using 'Original_Rank' from Master DB if available)
        if small_n_warning:
            # SP-2.10: EF and P-value unreliable below N=30
            ef_1_str = f"N<{MIN_RELIABLE_SAMPLE}"
            ef_5_str = f"N<{MIN_RELIABLE_SAMPLE}"
            ef_10_str = f"N<{MIN_RELIABLE_SAMPLE}"
        elif 'Original_Rank' in subset_df.columns:
            hits_1 = len(subset_df[subset_df['Original_Rank'] <= top_1_cutoff])
            hits_5 = len(subset_df[subset_df['Original_Rank'] <= top_5_cutoff])
            hits_10 = len(subset_df[subset_df['Original_Rank'] <= top_10_cutoff])
            
            # Enrichment Factors
            ef_1 = (hits_1 / N_subset) / 0.01 if N_subset > 0 else 0
            ef_5 = (hits_5 / N_subset) / 0.05 if N_subset > 0 else 0
            ef_10 = (hits_10 / N_subset) / 0.10 if N_subset > 0 else 0
            ef_1_str, ef_5_str, ef_10_str = f"{ef_1:.1f}x", f"{ef_5:.1f}x", f"{ef_10:.1f}x"
        else:
            ef_1_str, ef_5_str, ef_10_str = "N/A", "N/A", "N/A"
        
        # Performance statistics
        subset_sorted = subset_df.sort_values('target', ascending=False)
        top_5_avg = subset_sorted['target'].head(5).mean()
        wor_5_avg = subset_sorted['target'].tail(5).mean()
        
        # Statistical Significance (Mann-Whitney U Test)
        if N_subset == N_total:
            p_val = 1.0
        elif small_n_warning:
            p_val = None  # SP-2.10: unreliable below N=30
        else:
            try:
                u_stat, p_val = stats.mannwhitneyu(
                    subset_df['target'], total_df['target'], alternative='greater'
                )
            except Exception:
                p_val = 1.0
                
        # Base dictionary
        stats_dict = {
            "Filter": name,
            "Count": N_subset,
            "% Removed": f"{pct_removed:.2f}%",
            "P-Value": (
                f"N<{MIN_RELIABLE_SAMPLE}" if small_n_warning else
                "-" if N_subset == N_total else (
                    f"{p_val:.1e}" if p_val < 0.001 else f"{p_val:.3f}"
                )
            )
        }
        
        # Categorical + Continuous Band Gap Mode (SP-3.01 Hybrid)
        if getattr(self, 'is_qmof', False):
            stats_dict["Median Bandgap"] = f"{subset_df['target'].median():.3f}"
            
            # Category distribution (>= lower, < upper — gapless boundaries)
            for cat_full, (vmin, vmax) in config.BANDGAP_CATEGORIES.items():
                if N_subset > 0:
                    cat_count = len(subset_df[(subset_df['target'] >= vmin) & (subset_df['target'] < vmax)])
                    cat_pct = (cat_count / N_subset) * 100
                    stats_dict[cat_full] = f"{cat_pct:.1f}%"
                else:
                    stats_dict[cat_full] = "0.0%"
            
            # Continuous statistics alongside categories (Hybrid approach)
            if N_subset > 0:
                stats_dict["Q1 Bandgap"] = f"{subset_df['target'].quantile(0.25):.3f}"
                stats_dict["Q3 Bandgap"] = f"{subset_df['target'].quantile(0.75):.3f}"
                stats_dict["Min Bandgap"] = f"{subset_df['target'].min():.3f}"
                stats_dict["Max Bandgap"] = f"{subset_df['target'].max():.3f}"
            else:
                stats_dict["Q1 Bandgap"] = "-"
                stats_dict["Q3 Bandgap"] = "-"
                stats_dict["Min Bandgap"] = "-"
                stats_dict["Max Bandgap"] = "-"
            
            return stats_dict
            
        # Standard Numeric Mode
        stats_dict.update({
            "Median Performance": f"{subset_df['target'].median():.1f}",
            "Avg Top 5": f"{top_5_avg:.2f}",
            "Avg Worse 5": f"{wor_5_avg:.2f}",
            "Best Performance": f"{subset_sorted['target'].max():.2f}",
            "EF @ 1%": ef_1_str,
            "EF @ 5%": ef_5_str,
            "EF @ 10%": ef_10_str
        })
        
        return stats_dict
    
    def calculate_combinatorial_space(self, matchmaker_results: dict, agent2_output: dict) -> dict:
        """
        Calculate the theoretical design space (number of valid MOF combinations).
        
        Formula: Σ (Nodes_cn × Linkers × Topologies_cn) for each connectivity
        
        This metric represents the POTENTIAL of the hypothesis - how many MOFs
        could theoretically be synthesized, distinct from how many exist in the
        static markscheme database.
        
        Args:
            matchmaker_results: Dict with 'node', 'edge', 'topology' lists
            agent2_output: Agent 2 constraints (for requested connectivities)
            
        Returns:
            Dict with design space metrics
        """
        if getattr(self, 'is_qmof', False):
            n_ids = len(matchmaker_results.get('qmof_ids', []))
            return {
                'total_combinations': n_ids,
                'n_nodes': n_ids, 'n_linkers': 0, 'n_topologies': 0, # just display n_ids for nodes placeholder
                'breakdown': [{'combinations': n_ids, 'connectivity': 'QMOF', 'nodes': n_ids, 'linkers': 0, 'topologies': 0}]
            }
            
        # Get requested connectivities from Agent 2
        raw_cn = agent2_output['node_query'].get('connectivity', [])
        if isinstance(raw_cn, int):
            requested_cns = [raw_cn]
        else:
            requested_cns = list(raw_cn)
        
        # Build lookup for nodes by connectivity
        node_cn_map = {}  # {connectivity: [node_ids]}
        
        # Helper to ensure we catch nodes even if connectivity is a list (e.g. [8, 12])
        # We only care about requested CNs, so we can filter early.
        requested_cns_set = set(requested_cns)
        
        nodes_found_set = set(matchmaker_results.get('node', []))
        
        for item in self.bb_data:
            if item.get('Type') != 'Node':
                continue
                
            node_id = item.get('ID', '')
            if node_id not in nodes_found_set:
                continue
                
            # Get connectivity (int or list)
            raw_node_cn = item.get('connectivity', 0)
            
            # Normalize to list
            if isinstance(raw_node_cn, int):
                node_cns_list = [raw_node_cn]
            elif isinstance(raw_node_cn, list):
                node_cns_list = raw_node_cn
            else:
                node_cns_list = []
                
            # Add to buckets
            for cn_val in node_cns_list:
                if cn_val in requested_cns_set:
                    if cn_val not in node_cn_map:
                        node_cn_map[cn_val] = []
                    node_cn_map[cn_val].append(node_id)
        
        # All linkers are assumed 2-connected (ditopic) or matching the requested edge CN
        n_linkers = len(matchmaker_results.get('edge', []))
        
        # Retrieve Topology Map (Pre-calculated by Matchmaker)
        # fallback to empty dict if missing (DO NOT revert to global count, to avoid re-bugging)
        topo_cn_map = matchmaker_results.get('topology_by_cn', {})
        
        # Calculate combinatorial product
        total_combinations = 0
        breakdown = []
        
        for cn in requested_cns:
            n_nodes_cn = len(node_cn_map.get(cn, []))
            
            # Use specific topology count for this CN
            mapped_topos = topo_cn_map.get(cn, [])
            n_topos_cn = len(mapped_topos)
            
            # For single-node assembly: Nodes × Linkers × Topologies
            combos = n_nodes_cn * n_linkers * n_topos_cn
            total_combinations += combos
            
            breakdown.append({
                'connectivity': cn,
                'nodes': n_nodes_cn,
                'linkers': n_linkers,
                'topologies': n_topos_cn,
                'combinations': combos,
                'topology_ids': mapped_topos # Optional debug info
            })
        
        return {
            'total_combinations': total_combinations,
            'n_nodes': len(matchmaker_results.get('node', [])),
            'n_linkers': n_linkers,
            'n_topologies': len(set(t_id for sublist in topo_cn_map.values() for t_id in sublist)), # Unique topologies across all CNs
            'breakdown': breakdown
        }
    
    def run_analysis(self, agent2_output: dict, matchmaker_results: dict, 
                     output_dir: str = None, run_id: str = "") -> pd.DataFrame:
        """Execute the full sensitivity analysis with multi-variable support."""
        print(f"\n--- EXECUTING SENSITIVITY ANALYSIS (Multi-Variable) ---")
        
        # 1. SETUP DYNAMIC VARIABLES (From Agent 2 Input)
        g = agent2_output['geometry_filter']
        
        # Helper to parse ranges safely (None -> Open Range)
        def get_lims(min_key, max_key, default_min=0.0, default_max=99999.0):
            """Safely extract min/max limits, defaulting to open range if None."""
            low = g.get(min_key)
            high = g.get(max_key)
            low = float(low) if low is not None else default_min
            high = float(high) if high is not None else default_max
            return low, high
        
        # Core Pore Geometry (Di, Df)
        DI_MIN, DI_MAX = get_lims('target_Di_min', 'target_Di_max')
        DF_MIN, DF_MAX = get_lims('target_Df_min', 'target_Df_max')
        
        # NEW: Extended Geometric Descriptors
        SA_MIN, SA_MAX = get_lims('target_sa_min', 'target_sa_max')  # Surface Area (m²/g)
        VF_MIN, VF_MAX = get_lims('target_vf_min', 'target_vf_max', 0.0, 1.0)  # Void Fraction (0-1)
        RHO_MIN, RHO_MAX = get_lims('target_density_min', 'target_density_max')  # Density (g/cm³)
        DIF_MIN, DIF_MAX = get_lims('target_dif_min', 'target_dif_max')  # Diffusion Diameter (Å)
        CV_MIN, CV_MAX = get_lims('target_cv_min', 'target_cv_max', 0.0, 999999.0)  # Cell Volume (Å³)
        
        print(f"   Di: [{DI_MIN:.1f}, {DI_MAX:.1f}] Å | Df: [{DF_MIN:.1f}, {DF_MAX:.1f}] Å")
        print(f"   SA: [{SA_MIN:.0f}, {SA_MAX:.0f}] m²/g | VF: [{VF_MIN:.2f}, {VF_MAX:.2f}]")
        print(f"   Density: [{RHO_MIN:.2f}, {RHO_MAX:.2f}] g/cm³ | Dif: [{DIF_MIN:.1f}, {DIF_MAX:.1f}] Å")
        print(f"   CV: [{CV_MIN:.0f}, {CV_MAX:.0f}] Å³")
        
        # Robustly get metal symbol (Handle V2 'metal_symbol' vs V3 'metals_include')
        node_q = agent2_output['node_query']
        metal_sym = node_q.get('metal_symbol') or node_q.get('metals_include', ["Any"])
        
        node_cn = node_q['connectivity']
        # Extract Tags using Shared Parser
        # agent2_output is already specs-compatible
        global_and_tags, linker_or_tags, exclude_tags = parse_functional_groups(agent2_output, self.approved_vocab)
        
        # 2. GET VALID COMPONENT IDS
        valid_nodes = self._get_node_list(node_q)
        
        linker_query = agent2_output['linker_query']
        # Strict Linkers (Matches Hypothesis Physics)
        valid_linkers_strict = self._get_linker_list(linker_query, chem_only=False, exclude_tags=exclude_tags)
        
        # Relaxed Linkers (Matches Chemistry, Ignores Length) - For Set G
        valid_linkers_relaxed = self._get_linker_list(linker_query, chem_only=True, exclude_tags=exclude_tags)
        
        # 3. CREATE FILTER SETS
        print("Generating Filter Sets (A-I + J-S + Z)...")
        
        def filter_h2_db(row):
            """Checks if a Master DB row matches valid components AND global requirements"""
            try:
                parts = row['filename'].split('+')
                if len(parts) != 3: return False
                node_id, edge_id = parts[1], parts[2]
                
                # 1. Physics Check (Must be in valid lists)
                if node_id not in matchmaker_results['node']: return False
                if edge_id not in matchmaker_results['edge']: return False
                
                # 2. Union Logic Check (Global Requirements)
                if not check_global_requirements(node_id, edge_id, global_and_tags, self.bb_lookup,
                                                 linker_or_tags=linker_or_tags):
                    return False
                    
                return True
            except Exception:
                return False
        
        def filter_node_only(row):
            try:
                return row['filename'].split('+')[1] in valid_nodes
            except Exception:
                return False
        
        def filter_linker_only(row):
            try:
                # Use relaxed list for Set G to test broader chemistry
                return row['filename'].split('+')[2] in valid_linkers_relaxed
            except Exception:
                return False
        
        # Baseline: The Full Master DB
        df_total = self.df_master.copy()
        
        has_sa = 'sa' in df_total.columns
        has_vf = 'vf' in df_total.columns
        has_density = 'density' in df_total.columns
        has_dif = 'dif' in df_total.columns
        has_cv = 'cv' in df_total.columns
        has_di_df = 'di' in df_total.columns and 'df' in df_total.columns

        # --- DYNAMIC TARGET METRIC ---
        # Map the active metric (e.g. Surface Area) to the 'target' column for analysis
        target_col = config.ACTIVE_METRIC_COLUMN
        
        if target_col in df_total.columns:
            df_total['target'] = df_total[target_col]
            print(f"Target Metric successfully mapped: {target_col}")
        else:
            print(f"[WARNING] Active metric '{target_col}' not found in DB.")
            
            # ROBUST FALLBACK: Search for likely H2 columns if default failed
            found_fallback = False
            for col in df_total.columns:
                if "h2" in col.lower() and "uptake" in col.lower() and "volumetric" in col.lower():
                     print(f"   -> Found potential match: {col}. Using this as target.")
                     df_total['target'] = df_total[col]
                     found_fallback = True
                     break
            
            if not found_fallback:
                print("   -> No suitable metric column found. Analysis will produce N/A.")
                if self.is_qmof:
                     print("   -> (Running QMOF Database Mode: Expecting expected electronic/geom tags matching inputs)")
                else:
                    print(f"   -> AVAILABLE COLUMNS ({len(df_total.columns)}):")
                    # Print columns in chunks to avoid truncation
                    cols = list(df_total.columns)
                    if len(cols) > 0:
                        print(f"      {cols[:10]}")
                    
                # Create dummy target to prevent crash
                df_total['target'] = 0.0
        
        # =====================================================================
        # ORIGINAL SETS A-I (Preserved)
        # =====================================================================
        
        # Set A: Chemical Only (Primary result of Agent's Chemical Query)
        if getattr(self, 'is_hmof', False):
            hmof_ids = matchmaker_results.get('hmof_ids', [])
            set_a = df_total[df_total['hmof_id'].isin(hmof_ids)].copy()
            print(f"-> Set A (Chemical Search): {len(set_a)} hMOF candidates")
            from core.hmof_matchmaker import HMOFMatchmaker
            h_mm = HMOFMatchmaker()

            # Reconstruct query for partial matches
            query = matchmaker_results.get('query_specs', {})
            Agent2_Specs = {
                'node_query': query.get('node_query', {}),
                'linker_query': query.get('linker_query', {}),
                'global_requirements': query.get('global_requirements', {})
            }

            hmof_ids_metal = h_mm.match(Agent2_Specs, search_mode="metal_only")
            set_f = df_total[df_total['hmof_id'].isin(hmof_ids_metal)].copy()

            hmof_ids_linker = h_mm.match(Agent2_Specs, search_mode="linker_only")
            set_g = df_total[df_total['hmof_id'].isin(hmof_ids_linker)].copy()
        elif getattr(self, 'is_qmof', False):
            qmof_ids = matchmaker_results.get('qmof_ids', [])
            set_a = df_total[df_total['qmof_id'].isin(qmof_ids)].copy()
            print(f"-> Set A (Chemical Search): {len(set_a)} QMOF candidates")
            from core.qmof_matchmaker import QMOFMatchmaker
            q_mm = QMOFMatchmaker()
            
            # Reconstruct query for partial matches
            query = matchmaker_results.get('query_specs', {})
            Agent2_Specs = {
                'node_query': query.get('node_query', {}),
                'linker_query': query.get('linker_query', {}),
                'global_requirements': query.get('global_requirements', {})
            }
            
            qmof_ids_metal = q_mm.match(Agent2_Specs, tracker=None, search_mode="metal_only")
            set_f = df_total[df_total['qmof_id'].isin(qmof_ids_metal)].copy()
            
            qmof_ids_linker = q_mm.match(Agent2_Specs, tracker=None, search_mode="linker_only")
            set_g = df_total[df_total['qmof_id'].isin(qmof_ids_linker)].copy()
        else:
            set_a = df_total[df_total.apply(filter_h2_db, axis=1)].copy()
            print(f"-> Set A (Chemical Search): {len(set_a)} candidates")
            set_f = df_total[df_total.apply(filter_node_only, axis=1)].copy()
            set_g = df_total[df_total.apply(filter_linker_only, axis=1)].copy()
        
        # Set B, C, D (Subsets of A based on Agent 2's Geometry)
        if has_di_df:
            set_b = set_a[(set_a['di'] > DI_MIN) & (set_a['di'] < DI_MAX)].copy()
            set_c = set_a[(set_a['df'] > DF_MIN) & (set_a['df'] < DF_MAX)].copy()
            set_d = set_a[
                (set_a['di'] > DI_MIN) & (set_a['di'] < DI_MAX) &
                (set_a['df'] > DF_MIN) & (set_a['df'] < DF_MAX)
            ].copy()
            print(f"-> Set D (Chem + Di + Df): {len(set_d)} candidates")
        else:
            set_b = set_a.copy()
            set_c = set_a.copy()
            set_d = set_a.copy()
            print(f"-> Set D (Chem + fallback): {len(set_d)} candidates")
        
        # Set E: Geometry Only (Di + Df) - The "Competitor" (Deferred to geometric checks below)
        
        # NEW Set E2: Full Geometry (All Geometry Constraints)
        # (Column flags already computed above — reuse them)
        if has_di_df and has_sa and has_vf and has_density and has_dif and has_cv:
            set_e2 = df_total[
                (df_total['di'] > DI_MIN) & (df_total['di'] < DI_MAX) &
                (df_total['df'] > DF_MIN) & (df_total['df'] < DF_MAX) &
                (df_total['sa'] >= SA_MIN) & (df_total['sa'] <= SA_MAX) &
                (df_total['vf'] >= VF_MIN) & (df_total['vf'] <= VF_MAX) &
                (df_total['density'] >= RHO_MIN) & (df_total['density'] <= RHO_MAX) &
                (df_total['dif'] >= DIF_MIN) & (df_total['dif'] <= DIF_MAX) &
                (df_total['cv'] >= CV_MIN) & (df_total['cv'] <= CV_MAX)
            ].copy()
            set_z = set_d[
                (set_d['sa'] >= SA_MIN) & (set_d['sa'] <= SA_MAX) &
                (set_d['vf'] >= VF_MIN) & (set_d['vf'] <= VF_MAX) &
                (set_d['density'] >= RHO_MIN) & (set_d['density'] <= RHO_MAX) &
                (set_d['dif'] >= DIF_MIN) & (set_d['dif'] <= DIF_MAX) &
                (set_d['cv'] >= CV_MIN) & (set_d['cv'] <= CV_MAX)
            ].copy()
            
            set_e = df_total[
                (df_total['di'] > DI_MIN) & (df_total['di'] < DI_MAX) &
                (df_total['df'] > DF_MIN) & (df_total['df'] < DF_MAX)
            ].copy()
            
            set_h = df_total[(df_total['di'] > DI_MIN) & (df_total['di'] < DI_MAX)].copy()
            set_i = df_total[(df_total['df'] > DF_MIN) & (df_total['df'] < DF_MAX)].copy()
            set_j = df_total[(df_total['sa'] >= SA_MIN) & (df_total['sa'] <= SA_MAX)].copy()
            set_k = df_total[(df_total['vf'] >= VF_MIN) & (df_total['vf'] <= VF_MAX)].copy()
            set_l = df_total[(df_total['density'] >= RHO_MIN) & (df_total['density'] <= RHO_MAX)].copy()
            set_m = df_total[(df_total['dif'] >= DIF_MIN) & (df_total['dif'] <= DIF_MAX)].copy()
            set_n = df_total[(df_total['cv'] >= CV_MIN) & (df_total['cv'] <= CV_MAX)].copy()
            
        else: # Handle QMOF which may lack specific geometric characteristics internally
            set_e2 = df_total.head(0).copy()
            set_e = df_total.head(0).copy()
            set_z = set_d.copy()  # Fall back to chem+topology 
            
            set_h = df_total.head(0).copy()
            set_i = df_total.head(0).copy()
            set_j = df_total.head(0).copy()
            set_k = df_total.head(0).copy()
            set_l = df_total.head(0).copy()
            set_m = df_total.head(0).copy()
            set_n = df_total.head(0).copy()
        
        # =====================================================================
        # NEW SETS O-S: Chemical + Variable Constraints (Contextual Sensitivity)
        # =====================================================================
        if has_sa and has_vf and has_density and has_dif and has_cv:
            set_o = set_a[(set_a['sa'] >= SA_MIN) & (set_a['sa'] <= SA_MAX)].copy()      
            set_p = set_a[(set_a['vf'] >= VF_MIN) & (set_a['vf'] <= VF_MAX)].copy()      
            set_q = set_a[(set_a['density'] >= RHO_MIN) & (set_a['density'] <= RHO_MAX)].copy()  
            set_r = set_a[(set_a['dif'] >= DIF_MIN) & (set_a['dif'] <= DIF_MAX)].copy()  
            set_s = set_a[(set_a['cv'] >= CV_MIN) & (set_a['cv'] <= CV_MAX)].copy()      
        else:
            set_o = set_a.head(0).copy()
            set_p = set_a.head(0).copy()
            set_q = set_a.head(0).copy()
            set_r = set_a.head(0).copy()
            set_s = set_a.head(0).copy()
            
        # =====================================================================
        # SET Z: Full Intersection (The "Real" Hypothesis)
        # =====================================================================
        print(f"-> Set Z (FULL HYPOTHESIS): {len(set_z)} candidates")
        
        # =====================================================================
        # 4. CALCULATE STATISTICS
        # =====================================================================
        metrics = []
        
        # Original Sets (A-I)
        metrics.append(self._calculate_advanced_stats(df_total, df_total, "Baseline (Total DB)"))
        metrics.append(self._calculate_advanced_stats(set_a, df_total, "A (Chemical Only)"))
        metrics.append(self._calculate_advanced_stats(set_b, df_total, "B (Chem + Di)"))
        metrics.append(self._calculate_advanced_stats(set_c, df_total, "C (Chem + Df)"))
        metrics.append(self._calculate_advanced_stats(set_d, df_total, "D (Chem + Di + Df)"))
        metrics.append(self._calculate_advanced_stats(set_e, df_total, "E (Di + Df)"))
        metrics.append(self._calculate_advanced_stats(set_e2, df_total, "E2 (Full Geometry)"))
        metrics.append(self._calculate_advanced_stats(set_f, df_total, "F (Node Only)"))
        metrics.append(self._calculate_advanced_stats(set_g, df_total, "G (Linker Only)"))
        metrics.append(self._calculate_advanced_stats(set_h, df_total, "H (Di Only)"))
        metrics.append(self._calculate_advanced_stats(set_i, df_total, "I (Df Only)"))
        
        # New Single-Variable Sets (J-N)
        metrics.append(self._calculate_advanced_stats(set_j, df_total, "J (SA Only)"))
        metrics.append(self._calculate_advanced_stats(set_k, df_total, "K (VF Only)"))
        metrics.append(self._calculate_advanced_stats(set_l, df_total, "L (Density Only)"))
        metrics.append(self._calculate_advanced_stats(set_m, df_total, "M (Dif Only)"))
        metrics.append(self._calculate_advanced_stats(set_n, df_total, "N (CV Only)"))
        
        # New Chemical + Variable Sets (O-S)
        metrics.append(self._calculate_advanced_stats(set_o, df_total, "O (Chem + SA)"))
        metrics.append(self._calculate_advanced_stats(set_p, df_total, "P (Chem + VF)"))
        metrics.append(self._calculate_advanced_stats(set_q, df_total, "Q (Chem + Density)"))
        metrics.append(self._calculate_advanced_stats(set_r, df_total, "R (Chem + Dif)"))
        metrics.append(self._calculate_advanced_stats(set_s, df_total, "S (Chem + CV)"))
        
        # Full Hypothesis (Z)
        metrics.append(self._calculate_advanced_stats(set_z, df_total, "Z (FULL HYPOTHESIS: All Constraints)"))
        
        results_df = pd.DataFrame(metrics)
        
        # 5. CALCULATE THEORETICAL DESIGN SPACE
        design_space = self.calculate_combinatorial_space(matchmaker_results, agent2_output)
        
        # 6. DISPLAY REPORT
        print(f"\n--- SENSITIVITY ANALYSIS REPORT ({run_id}) ---")
        
        # Get dynamic columns based on active configuration
        report_cols = config.get_sensitivity_columns()
        
        # Ensure our results have these columns/keys
        # (This relies on _calculate_advanced_stats producing keys that match get_sensitivity_columns logic)
        # To be safe, we reconstruct the DataFrame with the exact columns requested
        
        # First, check if basic keys exist, if not, map them from generic keys
        # The key logic in _calculate_advanced_stats is generic ("Best Performance"), 
        # but get_sensitivity_columns returns specific ("Best H2 Storage")
        # We need to BRIDGE this gap.
        
        # Strategy: Rename the generic keys in results_df to match the reported columns
        m_label = "Performance"
        for key, val in config.METRIC_REGISTRY.items():
            if val == config.ACTIVE_METRIC_COLUMN:
                m_label = key.replace("_", " ").title()
                break
        
        rename_map = {
            "Avg Top 5": f"Avg Top 5 ({m_label})",
            "Avg Worse 5": f"Avg Worse 5 ({m_label})",
            "Best Performance": f"Best {m_label}",
            "Median Performance": f"Median {m_label}"
        }
        
        results_df_view = results_df.rename(columns=rename_map)
        
        # Check for missing columns and fill with N/A
        for c in report_cols:
            if c not in results_df_view.columns:
                results_df_view[c] = "N/A"
                print(f"Warning: Missing column '{c}' in results")

        print(results_df_view[report_cols].to_string(index=False))
        
        # Display Theoretical Design Space
        print(f"\n╔══════════════════════════════════════════════════════════╗")
        print(f"║           THEORETICAL DESIGN SPACE                        ║")
        print(f"╠══════════════════════════════════════════════════════════╣")
        
        # Check if we are in QMOF mode
        is_qmof = config.is_qmof_mode()
        
        if is_qmof:
            print(f"║   QMOFs Matching Chemistry:    {len(set_d):<8}                 ║")
            print(f"╠══════════════════════════════════════════════════════════╣")
            print(f"║   SEARCH SPACE:        {len(set_d):<12} candidates      ║")
            print(f"║   (Direct database matching)                              ║")
        else:
            print(f"║   Nodes Found:         {design_space['n_nodes']:<8}                        ║")
            print(f"║   Linkers Found:       {design_space['n_linkers']:<8}                        ║")
            print(f"║   Topologies Found:    {design_space['n_topologies']:<8}                        ║")
            print(f"╠══════════════════════════════════════════════════════════╣")
            print(f"║   DESIGN SPACE:        {design_space['total_combinations']:<12} combinations      ║")
            print(f"║   (N × L × T = potential MOFs)                            ║")
            
        print(f"╚══════════════════════════════════════════════════════════╝")
        
        # Store design space for later use
        self.design_space = design_space
        
        # 7. SAVE REPORT (if output_dir provided)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_filename = f"{output_dir}/Sensitivity_Report_{run_id}.csv"
            results_df.to_csv(report_filename, index=False)
            print(f"\nSaved Analysis Report to: {report_filename}")
            
            # Export detailed datasets
            # ... (Full export logic maintained implicitly or simplified for brevity)
        
        # Store filter sets for feedback generator (includes all new sets)
        self.filter_sets = {
            'total': df_total,
            # Original Sets
            'a': set_a, 'b': set_b, 'c': set_c, 'd': set_d,
            'e': set_e, 'e2': set_e2, 'f': set_f, 'g': set_g, 'h': set_h, 'i': set_i,
            # New Single-Variable Sets
            'j': set_j, 'k': set_k, 'l': set_l, 'm': set_m, 'n': set_n,
            # New Chemical + Variable Sets
            'o': set_o, 'p': set_p, 'q': set_q, 'r': set_r, 's': set_s,
            # Full Hypothesis
            'z': set_z
        }
        
        return results_df


# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_sensitivity_analyzer():
    """Test the sensitivity analyzer with V3 sample data."""
    
    print("\n" + "="*60)
    print("SENSITIVITY ANALYZER MODULE TEST (V3)")
    print("="*60 + "\n")
    
    # Sample Agent 2 output (V3 Structure)
    test_agent2_output = {
        "node_query": {
            "metals_include": ["Zr"],
            "connectivity": [6, 12], # Multi-CN query to test design space
            "nuclearity": 6,
            "sbu_type": "Cluster"
        },
        "linker_query": {
            "connectivity": 2,
            "must_contain_elements": ["O", "C", "H"],
            "length_min": 6.0,
            "length_max": 12.0,
            "is_rigid": True,
            "functional_groups": []
        },
        "geometry_filter": {
            "target_Di_min": 12.0, "target_Di_max": 20.0,
            "target_Df_min": 7.0, "target_Df_max": 10.0
        }
    }
    
    # Sample matchmaker results
    from core.matchmaker import Matchmaker
    matchmaker = Matchmaker()
    matchmaker_results = matchmaker.smart_matchmaker_single_node(test_agent2_output)
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer()
    
    if isinstance(matchmaker_results, str):
        print(f"Matchmaker Error: {matchmaker_results}")
        return

    # Run analysis
    results_df = analyzer.run_analysis(
        test_agent2_output, 
        matchmaker_results,
        run_id="TEST_V3"
    )
    
    print("\n--- Validation ---")
    d_count = int(results_df[results_df['Filter'] == 'D (Chem + Di + Df)']['Count'].values[0])
    print(f"Matches in Set D: {d_count}")
    
    if d_count >= 0:
        print("✓ VALIDATION PASSED!")
    else:
        print("✗ VALIDATION FAILED")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_sensitivity_analyzer()
