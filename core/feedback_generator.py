# =============================================================================
# LLM2POR Autonomous System - Feedback Generator
# =============================================================================
# Generates 6 scientifically-designed feedback types for Agent 1
# =============================================================================

import pandas as pd
import os
import sys
import random
import json

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEEDBACK_SAMPLE_SIZE, FEEDBACK_SAMPLE_SIZE_LARGE, STOCHASTIC_SAMPLING
import config
from core.name_resolver import get_name_resolver


class FeedbackGenerator:
    """
    Generates feedback prompts for Agent 1 based on filtered MOF candidates.
    
    6 Feedback Types (ranked by information value):
    1. 3-Beam Diagnostic     - Orthogonal diagnosis (D + A + E_pure)
    2. Universe Baseline     - Global calibration from full database
    3. Geometric Optimizer   - A/B test: Random Geo vs Constrained Geo
    4. Chemical Pivot        - A/B test: Your Metal vs Any Metal
    5. Best vs Worst         - Pattern discovery (Top 15 vs Bottom 15)
    6. Hypothesis Validation - Full hypothesis test (Set D only)
    
    Features:
    - Name Tags: Uses shared NameResolver for consistent ID → name translation.
    - Diagnostic Footer: Explains zero results with actionable hints.
    """
    
    def __init__(self):
        """Initialize feedback generator with shared name resolver."""
        self._resolver = get_name_resolver()
        
        # Expose bb_map for backward compatibility (read-only)
        
        # Random seed management
        self._iteration_count = 0
        self._last_seed = None
        self._last_sampled_ids = []
    
    def _get_random_state(self):
        """Get random state based on sampling mode. Stores seed for reproducibility."""
        if STOCHASTIC_SAMPLING:
            self._iteration_count += 1
            seed = random.randint(1, 100000)
            self._last_seed = seed
            print(f"   [SEED] Stochastic sampling seed: {seed}")
            return seed
        else:
            self._last_seed = 42
            return 42
    
    def get_last_sampling_info(self) -> dict:
        """Return the last seed and sampled structure IDs for reproducibility (SP-2.09)."""
        return {
            'seed': self._last_seed,
            'sampled_ids': list(self._last_sampled_ids)
        }
    
    def _translate_mof(self, filename_str: str) -> str:
        """Translates filename codes (ukd+N164+E70) to human readable names."""
        return self._resolver.translate_mof_filename(filename_str)

    @staticmethod
    def _describe_hmof_row(row) -> str:
        """Build a human-readable description from hMOF index fields."""
        name = row.get('readable_name', '')
        if name:
            return name
        # Fallback: construct from metals + topology + functional groups
        metals = row.get('metals', [])
        topo = row.get('topology', '')
        fg = row.get('functional_groups', [])
        # Pick the most chemically meaningful FGs (skip generic tags)
        skip = {'aromatic', 'aryl', 'ring', 'heterocycle', 'nitrogen', 'oxygen',
                'sulfur', 'halogen', 'carbonyl', 'carbon_framework', 'amine',
                'aliphatic_ring', 'azole', 'azolate', 'phosphorus'}
        specific_fg = [g for g in fg if g not in skip][:3]
        parts = []
        if metals:
            parts.append('/'.join(metals))
        if topo:
            parts.append(topo)
        if specific_fg:
            parts.append('+'.join(specific_fg))
        return ' '.join(parts) if parts else row.get('hmof_id', 'unknown')

    @staticmethod
    def _describe_qmof_row(row) -> str:
        """Build a human-readable description from QMOF index/CSV fields."""
        name = row.get('readable_name', '')
        if not name or str(name) == 'nan':
            # Fallback: formula + topology from qmof.csv columns
            parts = []
            formula = row.get('info.formula', '')
            if formula and str(formula) != 'nan':
                parts.append(str(formula))
            topo = row.get('info.mofid.topology', '')
            if topo and str(topo) != 'nan':
                parts.append(f'[{topo}]')
            nodes = row.get('info.mofid.smiles_nodes', '')
            if nodes and str(nodes) != 'nan':
                parts.append(f'Node:{nodes}')
            return ' '.join(parts) if parts else row.get('qmof_id', 'unknown')
        # Append node geometry if not already in the name
        geom = row.get('geometry', '')
        if geom and str(geom) != 'nan' and str(geom) != 'Unknown':
            geom_keywords = ['octahedral', 'tetrahedral', 'square planar',
                             'trigonal', 'cubic', 'dodecahedral', 'linear']
            name_lower = name.lower()
            if not any(gk in name_lower for gk in geom_keywords):
                name = f'{name} ({geom})'
        return name
    
    def _generate_table(self, df: pd.DataFrame, n: int, title: str = "Samples", metric_name: str = "Target Metric") -> str:
        """Creates the text table for the LLM prompt. QMOF-aware."""
        if df.empty:
            return f"--- {title} ---\nNo Data Available."
        
        random_state = self._get_random_state()
        samp = df.sample(min(n, len(df)), random_state=random_state).sort_values(
            'target', ascending=False
        )
        
        samp = samp.copy()
        # Store sampled IDs for reproducibility (SP-2.09)
        self._last_sampled_ids = samp['filename'].tolist() if 'filename' in samp.columns else []
        
        # Detect mode: columns vary by database
        is_qmof = config.is_qmof_mode()
        is_hmof = config.is_hmof_mode()
        
        # Apply mode-aware Structure Descriptions
        if is_hmof:
            samp['Structure Description'] = samp.apply(self._describe_hmof_row, axis=1)
        elif is_qmof:
            samp['Structure Description'] = samp.apply(self._describe_qmof_row, axis=1)
        else:
            samp['Structure Description'] = samp['filename'].apply(self._translate_mof)
        
        if is_qmof:
            # QMOF columns: readable structure description + bandgap + pore geometry
            available = ['Structure Description', 'target']
            display_names = ['Structure', metric_name]
            for col, name in [
                ('di', 'LCD (A)'), ('df', 'PLD (A)'), ('density', 'Density'),
            ]:
                if col in samp.columns:
                    available.append(col)
                    display_names.append(name)
            view = samp[available].copy()
            view.columns = display_names
        elif is_hmof:
            # hMOF columns: structural properties available (di/df/sa/vf/density but no dif/cv)
            available = ['Structure Description', 'target']
            display_names = ['Structure', metric_name]
            for col, name in [
                ('di', 'LCD (A)'), ('df', 'PLD (A)'),
                ('sa', 'SA (m2/g)'), ('vf', 'VF'), ('density', 'Density'),
            ]:
                if col in samp.columns:
                    available.append(col)
                    display_names.append(name)
            view = samp[available].copy()
            view.columns = display_names
        else:
            # PORMAKE columns (standard H2 mode)
            cols = ['Structure Description', 'target', 'di', 'df', 'sa', 'vf', 'density', 'dif', 'cv']
            view = samp[cols].copy()
            view.columns = ['Structure', metric_name, 'Di (A)', 'Df (A)', 'SA (m2/g)', 'VF', 'Density (g/cm3)', 'Dif (A)', 'CV (A3)']
        
        return f"--- {title} (N={len(samp)}) ---\n" + view.to_string(index=False)
    
    def _generate_diagnostic_footer(self, filter_sets: dict) -> str:
        """
        Generates a diagnostic footer if results are zero or low.
        Explains WHY the search failed based on filter counts.
        Mode-aware: adjusts messaging for QMOF vs PORMAKE.
        """
        is_qmof = config.is_qmof_mode()
        is_hmof = config.is_hmof_mode()
        
        set_a = filter_sets.get('a', pd.DataFrame()) # Chemical
        set_z = filter_sets.get('z', pd.DataFrame()) # Full Hypothesis
        set_e2 = filter_sets.get('e2', pd.DataFrame()) # Full Geometry Only
        
        # If Hypothesis (Z) has results, no need for major diagnostics
        if len(set_z) > 0:
            return ""
            
        footer = "\n\n*** DIAGNOSTIC FOOTER (NO MATCHES FOUND) ***\n"
        footer += "Your specific full hypothesis found ZERO matches.\n"
        footer += "Here is the breakdown of where it failed:\n"
        
        # 1. Check Chemistry (Set A)
        count_a = len(set_a)
        if count_a == 0:
            if is_qmof:
                footer += "[CRITICAL FAILURE] CHEMISTRY: No QMOFs match your Metal + Functional Group constraints.\n"
                footer += "  SUGGESTION: Broaden metal list or relax functional group requirements.\n"
            else:
                footer += "[CRITICAL FAILURE] CHEMISTRY: No MOFs exist with your specific Metal + Linker combination.\n"
                footer += "  Possible causes:\n"
                footer += "  -> Ligand Chemistry mismatch: Your Node's binding elements may not match Linker's functional groups.\n"
                footer += "  -> Functional Groups too strict: All specified tags must be present (AND logic).\n"
                footer += "  SUGGESTION: Relax linker constraints or broaden metals list.\n"
        else:
            footer += f"[PASS] CHEMISTRY: Found {count_a} candidates matching your chemical constraints.\n"
        
        # 2. Check Geometry (Set E2) — only relevant for PORMAKE
        if not is_qmof:
            count_e2 = len(set_e2)
            if count_e2 == 0:
                footer += "[CRITICAL FAILURE] GEOMETRY: No MOFs exist with your full set of physical property constraints.\n"
                footer += "  SUGGESTION: Your physical property constraints (Di, Df, SA, VF, etc.) may be too narrow.\n"
            else:
                footer += f"[PASS] GEOMETRY: Found {count_e2} MOFs matching your geometry constraints.\n"
                
            # 3. Intersection Failure
            if count_a > 0 and count_e2 > 0 and len(set_z) == 0:
                footer += "[INTERSECTION FAILURE] Chemistry and Geometry both pass individually but NEVER TOGETHER.\n"
                footer += "  SUGGESTION: Consider a different linker or changing your pore size targets.\n"

        return footer

    def generate_feedback(self, feedback_type: int, filter_sets: dict, metric_name: str = "H2 Uptake") -> str:
        """
        Generate feedback prompt based on selected type.
        """
        # Extract all filter sets
        set_total = filter_sets.get('total', pd.DataFrame())
        set_a = filter_sets.get('a', pd.DataFrame())
        set_d = filter_sets.get('d', pd.DataFrame())
        set_e = filter_sets.get('e', pd.DataFrame())
        set_e2 = filter_sets.get('e2', pd.DataFrame())
        set_f = filter_sets.get('f', pd.DataFrame())
        set_g = filter_sets.get('g', pd.DataFrame())
        set_z = filter_sets.get('z', pd.DataFrame())
        
        # Create "Pure" Geometric Control using E2 (Full Geometry)
        if not set_e2.empty and not set_a.empty:
            set_e_pure = set_e2[~set_e2['filename'].isin(set_a['filename'])].copy()
        elif not set_e2.empty:
            set_e_pure = set_e2.copy()
        else:
            set_e_pure = pd.DataFrame()
            
        is_qmof = config.is_qmof_mode()
        
        content = ""
        if feedback_type == 1:
            if is_qmof:
                content = self._generate_qmof_four_beam(set_z, set_f, set_g, set_total, metric_name)
            else:
                content = self._generate_three_beam(set_z, set_a, set_e_pure, metric_name)
        elif feedback_type == 2:
            content = self._generate_universe_baseline(set_total, metric_name)
        elif feedback_type == 3:
            content = self._generate_geometric_optimizer(set_a, set_d, metric_name)
        elif feedback_type == 4:
            content = self._generate_chemical_pivot(set_f, set_e_pure, metric_name)
        elif feedback_type == 5:
            content = self._generate_best_vs_worst(set_a, metric_name)
        elif feedback_type == 6:
            content = self._generate_hypothesis_validation(set_z, metric_name)
        elif feedback_type == 7:
            content = self._generate_virtual_synthesis_report(filter_sets, metric_name)
        else:
            content = "Error: Invalid feedback type selected."
            
        # Append Diagnostic Footer conditionally
        if feedback_type != 7:
            content += self._generate_diagnostic_footer(filter_sets)
        
        return content
    
    # =========================================================================
    # FEEDBACK TYPE 1: 3-BEAM DIAGNOSTIC (Rank 1 - Most Informative)
    # =========================================================================
    def _generate_three_beam(self, set_z: pd.DataFrame, set_a: pd.DataFrame, 
                              set_e_pure: pd.DataFrame, metric_name: str) -> str:
        """
        3-Beam Diagnostic: Orthogonal analysis to diagnose strategy.
        BEAM 1: Full Hypothesis (Z) - Is the complete hypothesis valid?
        BEAM 2: Chemical Control (A) - Does geometry add value?
        BEAM 3: Geometric Control (E_pure) - Is chemistry helping or hurting?
        """
        return f"""
*** EXPERIMENT: 3-BEAM DIAGNOSTIC ***
We ran 3 parallel search beams to diagnose your strategy.

BEAM 1: YOUR HYPOTHESIS (Your Metal + Your Linker + All Your Geometry Constraints)
{self._generate_table(set_z, FEEDBACK_SAMPLE_SIZE, "Beam 1", metric_name)}

BEAM 2: CHEMICAL CONTROL (Your Metal + Your Linker, Random Geometry)
{self._generate_table(set_a, FEEDBACK_SAMPLE_SIZE, "Beam 2", metric_name)} -> (Tests if Geometry adds value)

BEAM 3: GEOMETRIC CONTROL (Any Metal, All Your Geometry Constraints)
{self._generate_table(set_e_pure, FEEDBACK_SAMPLE_SIZE, "Beam 3", metric_name)} -> (Tests if Chemistry is limiting)
"""

    def _generate_qmof_four_beam(self, set_z: pd.DataFrame, set_f: pd.DataFrame, 
                                  set_g: pd.DataFrame, set_total: pd.DataFrame, metric_name: str) -> str:
        """
        Custom QMOF 4-Beam Diagnostic: Isolates the electronic contributions of Metals versus Linkers.
        BEAM 1: Full Hypothesis (Z)
        BEAM 2: Metal Control (F)
        BEAM 3: Linker Control (G)
        BEAM 4: Universe Baseline (total)
        """
        return f"""
*** EXPERIMENT: QMOF 4-BEAM ELECTRONIC DIAGNOSTIC ***
We ran 4 parallel search beams to isolate the electronic contributions of your components.

BEAM 1: FULL HYPOTHESIS (Your Metal(s) + Your Linker Functional Groups)
{self._generate_table(set_z, FEEDBACK_SAMPLE_SIZE, "Beam 1", metric_name)}

BEAM 2: METAL CONTROL (Your Metal(s), ANY Linker / No Func Group Constraints)
{self._generate_table(set_f, FEEDBACK_SAMPLE_SIZE, "Beam 2", metric_name)} -> (Tests the baseline capability of your chosen metal)

BEAM 3: LINKER CONTROL (ANY Metal, Your Functional Groups)
{self._generate_table(set_g, FEEDBACK_SAMPLE_SIZE, "Beam 3", metric_name)} -> (Tests the baseline electronic tunability of your linker substituents)

BEAM 4: GLOBAL BASELINE (ANY Metal, ANY Linker)
{self._generate_table(set_total, FEEDBACK_SAMPLE_SIZE, "Beam 4", metric_name)} -> (Global distribution of the entire QMOF database)
"""
    
    # =========================================================================
    # FEEDBACK TYPE 2: UNIVERSE BASELINE (Rank 2 - Critical for Set A=0)
    # =========================================================================
    def _generate_universe_baseline(self, set_total: pd.DataFrame, metric_name: str) -> str:
        """
        Universe Baseline: Sample from entire database for global calibration.
        Critical when Set A = 0 (agent's chemistry yields no matches).
        """
        is_qmof = config.is_qmof_mode()
        db_size = f"{len(set_total):,}" if not set_total.empty else "N/A"
        
        if is_qmof:
            context_text = (
                "Use this to understand:\n"
                "- What metals and functional groups exist in successful MOFs\n"
                "- What band gap ranges are achievable\n"
                "- Whether your hypothesis aligns with the QMOF database distribution"
            )
        else:
            context_text = (
                "Use this to understand:\n"
                "- What metals and linkers exist in successful MOFs\n"
                "- What geometry ranges (Di, Df) correlate with high performance\n"
                "- Whether your hypothesis aligns with the database distribution"
            )
        
        return f"""
*** EXPERIMENT: UNIVERSE BASELINE ***
We sampled the entire database ({db_size} MOFs) without constraints.
This establishes a global baseline for performance.

{self._generate_table(set_total, FEEDBACK_SAMPLE_SIZE_LARGE, "Global Random Samples", metric_name)}

{context_text}
"""
    
    # =========================================================================
    # FEEDBACK TYPE 3: GEOMETRIC OPTIMIZER (Rank 3 - A/B Test Geometry)
    # =========================================================================
    def _generate_geometric_optimizer(self, set_a: pd.DataFrame, set_d: pd.DataFrame, metric_name: str) -> str:
        """
        Geometric Optimizer: A/B test comparing random vs constrained geometry.
        Group A: Your chemistry, random geometry
        Group B: Your chemistry, your geometry constraints
        """
        return f"""
*** EXPERIMENT: A/B TEST (GEOMETRY) ***
We compared the same chemistry (Your Metal + Your Linker) with different geometries.

GROUP A: RANDOM GEOMETRY (Your Metal + Your Linker, No Di/Df constraints)
{self._generate_table(set_a, 15, "Group A (Random Geo)", metric_name)}

GROUP B: YOUR GEOMETRY (Your Metal + Your Linker + Your Di/Df constraints)
{self._generate_table(set_d, 15, "Group B (Your Geo)", metric_name)}

Analysis:
- If Group B >> Group A: Your geometry hypothesis is valuable
- If Group B ≈ Group A: Geometry constraints aren't the key differentiator
- If Group B << Group A: Geometry constraints may be too tight or wrong
"""
    
    # =========================================================================
    # FEEDBACK TYPE 4: CHEMICAL PIVOT (Rank 4 - A/B Test Chemistry)
    # =========================================================================
    def _generate_chemical_pivot(self, set_f: pd.DataFrame, set_e_pure: pd.DataFrame, metric_name: str) -> str:
        """
        Chemical Pivot: A/B test comparing your metal vs any metal.
        Group A: Your metal node (any geometry)
        Group B: Any metal + your geometry constraints
        """
        return f"""
*** EXPERIMENT: A/B TEST (CHEMISTRY) ***
We compared your metal choice against alternatives under your geometric constraints.

GROUP A: YOUR METAL (Your Metal Node, Any Geometry)
{self._generate_table(set_f, 15, "Group A (Your Metal)", metric_name)}

GROUP B: ANY METAL (Any Metal + Your Di/Df Geometry)
{self._generate_table(set_e_pure, 15, "Group B (Any Metal + Your Geo)", metric_name)}

Analysis:
- If Group A >> Group B: Your metal choice adds value
- If Group A ≈ Group B: Metal choice doesn't matter for this geometry
- If Group A << Group B: Consider alternative metals shown in Group B
"""
    
    # =========================================================================
    # FEEDBACK TYPE 5: BEST VS WORST (Rank 5 - Pattern Discovery)
    # =========================================================================
    def _generate_best_vs_worst(self, set_a: pd.DataFrame, metric_name: str) -> str:
        """
        Best vs Worst: Stratified sampling to discover patterns.
        Top 15 vs Bottom 15 performers.
        """
        if set_a.empty:
            return "*** EXPERIMENT: BEST vs WORST ***\nNo candidates found for comparison."
        
        s_best = set_a.nlargest(15, 'target')
        s_worst = set_a.nsmallest(15, 'target')
        
        # Generate separate tables for clarity
        best_table = self._generate_table(s_best, 15, "TOP 15 PERFORMERS", metric_name)
        worst_table = self._generate_table(s_worst, 15, "BOTTOM 15 PERFORMERS", metric_name)
        
        return f"""
*** EXPERIMENT: BEST vs WORST ANALYSIS ***
We compared the best and worst performers from your chemical search.

{best_table}

{worst_table}

Analysis Questions:
- What Di/Df values do the top performers have?
- What linker patterns appear in the best performers?
- What distinguishes the winners from the losers?
"""
    
    # =========================================================================
    # FEEDBACK TYPE 6: HYPOTHESIS VALIDATION (Rank 6 - Final Check)
    # =========================================================================
    def _generate_hypothesis_validation(self, set_z: pd.DataFrame, metric_name: str) -> str:
        """
        Hypothesis Validation: Test the complete hypothesis.
        """
        return f"""
*** EXPERIMENT: HYPOTHESIS VALIDATION ***
We tested candidates matching your complete hypothesis:
- Your Metal + Your Linker + All Your Geometry Constraints

{self._generate_table(set_z, FEEDBACK_SAMPLE_SIZE_LARGE, "Full Hypothesis Matches", metric_name)}
"""
    
    # =========================================================================
    # FEEDBACK TYPE 7: VIRTUAL SYNTHESIS & CHARACTERIZATION (2-Beam Stage-Gate)
    # =========================================================================
    def _generate_virtual_synthesis_report(self, filter_sets: dict, metric_name: str, sample_size: int = 10) -> str:
        """
        Virtual Synthesis: Solves survivorship bias by acting as a forward-synthesis lab simulation.
        State 1: Chemical Incompatibility (set_a empty)
        State 2: Geometric Impossibility (set_a not empty, set_z empty)
        State 3: Hypothesis Validated (set_z not empty, compares set_z vs set_a)
        """
        set_a = filter_sets.get('a', pd.DataFrame())
        set_z = filter_sets.get('z', pd.DataFrame())
        
        content = "*** EXPERIMENT: VIRTUAL SYNTHESIS & CHARACTERIZATION (2-Beam Stage-Gate) ***\n"
        
        # State 1: Chemical Incompatibility
        if set_a.empty:
            content += "STATUS: SYNTHESIS FAILED\n"
            content += "Your hypothesized Node and Linker combination is chemically or sterically incompatible.\n"
            content += "No such MOFs could be synthesized in our virtual lab. Please revise your chemical components.\n"
            return content
            
        # State 2: Geometric Impossibility
        if not set_a.empty and set_z.empty:
            content += "STATUS: SYNTHESIS SUCCESSFUL\n"
            content += "However, ZERO candidates met your hypothesized geometric constraints.\n\n"
            content += self._generate_table(set_a, sample_size, "CHARACTERIZED BATCH: CHEMICAL BASELINE", metric_name)
            content += "\n-> INSTRUCTION: Analyze the empirical geometries in the table above. Adjust your geometric expectations or pivot your chemistry to hit your target metrics.\n"
            return content
            
        # State 3: Hypothesis Validated
        if not set_z.empty:
            content += "STATUS: SYNTHESIS SUCCESSFUL & HYPOTHESIS VALIDATED\n\n"
            content += self._generate_table(set_z, sample_size, "BEAM 1: HYPOTHESIS SURVIVORS", metric_name)
            content += "\n"
            content += self._generate_table(set_a, sample_size, "BEAM 2: CHEMICAL BASELINE", metric_name)
            content += "\n-> INSTRUCTION: Perform an A/B test. Compare the target performance of Beam 1 against Beam 2 to evaluate the true causal impact of your geometric constraints.\n"
            
        return content


# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_feedback_generator():
    """Test the feedback generator with V3 sample data."""
    
    print("\n" + "="*60)
    print("FEEDBACK GENERATOR MODULE TEST (V3)")
    print("="*60 + "\n")
    
    # Requires matchmaker and sensitivity analyzer to be V3 ready
    from matchmaker import Matchmaker
    from sensitivity_analyzer import SensitivityAnalyzer
    
    test_agent2_output = {
        "node_query": {
            "metals_include": ["Zr"],
            "connectivity": [12],
            "nuclearity": 6,
            "sbu_type": "Cluster"
        },
        "linker_query": {
            "connectivity": 2,
            "must_contain_elements": ["O", "C", "H"],
            "length_min_angstrom": 6.0,
            "length_max_angstrom": 12.0,
            "is_rigid": True,
            "functional_groups": []
        },
        "geometry_filter": {
            "target_Di_min_A": 12.0, "target_Di_max_A": 20.0,
            "target_Df_min_A": 7.0, "target_Df_max_A": 10.0
        }
    }
    
    matchmaker = Matchmaker()
    matchmaker_results = matchmaker.smart_matchmaker_single_node(test_agent2_output)
    
    analyzer = SensitivityAnalyzer()
    analyzer.run_analysis(test_agent2_output, matchmaker_results, run_id="TEST_V3")
    
    generator = FeedbackGenerator()
    
    print("\n" + "="*60)
    print("TESTING ALL 7 FEEDBACK TYPES")
    print("="*60)
    
    type_names = [
        "3-Beam Diagnostic", "Universe Baseline", "Geometric Optimizer",
        "Chemical Pivot", "Best vs Worst", "Hypothesis Validation",
        "Virtual Synthesis (2-Beam)"
    ]
    
    for i, name in enumerate(type_names, 1):
        print(f"\n--- Type {i}: {name} ---")
        feedback = generator.generate_feedback(i, analyzer.filter_sets)
        # Print glimpse of feedback to verify "Name Tags" and Footer
        print(feedback[:400] + "..." if len(feedback) > 400 else feedback)
        
        # Check for Diagnostic Footer validation in a failed case
        if "DIAGNOSTIC FOOTER" in feedback:
            print(">> [VALIDATION] Diagnostic Footer detected.")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_feedback_generator()
