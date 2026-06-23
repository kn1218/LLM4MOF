# =============================================================================
# LLM4MOF Autonomous System - Agent 2 Handler
# =============================================================================
# Constraint Extractor (Stateless)
# =============================================================================

import os
import sys
from typing import Optional, Dict, Any
import json

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AGENT2_PROMPT_PATH
import config
from core.llm_client import LLMClient, load_prompt


# =============================================================================
# DATABASE MODE RULES (Injected into Agent 2 prompt at runtime)
# =============================================================================

_PORMAKE_MODE_RULES = """
> **⚠️ DATABASE MODE: PorMake Building-Block Assembly**
>
> In this mode, the database assembles MOFs from separate **Node** and **Edge** building blocks.
> PorMake's grammar splits the chemist's "linker molecule" into two parts:
>
> | Component | Contains | Example |
> |-----------|----------|---------|
> | **Node (SBU)** | Metal cluster **+ coordinating ligand groups** (carboxylate, azolate, phosphonate, carbonyl) | Zr₆O₄(OH)₄(carboxylate)₁₂ |
> | **Edge (Linker)** | **Only the organic backbone + substituents** between connection points | biphenyl, pyridine, amine-functionalized naphthalene |
>
> **CRITICAL CONSEQUENCE FOR BRANCH TAGS:**
> - Coordination-group tags (`Carboxyl`, `Carbonyl`, `Phosphonate`, `Sulfonate`) describe how the linker **binds** to the metal. These groups are on the **Node** side in PorMake.
> - **DO NOT** include coordination tags in `linker_branches.required_tags` or `functional_groups`. They will fail to match because edges don't carry them.
> - Instead, route coordination info to `node_query.ligand_chemistry` (e.g., `"Oxygen"` for carboxylate, `"Nitrogen"` for azolate).
> - In `linker_branches`, use **only backbone scaffold tags** (Benzene, Biphenyl, Pyridine, Naphthalene, etc.) and **substituent tags** (Amine, Methyl, Fluoro, Hydroxyl, etc.).
>
> **Example — "biphenyl dicarboxylate with amine groups":**
> - `node_query.ligand_chemistry`: `["Oxygen"]`  ← carboxylate coordination
> - `linker_branches`: `[{"description": "amine-biphenyl backbone", "required_tags": ["Biphenyl", "Amine"]}]`  ← backbone + substituent only
""".strip()

_HMOF_MODE_RULES = """
> **⚠️ DATABASE MODE: hMOF Whole-MOF Filtering**
>
> In this mode, the database contains **pre-assembled hypothetical MOFs** (51K entries).
> Each MOF entry carries a flat list of all chemical tags (metals, backbone scaffolds,
> coordination groups, substituents). Coordination tags like `Carboxyl` and `Azolate`
> ARE valid and present in the data — you may use them freely in `linker_branches`
> and `functional_groups`.
""".strip()

_QMOF_MODE_RULES = """
> **⚠️ DATABASE MODE: QMOF Whole-MOF Filtering (Electronic/Band Gap)**
>
> In this mode, the database contains **experimentally-derived MOFs** from the QMOF database.
> Each MOF entry carries chemical tags (metals, backbone scaffolds, coordination groups,
> substituents) plus electronic metadata (oxidation states, coordination geometry).
> Coordination tags like `Carboxyl` and `Azolate` ARE valid in the data — you may use
> them freely in `linker_branches` and `functional_groups`.
""".strip()


class Agent2Handler:
    """
    Agent 2: Data Bridge (Constraint Extractor)
    
    This agent converts Agent 1's qualitative hypothesis into
    quantitative database search specifications.
    
    Stateless: No memory of previous iterations.
    Pure translation layer: Agent 1 JSON -> Database Constraints JSON
    """
    
    def __init__(self, usage_log_path: Optional[str] = None):
        """Initialize Agent 2 with its system prompt and mode-specific rules.

        Args:
            usage_log_path: Optional path to usage_log.json for token/cost logging
        """
        # Load system prompt from file
        raw_prompt = load_prompt(AGENT2_PROMPT_PATH)

        # Inject database-mode-specific rules
        mode_rules = self._get_mode_rules()
        self.system_prompt = raw_prompt.replace("{DATABASE_MODE_RULES}", mode_rules)

        # Initialize LLM client WITHOUT multi-turn (stateless)
        self.client = LLMClient(self.system_prompt, multi_turn=False, usage_log_path=usage_log_path)

        mode_name = ("hMOF" if config.is_hmof_mode()
                     else "QMOF" if config.is_qmof_mode()
                     else "PorMake")
        print(f"[Agent 2] Initialized - Constraint Extractor (Stateless mode, {mode_name} rules)")

    @staticmethod
    def _get_mode_rules() -> str:
        """Return the database-mode-specific addendum for the Agent 2 prompt."""
        if config.is_hmof_mode():
            return _HMOF_MODE_RULES
        elif config.is_qmof_mode():
            return _QMOF_MODE_RULES
        else:
            return _PORMAKE_MODE_RULES
    
    def extract_constraints(self, hypothesis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract database constraints from Agent 1's hypothesis.
        """
        print("\n[Agent 2] Extracting database constraints...")
        
        # Convert hypothesis to JSON string for the prompt
        hypothesis_json = json.dumps(hypothesis, indent=2)
        
        response = self.client.send_message(hypothesis_json, temperature=config.AGENT2_TEMPERATURE)
        
        if not response:
            print("[Agent 2] ERROR: No response received")
            return None
        
        # Extract JSON from response
        constraints = LLMClient.extract_json(response)
        
        if constraints:
            self._print_constraints(constraints)
            # Validate structure
            if self._validate_constraints(constraints):
                return constraints
            else:
                print("[Agent 2] WARNING: Constraints structure incomplete or schema mismatch")
                return constraints  # Return anyway, matchmaker will handle
        else:
            print("[Agent 2] ERROR: Could not parse constraints JSON")
            print(f"   Raw response: {response[:500]}...")
            return None
    
    def _validate_constraints(self, constraints: Dict[str, Any]) -> bool:
        """Validate that the constraints have required structure (V3 Schema)."""
        required_keys = ['node_query', 'linker_query', 'geometry_filter']
        
        for key in required_keys:
            if key not in constraints:
                print(f"   [Validation] Missing key: {key}")
                return False
        
        # Check node_query (V3 Schema: metals_include)
        node_q = constraints.get('node_query', {})
        if 'metals_include' not in node_q:
            # Fallback check for old key 'metal_symbol' just in case
            if 'metal_symbol' not in node_q:
                print("   [Validation] node_query missing 'metals_include'")
                return False
        
        # Check V3.3 fields (Soft Validation)
        if 'ligand_chemistry' not in node_q:
             print("   [Validation] WARNING: node_query missing 'ligand_chemistry' (V3.3 Schema)")

        # Check linker_query
        l_q = constraints.get('linker_query', {})
        if 'functional_groups' not in l_q:
             print("   [Validation] WARNING: linker_query missing 'functional_groups' (V3.3 Schema)")
        
        # Check abstract_features (soft validation --optional field)
        node_af = node_q.get('abstract_features', {})
        if node_af and not isinstance(node_af, dict):
            print("   [Validation] WARNING: node_query.abstract_features should be a dict, got:", type(node_af))
        linker_af = l_q.get('abstract_features', {})
        if linker_af and not isinstance(linker_af, dict):
            print("   [Validation] WARNING: linker_query.abstract_features should be a dict, got:", type(linker_af))
        
        # Check categorized functional group fields (soft validation --optional)
        bb_reqs = l_q.get('backbone_requirements')
        if bb_reqs is not None and not isinstance(bb_reqs, list):
            print("   [Validation] WARNING: linker_query.backbone_requirements should be a list, got:", type(bb_reqs))
        sub_reqs = l_q.get('substituent_requirements')
        if sub_reqs is not None and not isinstance(sub_reqs, list):
            print("   [Validation] WARNING: linker_query.substituent_requirements should be a list, got:", type(sub_reqs))
        mgc = l_q.get('min_group_counts')
        if mgc is not None and not isinstance(mgc, dict):
            print("   [Validation] WARNING: linker_query.min_group_counts should be a dict, got:", type(mgc))
        
        # Check Phase 3 electronic metadata fields (soft validation --optional)
        ox = node_q.get('oxidation_state')
        if ox is not None and not isinstance(ox, dict):
            print("   [Validation] WARNING: node_query.oxidation_state should be a dict, got:", type(ox))
        geom = node_q.get('geometry_preference')
        if geom is not None and not isinstance(geom, str):
            print("   [Validation] WARNING: node_query.geometry_preference should be a string, got:", type(geom))
        
        # Check linker_branches (soft validation --optional field)
        branches = l_q.get('linker_branches')
        if branches is not None:
            if not isinstance(branches, list):
                print("   [Validation] WARNING: linker_query.linker_branches should be a list, got:", type(branches))
            else:
                for i, branch in enumerate(branches):
                    if not isinstance(branch, dict):
                        print(f"   [Validation] WARNING: linker_branches[{i}] should be a dict, got:", type(branch))
                    elif 'required_tags' not in branch:
                        print(f"   [Validation] WARNING: linker_branches[{i}] missing 'required_tags'")
                    elif not isinstance(branch['required_tags'], list):
                        print(f"   [Validation] WARNING: linker_branches[{i}].required_tags should be a list")
                if len(branches) > 5:
                    print(f"   [Validation] NOTE: {len(branches)} branches detected (>5 may indicate over-decomposition)")

        # Check geometry_filter (soft validation — empty is valid for chemistry-first mode)
        geo = constraints.get('geometry_filter', {})
        if not geo:
             print("   [Validation] geometry_filter is empty (OK - geometry is a second-stage prediction, not required)")
        
        return True
    
    def _print_constraints(self, constraints: Dict[str, Any]):
        """Pretty print the constraints for CLI display."""
        print("\n" + "-"*50)
        print("AGENT 2 CONSTRAINTS")
        print("-"*50)
        
        node_q = constraints.get('node_query', {})
        linker_q = constraints.get('linker_query', {})
        geo = constraints.get('geometry_filter', {})
        
        # Handle V3 vs V2 keys safely
        metals = node_q.get('metals_include', node_q.get('metal_symbol', 'N/A'))
        node_cn = node_q.get('connectivity', 'N/A')
        node_chem = node_q.get('ligand_chemistry', 'N/A')
        
        # Linker V3 Fields
        linker_cn = linker_q.get('connectivity', 'N/A')
        len_min = linker_q.get('length_min', '?')
        len_max = linker_q.get('length_max', '?')
        rigid = linker_q.get('is_rigid', 'N/A')
        
        linker_funcs = linker_q.get('functional_groups', [])

        print(f"Metals: {metals}")
        print(f"Node Connectivity: {node_cn}")
        print(f"Node Ligand Chemistry: {node_chem}")
        print(f"Linker Connectivity: {linker_cn}")
        print(f"Linker Length: {len_min} - {len_max} A")
        print(f"Linker Rigid: {rigid}")
        print(f"Linker Func Groups: {linker_funcs}")
        # Categorized functional group requirements (if present)
        bb_reqs = linker_q.get('backbone_requirements', [])
        if bb_reqs:
            print(f"Backbone Requirements: {bb_reqs}")
        sub_reqs = linker_q.get('substituent_requirements', [])
        if sub_reqs:
            print(f"Substituent Requirements: {sub_reqs}")
        min_gc = linker_q.get('min_group_counts', {})
        if min_gc:
            print(f"Min Group Counts: {min_gc}")
        # Abstract Features (if present)
        node_af = node_q.get('abstract_features', {})
        if node_af:
            active = {k: v for k, v in node_af.items() if v is not None}
            if active:
                print(f"Node Abstract Features: {active}")
        linker_af = linker_q.get('abstract_features', {})
        if linker_af:
            active = {k: v for k, v in linker_af.items() if v is not None}
            if active:
                print(f"Linker Abstract Features: {active}")
        # Linker branches (if present)
        branches = linker_q.get('linker_branches', [])
        if branches:
            print(f"Linker Branches ({len(branches)} alternatives):")
            for i, branch in enumerate(branches):
                desc = branch.get('description', '')
                tags = branch.get('required_tags', [])
                print(f"  Branch {i+1}: {desc} -> required: {tags}")
        # Phase 3 electronic metadata (QMOF-only)
        ox_state = node_q.get('oxidation_state')
        if ox_state:
            print(f"Oxidation State: {ox_state}")
        geom_pref = node_q.get('geometry_preference')
        if geom_pref:
            print(f"Geometry Preference: {geom_pref}")
        print(f"Di Range: {geo.get('target_Di_min', '?')} - {geo.get('target_Di_max', '?')} A")
        print(f"Df Range: {geo.get('target_Df_min', '?')} - {geo.get('target_Df_max', '?')} A")
        print(f"SA Range: {geo.get('target_sa_min', '?')} - {geo.get('target_sa_max', '?')} m2/cm3")
        print(f"VF Range: {geo.get('target_vf_min', '?')} - {geo.get('target_vf_max', '?')}")
        print(f"Density Range: {geo.get('target_density_min', '?')} - {geo.get('target_density_max', '?')} g/cm3")
        print(f"Dif Range: {geo.get('target_dif_min', '?')} - {geo.get('target_dif_max', '?')} A")
        print(f"CV Range: {geo.get('target_cv_min', '?')} - {geo.get('target_cv_max', '?')} A^3")
        
        print("-"*50)


# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_agent2():
    """Test Agent 2 with sample Agent 1 output."""
    
    print("\n" + "="*60)
    print("AGENT 2 HANDLER TEST")
    print("="*60 + "\n")
    
    # Sample Agent 1 output
    sample_hypothesis = {
        "target_application": "Maximize gravimetric H2 uptake in a MOF",
        "hypothesis_mechanism": "Test mixed nuclearity handling.",
        "ideal_pore_geometry": "Standard.",
        "node_composition": "Use a Zr6 cluster OR a Zn dimer node.",
        "linker_composition": "Rigid Biphenyl- or terphenyl-dicarboxylate.",
        "novelty_justification": "Testing fix."
    }
    
    # Initialize agent
    agent2 = Agent2Handler()
    
    # Test constraint extraction
    constraints = agent2.extract_constraints(sample_hypothesis)
    
    if constraints:
        print("\n--- Validation ---")
        node_q = constraints.get('node_query', {})
        metals = node_q.get('metals_include', node_q.get('metal_symbol'))
        
        # Test V3 list structure
        if isinstance(metals, list) and ('Zr' in metals or 'Hf' in metals):
             print("[OK] AGENT 2 TEST PASSED - Metals extracted as list")
        else:
             print(f"[FAIL] Unexpected metal format: {metals}")
             
        # Test Angstrom extraction
        l_q = constraints.get('linker_query', {})
        if l_q.get('length_min'):
            print("[OK] AGENT 2 TEST PASSED - Angstrom length extracted")
        else:
            print("[FAIL] Angstrom length missing (Check prompt?)")

    else:
        print("[FAIL] AGENT 2 TEST FAILED - No constraints extracted")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_agent2()
