# =============================================================================
# LLM2POR Autonomous System - Feedback Generator
# =============================================================================
# Generates 6 scientifically-designed feedback types for Agent 1
# =============================================================================

import pandas as pd
import numpy as np
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

    def _parse_pormake_filename(self, filename_str: str) -> tuple[str | None, str | None, str | None]:
        """Parse PORMAKE filename into (topology, node_id, linker_id).
        
        Returns (None, None, None) if parsing fails.
        """
        try:
            if '+' not in str(filename_str):
                return (None, None, None)
            parts = str(filename_str).split('+')
            if len(parts) < 3:
                return (None, None, None)
            return (parts[0], parts[1], parts[2])
        except (IndexError, TypeError):
            return (None, None, None)

    def _enrich_pormake_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich PORMAKE DataFrame rows with BB-level chemical metadata.
        
        Joins filename → node_id/linker_id → bb_lookup to add:
        - _node_metals: list of metal symbols from the node BB
        - _node_af: dict of abstract_features for the node
        - _linker_af: dict of abstract_features for the linker
        - _linker_fg_cat: dict of functional_groups_categorized for the linker
        
        Columns prefixed with '_' to avoid collision with existing DataFrame columns.
        Returns a copy with enrichment columns added.
        """
        df = df.copy()
        bb = self._resolver.bb_lookup
        
        node_metals_list = []
        node_af_list = []
        linker_af_list = []
        linker_fg_cat_list = []
        
        for _, row in df.iterrows():
            _, node_id, linker_id = self._parse_pormake_filename(str(row.get('filename', '')))
            
            node_data = bb.get(node_id, {}) if node_id else {}
            linker_data = bb.get(linker_id, {}) if linker_id else {}
            
            node_metals_list.append(node_data.get('metals', []))
            node_af_list.append(node_data.get('abstract_features', {}))
            linker_af_list.append(linker_data.get('abstract_features', {}))
            linker_fg_cat_list.append(linker_data.get('functional_groups_categorized', {}))
        
        df['_node_metals'] = node_metals_list
        df['_node_af'] = node_af_list
        df['_linker_af'] = linker_af_list
        df['_linker_fg_cat'] = linker_fg_cat_list
        
        return df

    @staticmethod
    def _describe_hmof_row(row) -> str:
        """Build a human-readable description from hMOF index fields.
        
        Uses functional_groups_categorized (backbone/substituents) when available
        for chemically informative descriptions that Agent 1 can learn from.
        Falls back to readable_name or metals+topology if categorized data missing.
        """
        metals = row.get('metals', [])
        topo = row.get('topology', '')
        fg_cat = row.get('functional_groups_categorized', None)
        
        # Build metal + topology prefix (always present)
        prefix_parts = []
        if metals:
            prefix_parts.append('/'.join(metals if isinstance(metals, list) else [metals]))
        if topo:
            prefix_parts.append(str(topo))
        prefix = ' '.join(prefix_parts) if prefix_parts else ''
        
        # Use categorized FGs when available (backbone/substituent separation)
        if isinstance(fg_cat, dict) and (fg_cat.get('backbone') or fg_cat.get('substituents')):
            backbone = fg_cat.get('backbone', [])
            substituents = fg_cat.get('substituents', [])
            # Deduplicate carboxyl variants: keep only 'carboxylate' if both present
            if 'carboxylate' in backbone and 'carboxyl_any' in backbone:
                backbone = [g for g in backbone if g != 'carboxyl_any']
            chem_parts = []
            if backbone:
                chem_parts.append(f"Bkbn:[{','.join(backbone[:4])}]")
            if substituents:
                chem_parts.append(f"Subs:[{','.join(substituents[:3])}]")
            chem_str = ' '.join(chem_parts)
            return f"{prefix} | {chem_str}" if prefix else chem_str
        
        # Fallback to readable_name if available
        name = row.get('readable_name', '')
        if name:
            return f"{prefix} | {name}" if prefix else name
        
        return prefix if prefix else 'unknown'

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
            return ' '.join(parts) if parts else 'unknown'
        # Append node geometry if not already in the name
        geom = row.get('geometry', '')
        if geom and str(geom) != 'nan' and str(geom) != 'Unknown':
            geom_keywords = ['octahedral', 'tetrahedral', 'square planar',
                             'trigonal', 'cubic', 'dodecahedral', 'linear']
            name_lower = name.lower()
            if not any(gk in name_lower for gk in geom_keywords):
                name = f'{name} ({geom})'
        return name
    
    # =========================================================================
    # ENRICHED FEEDBACK: Chemistry Profile + Pattern Summary (Phase 7)
    # =========================================================================
    
    def _generate_chemistry_profile(self, df: pd.DataFrame, title: str = "Chemistry Profile") -> str:
        """Generate per-sample chemistry profile showing abstract_features and categorized FGs.
        
        Separate from the geometry table to maintain readability. Shows only TRUE
        abstract features (compact) and backbone/substituent breakdown per MOF.
        
        Works for all modes:
        - PORMAKE: Uses enriched _node_af/_linker_af/_linker_fg_cat columns
        - hMOF: Uses functional_groups_categorized column directly  
        - QMOF: Uses metals + functional_groups columns
        
        Returns empty string if no chemistry metadata available.
        """
        if df.empty:
            return ""
        
        is_qmof = config.is_qmof_mode()
        is_hmof = config.is_hmof_mode()
        
        lines = [f"--- {title} ---"]

        # Anonymous labels to prevent Agent 1 from inferring database identity
        for idx, (_, row) in enumerate(df.iterrows(), 1):
            label = f"MOF-{idx}"

            if is_hmof:
                # hMOF: use functional_groups_categorized directly
                fg_cat = row.get('functional_groups_categorized', None)
                parts = []
                if isinstance(fg_cat, dict):
                    backbone = fg_cat.get('backbone', [])
                    subs = fg_cat.get('substituents', [])
                    if 'carboxylate' in backbone and 'carboxyl_any' in backbone:
                        backbone = [g for g in backbone if g != 'carboxyl_any']
                    if backbone:
                        parts.append(f"Bkbn:[{','.join(backbone[:4])}]")
                    if subs:
                        parts.append(f"Subs:[{','.join(subs[:3])}]")
                # Fallback: show metals + topology when no FG data
                if not parts:
                    metals = row.get('metals', [])
                    topo = row.get('topology', '')
                    if metals and isinstance(metals, list):
                        parts.append(f"Metals:[{','.join(metals)}]")
                    if topo:
                        parts.append(f"Topo:{topo}")
                lines.append(f"  {label}: {' '.join(parts)}" if parts else f"  {label}: (bare framework)")

            elif is_qmof:
                # QMOF: use metals + functional_groups + oxidation_states
                metals = row.get('metals', [])
                fg = row.get('functional_groups', [])
                ox = row.get('oxidation_states', {})
                geom = row.get('geometry', '')
                parts = []
                if metals and isinstance(metals, list):
                    parts.append(f"Metals:[{','.join(metals[:3])}]")
                if isinstance(ox, dict) and ox:
                    ox_str = ','.join(f'{m}({v}+)' for m, v in list(ox.items())[:2])
                    parts.append(f"Ox:[{ox_str}]")
                if geom and str(geom) not in ('nan', 'Unknown', ''):
                    parts.append(f"Geom:{geom}")
                if fg and isinstance(fg, list):
                    parts.append(f"FG:[{','.join(fg[:4])}]")
                lines.append(f"  {label}: {' '.join(parts)}" if parts else f"  {label}: (no chem data)")

            else:
                # PORMAKE: use enriched columns from _enrich_pormake_rows()
                node_af = row.get('_node_af', {})
                linker_af = row.get('_linker_af', {})
                linker_fg = row.get('_linker_fg_cat', {})
                node_metals = row.get('_node_metals', [])

                # Only show TRUE abstract features (compact format)
                true_feats = set()
                for af_dict in [node_af, linker_af]:
                    if isinstance(af_dict, dict):
                        for k, v in af_dict.items():
                            if v is True:
                                # Clean up feature names for readability
                                clean = k.replace('is_', '').replace('has_', '')
                                true_feats.add(clean)

                parts = []
                if node_metals and isinstance(node_metals, list):
                    parts.append(f"Metals:[{','.join(node_metals[:3])}]")
                if true_feats:
                    parts.append(f"Features:[{','.join(sorted(true_feats)[:5])}]")
                if isinstance(linker_fg, dict):
                    bk = linker_fg.get('backbone', [])
                    sb = linker_fg.get('substituents', [])
                    if bk:
                        parts.append(f"Bkbn:[{','.join(bk[:3])}]")
                    if sb:
                        parts.append(f"Subs:[{','.join(sb[:3])}]")

                lines.append(f"  {label}: {' '.join(parts)}" if parts else f"  {label}: (no chem data)")
        
        # Only return if we actually generated chemistry data
        if len(lines) <= 1:
            return ""
        return '\n'.join(lines)
    
    def _generate_pattern_summary(self, df: pd.DataFrame, title: str = "Pattern Summary",
                                  max_chars: int = 500) -> str:
        """Generate aggregate chemistry statistics across sampled MOFs in a beam.
        
        Computes: metal frequency, backbone frequency, abstract feature frequency,
        and geometry ranges. Capped at max_chars to control context window budget.
        Shows only top-3 entries per category.
        
        Works for all modes using available metadata.
        Returns empty string if no meaningful patterns found.
        """
        if df.empty or len(df) < 2:
            return ""
        
        is_qmof = config.is_qmof_mode()
        is_hmof = config.is_hmof_mode()
        n = len(df)
        
        lines = [f"--- {title} (N={n}) ---"]
        
        # === Metal frequency ===
        metals_counter: dict[str, int] = {}
        # === Backbone frequency ===
        backbone_counter: dict[str, int] = {}
        # === Substituent frequency ===
        subs_counter: dict[str, int] = {}
        # === Abstract feature frequency (only for PORMAKE) ===
        feat_counter: dict[str, int] = {}
        
        for _, row in df.iterrows():
            if is_hmof:
                metals = row.get('metals', [])
                if isinstance(metals, list):
                    for m in metals:
                        metals_counter[m] = metals_counter.get(m, 0) + 1
                fg_cat = row.get('functional_groups_categorized', None)
                if isinstance(fg_cat, dict):
                    for g in fg_cat.get('backbone', []):
                        if g != 'carboxyl_any':  # skip duplicate
                            backbone_counter[g] = backbone_counter.get(g, 0) + 1
                    for g in fg_cat.get('substituents', []):
                        subs_counter[g] = subs_counter.get(g, 0) + 1
                        
            elif is_qmof:
                metals = row.get('metals', [])
                if isinstance(metals, list):
                    for m in metals:
                        metals_counter[m] = metals_counter.get(m, 0) + 1
                fg = row.get('functional_groups', [])
                if isinstance(fg, list):
                    for g in fg:
                        backbone_counter[g] = backbone_counter.get(g, 0) + 1
                        
            else:
                # PORMAKE: use enriched columns
                node_metals = row.get('_node_metals', [])
                if isinstance(node_metals, list):
                    for m in node_metals:
                        metals_counter[m] = metals_counter.get(m, 0) + 1
                linker_fg = row.get('_linker_fg_cat', {})
                if isinstance(linker_fg, dict):
                    for g in linker_fg.get('backbone', []):
                        backbone_counter[g] = backbone_counter.get(g, 0) + 1
                    for g in linker_fg.get('substituents', []):
                        subs_counter[g] = subs_counter.get(g, 0) + 1
                # Aggregate abstract features (per-MOF, not per-BB)
                # Use a set so each feature counts at most once per MOF
                mof_feats: set[str] = set()
                for af_dict in [row.get('_node_af', {}), row.get('_linker_af', {})]:
                    if isinstance(af_dict, dict):
                        for k, v in af_dict.items():
                            if v is True:
                                mof_feats.add(k.replace('is_', '').replace('has_', ''))
                for feat in mof_feats:
                    feat_counter[feat] = feat_counter.get(feat, 0) + 1
        
        # Format top-3 per category as "name(XX%)" 
        def _top3(counter: dict[str, int]) -> str:
            if not counter:
                return ""
            sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:3]
            return ', '.join(f"{k}({v*100//n}%)" for k, v in sorted_items)
        
        metals_str = _top3(metals_counter)
        backbone_str = _top3(backbone_counter)
        subs_str = _top3(subs_counter)
        feats_str = _top3(feat_counter)
        
        if metals_str:
            lines.append(f"  Metals: {metals_str}")
        if backbone_str:
            lines.append(f"  Backbone: {backbone_str}")
        if subs_str:
            lines.append(f"  Substituents: {subs_str}")
        if feats_str:
            lines.append(f"  Features: {feats_str}")
        
        # Geometry ranges (available in all modes)
        for col, label in [('di', 'Di'), ('df', 'Df'), ('sa', 'SA'), ('vf', 'VF')]:
            if col in df.columns:
                try:
                    raw = [float(x) for x in df[col] if pd.notna(x)]
                    if len(raw) >= 2 and max(raw) > 0:
                        raw_sorted = sorted(raw)
                        mid = len(raw_sorted) // 2
                        med = (raw_sorted[mid] + raw_sorted[~mid]) / 2
                        lines.append(
                            f"  {label}: {min(raw):.1f}-{max(raw):.1f} (med {med:.1f})"
                        )
                except (ValueError, TypeError):
                    pass  # Skip non-numeric columns gracefully
        
        # Only return if we generated meaningful content beyond the header
        if len(lines) <= 1:
            return ""
        
        result = '\n'.join(lines)
        # Enforce size cap
        if len(result) > max_chars:
            result = result[:max_chars - 3] + "..."
        return result

    def _generate_table(self, df: pd.DataFrame, n: int, title: str = "Samples", metric_name: str = "Target Metric") -> str:
        """Creates the text table for the LLM prompt. QMOF-aware.
        
        Returns only the table text. For enriched output with chemistry profile
        and pattern summary, use _generate_enriched_beam() instead.
        """
        table_text, _ = self._generate_table_with_samples(df, n, title, metric_name)
        return table_text
    
    def _generate_table_with_samples(self, df: pd.DataFrame, n: int, title: str = "Samples",
                                      metric_name: str = "Target Metric") -> tuple[str, pd.DataFrame]:
        """Creates the text table and returns both table text and the sampled DataFrame.
        
        The returned DataFrame is enriched with BB metadata for PORMAKE mode,
        enabling downstream chemistry profile and pattern summary generation.
        """
        if df.empty:
            return f"--- {title} ---\nNo Data Available.", pd.DataFrame()
        
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
        
        # Enrich PORMAKE rows with BB metadata for chemistry profile/pattern summary
        if not is_qmof and not is_hmof:
            samp = self._enrich_pormake_rows(samp)
        
        # Apply mode-aware Structure Descriptions
        if is_hmof:
            samp['Structure Description'] = samp.apply(self._describe_hmof_row, axis=1)
        elif is_qmof:
            samp['Structure Description'] = samp.apply(self._describe_qmof_row, axis=1)
        else:
            samp['Structure Description'] = samp['filename'].apply(self._translate_mof)
        
        if is_qmof:
            # QMOF columns: readable structure description + bandgap + pore geometry + electronic metadata
            available = ['Structure Description', 'target']
            display_names = ['Structure', metric_name]
            # Format oxidation_states dict for compact display
            if 'oxidation_states' in samp.columns:
                samp['oxidation_states'] = samp['oxidation_states'].apply(
                    lambda x: ', '.join(f'{m}({v}+)' for m, v in x.items())
                    if isinstance(x, dict) else (str(x) if pd.notna(x) else '')
                )
            for col, name in [
                ('di', 'LCD (A)'), ('df', 'PLD (A)'), ('density', 'Density'),
                ('geometry', 'Geometry'), ('oxidation_states', 'Ox. States'),
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
        
        table_text = f"--- {title} ---\n" + view.to_string(index=False)
        return table_text, samp
    
    def _generate_enriched_beam(self, df: pd.DataFrame, n: int, title: str = "Samples",
                                 metric_name: str = "Target Metric") -> str:
        """Generate a complete enriched beam: geometry table + chemistry profile + pattern summary.
        
        This is the Phase 7 enriched output that replaces plain _generate_table() calls
        in feedback type methods. It combines:
        1. The geometry table (existing behavior, unchanged)
        2. Chemistry Profile: per-sample abstract_features + categorized FGs
        3. Pattern Summary: aggregate statistics with 500-char cap
        """
        table_text, samp = self._generate_table_with_samples(df, n, title, metric_name)
        
        if samp.empty:
            return table_text
        
        parts = [table_text]
        
        profile = self._generate_chemistry_profile(samp, f"Chemistry Profile ({title})")
        if profile:
            parts.append(profile)
        
        summary = self._generate_pattern_summary(samp, f"Pattern Summary ({title})")
        if summary:
            parts.append(summary)
        
        return '\n'.join(parts)
    
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
            footer += "[CRITICAL FAILURE] CHEMISTRY: No entries match your Metal + Functional Group constraints.\n"
            footer += "  Possible causes:\n"
            footer += "  -> Functional Groups too strict: All specified tags must be present (AND logic).\n"
            footer += "  -> Metal + Linker combination may not exist in this search space.\n"
            footer += "  SUGGESTION: Broaden metal list or relax functional group requirements.\n"
        else:
            footer += "[PASS] CHEMISTRY: Your chemical constraints match entries in the database.\n"

        # 2. Check Geometry (Set E2) — only relevant for PORMAKE markscheme mode
        # In live simulation mode, e2 is always empty (no geometry-only filter set)
        # so skip the geometry diagnostic entirely to avoid misleading messages.
        is_live_mode = (len(set_e2) == 0
                        and len(filter_sets.get('d', pd.DataFrame())) == 0
                        and len(filter_sets.get('g', pd.DataFrame())) == 0)
        if not is_qmof and not is_live_mode:
            count_e2 = len(set_e2)
            if count_e2 == 0:
                footer += "[CRITICAL FAILURE] GEOMETRY: No MOFs exist with your full set of physical property constraints.\n"
                footer += "  SUGGESTION: Your physical property constraints (Di, Df, SA, VF, etc.) may be too narrow.\n"
            else:
                footer += "[PASS] GEOMETRY: Your geometry constraints match entries in the database.\n"

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
            # Unified 4-beam diagnostic for ALL databases (chemistry-first)
            # For QMOF: Beam 1 ~ Beam 2 (no geometry gate), which correctly signals "geometry irrelevant"
            content = self._generate_four_beam(set_z, set_a, set_f, set_total, metric_name)
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
    # FEEDBACK TYPE 1: 4-BEAM DIAGNOSTIC (Chemistry-First with Geometry Gate)
    # =========================================================================
    def _generate_four_beam(self, set_z: pd.DataFrame, set_a: pd.DataFrame,
                             set_f: pd.DataFrame, set_total: pd.DataFrame,
                             metric_name: str) -> str:
        """
        4-Beam Diagnostic: Chemistry-first design with geometry as second-stage gate.
        BEAM 1: Full Hypothesis (Z) - Chemistry + Geometry gate applied
        BEAM 2: Chemistry Only (A) - Your chemistry, any geometry (what assembly produces)
        BEAM 3: Metal Only (F) - Your metals, any linker (isolates metal contribution)
        BEAM 4: Global Baseline (total) - Random sample from entire database

        Diagnostic chain:
          Beam 1 vs Beam 2 → Is your geometry prediction helping or hurting?
          Beam 2 vs Beam 3 → Is your linker selection adding value beyond metal choice?
          Beam 2 vs Beam 4 → Is your chemistry better than random?
        """
        return f"""
*** EXPERIMENT: 4-BEAM DIAGNOSTIC ***
We ran 4 parallel search beams to diagnose your strategy.

BEAM 1: FULL HYPOTHESIS (Your Chemistry + Your Geometry Prediction as Second-Stage Gate)
{self._generate_enriched_beam(set_z, FEEDBACK_SAMPLE_SIZE, "Beam 1", metric_name)}
-> Compare with Beam 2: does your geometry prediction improve or hurt performance?

BEAM 2: CHEMISTRY ONLY (Your Metals + Your Linker Constraints, Any Geometry)
{self._generate_enriched_beam(set_a, FEEDBACK_SAMPLE_SIZE, "Beam 2", metric_name)}
-> This is what your chemistry produces before any geometry gate. Compare with Beam 3.

BEAM 3: METAL ONLY (Your Metals, Any Linker, Any Geometry)
{self._generate_enriched_beam(set_f, FEEDBACK_SAMPLE_SIZE, "Beam 3", metric_name)}
-> Isolates metal contribution. If Beam 2 >> Beam 3, your linker choice is adding value.

BEAM 4: GLOBAL BASELINE (Random Sample from Entire Database)
{self._generate_enriched_beam(set_total, FEEDBACK_SAMPLE_SIZE, "Beam 4", metric_name)}
-> Calibration: if Beam 2 >> Beam 4, your chemistry is better than random selection.
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
*** EXPERIMENT: 4-BEAM ELECTRONIC DIAGNOSTIC ***
We ran 4 parallel search beams to isolate the electronic contributions of your components.

BEAM 1: FULL HYPOTHESIS (Your Metal(s) + Your Linker Functional Groups)
{self._generate_enriched_beam(set_z, FEEDBACK_SAMPLE_SIZE, "Beam 1", metric_name)}

BEAM 2: METAL CONTROL (Your Metal(s), ANY Linker / No Func Group Constraints)
{self._generate_enriched_beam(set_f, FEEDBACK_SAMPLE_SIZE, "Beam 2", metric_name)} -> (Tests the baseline capability of your chosen metal)

BEAM 3: LINKER CONTROL (ANY Metal, Your Functional Groups)
{self._generate_enriched_beam(set_g, FEEDBACK_SAMPLE_SIZE, "Beam 3", metric_name)} -> (Tests the baseline electronic tunability of your linker substituents)

BEAM 4: GLOBAL BASELINE (ANY Metal, ANY Linker)
{self._generate_enriched_beam(set_total, FEEDBACK_SAMPLE_SIZE, "Beam 4", metric_name)} -> (Global distribution of the entire QMOF database)
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

{self._generate_enriched_beam(set_total, FEEDBACK_SAMPLE_SIZE_LARGE, "Global Random Samples", metric_name)}

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
{self._generate_enriched_beam(set_a, 15, "Group A (Random Geo)", metric_name)}

GROUP B: YOUR GEOMETRY (Your Metal + Your Linker + Your Di/Df constraints)
{self._generate_enriched_beam(set_d, 15, "Group B (Your Geo)", metric_name)}

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
{self._generate_enriched_beam(set_f, 15, "Group A (Your Metal)", metric_name)}

GROUP B: ANY METAL (Any Metal + Your Di/Df Geometry)
{self._generate_enriched_beam(set_e_pure, 15, "Group B (Any Metal + Your Geo)", metric_name)}

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
        
        # Generate separate enriched beams for clarity
        best_table = self._generate_enriched_beam(s_best, 15, "TOP 15 PERFORMERS", metric_name)
        worst_table = self._generate_enriched_beam(s_worst, 15, "BOTTOM 15 PERFORMERS", metric_name)
        
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

{self._generate_enriched_beam(set_z, FEEDBACK_SAMPLE_SIZE_LARGE, "Full Hypothesis Matches", metric_name)}
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
            content += self._generate_enriched_beam(set_a, sample_size, "CHARACTERIZED BATCH: CHEMICAL BASELINE", metric_name)
            content += "\n-> INSTRUCTION: Analyze the empirical geometries and chemistry profiles in the data above. Adjust your geometric expectations or pivot your chemistry to hit your target metrics.\n"
            return content
            
        # State 3: Hypothesis Validated
        if not set_z.empty:
            content += "STATUS: SYNTHESIS SUCCESSFUL & HYPOTHESIS VALIDATED\n\n"
            content += self._generate_enriched_beam(set_z, sample_size, "BEAM 1: HYPOTHESIS SURVIVORS", metric_name)
            content += "\n"
            content += self._generate_enriched_beam(set_a, sample_size, "BEAM 2: CHEMICAL BASELINE", metric_name)
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
