"""SMARTS-based functional group detection engine for QMOF linker SMILES.

Imports all SMARTS patterns directly from PORMAKE's bb_pipeline.smarts_library
(zero pattern duplication). Applies them to linker SMILES extracted from
QMOF MOFid decomposition.

Functions:
  enrich_single_linker   — single SMILES → LinkerEnrichment
  enrich_all_linkers     — list[SMILES]  → (per-linker, MOF-level FG)
  compute_abstract_features — combined node+linker SMILES → AbstractFeatures
  infer_properties       — abstract + FG → list[str] property tags
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import RWMol

# ── Path setup for imports ────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent  # database_construction/
sys.path.insert(0, str(BASE_DIR / "PORMAKE"))
sys.path.insert(0, str(BASE_DIR))

# ── Import PORMAKE's SMARTS library (no pattern duplication) ──────────
from bb_pipeline.smarts_library import (  # noqa: E402
    ALL_PATTERNS,
    FUNCTIONAL_GROUPS,
    SUBSTITUENTS,
    SCAFFOLDS,
    HETEROCYCLES,
    ABSTRACT_PATTERNS,
    SmartsPattern,
)
from bb_pipeline.config import METAL_SET  # noqa: E402

# ── Import QMOF schema dataclasses ───────────────────────────────────
from qmof.qmof_pipeline.schema_v2 import (  # noqa: E402
    LinkerEnrichment,
    LinkerFunctionalGroups,
    FunctionalGroups,
    AbstractFeatures,
)


# =====================================================================
# Helpers (mirrors PORMAKE bb_pipeline/smarts_enricher.py exactly)
# =====================================================================

def _get_mol(smiles: Optional[str]) -> Optional[Chem.Mol]:
    """Parse SMILES into RDKit mol, handling ``[*]`` wildcards.

    Returns None if the SMILES is empty/None or sanitization fails
    (common for metal-containing SMILES).
    """
    if not smiles:
        return None
    # Replace [*] with [Lr] for RDKit
    lr_smiles = smiles.replace("[*]", "[Lr]")
    mol = Chem.MolFromSmiles(lr_smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        return mol
    except Exception:
        return None


def _get_mol_strip_metals(smiles: str) -> Optional[Chem.Mol]:
    """Strip metal atoms from SMILES to enable SMARTS matching on organic fragments.

    When metal nodes fail RDKit sanitization (valence errors), removing
    metal atoms yields organic fragments (carboxylate, azolate, nitrile, etc.)
    that can be matched by SMARTS patterns.

    Follows PORMAKE bb_pipeline/smarts_enricher.py lines 144-195 exactly.
    """
    lr_smiles = smiles.replace("[*]", "[Lr]")
    mol = Chem.MolFromSmiles(lr_smiles, sanitize=False)
    if mol is None:
        return None

    # Find metal atom indices
    metal_indices = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in METAL_SET:
            metal_indices.add(atom.GetIdx())

    if not metal_indices:
        return None  # no metals to strip — original failure was something else

    # Remove metals (reverse order to preserve indices)
    rw = RWMol(mol)
    for idx in sorted(metal_indices, reverse=True):
        rw.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(
            rw,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        return rw.GetMol()
    except Exception:
        # Last resort: try to get fragments from SMILES string
        try:
            stripped_smi = Chem.MolToSmiles(rw)
            if stripped_smi:
                frag_mol = Chem.MolFromSmiles(stripped_smi, sanitize=False)
                if frag_mol:
                    try:
                        Chem.SanitizeMol(
                            frag_mol,
                            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                        )
                        return frag_mol
                    except Exception:
                        pass
        except Exception:
            pass
        return None


def _has_metal_atoms(smiles: str) -> bool:
    """Check if a SMILES string contains metal atom symbols."""
    mol = Chem.MolFromSmiles(smiles.replace("[*]", "[Lr]"), sanitize=False)
    if mol is None:
        return False
    return any(atom.GetSymbol() in METAL_SET for atom in mol.GetAtoms())


def _match_patterns(
    mol: Optional[Chem.Mol],
    patterns: list[SmartsPattern],
) -> list[SmartsPattern]:
    """Match all SMARTS patterns against a molecule.

    Mirrors PORMAKE bb_pipeline/smarts_enricher.py lines 198-217.
    """
    if mol is None:
        return []

    matched: list[SmartsPattern] = []
    for pat in patterns:
        try:
            query = Chem.MolFromSmarts(pat.smarts)
            if query is None:
                continue
            if mol.HasSubstructMatch(query):
                matched.append(pat)
        except Exception:
            continue

    return matched


def _count_pattern_matches(
    mol: Optional[Chem.Mol],
    patterns: list[SmartsPattern],
) -> dict[str, int]:
    """Count non-overlapping matches for each pattern.

    Returns ``{pattern_name: count}`` for patterns with ≥1 match.
    Uses ``GetSubstructMatches(query, uniquify=True)`` for unique match sets.

    Mirrors PORMAKE bb_pipeline/smarts_enricher.py lines 220-246.
    """
    if mol is None:
        return {}

    counts: dict[str, int] = {}
    for pat in patterns:
        try:
            query = Chem.MolFromSmarts(pat.smarts)
            if query is None:
                continue
            matches = mol.GetSubstructMatches(query, uniquify=True)
            if matches:
                # Use pattern name; if duplicate name, keep the higher count
                existing = counts.get(pat.name, 0)
                counts[pat.name] = max(existing, len(matches))
        except Exception:
            continue

    return counts


# =====================================================================
# Core functions
# =====================================================================

def enrich_single_linker(smiles: str) -> LinkerEnrichment:
    """Detect functional groups in a single linker SMILES.

    Parameters
    ----------
    smiles : str
        Linker SMILES string (may be empty, invalid, or contain metals).

    Returns
    -------
    LinkerEnrichment
        Per-linker enrichment result with backbone, substituents, counts.
    """
    # Edge case: empty SMILES
    if not smiles or not smiles.strip():
        return LinkerEnrichment(smiles=smiles, smiles_valid=False)

    smiles = smiles.strip()

    # Parse SMILES
    mol = _get_mol(smiles)

    # If parse fails and SMILES contains metal atoms: strip metals and retry
    if mol is None and _has_metal_atoms(smiles):
        mol = _get_mol_strip_metals(smiles)

    # If still None, return invalid result
    if mol is None:
        return LinkerEnrichment(smiles=smiles, smiles_valid=False)

    # Run ALL_PATTERNS against mol
    detected = _match_patterns(mol, ALL_PATTERNS)

    # Classify into backbone vs substituents (follows PORMAKE exactly)
    fg_names = sorted(set(
        p.name for p in detected if p.category == "functional_group"
    ))
    sub_names = sorted(set(
        p.name for p in detected if p.category == "substituent"
    ))
    scaffold_names = sorted(set(
        p.name for p in detected if p.category == "scaffold"
    ))
    heterocycle_names = sorted(set(
        p.name for p in detected if p.category == "heterocycle"
    ))

    # Backbone = scaffolds + heterocycles + core FGs
    # Substituents = halogens, methyl, methoxy, CF3, etc.
    backbone = sorted(set(fg_names + scaffold_names + heterocycle_names))
    substituents = sorted(set(sub_names))
    rule_based = sorted(set(backbone + substituents))

    # Count matches for non-abstract patterns
    non_abstract = [p for p in ALL_PATTERNS if p.category != "abstract"]
    match_counts = _count_pattern_matches(mol, non_abstract)

    # Build counts dict only for detected groups
    rule_based_counts = {
        name: match_counts[name]
        for name in rule_based
        if name in match_counts
    }

    return LinkerEnrichment(
        smiles=smiles,
        smiles_valid=True,
        functional_groups=LinkerFunctionalGroups(
            backbone=backbone,
            substituents=substituents,
            rule_based=rule_based,
            rule_based_counts=rule_based_counts,
        ),
        core_scaffold=scaffold_names,
        heterocycles=heterocycle_names,
    )


def enrich_all_linkers(
    smiles_list: list[str],
) -> tuple[list[LinkerEnrichment], FunctionalGroups]:
    """Process all linker SMILES and aggregate MOF-level functional groups.

    Parameters
    ----------
    smiles_list : list[str]
        List of linker SMILES from MOFid decomposition.

    Returns
    -------
    tuple[list[LinkerEnrichment], FunctionalGroups]
        (per-linker results, aggregated MOF-level functional groups)
    """
    per_linker: list[LinkerEnrichment] = []
    all_backbone: set[str] = set()
    all_substituents: set[str] = set()
    all_rule_based: set[str] = set()
    aggregated_counts: dict[str, int] = {}

    for smi in smiles_list:
        enrichment = enrich_single_linker(smi)
        per_linker.append(enrichment)

        if enrichment.smiles_valid:
            fg = enrichment.functional_groups
            all_backbone.update(fg.backbone)
            all_substituents.update(fg.substituents)
            all_rule_based.update(fg.rule_based)

            # Sum counts across linkers
            for name, count in fg.rule_based_counts.items():
                aggregated_counts[name] = aggregated_counts.get(name, 0) + count

    mof_fg = FunctionalGroups(
        backbone=sorted(all_backbone),
        substituents=sorted(all_substituents),
        rule_based=sorted(all_rule_based),
        rule_based_counts=aggregated_counts,
        llm_additions=[],  # populated later by LLM enricher
    )

    return per_linker, mof_fg


def compute_abstract_features(
    smiles_nodes: list[str],
    smiles_linkers: list[str],
) -> AbstractFeatures:
    """Compute abstract boolean features from combined node + linker SMILES.

    Parameters
    ----------
    smiles_nodes : list[str]
        Node SMILES from MOFid decomposition.
    smiles_linkers : list[str]
        Linker SMILES from MOFid decomposition.

    Returns
    -------
    AbstractFeatures
        Boolean feature flags computed from all SMILES.
    """
    # Parse all SMILES into RDKit mols
    all_smiles = smiles_nodes + smiles_linkers
    mols: list[Chem.Mol] = []
    for smi in all_smiles:
        mol = _get_mol(smi)
        if mol is None and smi and _has_metal_atoms(smi):
            mol = _get_mol_strip_metals(smi)
        if mol is not None:
            mols.append(mol)

    if not mols:
        return AbstractFeatures()

    # Use PORMAKE's ABSTRACT_PATTERNS for electron effects + H-bond detection
    all_abstract_detected: set[str] = set()
    all_detected_names: set[str] = set()
    has_f = False
    has_metal = False
    has_charge = False
    has_aromatic = False

    for mol in mols:
        # Detect abstract patterns
        detected = _match_patterns(mol, ABSTRACT_PATTERNS)
        all_abstract_detected.update(p.name for p in detected)

        # Also detect non-abstract patterns for azo check
        all_det = _match_patterns(mol, ALL_PATTERNS)
        all_detected_names.update(p.name for p in all_det)

        # Check for fluorine atom
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym == "F":
                has_f = True
            if sym in METAL_SET:
                has_metal = True
            if atom.GetFormalCharge() != 0:
                has_charge = True

        # Check for aromatic bonds (conjugation)
        for bond in mol.GetBonds():
            if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() == 1.5:
                has_aromatic = True
                break

    return AbstractFeatures(
        is_fluorinated=has_f,
        is_electron_deficient=bool(all_abstract_detected & {
            "electron_withdrawing_nitro",
            "electron_withdrawing_cyano",
            "electron_withdrawing_cf3",
            "electron_withdrawing_sulfonyl",
        }),
        is_electron_rich=bool(all_abstract_detected & {
            "electron_donating_amino",
            "electron_donating_hydroxyl",
            "electron_donating_alkoxy",
        }),
        has_hydrogen_bond_donor="hbond_donor" in all_abstract_detected,
        has_hydrogen_bond_acceptor=(
            "hbond_acceptor" in all_abstract_detected
            or "hbond_acceptor_broad" in all_abstract_detected
        ),
        is_conjugated=has_aromatic,
        is_metalated=has_metal,
        is_charged=has_charge,
        is_photoswitchable="azo" in all_detected_names,
        is_symmetric=None,          # can't determine without connection points
        has_open_metal_site=None,   # filled from source JSON by metal_enricher
    )


def infer_properties(
    abstract: AbstractFeatures,
    fg: FunctionalGroups,
) -> list[str]:
    """Derive high-level property tags from abstract features and functional groups.

    Parameters
    ----------
    abstract : AbstractFeatures
        Boolean abstract feature flags.
    fg : FunctionalGroups
        MOF-level aggregated functional groups.

    Returns
    -------
    list[str]
        Sorted unique list of inferred property tags.
    """
    props: set[str] = set()

    # From abstract features
    if abstract.is_electron_deficient:
        props.add("electron_deficient")
    if abstract.is_electron_rich:
        props.add("electron_rich")
    if abstract.is_fluorinated:
        props.add("fluorinated")
    if abstract.is_conjugated:
        props.add("conjugated")
    if abstract.has_hydrogen_bond_donor:
        props.add("hbd")
    if abstract.has_hydrogen_bond_acceptor:
        props.add("hba")
    if abstract.is_charged:
        props.add("charged")
    if abstract.is_metalated:
        props.add("metalated")
    if abstract.is_photoswitchable:
        props.add("photoswitchable")

    # From functional groups (rule_based list)
    rb = set(fg.rule_based)
    if rb & {"carboxylic_acid", "carboxylate", "carboxyl_any"}:
        props.add("carboxylate_based")
    if rb & {"imidazole", "pyrazole", "triazole_any",
             "triazole_1_2_3", "triazole_1_2_4", "tetrazole"}:
        props.add("azolate_based")
    if "pyridine" in rb:
        props.add("pyridine_based")

    return sorted(props)


# =====================================================================
# Convenience: full pipeline for a single MOF
# =====================================================================

def enrich_mof_smarts(
    smiles_nodes: list[str],
    smiles_linkers: list[str],
) -> tuple[list[LinkerEnrichment], FunctionalGroups, AbstractFeatures, list[str]]:
    """Run the full SMARTS enrichment pipeline for one MOF.

    Parameters
    ----------
    smiles_nodes : list[str]
        Node SMILES from MOFid.
    smiles_linkers : list[str]
        Linker SMILES from MOFid.

    Returns
    -------
    tuple
        (per_linker, mof_fg, abstract_features, inferred_properties)
    """
    per_linker, mof_fg = enrich_all_linkers(smiles_linkers)
    abstract = compute_abstract_features(smiles_nodes, smiles_linkers)
    properties = infer_properties(abstract, mof_fg)
    return per_linker, mof_fg, abstract, properties


# =====================================================================
# CLI smoke test
# =====================================================================

if __name__ == "__main__":
    import json
    from dataclasses import asdict

    test_cases = [
        ("[O-]C(=O)c1ccncc1", "carboxylate + pyridine linker"),
        ("c1ccc(-c2ccncc2)cc1", "biphenyl + pyridine linker"),
        ("", "empty SMILES"),
        ("[Cu]", "metal-only SMILES"),
    ]

    for smiles, label in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {label}")
        print(f"SMILES: {smiles!r}")
        print(f"{'='*60}")

        result = enrich_single_linker(smiles)
        print(f"  valid:        {result.smiles_valid}")
        print(f"  backbone:     {result.functional_groups.backbone}")
        print(f"  substituents: {result.functional_groups.substituents}")
        print(f"  rule_based:   {result.functional_groups.rule_based}")
        print(f"  counts:       {result.functional_groups.rule_based_counts}")
        print(f"  scaffolds:    {result.core_scaffold}")
        print(f"  heterocycles: {result.heterocycles}")

    # Full MOF-level test
    print(f"\n{'='*60}")
    print("MOF-level aggregation test")
    print(f"{'='*60}")
    linkers = ["[O-]C(=O)c1ccncc1", "c1ccc(-c2ccncc2)cc1"]
    nodes = ["[Cu]"]
    per_linker, mof_fg, abstract, properties = enrich_mof_smarts(nodes, linkers)
    print(f"  MOF backbone:     {mof_fg.backbone}")
    print(f"  MOF substituents: {mof_fg.substituents}")
    print(f"  MOF rule_based:   {mof_fg.rule_based}")
    print(f"  MOF counts:       {mof_fg.rule_based_counts}")
    print(f"  Abstract:         {json.dumps(asdict(abstract), indent=4)}")
    print(f"  Properties:       {properties}")
