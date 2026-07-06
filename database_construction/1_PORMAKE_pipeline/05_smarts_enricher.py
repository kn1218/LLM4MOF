"""Phase 4f: Rule-based Layer 2 enrichment using SMARTS matching.

Applies SMARTS patterns to SMILES strings to detect functional groups,
scaffolds, heterocycles, and abstract features. All results are tagged
as "rule-based" in the Layer 2 semantics.
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import RWMol

from .config import METAL_SET, HALOGEN_SET, TYPICAL_COORDINATION
from .schema import (
    BBRecord,
    Layer1Facts,
    Layer2Semantics,
    FunctionalGroups,
    AbstractFeatures,
    SemanticSource,
)
from .smarts_library import (
    ALL_PATTERNS,
    FUNCTIONAL_GROUPS,
    SUBSTITUENTS,
    SCAFFOLDS,
    HETEROCYCLES,
    ABSTRACT_PATTERNS,
    SmartsPattern,
)


def enrich_layer2(record: BBRecord) -> BBRecord:
    """Apply rule-based SMARTS enrichment to a BBRecord.

    Modifies record.layer2_semantics in-place and returns the record.
    Does NOT overwrite any existing llm_additions.
    """
    facts = record.layer1_facts
    l2 = record.layer2_semantics

    # Build RDKit mol from SMILES for SMARTS matching
    # Try full mol first; if sanitization fails (metals), strip metals and retry
    mol = _get_mol(facts.smiles)
    if mol is None and facts.has_metal and facts.smiles:
        mol = _get_mol_strip_metals(facts.smiles)

    # ── Functional group detection ──
    detected = _match_patterns(mol, ALL_PATTERNS)
    fg_names = sorted(set(p.name for p in detected if p.category == "functional_group"))
    sub_names = sorted(set(p.name for p in detected if p.category == "substituent"))
    scaffold_names = sorted(set(p.name for p in detected if p.category == "scaffold"))
    heterocycle_names = sorted(set(p.name for p in detected if p.category == "heterocycle"))

    # Count matches for non-abstract patterns
    non_abstract = [p for p in ALL_PATTERNS if p.category != "abstract"]
    match_counts = _count_pattern_matches(mol, non_abstract)

    # Separate backbone vs substituent FGs
    # Backbone = scaffolds + heterocycles + core FGs (carboxyl, amine at connection points)
    # Substituents = halogens, methyl, methoxy, CF3, etc.
    backbone = sorted(set(fg_names + scaffold_names + heterocycle_names))
    substituents = sorted(set(sub_names))
    all_rule_based = sorted(set(backbone + substituents))

    # Build counts dict only for detected groups
    rule_based_counts = {name: match_counts[name]
                         for name in all_rule_based
                         if name in match_counts}

    l2.functional_groups = FunctionalGroups(
        backbone=backbone,
        substituents=substituents,
        rule_based=all_rule_based,
        rule_based_counts=rule_based_counts,
        llm_additions=l2.functional_groups.llm_additions,  # preserve existing
    )

    # ── Core scaffold ──
    l2.core_scaffold = scaffold_names + heterocycle_names

    # ── Abstract features ──
    abstract_detected = set(p.name for p in detected if p.category == "abstract")

    l2.abstract_features = AbstractFeatures(
        is_fluorinated=_has_element(facts, "F"),
        is_electron_deficient=bool(abstract_detected & {
            "electron_withdrawing_nitro",
            "electron_withdrawing_cyano",
            "electron_withdrawing_cf3",
            "electron_withdrawing_sulfonyl",
        }),
        is_electron_rich=bool(abstract_detected & {
            "electron_donating_amino",
            "electron_donating_hydroxyl",
            "electron_donating_alkoxy",
        }),
        is_symmetric=_check_symmetry(facts),
        is_conjugated=_check_conjugation(facts),
        is_metalated=facts.has_metal,
        has_hydrogen_bond_donor="hbond_donor" in abstract_detected,
        has_hydrogen_bond_acceptor=(
            "hbond_acceptor" in abstract_detected
            or "hbond_acceptor_broad" in abstract_detected
        ),
        is_charged=facts.net_charge != 0,
        is_photoswitchable="azo" in set(p.name for p in detected),
        has_open_metal_site=_check_open_metal_site(facts),
    )

    # ── Source tag ──
    l2.source = SemanticSource.RULE_BASED

    return record


# ── Helpers ────────────────────────────────────────────────────────────

def _get_mol(smiles: Optional[str]) -> Optional[Chem.Mol]:
    """Parse SMILES into RDKit mol, handling [*] wildcards.

    Returns None if sanitization fails (common for metal-containing SMILES).
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


def _match_patterns(
    mol: Optional[Chem.Mol],
    patterns: list[SmartsPattern],
) -> list[SmartsPattern]:
    """Match all SMARTS patterns against a molecule."""
    if mol is None:
        return []

    matched = []
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

    Returns {pattern_name: count} for patterns with >=1 match.
    Uses GetSubstructMatches with uniquify=True for unique match sets.
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


def _has_element(facts: Layer1Facts, element: str) -> bool:
    """Check if an element is present in atom counts."""
    return facts.atom_counts.get(element, 0) > 0


def _check_symmetry(facts: Layer1Facts) -> bool:
    """Heuristic symmetry check.

    A BB is likely symmetric if all connection points bond to the
    same element type and the distance matrix has repeated values.
    """
    cp = facts.connection_points
    if cp.count < 2:
        return False
    # All connection chemistry same?
    if len(set(cp.connection_chemistry)) != 1:
        return False
    # For 2-connected: always "symmetric" in this sense
    if cp.count == 2:
        return True
    # For higher connectivity: check distance matrix regularity
    # All off-diagonal distances similar (within 10%)?
    distances = []
    for i in range(cp.count):
        for j in range(i + 1, cp.count):
            distances.append(cp.distance_matrix[i][j])
    if not distances:
        return False
    mean_d = sum(distances) / len(distances)
    if mean_d == 0:
        return False
    max_deviation = max(abs(d - mean_d) / mean_d for d in distances)
    return max_deviation < 0.15  # 15% tolerance


def _check_conjugation(facts: Layer1Facts) -> bool:
    """Heuristic: conjugated if aromatic bonds exist or DoU is high."""
    aromatic_count = facts.bond_graph.bond_type_counts.get("A", 0)
    return aromatic_count > 0


def _check_open_metal_site(facts: Layer1Facts) -> Optional[bool]:
    """Check if any metal has fewer bonds than its typical coordination number."""
    if not facts.has_metal or not facts.metal_coordination:
        return None
    for mc in facts.metal_coordination:
        typical = TYPICAL_COORDINATION.get(mc.element)
        if typical is not None and mc.coordination_number < typical:
            return True
    return False
