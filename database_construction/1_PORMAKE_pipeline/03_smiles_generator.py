"""Phase 4b: SMILES and SELFIES generation from parsed XYZ data.

Builds RDKit molecules from the explicit bond block, generates SMILES
with connection points represented as [Lr] (Lawrencium, atomic number 103),
then post-processes to replace [Lr] with [*] for standard notation.

Strategy hierarchy:
  1. explicit_bond_block  — Build mol from bond block directly (preferred)
  2. geometry_fallback    — Use xyz2mol for coordinate-based perception
  3. partial_organic      — Strip metals, generate SMILES for organic fragment
  4. failed               — Record failure with reason

Design decisions (from senior agent review):
  - Kekulization OFF: aromatic bonds stay lowercase in SMILES
  - Dummy atoms → Lr (Z=103) to avoid RDKit valence errors
  - Post-process [Lr] → [*] for standard wildcard notation
  - Metal-containing molecules: try explicit bonds first, fall back to partial
  - SELFIES encoding always attempted on successful SMILES
"""

from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from .config import BOND_TYPE_MAP, METAL_SET, LR_ATOMIC_NUM, ATOMIC_WEIGHTS
from .xyz_parser import ParsedXYZ

# RDKit bond type mapping
_RDKIT_BOND_TYPES = {
    "S": Chem.BondType.SINGLE,
    "D": Chem.BondType.DOUBLE,
    "T": Chem.BondType.TRIPLE,
    "A": Chem.BondType.AROMATIC,
}

# Atomic number lookup (subset — extend if needed)
_ELEMENT_TO_ATOMIC_NUM = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95,
}


class SmilesResult:
    """Result of SMILES generation for a building block."""

    __slots__ = ("smiles", "smiles_canonical", "selfies", "method",
                 "net_charge", "error")

    def __init__(
        self,
        smiles: Optional[str] = None,
        smiles_canonical: Optional[str] = None,
        selfies: Optional[str] = None,
        method: str = "failed",
        net_charge: int = 0,
        error: Optional[str] = None,
    ):
        self.smiles = smiles
        self.smiles_canonical = smiles_canonical
        self.selfies = selfies
        self.method = method
        self.net_charge = net_charge
        self.error = error


def generate_smiles(parsed: ParsedXYZ) -> SmilesResult:
    """Generate SMILES and SELFIES for a parsed building block.

    Tries methods in order:
      1. explicit_bond_block — build RDKit mol from bond block
      2. partial_organic     — strip metals, build organic fragment
      3. failed              — return error info

    Args:
        parsed: Output from xyz_parser.parse_xyz().

    Returns:
        SmilesResult with SMILES, canonical SMILES, SELFIES, method used.
    """
    # Strategy 1: Build from explicit bond block
    result = _from_bond_block(parsed)
    if result.smiles is not None:
        return result

    # Strategy 2: Partial organic (strip metals, try organic fragment)
    if any(a.element in METAL_SET for a in parsed.real_atoms):
        result = _partial_organic(parsed)
        if result.smiles is not None:
            return result

    # All strategies failed
    return SmilesResult(
        method="failed",
        error=result.error or "All SMILES generation strategies failed",
    )


def _from_bond_block(parsed: ParsedXYZ) -> SmilesResult:
    """Build RDKit mol directly from the bond block.

    Dummy atoms (X) are represented as Lawrencium (Lr, Z=103) to avoid
    valence errors. Post-processed to [*] in final SMILES.
    """
    try:
        mol = Chem.RWMol()

        # Map: xyz_index → rdkit_index
        idx_map: dict[int, int] = {}

        # Add all atoms
        for atom in parsed.atoms:
            if atom.is_dummy:
                rd_atom = Chem.Atom(LR_ATOMIC_NUM)  # Lr placeholder
                rd_atom.SetNoImplicit(True)
            else:
                atomic_num = _ELEMENT_TO_ATOMIC_NUM.get(atom.element)
                if atomic_num is None:
                    return SmilesResult(
                        error=f"Unknown element: {atom.element}"
                    )
                rd_atom = Chem.Atom(atomic_num)

                # For metals, set no implicit H and zero formal charge
                if atom.element in METAL_SET:
                    rd_atom.SetNoImplicit(True)
                    rd_atom.SetFormalCharge(0)

            rd_idx = mol.AddAtom(rd_atom)
            idx_map[atom.index] = rd_idx

        # Add all bonds (deduplicate — some XYZ files have duplicate entries)
        seen_bonds: set[tuple[int, int]] = set()
        for bond in parsed.bonds:
            bond_key = (min(bond.atom1, bond.atom2), max(bond.atom1, bond.atom2))
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)
            rd_idx1 = idx_map[bond.atom1]
            rd_idx2 = idx_map[bond.atom2]
            bond_type = _RDKIT_BOND_TYPES.get(bond.bond_type)
            if bond_type is None:
                return SmilesResult(
                    error=f"Unknown bond type: {bond.bond_type}"
                )
            mol.AddBond(rd_idx1, rd_idx2, bond_type)

        # Set coordinates (helps with stereo perception)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom in parsed.atoms:
            rd_idx = idx_map[atom.index]
            conf.SetAtomPosition(rd_idx, (atom.x, atom.y, atom.z))
        mol.AddConformer(conf, assignId=True)

        # Try to sanitize — but skip Kekulization for aromatic systems
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
        except Exception as e:
            # Try more permissive: skip properties and kekulize
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=(
                        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                    ),
                )
            except Exception as e2:
                return SmilesResult(
                    error=f"Sanitization failed: {e2}"
                )

        # Generate SMILES (no Kekulization)
        raw_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False)
        if not raw_smiles:
            return SmilesResult(error="MolToSmiles returned empty string")

        # Post-process: [Lr] → [*]
        smiles = _replace_lr_with_wildcard(raw_smiles)

        # Canonical SMILES
        canonical = _canonicalize(smiles)

        # Net charge
        net_charge = _compute_net_charge(mol)

        # SELFIES
        selfies_str = _encode_selfies(smiles)

        return SmilesResult(
            smiles=smiles,
            smiles_canonical=canonical,
            selfies=selfies_str,
            method="explicit_bond_block",
            net_charge=net_charge,
        )

    except Exception as e:
        return SmilesResult(error=f"Bond block method failed: {e}")


def _partial_organic(parsed: ParsedXYZ) -> SmilesResult:
    """Strip metal atoms and their bonds, generate SMILES for organic fragment.

    For metal-containing building blocks where full mol construction fails.
    The resulting SMILES represents only the organic ligand framework.
    """
    try:
        metal_indices = {a.index for a in parsed.real_atoms if a.element in METAL_SET}
        mol = Chem.RWMol()
        idx_map: dict[int, int] = {}

        # Add non-metal atoms only
        for atom in parsed.atoms:
            if atom.index in metal_indices:
                continue
            if atom.is_dummy:
                rd_atom = Chem.Atom(LR_ATOMIC_NUM)
                rd_atom.SetNoImplicit(True)
            else:
                atomic_num = _ELEMENT_TO_ATOMIC_NUM.get(atom.element)
                if atomic_num is None:
                    continue  # skip unknown elements
                rd_atom = Chem.Atom(atomic_num)
            rd_idx = mol.AddAtom(rd_atom)
            idx_map[atom.index] = rd_idx

        # Add bonds between non-metal atoms only (deduplicate)
        seen_bonds: set[tuple[int, int]] = set()
        for bond in parsed.bonds:
            if bond.atom1 in metal_indices or bond.atom2 in metal_indices:
                continue
            if bond.atom1 not in idx_map or bond.atom2 not in idx_map:
                continue
            bond_key = (min(bond.atom1, bond.atom2), max(bond.atom1, bond.atom2))
            if bond_key in seen_bonds:
                continue
            seen_bonds.add(bond_key)
            bond_type = _RDKIT_BOND_TYPES.get(bond.bond_type)
            if bond_type is None:
                continue
            mol.AddBond(idx_map[bond.atom1], idx_map[bond.atom2], bond_type)

        # Set coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom in parsed.atoms:
            if atom.index in idx_map:
                conf.SetAtomPosition(idx_map[atom.index], (atom.x, atom.y, atom.z))
        mol.AddConformer(conf, assignId=True)

        # Sanitize (skip kekulization)
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
        except Exception:
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=(
                        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                    ),
                )
            except Exception as e:
                return SmilesResult(error=f"Partial organic sanitization failed: {e}")

        raw_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False)
        if not raw_smiles:
            return SmilesResult(error="Partial organic MolToSmiles returned empty")

        smiles = _replace_lr_with_wildcard(raw_smiles)
        canonical = _canonicalize(smiles)
        net_charge = _compute_net_charge(mol)
        selfies_str = _encode_selfies(smiles)

        return SmilesResult(
            smiles=smiles,
            smiles_canonical=canonical,
            selfies=selfies_str,
            method="partial_organic",
            net_charge=net_charge,
        )

    except Exception as e:
        return SmilesResult(error=f"Partial organic method failed: {e}")


# ── Helpers ────────────────────────────────────────────────────────────

def _replace_lr_with_wildcard(smiles: str) -> str:
    """Replace [Lr] with [*] in SMILES string."""
    return smiles.replace("[Lr]", "[*]")


def _canonicalize(smiles: str) -> Optional[str]:
    """Get RDKit canonical SMILES. Returns None if parsing fails.

    Uses a round-trip: parse the wildcard SMILES back (with [*] as dummy),
    then regenerate canonical form.
    """
    try:
        # Replace [*] back to [Lr] for RDKit parsing
        lr_smiles = smiles.replace("[*]", "[Lr]")
        mol = Chem.MolFromSmiles(lr_smiles, sanitize=False)
        if mol is None:
            return smiles  # fall back to original
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
        except Exception:
            pass
        canonical = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)
        return _replace_lr_with_wildcard(canonical) if canonical else smiles
    except Exception:
        return smiles


def _compute_net_charge(mol: Chem.Mol) -> int:
    """Compute net formal charge from RDKit mol."""
    return Chem.GetFormalCharge(mol)


def _encode_selfies(smiles: str) -> Optional[str]:
    """Encode SMILES to SELFIES. Returns None if encoding fails.

    Note: SELFIES may not handle all metal-containing SMILES.
    """
    try:
        import selfies as sf
        # SELFIES doesn't handle [*] — replace with [C] temporarily
        # or use a known-safe placeholder
        selfies_smiles = smiles.replace("[*]", "[Au]")  # Gold as placeholder
        encoded = sf.encoder(selfies_smiles)
        if encoded is not None:
            # Replace [Au] back with [*] in SELFIES
            encoded = encoded.replace("[Au]", "[*]")
        return encoded
    except Exception:
        return None
