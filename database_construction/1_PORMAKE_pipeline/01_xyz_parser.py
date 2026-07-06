"""Phase 3: XYZ file parser for PORMAKE building blocks.

Reads .xyz files with the PORMAKE-specific format:
  Line 1:    atom count (integer)
  Line 2:    space-separated 0-indexed dummy atom indices (may be blank)
  Lines 3+:  Element  x  y  z  (one per atom)
  Remaining: idx1  idx2  bond_type  (bond block)

Bond types: S=Single, D=Double, T=Triple, A=Aromatic
Dummy atoms: Element 'X', mark connection points to other building blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DUMMY_ELEMENT


@dataclass(frozen=True)
class Atom:
    """A single atom from the XYZ file."""
    index: int
    element: str
    x: float
    y: float
    z: float

    @property
    def is_dummy(self) -> bool:
        return self.element == DUMMY_ELEMENT


@dataclass(frozen=True)
class Bond:
    """A single bond from the bond block."""
    atom1: int    # 0-indexed
    atom2: int    # 0-indexed
    bond_type: str  # "S", "D", "T", or "A"


@dataclass(frozen=True)
class ParsedXYZ:
    """Complete parsed representation of a PORMAKE XYZ file."""
    bb_id: str                        # e.g. "E1" or "N100"
    bb_type: str                      # "edge" or "node"
    source_path: str                  # absolute path to the .xyz file
    atom_count_declared: int          # from line 1
    dummy_indices_from_header: tuple[int, ...] | None  # from line 2 (None if blank)
    atoms: tuple[Atom, ...]           # all atoms including X
    bonds: tuple[Bond, ...]           # all bonds from bond block
    real_atoms: tuple[Atom, ...]      # atoms excluding X
    dummy_atoms: tuple[Atom, ...]     # only X atoms
    connection_count: int             # number of X atoms (connectivity)


class XYZParseError(Exception):
    """Raised when an XYZ file cannot be parsed."""


def parse_xyz(filepath: Path) -> ParsedXYZ:
    """Parse a single PORMAKE XYZ file into a ParsedXYZ record.

    Args:
        filepath: Path to a .xyz file (e.g., E1.xyz or N100.xyz).

    Returns:
        ParsedXYZ with all atoms, bonds, and metadata.

    Raises:
        XYZParseError: If the file cannot be parsed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise XYZParseError(f"File not found: {filepath}")

    # Extract BB ID and type from filename
    stem = filepath.stem  # e.g. "E1", "N100"
    bb_id = stem
    if stem.startswith("E"):
        bb_type = "edge"
    elif stem.startswith("N"):
        bb_type = "node"
    else:
        raise XYZParseError(f"Cannot determine BB type from filename: {stem}")

    lines = filepath.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise XYZParseError(f"File too short ({len(lines)} lines): {filepath}")

    # ── Line 1: atom count ──
    try:
        atom_count_declared = int(lines[0].strip())
    except ValueError:
        raise XYZParseError(f"Line 1 is not an integer atom count: '{lines[0].strip()}'")

    # ── Line 2: dummy atom indices (may be blank) ──
    line2 = lines[1].strip()
    if line2:
        try:
            dummy_indices_from_header = tuple(int(x) for x in line2.split())
        except ValueError:
            raise XYZParseError(f"Line 2 cannot be parsed as integer indices: '{line2}'")
    else:
        dummy_indices_from_header = None

    # ── Lines 3 to (2 + atom_count): atom coordinates ──
    atoms = []
    atom_start = 2  # 0-indexed line number
    atom_end = atom_start + atom_count_declared

    if atom_end > len(lines):
        raise XYZParseError(
            f"Expected {atom_count_declared} atoms but file has only "
            f"{len(lines) - atom_start} lines after header"
        )

    for i in range(atom_start, atom_end):
        parts = lines[i].split()
        if len(parts) < 4:
            raise XYZParseError(f"Line {i+1}: expected 'Element x y z', got: '{lines[i].strip()}'")
        element = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise XYZParseError(f"Line {i+1}: cannot parse coordinates: '{lines[i].strip()}'")
        atoms.append(Atom(index=i - atom_start, element=element, x=x, y=y, z=z))

    # ── Validate atom count ──
    if len(atoms) != atom_count_declared:
        raise XYZParseError(
            f"Declared {atom_count_declared} atoms but parsed {len(atoms)}"
        )

    # ── Remaining lines: bond block ──
    bonds = []
    for i in range(atom_end, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue  # skip malformed lines
        try:
            idx1, idx2 = int(parts[0]), int(parts[1])
            bond_type = parts[2]
        except (ValueError, IndexError):
            continue  # skip unparseable lines

        if bond_type not in ("S", "D", "T", "A"):
            raise XYZParseError(
                f"Line {i+1}: unknown bond type '{bond_type}' "
                f"(expected S, D, T, or A)"
            )

        # Validate bond indices
        if idx1 < 0 or idx1 >= atom_count_declared:
            raise XYZParseError(f"Line {i+1}: bond index {idx1} out of range [0, {atom_count_declared})")
        if idx2 < 0 or idx2 >= atom_count_declared:
            raise XYZParseError(f"Line {i+1}: bond index {idx2} out of range [0, {atom_count_declared})")

        bonds.append(Bond(atom1=idx1, atom2=idx2, bond_type=bond_type))

    # ── Separate real atoms from dummy atoms ──
    real_atoms = tuple(a for a in atoms if not a.is_dummy)
    dummy_atoms = tuple(a for a in atoms if a.is_dummy)

    # ── Cross-validate header dummy indices vs X atoms found ──
    dummy_indices_from_scan = tuple(a.index for a in dummy_atoms)
    if dummy_indices_from_header is not None:
        header_set = set(dummy_indices_from_header)
        scan_set = set(dummy_indices_from_scan)
        if header_set != scan_set:
            # Not an error — header might list them differently.
            # We trust the actual X atoms found in coordinates.
            pass

    return ParsedXYZ(
        bb_id=bb_id,
        bb_type=bb_type,
        source_path=str(filepath.resolve()),
        atom_count_declared=atom_count_declared,
        dummy_indices_from_header=dummy_indices_from_header,
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        real_atoms=real_atoms,
        dummy_atoms=dummy_atoms,
        connection_count=len(dummy_atoms),
    )


def parse_all_xyz(bbs_dir: Path) -> list[ParsedXYZ]:
    """Parse all .xyz files in a directory.

    Returns:
        List of ParsedXYZ records, sorted by bb_id.
    """
    bbs_dir = Path(bbs_dir)
    results = []
    errors = []

    for xyz_file in sorted(bbs_dir.glob("*.xyz")):
        try:
            parsed = parse_xyz(xyz_file)
            results.append(parsed)
        except XYZParseError as e:
            errors.append((xyz_file.name, str(e)))

    if errors:
        print(f"Parse errors ({len(errors)}):")
        for fname, err in errors:
            print(f"  {fname}: {err}")

    return results
