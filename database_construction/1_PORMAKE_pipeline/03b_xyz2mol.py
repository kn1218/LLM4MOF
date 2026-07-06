"""
xyz2mol: Converts XYZ coordinates to RDKit Mol object
Based on the implementation by Jan H. Jensen et al.
"""
import sys
import copy
from collections import defaultdict
import numpy as np
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Atomic radii (Angstrom) - standard covalent radii
# Using a simplified set for common elements in MOFs
COVALENT_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 
    'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 
    'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 
    'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16, 
    'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 
    'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40, 
    'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 
    'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 
    'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 
    'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 
    'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40, 
    'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 
    'Pa': 2.00, 'U': 1.96, 'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69
}

def get_covalent_radius(symbol):
    return COVALENT_RADII.get(symbol, 1.5) # Default to 1.5 if unknown

def xyz2AC(atoms, xyz):
    """
    Generate Adjacency Matrix (AC) from coordinates.
    Simple distance check: d < r1 + r2 + tolerance
    tolerance is 0.4 Angstrom usually.
    """
    n_atoms = len(atoms)
    AC = np.zeros((n_atoms, n_atoms), dtype=int)
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r_sum = get_covalent_radius(atoms[i]) + get_covalent_radius(atoms[j])
            dist = np.linalg.norm(xyz[i] - xyz[j])
            
            # Tolerance factor. 1.3 is a generous multiplier for bond formation
            # Standard xyz2mol uses (r1+r2)*1.3 or similar logic
            # Here we use standard RDKit-like cutoff: r1+r2 + 0.4
            if dist < (r_sum + 0.4):
                AC[i, j] = 1
                AC[j, i] = 1
                
    return AC

def AC2mol(atoms, AC, xyz, charge=0, use_graph=True, allow_charged_fragments=True):
    """
    Convert Adjacency Matrix to RDKit Mol.
    Note: A full xyz2mol implementations solves for optimal bond orders/charges.
    This is a simplified version that relies on RDKit's ability to guess or simple single bonds initially.
    
    For MOF nodes with metals, strict valence rules are often violated.
    We will create a specific version that handles 'X' atoms and metals leniently.
    """
    
    mol = Chem.RWMol()
    atom_map = {}
    
    for i, symbol in enumerate(atoms):
        if symbol == "X":
            # For connectivity points, we can treat them as dummy atoms
            a = Chem.Atom(0) 
            a.SetIsotope(0)
            a.SetNoImplicit(True) # Don't add hydrogens to dummy
        else:
            a = Chem.Atom(symbol)
            
        idx = mol.AddAtom(a)
        atom_map[i] = idx
        
    # Add bonds from AC
    # We start with SINGLE bonds.
    # Advanced: Heuristic to upgrade to Double/Triple based on valency?
    # For now, simplistic connectivity is better than the "hallucinated" bonds.
    n_atoms = len(atoms)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if AC[i, j] == 1:
                # Check for O-O bonds in metal clusters presence
                # Heuristic: If both are O, and both are bonded to a Metal... 
                # This is hard to do without more graph analysis. 
                # Ideally the radii check + no artificial bond list solves it.
                mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.SINGLE)

    # Set 3D coordinates
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        conf.SetAtomPosition(atom_map[i], (float(xyz[i][0]), float(xyz[i][1]), float(xyz[i][2])))
    mol.AddConformer(conf)
    
    return mol

def xyz2mol(atoms, xyz, charge=0, use_huckel=False, verbose=False):
    """
    Main driver function.
    """
    AC = xyz2AC(atoms, xyz)
    mol = AC2mol(atoms, AC, xyz, charge)
    
    # Optional: Sanitization
    # But for metals, we might want to skip standard valency checks
    try:
        # We try to sanitize to compute valences, but catch errors
        Chem.SanitizeMol(mol)
    except Exception as e:
        if verbose:
            print(f"Sanitization warning: {e}")
            
    return mol
