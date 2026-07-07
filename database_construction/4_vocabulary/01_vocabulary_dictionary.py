"""
vocabulary_dictionary.py  –  SMARTS-grounded vocabulary dictionary for edge classification.

Hierarchical mapping:  Class → Subclass → SMARTS string.

All patterns are RDKit-compatible.  Anchor-detection patterns use [#0] to
match the dummy atom (atomic number 0) that represents the topological
connection point (*) in SMILES.

This module is imported by generate_edge_metadata.py and should NOT depend
on any LLM-generated data files.
"""

# ============================================================================
# 1.  ANCHOR-DETECTION PATTERNS
#     Detect what atom type is directly bonded to each * (dummy, atomic num 0).
#     These tell the system what kind of metal-node coordination is possible.
# ============================================================================

ANCHOR_SMARTS = {
    # --- Carbon anchors ---
    "aromatic_carbon":   "[#0]-[c]",        # * bonded to aromatic C  (most common)
    "sp3_carbon":        "[#0]-[CX4]",      # * bonded to sp3 C
    "sp2_carbon":        "[#0]-[CX3]",      # * bonded to sp2 C  (vinyl / carbonyl)
    "sp_carbon":         "[#0]-[CX2]",      # * bonded to sp C   (alkyne terminal)

    # --- Heteroatom anchors ---
    "nitrogen_anchor":   "[#0]-[#7]",       # * bonded to any nitrogen
    "oxygen_anchor":     "[#0]-[#8]",       # * bonded to any oxygen
    "sulfur_anchor":     "[#0]-[#16]",      # * bonded to any sulfur

    # --- Coordinating-group anchors (bond context) ---
    "carboxylate_anchor":  "[#0]-[CX3](=O)[O,O-]",         # *-C(=O)O
    "pyridine_anchor":     "[#0]-c1ccncc1",                 # * bonded to pyridine ring
    "amide_anchor":        "[#0]-[NX3][CX3](=[OX1])",       # *-NHC(=O)
    "amine_anchor":        "[#0]-[NX3;H1,H2]",              # *-NH2  or  *-NH-
}


# ============================================================================
# 2.  CORE-SCAFFOLD PATTERNS
#     Identify the primary structural backbone of the linker.
#     Ordered from most specific → least specific so that e.g. "terphenyl"
#     is tested before "biphenyl" before "benzene".
# ============================================================================

SCAFFOLD_SMARTS = {
    # --- Polycyclic aromatic hydrocarbons ---
    "pyrene":        "c1cc2ccc3cccc4ccc(c1)c2c34",
    "anthracene":    "c1ccc2cc3ccccc3cc2c1",
    "phenanthrene":  "c1ccc2c(c1)ccc1ccccc12",
    "naphthalene":   "c1ccc2ccccc2c1",

    # --- Multi-ring linear ---
    "terphenyl":     "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",
    "biphenyl":      "c1ccc(-c2ccccc2)cc1",

    # --- Macrocycles / cages ---
    "porphyrin_core":  "[#7]1~[#6]~[#6]~[#6]~1",    # pyrrole subunit (x4 in porphyrin)
    "cubane":          "[CH,CH0]12[CH,CH0]3[CH,CH0]4[CH,CH0]1[CH,CH0]5[CH,CH0]4[CH,CH0]3[CH,CH0]25",
    "adamantane":      "C1C2CC3CC1CC(C2)C3",         # tricyclo[3.3.1.1] cage
    "barrelene":       "C12C=CC(C=C1)C=C2",
    "triptycene":      "C12c3ccccc3C(c3ccccc31)c1ccccc12",
    "BODIPY":          "[#5]1([F])[F]~[#7]~[#6]~[#6]~[#6]~[#7]~1",  # Boron dipyrromethene core

    # --- Simple rings ---
    "benzene":       "c1ccccc1",
    "cyclohexane":   "C1CCCCC1",
    "cyclopentane":  "C1CCCC1",

    # --- Saturated heterocyclic rings ---
    "piperazine":    "C1CNCCN1",
    "piperidine":    "C1CCNCC1",
    "morpholine":    "C1COCCN1",
    "DABCO":         "C1CN2CCN1CC2",

    # --- Fused heterocycle systems ---
    "benzodioxole":  "c1ccc2OCOc2c1",
    "indane":        "C1Cc2ccccc2C1",
    "fluorene":      "c1ccc2c(c1)Cc1ccccc1-2",
}


# ============================================================================
# 3.  HETEROCYCLE PATTERNS
#     Aromatic nitrogen / sulfur / oxygen heterocycles.
# ============================================================================

HETEROCYCLE_SMARTS = {
    # --- 6-membered N-heterocycles ---
    "pyridine":       "c1ccncc1",
    "pyrimidine":     "c1cncnc1",
    "pyridazine":     "c1ccnnc1",
    "triazine":       "c1ncncn1",
    "tetrazine":      "c1nncnn1",
    "pyrazine":       "c1cnccn1",

    # --- 5-membered N-heterocycles (generic bond queries for N-substituted forms) ---
    "imidazole":      "[#6]1~[#6]~[#7]~[#6]~[#7]~1",   # 2N in 5-ring (matches N-substituted)
    "pyrazole":       "[#6]1~[#6]~[#6]~[#7]~[#7]~1",   # adjacent NN in 5-ring
    "triazole_124":   "[#6]1~[#7]~[#7]~[#6]~[#7]~1",   # 1,2,4-triazole
    "triazole_123":   "[#6]1~[#7]~[#7]~[#7]~[#6]~1",   # 1,2,3-triazole
    "tetrazole":      "[#6]1~[#7]~[#7]~[#7]~[#7]~1",   # 4N in 5-ring

    # --- 5-membered S/O-heterocycles ---
    "thiophene":      "c1ccsc1",
    "furan":          "c1ccoc1",
    "thiazole":       "c1ncsc1",
    "oxazole":        "c1ncoc1",
    "thiadiazole":    "c1nncs1",
    "oxadiazole":     "c1nnco1",
    "isoxazole":      "c1ccno1",
    "isothiazole":    "c1ccns1",

    # --- Fused heterocycles ---
    "benzimidazole":  "c1ccc2[nH]cnc2c1",
    "benzothiazole":  "c1ccc2ncsc2c1",
    "benzoxazole":    "c1ccc2ncoc2c1",
    "quinoline":      "c1ccc2ncccc2c1",
    "isoquinoline":   "c1ccc2cnccc2c1",
    "indole":         "c1ccc2[nH]ccc2c1",
    "benzofuran":     "c1ccc2occc2c1",
    "benzothiophene": "c1ccc2sccc2c1",

    # Imide / diimide ring systems (planar electron-deficient cores)
    "imide_ring":            "C(=O)[NX3]C(=O)",             # generic imide (catches NDI, PMDI)
}


# ============================================================================
# 4.  SUBSTITUENT PATTERNS
#     Pendant / pore-modifier groups that alter electronic / steric properties.
# ============================================================================

SUBSTITUENT_SMARTS = {
    # --- Halogens ---
    "fluorine":        "[FX1]",
    "chlorine":        "[ClX1]",
    "bromine":         "[BrX1]",
    "iodine":          "[IX1]",

    # --- Alkyl groups ---
    "methyl":          "[CH3;$([CH3][c,C])]",          # -CH3 bound to any C
    "ethyl":           "[CH2;$([CH2][CH3])]",           # -CH2CH3
    "isopropyl":       "[CH;$([CH]([CH3])[CH3])]",     # -CH(CH3)2
    "tert_butyl":      "[CX4;$([CX4]([CH3])([CH3])[CH3])]",  # -C(CH3)3
    "methylene_bridge": "[CH2;$([CH2]([c,C])[c,C])]",  # -CH2- bridging

    # --- Oxygen-containing ---
    "hydroxyl":        "[OX2H;!$([OX2H][CX3]=[OX1])]",  # -OH (excludes carboxylic)
    "methoxy":         "[OX2]([CH3])[c,C]",               # -OCH3
    "ether":           "[OD2;!$([OD2][CX3]=[OX1])]([#6])[#6]",  # generic -O-
    "carboxylic_acid": "[CX3](=O)[OX2H]",                # -COOH
    "ketone":          "[CX3](=O)([#6])[#6]",             # C(=O)
    "aldehyde":        "[CX3H1](=O)",                     # -CHO

    # --- Nitrogen-containing ---
    "amine":           "[NX3;H2;!$([NX3][#6]=[#8])]",    # -NH2 (primary, non-amide)
    "secondary_amine": "[NX3;H1;!$([NX3][CX3]=[OX1])]",  # -NH- (non-amide)
    "nitro":           "[$([NX3](=[OX1])=[OX1]),$([NX3+](=[OX1])[O-])]",  # -NO2 (both forms)
    "amide":           "[NX3][CX3](=[OX1])[#6]",          # -NHC(O)-
    "sulfonamide":     "[SX4](=O)(=O)[NX3]",              # -SO2NH-

    # --- Sulfur-containing ---
    "thiol":           "[SX2H]",                          # -SH
    "thioether":       "[#16X2]([#6])[#6]",               # -S- (generic)
    "sulfonic_acid":   "[SX4](=O)(=O)[OX2H]",            # -SO3H
    "sulfonyl":        "[SX4](=O)(=O)",                   # -SO2-

    # --- Fluorinated ---
    "trifluoromethyl": "[CX4](F)(F)F",                    # -CF3
    "perfluoro":       "[CX4](F)(F)",                     # -CF2- (partial)

    # --- Charged / ionic ---
    "pyridinium":      "[n+]",                            # protonated / quaternary N
    "quaternary_N":    "[NX4+]",                          # R4N+
}


# ============================================================================
# 5.  BACKBONE / LINKER-TYPE PATTERNS
#     Identify the primary inter-ring bonds or spacer motifs.
# ============================================================================

LINKER_TYPE_SMARTS = {
    "alkyne":       "[CX2]#[CX2]",                # C≡C
    "butadiyne":    "[CX2]#[CX2][CX2]#[CX2]",    # C≡C-C≡C
    "hexatriyne":   "[CX2]#[CX2][CX2]#[CX2][CX2]#[CX2]",  # C≡C-C≡C-C≡C
    "alkene":       "[CX3]=[CX3]",                 # C=C  (non-aromatic)
    "butadiene":    "C=CC=C",                      # C=C-C=C
    "azo":          "[NX2]=[NX2]",                 # N=N
    "azine":        "[CX3]=[NX2][NX2]=[CX3]",     # C=N-N=C (hydrazone/azine linker)
    "hydrazone":    "[CX3]=[NX2][NX3]",            # C=N-N
    "imine":        "[CX3]=[NX2;!$([NX2][NX2])]", # C=N (not azo)
    "ether_link":   "[#6][OX2][#6]",               # C-O-C
    "thioether_link": "[#6][SX2][#6]",             # C-S-C
    "amide_link":   "[NX3][CX3](=[OX1])",          # -NHC(O)-
    "alkane_chain": "[CX4][CX4]",                  # C(sp3)-C(sp3) saturated
}


# ============================================================================
# 6.  METAL DETECTION PATTERNS
#     Catch metals embedded in the edge SMILES (metallolinkers).
#     Uses element symbol detection since SMARTS metal matching is limited.
# ============================================================================

METALS_IN_SMILES = [
    "Cu", "Ag", "Ni", "Cd", "Mn", "Fe", "Co", "Zn", "Ir", "Rh",
    "Pd", "Pt", "Ru", "In", "Al", "Ti", "Zr", "Hf",
]


# ============================================================================
# 7.  RELATIONAL LOGIC RULES
#     Maps detected substructures → inferred physical properties.
# ============================================================================

RELATIONAL_RULES: dict[str, list[str]] = {
    # --- Halogen substituents ---
    "fluorine":           ["hydrophobic", "electron_deficient", "chemically_inert"],
    "chlorine":           ["hydrophobic", "polarizable"],
    "bromine":            ["hydrophobic", "polarizable", "heavy_atom_effect"],
    "iodine":             ["hydrophobic", "highly_polarizable", "heavy_atom_effect"],
    "trifluoromethyl":    ["hydrophobic", "electron_deficient", "chemically_inert"],
    "perfluoro":          ["hydrophobic", "electron_deficient", "chemically_inert"],

    # --- Nitrogen substituents ---
    "nitro":              ["electron_deficient", "strong_dipole"],
    "amine":              ["hydrogen_bond_donor", "basic", "electron_donating", "hydrophilic"],
    "secondary_amine":    ["hydrogen_bond_donor", "basic", "electron_donating"],
    "amide":              ["hydrogen_bond_donor", "hydrogen_bond_acceptor", "polar"],

    # --- Oxygen substituents ---
    "hydroxyl":           ["hydrogen_bond_donor", "hydrogen_bond_acceptor", "hydrophilic"],
    "methoxy":            ["electron_donating", "hydrogen_bond_acceptor"],
    "carboxylic_acid":    ["hydrogen_bond_donor", "hydrogen_bond_acceptor", "acidic"],
    "ketone":             ["hydrogen_bond_acceptor", "polar"],
    "aldehyde":           ["hydrogen_bond_acceptor", "polar", "reactive"],
    "sulfonic_acid":      ["strongly_acidic", "hydrophilic", "strong_dipole"],
    "sulfonyl":           ["electron_deficient", "hydrophilic", "strong_dipole"],

    # --- Alkyl groups ---
    "methyl":             ["hydrophobic", "steric_bulk", "electron_donating"],
    "ethyl":              ["hydrophobic", "steric_bulk"],
    "isopropyl":          ["hydrophobic", "steric_bulk", "electron_donating"],
    "tert_butyl":         ["hydrophobic", "large_steric_bulk", "electron_donating"],
    "thioether":          ["polarizable", "electron_donating"],

    # --- Backbone / linker types ---
    "alkyne":             ["rigid", "linear", "conjugated"],
    "butadiyne":          ["rigid", "linear", "conjugated", "extended_rod"],
    "hexatriyne":         ["rigid", "linear", "conjugated", "molecular_wire"],
    "alkene":             ["conjugated", "potentially_reactive"],
    "butadiene":          ["conjugated", "potentially_reactive"],
    "azo":                ["photoswitchable", "conjugated"],
    "azine":              ["conjugated", "metal_coordination"],
    "hydrazone":          ["conjugated", "metal_coordination"],
    "imine":              ["conjugated", "metal_coordination"],
    "alkane_chain":       ["flexible", "aliphatic"],
    "ether_link":         ["flexible", "hydrogen_bond_acceptor"],
    "thioether_link":     ["flexible", "polarizable"],
    "amide_link":         ["hydrogen_bond_donor", "hydrogen_bond_acceptor", "polar"],

    # --- Heterocycles ---
    "tetrazine":          ["electron_deficient", "click_reactive", "nitrogen_dense"],
    "triazine":           ["electron_deficient", "nitrogen_dense"],
    "pyridine":           ["basic", "metal_coordination", "hydrogen_bond_acceptor"],
    "pyrimidine":         ["electron_deficient", "metal_coordination"],
    "imidazole":          ["metal_coordination", "hydrogen_bond_donor", "basic"],
    "triazole_124":       ["metal_coordination", "hydrogen_bond_donor", "nitrogen_dense"],
    "triazole_123":       ["metal_coordination", "hydrogen_bond_donor", "nitrogen_dense"],
    "pyrazole":           ["metal_coordination", "hydrogen_bond_donor"],
    "thiophene":          ["electron_rich", "conductive", "conjugated"],
    "furan":              ["electron_rich", "hydrogen_bond_acceptor"],
    "thiadiazole":        ["electron_deficient", "polar", "metal_coordination"],
    "oxadiazole":         ["electron_deficient", "polar"],

    # --- Scaffolds ---
    "porphyrin_core":     ["light_absorbing", "redox_active", "planar", "large_pi_system"],
    "pyrene":             ["light_absorbing", "fluorescent", "large_pi_system"],
    "anthracene":         ["fluorescent", "large_pi_system", "conjugated"],
    "adamantane":         ["rigid", "three_dimensional", "high_free_volume"],
    "cubane":             ["rigid", "three_dimensional", "high_strain"],
    "BODIPY":             ["fluorescent", "light_absorbing", "electron_deficient"],
    "imide_ring":         ["electron_deficient", "planar", "n_type"],

    # --- Charged ---
    "pyridinium":         ["cationic", "electron_deficient"],
    "quaternary_N":       ["cationic"],
}


# ============================================================================
# 8.  RIGIDITY RULES (Refinement #5)
#     Features that force the overall linker to be classified as rigid or flexible.
#     "flexible" overrides any "rigid" from the scaffold.
# ============================================================================

RIGID_FEATURES: set[str] = {
    # Scaffolds that enforce rigidity
    "benzene", "naphthalene", "anthracene", "phenanthrene", "pyrene",
    "biphenyl", "terphenyl", "porphyrin_core", "cubane", "adamantane",
    "barrelene", "triptycene", "fluorene", "DABCO", "BODIPY",
    # Heterocycles that enforce rigidity
    "pyridine", "pyrimidine", "pyrazine", "triazine", "tetrazine",
    "imidazole", "pyrazole", "triazole_124", "triazole_123",
    "thiophene", "furan", "thiadiazole", "oxadiazole", "thiazole",
    "benzimidazole", "benzothiazole", "quinoline", "indole",
    # Linker types that enforce rigidity
    "alkyne", "butadiyne", "hexatriyne",
    # Diimide cores
    "imide_ring",
}

FLEXIBLE_FEATURES: set[str] = {
    # Features that force flexibility (override rigid)
    "alkane_chain", "ether_link", "thioether_link",
    "piperazine", "piperidine", "morpholine",
    "cyclohexane", "cyclopentane",
    "methylene_bridge",
}
