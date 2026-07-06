"""Phase 4f: SMARTS pattern library for functional group detection.

Curated SMARTS patterns organized by category. Each pattern is tagged
with a canonical name and category for consistent Layer 2 annotation.

Sources:
  - RDKit built-in patterns (Functional_Group_Hierarchy.txt)
  - Daylight SMARTS theory manual
  - SMARTS-RX database
  - Manual curation for MOF-relevant groups

Design:
  - Patterns are ordered from most specific to most general within categories
  - Each pattern has a unique canonical name (lowercase_with_underscores)
  - Backbone vs substituent classification is handled by the enrichment engine
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmartsPattern:
    """A single SMARTS pattern with metadata."""
    name: str           # canonical name (e.g., "carboxylic_acid")
    smarts: str         # SMARTS string
    category: str       # "functional_group", "scaffold", "substituent", "heterocycle", "abstract"
    description: str    # human-readable description


# ── Functional Groups ──────────────────────────────────────────────────

FUNCTIONAL_GROUPS: list[SmartsPattern] = [
    # Carboxyl / Carboxylate
    # Note: PORMAKE XYZ encodes resonance-delocalized C-O bonds as "A" (aromatic),
    # so RDKit sees carboxylate as aromatic c(o)(o) not C(=O)(O-). We use bond-
    # order-agnostic patterns [#6]([#8])([#8]) alongside textbook patterns.
    SmartsPattern("carboxylic_acid", "[CX3](=O)[OX2H1]", "functional_group",
                  "Carboxylic acid (-COOH)"),
    SmartsPattern("carboxylate", "[$([CX3](=O)[OX1-]),$([#6]([#8])(~[#8])~[#6,#7])]", "functional_group",
                  "Carboxylate anion (-COO-) or aromatic-encoded carboxylate"),
    SmartsPattern("carboxyl_any", "[$([CX3](=O)[OX2H1,OX1-]),$([#6X3](~[#8])(~[#8]))]", "functional_group",
                  "Any carboxyl group (acid, anion, or aromatic-encoded)"),

    # Amines
    SmartsPattern("primary_amine", "[NX3;H2;!$(NC=O);!$(NS=O)]", "functional_group",
                  "Primary amine (-NH2), not amide/sulfonamide"),
    SmartsPattern("secondary_amine", "[NX3;H1;!$(NC=O);!$(NS=O)]([#6])[#6]", "functional_group",
                  "Secondary amine (-NHR)"),
    SmartsPattern("tertiary_amine", "[NX3;H0;!$(NC=O);!$(NS=O)]([#6])([#6])[#6]", "functional_group",
                  "Tertiary amine (-NR3)"),
    SmartsPattern("amine_any", "[NX3;H2,H1;!$(NC=O);!$(NS=O)]", "functional_group",
                  "Any amine (primary or secondary)"),

    # Amide
    SmartsPattern("amide", "[NX3][CX3](=[OX1])[#6]", "functional_group",
                  "Amide (-CONR)"),
    SmartsPattern("primary_amide", "[NX3H2][CX3](=[OX1])", "functional_group",
                  "Primary amide (-CONH2)"),

    # Hydroxyl
    SmartsPattern("hydroxyl", "[OX2H1][#6;!$(C=O)]", "functional_group",
                  "Hydroxyl (-OH), not carboxylic acid"),
    SmartsPattern("phenol", "[OX2H1]c", "functional_group",
                  "Phenol (ArOH)"),

    # Nitro (standard and aromatic-encoded forms)
    SmartsPattern("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-]),$([#7]([#8])([#8]))]", "functional_group",
                  "Nitro group (-NO2)"),

    # Cyano / Nitrile
    SmartsPattern("nitrile", "[CX2]#[NX1]", "functional_group",
                  "Nitrile / cyano (-C#N)"),

    # Thiol
    SmartsPattern("thiol", "[SX2H1]", "functional_group",
                  "Thiol (-SH)"),

    # Sulfonate (standard + aromatic-encoded)
    SmartsPattern("sulfonic_acid", "[SX4](=O)(=O)[OX2H1]", "functional_group",
                  "Sulfonic acid (-SO3H)"),
    SmartsPattern("sulfonate", "[$([SX4](=O)(=O)[OX1-]),$([#16](~[#8])(~[#8])(~[#8]))]", "functional_group",
                  "Sulfonate (-SO3-) or aromatic-encoded"),
    SmartsPattern("sulfonamide", "[$([SX4](=O)(=O)[NX3]),$([#16](~[#8])(~[#8])(~[#7]))]", "functional_group",
                  "Sulfonamide (-SO2NR)"),

    # Phosphonate (standard + aromatic-encoded)
    SmartsPattern("phosphonic_acid", "[PX4](=O)([OX2H1])[OX2H1]", "functional_group",
                  "Phosphonic acid (-PO(OH)2)"),
    SmartsPattern("phosphonate", "[$([PX4](=O)([OX1-])[OX1-]),$([#15](~[#8])(~[#8])(~[#8]))]", "functional_group",
                  "Phosphonate (-PO3^2-) or aromatic-encoded"),

    # Ether
    SmartsPattern("ether", "[OX2]([#6;!$(C=O)])[#6;!$(C=O)]", "functional_group",
                  "Ether (C-O-C)"),
    SmartsPattern("aryl_ether", "[OX2](c)c", "functional_group",
                  "Aryl ether (Ar-O-Ar)"),

    # Ester
    SmartsPattern("ester", "[CX3](=O)[OX2][#6]", "functional_group",
                  "Ester (-COOR)"),

    # Ketone
    SmartsPattern("ketone", "[CX3](=O)([#6])[#6]", "functional_group",
                  "Ketone (R-CO-R)"),

    # Aldehyde
    SmartsPattern("aldehyde", "[CX3H1](=O)[#6]", "functional_group",
                  "Aldehyde (-CHO)"),

    # Imine / Schiff base
    SmartsPattern("imine", "[CX3;$([C]([#6])[#6])]=[NX2][#6]", "functional_group",
                  "Imine / Schiff base (C=N-R)"),
    SmartsPattern("imine_any", "[CX3]=[NX2]", "functional_group",
                  "Any C=N bond (incl. hydrazones)"),

    # Azo
    SmartsPattern("azo", "[NX2]=[NX2]", "functional_group",
                  "Azo group (-N=N-)"),
    SmartsPattern("azoxy", "[NX2]=[NX3+]([O-])", "functional_group",
                  "Azoxy group"),

    # Hydrazide
    SmartsPattern("hydrazide", "[NX3][NX3][CX3](=O)", "functional_group",
                  "Hydrazide (-CONHNH-)"),
    SmartsPattern("hydrazine", "[NX3H2][NX3H2]", "functional_group",
                  "Hydrazine (-NHNH-)"),

    # Urea / Thiourea
    SmartsPattern("urea", "[NX3][CX3](=[OX1])[NX3]", "functional_group",
                  "Urea (-NHCONH-)"),
    SmartsPattern("thiourea", "[NX3][CX3](=[SX1])[NX3]", "functional_group",
                  "Thiourea (-NHCSNH-)"),

    # Boronic acid / Boronate ester
    SmartsPattern("boronic_acid", "[BX3]([OX2H1])[OX2H1]", "functional_group",
                  "Boronic acid (-B(OH)2)"),
    SmartsPattern("boronate_ester", "[BX3]([OX2][#6])[OX2][#6]", "functional_group",
                  "Boronate ester (-B(OR)2)"),
    SmartsPattern("boron_any", "[#5]", "functional_group",
                  "Any boron atom"),

    # Isocyanate / Isothiocyanate
    SmartsPattern("isocyanate", "[NX2]=[CX2]=[OX1]", "functional_group",
                  "Isocyanate (-N=C=O)"),
    SmartsPattern("isothiocyanate", "[NX2]=[CX2]=[SX1]", "functional_group",
                  "Isothiocyanate (-N=C=S)"),

    # Anhydride
    SmartsPattern("anhydride", "[CX3](=O)[OX2][CX3](=O)", "functional_group",
                  "Anhydride (-CO-O-CO-)"),

    # Thioether
    SmartsPattern("thioether", "[SX2]([#6])[#6]", "functional_group",
                  "Thioether (C-S-C)"),

    # Sulfoxide / Sulfone
    SmartsPattern("sulfoxide", "[SX3](=O)([#6])[#6]", "functional_group",
                  "Sulfoxide (-SO-)"),
    SmartsPattern("sulfone", "[SX4](=O)(=O)([#6])[#6]", "functional_group",
                  "Sulfone (-SO2-)"),
]


# ── Substituents ──────────────────────────────────────────────────────

SUBSTITUENTS: list[SmartsPattern] = [
    SmartsPattern("methyl", "[CH3;!$(C=O)]", "substituent",
                  "Methyl group (-CH3)"),
    SmartsPattern("trifluoromethyl", "[CX4](F)(F)F", "substituent",
                  "Trifluoromethyl (-CF3)"),
    SmartsPattern("methoxy", "[OX2]([CH3])[#6]", "substituent",
                  "Methoxy (-OCH3)"),
    SmartsPattern("tert_butyl", "[CX4]([CH3])([CH3])[CH3]", "substituent",
                  "Tert-butyl (-C(CH3)3)"),

    # Halogens
    SmartsPattern("fluorine", "[FX1]", "substituent", "Fluorine (-F)"),
    SmartsPattern("chlorine", "[ClX1]", "substituent", "Chlorine (-Cl)"),
    SmartsPattern("bromine", "[BrX1]", "substituent", "Bromine (-Br)"),
    SmartsPattern("iodine", "[IX1]", "substituent", "Iodine (-I)"),

    # Vinyl / Acetylene
    SmartsPattern("vinyl", "[CX3]=[CX3]", "substituent",
                  "Vinyl / alkene (C=C)"),
    SmartsPattern("acetylene", "[CX2]#[CX2]", "substituent",
                  "Acetylene / alkyne (C#C)"),
    SmartsPattern("butadiyne", "[CX2]#[CX2][CX2]#[CX2]", "substituent",
                  "Butadiyne / diacetylene (C#C-C#C)"),
    SmartsPattern("nitrile_sub", "[CX2]#[NX1]", "substituent",
                  "Nitrile / cyano (-C#N) as substituent"),
]


# ── Core Scaffolds ────────────────────────────────────────────────────

SCAFFOLDS: list[SmartsPattern] = [
    # Specific polycyclic systems (most specific first)
    SmartsPattern("porphyrin",
                  "[#7]1~[#6]~[#6]~[#6]~1~[#6]~[#6]~1~[#7]~[#6]~[#6]~[#6]~1",
                  "scaffold", "Porphyrin core (partial match)"),
    SmartsPattern("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34", "scaffold",
                  "Pyrene"),
    SmartsPattern("fluorene", "c1ccc2c(c1)Cc1ccccc12", "scaffold",
                  "Fluorene"),
    SmartsPattern("carbazole", "c1ccc2c(c1)[nH]c1ccccc12", "scaffold",
                  "Carbazole"),
    SmartsPattern("acridine", "c1ccc2nc3ccccc3c(c1)c2", "scaffold",
                  "Acridine"),
    SmartsPattern("phenanthroline", "c1cnc2c(c1)ccc1cccnc12", "scaffold",
                  "1,10-Phenanthroline"),
    SmartsPattern("naphthalene", "c1ccc2ccccc2c1", "scaffold",
                  "Naphthalene"),
    SmartsPattern("anthracene", "c1ccc2cc3ccccc3cc2c1", "scaffold",
                  "Anthracene"),
    SmartsPattern("biphenyl", "c1ccccc1-c1ccccc1", "scaffold",
                  "Biphenyl"),
    SmartsPattern("terphenyl", "c1ccc(-c2ccccc2)cc1-c1ccccc1", "scaffold",
                  "Terphenyl (partial)"),

    # Simple rings
    SmartsPattern("benzene_ring", "c1ccccc1", "scaffold",
                  "Benzene ring"),
    SmartsPattern("cyclohexane", "C1CCCCC1", "scaffold",
                  "Cyclohexane ring"),
    SmartsPattern("cyclopentane", "C1CCCC1", "scaffold",
                  "Cyclopentane ring"),
]


# ── Heterocycles ──────────────────────────────────────────────────────

HETEROCYCLES: list[SmartsPattern] = [
    # 6-membered
    SmartsPattern("pyridine", "c1ccncc1", "heterocycle",
                  "Pyridine"),
    SmartsPattern("pyrimidine", "c1ncncc1", "heterocycle",
                  "Pyrimidine (1,3-diazine)"),
    SmartsPattern("pyridazine", "c1ccnnc1", "heterocycle",
                  "Pyridazine (1,2-diazine)"),
    SmartsPattern("pyrazine", "c1cnccn1", "heterocycle",
                  "Pyrazine (1,4-diazine)"),
    SmartsPattern("triazine", "c1ncncn1", "heterocycle",
                  "1,3,5-Triazine"),
    SmartsPattern("tetrazine", "c1nncnn1", "heterocycle",
                  "s-Tetrazine (1,2,4,5-tetrazine)"),

    # 5-membered
    SmartsPattern("imidazole", "[$([#7]1~[#6]~[#7]~[#6]~[#6]~1)]", "heterocycle",
                  "Imidazole (incl. N-substituted)"),
    SmartsPattern("pyrazole", "c1cc[nH]n1", "heterocycle",
                  "Pyrazole"),
    SmartsPattern("triazole_1_2_4", "c1nnc[nH]1", "heterocycle",
                  "1,2,4-Triazole"),
    SmartsPattern("triazole_1_2_3", "c1cn[nH]n1", "heterocycle",
                  "1,2,3-Triazole"),
    SmartsPattern("triazole_any", "[#7]1~[#7]~[#6]~[#7]~[#6]~1", "heterocycle",
                  "Any triazole (incl. N-substituted)"),
    SmartsPattern("tetrazole", "c1nn[nH]n1", "heterocycle",
                  "Tetrazole"),
    SmartsPattern("oxazole", "c1cocn1", "heterocycle",
                  "Oxazole"),
    SmartsPattern("thiazole", "c1cscn1", "heterocycle",
                  "Thiazole"),
    SmartsPattern("isoxazole", "c1conc1", "heterocycle",
                  "Isoxazole"),
    SmartsPattern("isothiazole", "c1csnc1", "heterocycle",
                  "Isothiazole"),
    SmartsPattern("oxadiazole", "[#6]1~[#7]~[#7]~[#6]~[#8]~1", "heterocycle",
                  "1,3,4-Oxadiazole"),
    SmartsPattern("thiadiazole", "[#6]1~[#7]~[#7]~[#6]~[#16]~1", "heterocycle",
                  "1,3,4-Thiadiazole"),
    SmartsPattern("thiophene", "c1ccsc1", "heterocycle",
                  "Thiophene"),
    SmartsPattern("furan", "c1ccoc1", "heterocycle",
                  "Furan"),
    SmartsPattern("pyrrole", "c1cc[nH]c1", "heterocycle",
                  "Pyrrole"),
    SmartsPattern("piperazine", "C1CNCCN1", "heterocycle",
                  "Piperazine (saturated 6-ring with 2 N)"),

    # Bicyclic heterocycles
    SmartsPattern("quinoline", "c1ccc2ncccc2c1", "heterocycle",
                  "Quinoline"),
    SmartsPattern("isoquinoline", "c1ccc2cnccc2c1", "heterocycle",
                  "Isoquinoline"),
    SmartsPattern("indole", "c1ccc2[nH]ccc2c1", "heterocycle",
                  "Indole"),
    SmartsPattern("benzimidazole", "c1ccc2[nH]cnc2c1", "heterocycle",
                  "Benzimidazole"),
    SmartsPattern("benzothiazole", "c1ccc2scnc2c1", "heterocycle",
                  "Benzothiazole"),
    SmartsPattern("benzoxazole", "c1ccc2ocnc2c1", "heterocycle",
                  "Benzoxazole"),
    SmartsPattern("bipyridine", "c1ccnc(-c2ccccn2)c1", "heterocycle",
                  "2,2'-Bipyridine"),
]


# ── Abstract Feature Patterns ─────────────────────────────────────────

ABSTRACT_PATTERNS: list[SmartsPattern] = [
    # H-bond donors/acceptors
    SmartsPattern("hbond_donor", "[#7H,#8H,#16H]", "abstract",
                  "H-bond donor (NH, OH, SH)"),
    SmartsPattern("hbond_acceptor", "[#7,#8,#16;!H0;v2,v3]", "abstract",
                  "H-bond acceptor (N, O, S with lone pairs)"),
    SmartsPattern("hbond_acceptor_broad",
                  "[$([#7;!$([nH]);!$([#7]~[#8])]),$([#8;!$([OH]);!$([O]~[#7])]),$([#16;!$([SH])])]",
                  "abstract", "Broad H-bond acceptor"),

    # Electron effects
    SmartsPattern("electron_withdrawing_nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-]),$([#7]([#8])([#8]))]",
                  "abstract", "EWG: nitro"),
    SmartsPattern("electron_withdrawing_cyano", "[CX2]#[NX1]", "abstract",
                  "EWG: cyano"),
    SmartsPattern("electron_withdrawing_cf3", "[CX4](F)(F)F", "abstract",
                  "EWG: trifluoromethyl"),
    SmartsPattern("electron_withdrawing_sulfonyl", "[SX4](=O)(=O)", "abstract",
                  "EWG: sulfonyl"),
    SmartsPattern("electron_donating_amino", "[NX3;H2,H1;!$(NC=O)]c", "abstract",
                  "EDG: amino on aromatic"),
    SmartsPattern("electron_donating_hydroxyl", "[OX2H1]c", "abstract",
                  "EDG: hydroxyl on aromatic"),
    SmartsPattern("electron_donating_alkoxy", "[OX2]([CX4])c", "abstract",
                  "EDG: alkoxy on aromatic"),
]


# ── Master list (all patterns) ────────────────────────────────────────

ALL_PATTERNS: list[SmartsPattern] = (
    FUNCTIONAL_GROUPS
    + SUBSTITUENTS
    + SCAFFOLDS
    + HETEROCYCLES
    + ABSTRACT_PATTERNS
)


def get_patterns_by_category(category: str) -> list[SmartsPattern]:
    """Get all patterns in a specific category."""
    return [p for p in ALL_PATTERNS if p.category == category]


def get_pattern_names() -> list[str]:
    """Get all canonical pattern names."""
    return [p.name for p in ALL_PATTERNS]
