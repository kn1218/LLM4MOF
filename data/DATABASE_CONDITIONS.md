# Database Simulation Conditions

Reference document for all evaluation databases used in LLM4MOF.

---

## 1. PORMAKE Markscheme (H2 Storage)

Pre-computed H2 uptake for PORMAKE-assemblable MOFs. GCMC simulations at cryogenic conditions.

| CSV File | Gas | T (K) | P (bar) | Unit | Valid Rows | Target Range |
|----------|-----|--------|---------|------|------------|-------------|
| `total_characteristics_h2_100bar_77K.csv` | H2 | 77 | 100 | cm3(STP)/cm3 | 12,188 | 0.74 -- 616.90 |
| `total_characteristics_h2_5bar_77K.csv` | H2 | 77 | 5 | cm3(STP)/cm3 | 19,997 | 0.07 -- 419.97 |
| `total_characteristics_h2_100bar_77K_mol_kg.csv` | H2 | 77 | 100 | mol/kg | 12,188 | 0.02 -- 369.17 |
| `total_characteristics_h2_5bar_77K_mol_kg.csv` | H2 | 77 | 5 | mol/kg | 19,997 | 0.001 -- 37.16 |
| `total_characteristics_h2_100bar_77K_gperL.csv` | H2 | 77 | 100 | g/L | 12,188 | 0.07 -- 55.49 |
| `total_characteristics_h2_5bar_77K_gperL.csv` | H2 | 77 | 5 | g/L | 19,997 | 0.006 -- 37.77 |

**Geometric columns in all CSVs:** `sa` (surface area), `cv` (cell volume), `density`, `vf` (void fraction), `di` (largest included sphere diameter)

**Note:** The 100 bar CSVs have 12,190 rows (12,188 with valid target); the 5 bar CSVs have 20,000 rows (19,997 valid). The difference is because PORMAKE assemblies vary by which structures converge at each pressure.

---

## 2. hMOF (Hypothetical MOF Database)

Source: Wilmer, C. E.; Leaf, M.; Lee, C. Y.; Farha, O. K.; Hauser, B. G.; Hupp, J. T.; Snurr, R. Q. *Nature Chemistry* **2012**, 4 (2), 83--89.

137,953 hypothetical MOFs generated from 102 building blocks (84 linkers, 5 metal nodes, 13 functional groups, 6 topologies). Our index contains 51,163 entries.

File: `data/hMOF/hmof_index.json`

### Gas Adsorption Properties (GCMC)

| Property | Key in JSON | Gas | T (K) | P (bar) | Unit | Valid Entries | Range |
|----------|------------|-----|--------|---------|------|--------------|-------|
| CH4 uptake | `ch4_uptake_35bar_298K` | CH4 | **298** | **35** | cm3(STP)/cm3 (volumetric) | 51,162 | 0.004 -- 261.25 |
| CO2 uptake | `co2_uptake_2_5bar_298K` | CO2 | **298** | **2.5** | mol/kg | 51,015 | 0.00 -- 43.82 |
| H2 uptake (high-P) | `h2_uptake_100bar_77K` | H2 | **77** | **100** | g/L | 51,110 | 0.0001 -- 58.58 |
| H2 uptake (low-P) | `h2_uptake_2bar_77K` | H2 | **77** | **2** | g/L | 51,100 | 0.00 -- 42.08 |
| Xe loading | `xe_loading_1bar_273K` | Xe | **273** | **1** | mol/kg | 50,392 | 0.00 -- 8.68 |
| Kr loading | `kr_loading_1bar_273K` | Kr | **273** | **1** | mol/kg | 50,924 | 0.0001 -- 6.99 |
| Xe/Kr selectivity | `xekr_selectivity_1bar` | Xe/Kr | **273** | **1** | dimensionless | 50,293 | 0.0002 -- 662.92 |

### Geometric Properties

| Property | Key | Unit | Range |
|----------|-----|------|-------|
| LCD | `lcd` | Angstrom | 2.25 -- 24.75 |
| PLD | `pld` | Angstrom | 1.25 -- 24.75 |
| Surface area | `surface_area_m2g` | m2/g | 0.1 -- 6920.8 |
| Void fraction | `void_fraction` | -- | 0.033 -- 0.960 |
| Density | `density` | g/cm3 | 0.12 -- 4.06 |

### Source Papers by Property (verified via MOFX-DB API, 2026-05-10)

| Gas | DOI | Reference | Force Fields | GCMC Notes |
|-----|-----|-----------|-------------|------------|
| **CH4** (35 bar, 298 K) | 10.1038/nchem.1192 | Wilmer et al., *Nature Chemistry* **2012**, 4, 83--89 | UFF (framework), TraPPE (CH4) | Three-stage GCMC (500/2500/12500 cycles) |
| **CO2** (0.01--2.5 bar, 298 K) | 10.1039/C2EE23201D | Wilmer et al., *Energy Environ. Sci.* **2012**, 5, 9849--9856 | UFF (framework), TraPPE (CO2) | 5 pressure points: 0.01, 0.05, 0.1, 0.5, 2.5 bar |
| **H2** (2 & 100 bar, 77 K) | 10.1021/acs.jpcc.6b08729 | Bobbitt et al., *J. Phys. Chem. C* **2016**, 120, 27328--27341 | UFF (framework), Darkrim-Levesque (H2) | 2 pressure points |
| **Xe/Kr** (1, 5, 10 bar, 273 K) | 10.1039/C2SC01097F | Sikora et al., *Chem. Sci.* **2012**, 3, 2217--2223 | UFF (framework), Hirschfelder/Talu (Xe, Kr) | Binary mixture (20/80 Xe/Kr), 3 pressure points |

**Unit note (verified against the hMOF raw source data):** MOFX-DB stores CH4 in cm3(STP)/cm3, CO2 in mol/kg, H2 in g/L, Xe/Kr in mol/kg. Our `hmof_index.json` stores these values **unchanged** — the build pipeline is pure pass-through, no unit conversion. The previous version of this document (and `config.py`) wrongly asserted "cm3(STP)/g" for CH4/CO2/H2 — that was a mislabel, not a real conversion.

---

## 3. QMOF (Quantum MOF Database)

Source: Rosen, A. S.; Iyer, S. M.; Ray, D.; Yao, Z.; Aspuru-Guzik, A.; Gagliardi, L.; Notestein, J. M.; Snurr, R. Q. *Matter* **2021**, 4 (5), 1578--1597.

DFT-computed electronic properties for experimentally reported MOFs.

File: `data/qmof_index_v2.json`

| Property | Key | Method | Unit | Entries | Range |
|----------|-----|--------|------|---------|-------|
| Band gap (PBE) | `bandgap` | DFT-PBE | eV | 20,373 | 0.0001 -- 6.45 |
| Band gap (HLE17) | `bandgap_hle17` | DFT-HLE17 | eV | varies | -- |
| Band gap (HSE06) | `bandgap_hse06` | DFT-HSE06 | eV | varies | -- |
| Band gap (HSE06-10HF) | `bandgap_hse06_10hf` | DFT-HSE06(10%HF) | eV | varies | -- |

**Note:** We use `bandgap` (PBE functional) as the primary target. PBE systematically underestimates band gaps, but the ranking is generally preserved. The QMOF database contains experimentally synthesized MOFs (not hypothetical), unlike hMOF.

### Other Metadata
- `metals`: list of metal elements
- `functional_groups`: list of functional group tags
- `topology`: framework topology
- `oxidation_states`, `spin_state`, `coordinating_groups`
- `has_open_metal_sites`: boolean (when available)
- `synthesized`: always True (experimentally reported structures)

---

## 4. Condition Summary — What We Use in LLM4MOF Queries

| Query | Database | T | P | Unit in DB | Matches DB? |
|-------|----------|---|---|-----------|-------------|
| "volumetric H2 at 77K and 100 bar" | PORMAKE | 77 K | 100 bar | cm3(STP)/cm3 | Exact |
| "volumetric H2 at 77K and 5 bar" | PORMAKE | 77 K | 5 bar | cm3(STP)/cm3 | Exact |
| "gravimetric H2 at 77K and 100 bar" | PORMAKE | 77 K | 100 bar | mol/kg | Exact |
| "gravimetric H2 at 77K and 5 bar" | PORMAKE | 77 K | 5 bar | mol/kg | Exact |
| "H2 in g/L at 77K and 100 bar" | PORMAKE | 77 K | 100 bar | g/L | Exact |
| "H2 in g/L at 77K and 5 bar" | PORMAKE | 77 K | 5 bar | g/L | Exact |
| "H2 in cm3(STP)/cm3 at 77K and 100 bar" | PORMAKE | 77 K | 100 bar | cm3(STP)/cm3 | Exact |
| "H2 in cm3(STP)/cm3 at 77K and 5 bar" | PORMAKE | 77 K | 5 bar | cm3(STP)/cm3 | Exact |
| "H2 in mol/kg at 77K and 100 bar" | PORMAKE | 77 K | 100 bar | mol/kg | Exact |
| "H2 in mol/kg at 77K and 5 bar" | PORMAKE | 77 K | 5 bar | mol/kg | Exact |
| "volumetric CH4 uptake at 298K and 35 bar" | hMOF | 298 K | 35 bar | cm3(STP)/cm3 | Exact (corrected 2026-05-27; old "gravimetric" query was misleading) |
| "CO2 uptake at 298K and 2.5 bar" | hMOF | 298 K | 2.5 bar | mol/kg | Exact (label was "cm3(STP)/g" pre-2026-05-27 — wrong) |
| "Xe/Kr selectivity at 273K and 1 bar" | hMOF | 273 K | 1 bar | dimensionless | Exact |
| "band gap below 0.1 eV" | QMOF | -- | -- | eV | Exact (DFT-PBE) |
| "band gap above 4 eV" | QMOF | -- | -- | eV | Exact (DFT-PBE) |

All 15 queries match their database conditions exactly. No temperature/pressure mismatch exists.
