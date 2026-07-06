"""QMOF Enriched v2 Schema — 2-layer architecture following PORMAKE standard.

Design decision: One record per MOF (Option C) with per-linker sub-entries
in layer2_semantics.linker_enrichment[]. This preserves MOF-level properties
(bandgap, density) while enabling per-linker functional group detection.

Mirrors PORMAKE's bb_metadata_v8 schema where applicable:
  layer1_facts      — Deterministic data (CSV + source JSONs)
  layer2_semantics  — Interpreted data (SMARTS + LLM)
  provenance        — Full reproducibility metadata
  validation_report — 14 automated checks
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# =============================================================================
# Layer 1: Facts (deterministic, verifiable)
# =============================================================================

@dataclass
class MetalNode:
    """Metal node data extracted from QMOF source JSONs.

    Maps to PORMAKE's node-specific layer1 fields (nuclearity, geometry,
    metal_coordination) adapted for whole-MOF context.
    """
    metals: list[str] = field(default_factory=list)
    nuclearity: int = 0
    connectivity: Optional[int] = None
    geometry: Optional[str] = None
    sbu_type: Optional[str] = None
    oxidation_states: Optional[dict[str, int]] = None
    ligand_chemistry: list[str] = field(default_factory=list)
    has_open_metal_sites: Optional[bool] = None
    spin_state: Optional[str] = None
    net_charge: int = 0
    # Functional groups from source JSON (metal coordination sphere)
    coordinating_groups: list[str] = field(default_factory=list)
    linker_substituents: list[str] = field(default_factory=list)
    linker_backbone: list[str] = field(default_factory=list)
    metal_terminal_ligands: list[str] = field(default_factory=list)


@dataclass
class Layer1Facts:
    """Deterministic facts extracted from QMOF CSV and source JSONs.

    All fields are reproducible from the source data. No interpretation.
    """
    # Identity
    formula: str = ""

    # Crystal properties (from CSV)
    natoms: Optional[int] = None
    density: Optional[float] = None          # g/cm³
    volume: Optional[float] = None           # Å³
    pld: Optional[float] = None              # pore limiting diameter, Å
    lcd: Optional[float] = None              # largest cavity diameter, Å

    # Topology and symmetry
    topology: Optional[str] = None           # RCSR symbol
    spacegroup: Optional[str] = None
    crystal_system: Optional[str] = None

    # Identifiers
    mofid: Optional[str] = None
    mofkey: Optional[str] = None
    synthesized: Optional[bool] = None
    doi: Optional[str] = None

    # SMILES (from MOFid decomposition in CSV)
    smiles_nodes: list[str] = field(default_factory=list)
    smiles_linkers: list[str] = field(default_factory=list)

    # Per-SMILES validation results
    smiles_nodes_valid: list[bool] = field(default_factory=list)
    smiles_linkers_valid: list[bool] = field(default_factory=list)
    smiles_validation_warnings: list[str] = field(default_factory=list)

    # Bandgap data (from DFT calculations in CSV)
    bandgap_pbe: Optional[float] = None
    bandgap_hle17: Optional[float] = None
    bandgap_hse06_10hf: Optional[float] = None
    bandgap_hse06: Optional[float] = None

    # Metal node (from source JSON)
    metal_node: MetalNode = field(default_factory=MetalNode)


# =============================================================================
# Layer 2: Semantics (interpreted, high confidence)
# =============================================================================

@dataclass
class LinkerFunctionalGroups:
    """Functional groups detected in a single linker SMILES via SMARTS.

    Mirrors PORMAKE's FunctionalGroups structure:
      backbone     = scaffolds + heterocycles + core FGs
      substituents = halogens, methyl, methoxy, CF3, etc.
      rule_based   = union of backbone + substituents
    """
    backbone: list[str] = field(default_factory=list)
    substituents: list[str] = field(default_factory=list)
    rule_based: list[str] = field(default_factory=list)
    rule_based_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class LinkerEnrichment:
    """Per-linker SMARTS enrichment result.

    Each linker SMILES gets its own FG detection block, which are then
    aggregated into the MOF-level functional_groups field.
    """
    smiles: str = ""
    smiles_valid: bool = False
    functional_groups: LinkerFunctionalGroups = field(
        default_factory=LinkerFunctionalGroups
    )
    core_scaffold: list[str] = field(default_factory=list)
    heterocycles: list[str] = field(default_factory=list)


@dataclass
class FunctionalGroups:
    """MOF-level functional groups aggregated from all linkers.

    Same structure as PORMAKE's FunctionalGroups with added llm_additions.
    """
    backbone: list[str] = field(default_factory=list)
    substituents: list[str] = field(default_factory=list)
    rule_based: list[str] = field(default_factory=list)
    rule_based_counts: dict[str, int] = field(default_factory=dict)
    llm_additions: list[str] = field(default_factory=list)


@dataclass
class AbstractFeatures:
    """Boolean abstract features computed from combined linker + node SMILES.

    Matches PORMAKE's abstract_features exactly, plus has_open_metal_site.
    """
    is_fluorinated: Optional[bool] = None
    is_electron_deficient: Optional[bool] = None
    is_electron_rich: Optional[bool] = None
    is_symmetric: Optional[bool] = None
    is_conjugated: Optional[bool] = None
    is_metalated: Optional[bool] = None
    has_hydrogen_bond_donor: Optional[bool] = None
    has_hydrogen_bond_acceptor: Optional[bool] = None
    is_charged: Optional[bool] = None
    is_photoswitchable: Optional[bool] = None
    has_open_metal_site: Optional[bool] = None


@dataclass
class Layer2Semantics:
    """Interpreted/enriched data from SMARTS matching and LLM.

    Contains both per-linker detail and MOF-level aggregations.
    """
    source: str = "smarts_rules"   # "smarts_rules", "smarts_rules+llm"

    # Per-linker enrichment (the detailed view)
    linker_enrichment: list[LinkerEnrichment] = field(default_factory=list)

    # MOF-level aggregated FGs (union across all linkers)
    functional_groups: FunctionalGroups = field(default_factory=FunctionalGroups)
    core_scaffold: list[str] = field(default_factory=list)

    # Abstract features (computed from all SMILES combined)
    abstract_features: AbstractFeatures = field(default_factory=AbstractFeatures)

    # Coordinating groups (from source JSON metal node chemistry)
    coordinating_groups: list[str] = field(default_factory=list)

    # Inferred properties (derived from abstract features + structural)
    inferred_properties: list[str] = field(default_factory=list)

    # LLM-generated fields
    readable_name: Optional[str] = None
    design_hints: Optional[str] = None


# =============================================================================
# Provenance
# =============================================================================

@dataclass
class Provenance:
    """Full reproducibility metadata for the enrichment pipeline.

    Tracks source files, tool versions, and which source each field came from.
    """
    pipeline_version: str = "2.0.0"
    schema_version: str = "2.0.0"
    generated_at: str = ""

    # Source file tracking
    source_csv: str = "qmof.csv"
    source_csv_row: Optional[int] = None
    source_json: Optional[str] = None
    source_json_sha256: Optional[str] = None

    # Tool versions (populated at runtime)
    tool_versions: dict[str, str] = field(default_factory=dict)

    # SMARTS library info
    smarts_library: str = "pormake_bb_pipeline_v1.0"
    smarts_pattern_count: int = 0

    # Method tracking
    layer1_method: str = "csv_extraction+source_json"
    layer2_method: str = "smarts_rules"  # or "smarts_rules+llm"

    # Field-level provenance: which source each field came from
    field_sources: dict[str, str] = field(default_factory=dict)


# =============================================================================
# Validation
# =============================================================================

@dataclass
class IssueLogEntry:
    """A single issue found during enrichment or validation.

    Every quality issue is logged with field, severity, and source for
    full traceability.
    """
    field: str = ""
    severity: str = "info"             # "error", "warning", "info"
    message: str = ""
    source: Optional[str] = None       # "csv", "source_json", "smarts", "llm"

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "severity": self.severity,
            "message": self.message,
            "source": self.source,
        }


@dataclass
class ValidationReport:
    """14 automated checks adapted for QMOF MOF-level records.

    Status logic:
      - "pass"    = all checks True, no errors
      - "warning" = all checks True but warnings present
      - "error"   = at least one check False (error-severity)
    """
    status: str = "pass"

    # Named boolean checks (14 total)
    checks: dict[str, bool] = field(default_factory=lambda: {
        "formula_nonempty": False,
        "has_smiles_nodes": False,
        "has_smiles_linkers": False,
        "smiles_nodes_all_valid": False,
        "smiles_linkers_all_valid": False,
        "has_metal_node_data": False,
        "metal_composition_consistent": False,
        "topology_present": False,
        "bandgap_nonneg": False,
        "density_positive": False,
        "volume_positive": False,
        "fg_detection_ran": False,
        "source_json_available": False,
        "provenance_complete": False,
    })

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Per-record issue log (all issues found during enrichment)
    issue_log: list[IssueLogEntry] = field(default_factory=list)

    def compute_status(self) -> str:
        """Derive status from checks and errors."""
        # Error-severity checks: formula, bandgap, density, volume
        error_checks = [
            "formula_nonempty", "bandgap_nonneg",
            "density_positive", "volume_positive",
        ]
        has_error = any(
            not self.checks.get(c, True) for c in error_checks
        )
        if has_error or self.errors:
            self.status = "error"
        elif self.warnings:
            self.status = "warning"
        else:
            self.status = "pass"
        return self.status


# =============================================================================
# Top-level record
# =============================================================================

@dataclass
class QMOFRecordV2:
    """Top-level QMOF enriched record following PORMAKE's 2-layer architecture.

    One record per MOF with per-linker sub-entries for functional group detail.
    """
    qmof_id: str = ""
    record_type: str = "mof"          # always "mof" (vs PORMAKE's "edge"/"node")

    layer1_facts: Layer1Facts = field(default_factory=Layer1Facts)
    layer2_semantics: Layer2Semantics = field(default_factory=Layer2Semantics)
    provenance: Provenance = field(default_factory=Provenance)
    validation_report: ValidationReport = field(default_factory=ValidationReport)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict.

        Uses dataclasses.asdict() with custom handling for IssueLogEntry.
        """
        d = asdict(self)
        # Ensure issue_log entries are plain dicts
        if "validation_report" in d and "issue_log" in d["validation_report"]:
            d["validation_report"]["issue_log"] = [
                entry if isinstance(entry, dict) else asdict(entry)
                for entry in d["validation_report"]["issue_log"]
            ]
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "QMOFRecordV2":
        """Reconstruct from a dict (e.g., loaded from JSON).

        Handles nested dataclass reconstruction.
        """
        # Reconstruct MetalNode
        l1_data = d.get("layer1_facts", {})
        mn_data = l1_data.pop("metal_node", {})
        metal_node = MetalNode(**mn_data) if mn_data else MetalNode()

        # Reconstruct Layer1Facts
        layer1 = Layer1Facts(**l1_data, metal_node=metal_node)

        # Reconstruct Layer2Semantics
        l2_data = d.get("layer2_semantics", {})

        # Reconstruct linker_enrichment list
        le_list = []
        for le_data in l2_data.pop("linker_enrichment", []):
            fg_data = le_data.pop("functional_groups", {})
            fg = LinkerFunctionalGroups(**fg_data)
            le_list.append(LinkerEnrichment(**le_data, functional_groups=fg))

        # Reconstruct MOF-level FGs
        fg_data = l2_data.pop("functional_groups", {})
        mof_fg = FunctionalGroups(**fg_data)

        # Reconstruct abstract features
        af_data = l2_data.pop("abstract_features", {})
        abstract = AbstractFeatures(**af_data)

        layer2 = Layer2Semantics(
            **l2_data,
            linker_enrichment=le_list,
            functional_groups=mof_fg,
            abstract_features=abstract,
        )

        # Reconstruct Provenance
        prov_data = d.get("provenance", {})
        provenance = Provenance(**prov_data)

        # Reconstruct ValidationReport
        vr_data = d.get("validation_report", {})
        issue_log_data = vr_data.pop("issue_log", [])
        issue_log = [
            IssueLogEntry(**entry) if isinstance(entry, dict) else entry
            for entry in issue_log_data
        ]
        validation = ValidationReport(**vr_data, issue_log=issue_log)

        return cls(
            qmof_id=d.get("qmof_id", ""),
            record_type=d.get("record_type", "mof"),
            layer1_facts=layer1,
            layer2_semantics=layer2,
            provenance=provenance,
            validation_report=validation,
        )


# =============================================================================
# Factory helpers
# =============================================================================

def make_empty_record(qmof_id: str) -> QMOFRecordV2:
    """Create a minimal valid record with defaults."""
    return QMOFRecordV2(
        qmof_id=qmof_id,
        provenance=Provenance(
            generated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


def make_provenance(
    source_csv_row: Optional[int] = None,
    source_json: Optional[str] = None,
    source_json_sha256: Optional[str] = None,
    layer2_method: str = "smarts_rules",
    smarts_pattern_count: int = 0,
) -> Provenance:
    """Create a Provenance with current tool versions."""
    import sys
    tool_versions = {"python": sys.version.split()[0]}

    try:
        import rdkit
        tool_versions["rdkit"] = rdkit.__version__
    except (ImportError, AttributeError):
        pass

    try:
        import pandas
        tool_versions["pandas"] = pandas.__version__
    except (ImportError, AttributeError):
        pass

    try:
        import networkx
        tool_versions["networkx"] = networkx.__version__
    except (ImportError, AttributeError):
        pass

    return Provenance(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_csv_row=source_csv_row,
        source_json=source_json,
        source_json_sha256=source_json_sha256,
        tool_versions=tool_versions,
        layer2_method=layer2_method,
        smarts_pattern_count=smarts_pattern_count,
    )
