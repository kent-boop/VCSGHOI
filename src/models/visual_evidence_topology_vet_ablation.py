"""Isolated Visual Evidence Topology components for V0--V5."""

from typing import Any

from src.models.candidate_evidence_topology_vet_ablation import (
    CandidateEvidenceTopologyVETAblation,
    get_vet_ablation_spec,
)
from src.models.visual_evidence_topology_vet import VisualEvidenceTopology


class VisualEvidenceTopologyVETAblation(VisualEvidenceTopology):
    """Common VET state and helpers used by the layerwise ablation runner."""

    def __init__(
        self,
        *args: Any,
        vet_ablation_mode: str = "path_reweight",
        vet_selection_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        get_vet_ablation_spec(vet_ablation_mode)
        super().__init__(*args, **kwargs)
        self.vet_ablation_mode = str(vet_ablation_mode)
        self.vet_selection_temperature = float(vet_selection_temperature)
        # Replace the original CETI module without changing the surrounding VET.
        self.candidate_evidence = CandidateEvidenceTopologyVETAblation(
            d_model=self.d_model,
            max_context_paths=self.candidate_evidence.max_context_paths,
            dropout=float(kwargs.get("graph_evidence_dropout", kwargs.get("dropout", 0.1))),
            gate_init=float(kwargs.get("graph_evidence_gate_init", 0.0)),
            allow_node_only_context=bool(
                kwargs.get("graph_evidence_allow_node_only_context", False)
            ),
            ablation_mode=self.vet_ablation_mode,
            selection_temperature=self.vet_selection_temperature,
            debug=bool(kwargs.get("graph_evidence_debug", False)),
        )

    @property
    def ablation_spec(self):
        return get_vet_ablation_spec(self.vet_ablation_mode)

    def forward(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "VET internal ablations require layerwise SETM scheduling; "
            "use VisualEvidenceTopologyVETAblationLayerwiseSETM."
        )


def vet_ablation_config_from_args(args: Any) -> dict:
    dropout = getattr(args, "graph_dropout", 0.1)
    graph_oo_mode = getattr(args, "graph_oo_mode", "hybrid")
    if graph_oo_mode == "text_hybrid":
        raise ValueError("VET internal ablation does not allow text_hybrid OO mode")
    return {
        "d_model": getattr(args, "hidden_dim", 256),
        "num_obj_classes": getattr(args, "num_obj_classes", 80),
        "num_edge_types": getattr(args, "graph_num_edge_types", 4),
        "num_node_types": getattr(args, "graph_num_node_types", 3),
        "num_layers": getattr(args, "graph_num_layers", 2),
        "num_heads": getattr(args, "graph_num_heads", 8),
        "k_oo": getattr(args, "graph_k_oo", 3),
        "enable_oo_edges": getattr(args, "graph_enable_oo_edges", True),
        "enable_hh_edges": getattr(args, "graph_enable_hh_edges", False),
        "dropout": dropout,
        "spatial_sigma": getattr(args, "graph_spatial_sigma", 0.5),
        "graph_oo_mode": graph_oo_mode,
        "graph_oo_lambda_geo": getattr(args, "graph_oo_lambda_geo", 1.0),
        "graph_oo_lambda_share": getattr(args, "graph_oo_lambda_share", 0.5),
        "graph_oo_min_score": getattr(args, "graph_oo_min_score", 0.0),
        "graph_oo_fallback_to_spatial": getattr(args, "graph_oo_fallback_to_spatial", True),
        "graph_oo_share_eps": getattr(args, "graph_oo_share_eps", 1e-6),
        "graph_evidence_topk": getattr(args, "graph_evidence_topk", 4),
        "graph_evidence_gate_init": getattr(args, "graph_evidence_gate_init", 0.0),
        "graph_evidence_dropout": getattr(args, "graph_evidence_dropout", dropout),
        "graph_evidence_allow_node_only_context": getattr(
            args, "graph_evidence_allow_node_only_context", False
        ),
        "graph_evidence_debug": getattr(args, "graph_evidence_debug", False),
        "debug": getattr(args, "vet_debug", getattr(args, "graph_debug", False)),
        "store_dense_debug_tensors": getattr(args, "vet_store_dense_debug_tensors", False),
        "vet_ablation_mode": getattr(args, "graph_vet_ablation_mode", "path_reweight"),
        "vet_selection_temperature": getattr(args, "graph_vet_selection_temperature", 1.0),
    }

