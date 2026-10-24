"""Isolated STIP entry for VET internal ablations V0--V5.

The detector, LPT, SETM, decoder, criterion, and post-process are inherited
unchanged from B5. Only the VET reasoner is replaced.
"""

from src.models.stip_b5_setm_ablation import (
    STIP as _B5STIP,
    STIPCriterion,
    STIPPostProcess,
)
from src.models.visual_evidence_topology_vet_ablation_layerwise_setm import (
    build_vet_internal_ablation_reasoner,
)


class STIP(_B5STIP):
    """B5 layerwise architecture with VET evidence as the only ablated part."""

    def __init__(self, args, detr, blip, detr_matcher):
        self._validate_frozen_ablation_config(args)
        super().__init__(args, detr=detr, blip=blip, detr_matcher=detr_matcher)
        self.vet = build_vet_internal_ablation_reasoner(args)
        self.vet_internal_ablation_mode = args.graph_vet_ablation_mode
        print(
            "[STIP-VET-Internal-Ablation] "
            f"mode={self.vet_internal_ablation_mode}, "
            f"K={getattr(args, 'graph_evidence_topk', 4)}, "
            f"visual_layers={getattr(args, 'graph_num_layers', 2)}, "
            "SETM=stage4 matcher after every visual layer, LPT=fixed"
        )

    @staticmethod
    def _validate_frozen_ablation_config(args) -> None:
        errors = []
        if int(getattr(args, "setm_ablation_stage", 5)) != 5:
            errors.append("setm_ablation_stage must be 5")
        if not bool(getattr(args, "use_vet", True)):
            errors.append("use_vet must be True")
        if not bool(getattr(args, "use_lpt", True)):
            errors.append("use_lpt must be True")
        if not bool(getattr(args, "use_setm_ablation", True)):
            errors.append("use_setm_ablation must be True")
        if bool(getattr(args, "graph_text_dynamic_retrieval", False)):
            errors.append("graph_text_dynamic_retrieval must be False")
        if not bool(getattr(args, "graph_layerwise_sync_write_visual", True)):
            errors.append("graph_layerwise_sync_write_visual must be True")
        if not bool(getattr(args, "graph_layerwise_sync_write_text", True)):
            errors.append("graph_layerwise_sync_write_text must be True")
        if bool(getattr(args, "graph_layerwise_sync_final_matching", False)):
            errors.append("graph_layerwise_sync_final_matching must be False")
        if errors:
            raise ValueError("Invalid frozen VET ablation configuration: " + "; ".join(errors))

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out["vet_internal_ablation_debug"] = {
            "model_variant": "VET_INTERNAL_ABLATION",
            "mode": self.vet_internal_ablation_mode,
            "setm_ablation_stage": self.setm_ablation_stage,
            "layerwise_setm": self.setm_layerwise,
            "graph_num_layers": int(getattr(self.args, "graph_num_layers", 2)),
            "graph_evidence_topk": int(getattr(self.args, "graph_evidence_topk", 4)),
            "fixed_lpt": True,
            "fixed_setm": True,
            "fixed_classifier": True,
        }
        return out

