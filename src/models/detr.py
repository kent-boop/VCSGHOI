# ------------------------------------------------------------------------
# HOTR official code : src/models/detr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR & HOTR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from src.util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from src.lm.BLIP.build_blip import blip_itm

from .backbone import build_backbone
from .detr_matcher import build_matcher
from .hotr_matcher import build_hoi_matcher
from .transformer import build_transformer, build_hoi_transformer
from .criterion import SetCriterion
from .post_process import PostProcess
from .feed_forward import MLP

from .hotr import HOTR
# from .stip_static import STIP, STIPPostProcess, STIPCriterion
from .stip import STIP, STIPPostProcess, STIPCriterion
from .stip_evidence_trace import (
    STIP as STIPEvidenceTrace,
    STIPPostProcess as STIPEvidenceTracePostProcess,
    STIPCriterion as STIPEvidenceTraceCriterion,
)
from .stip_b1_blip_detr import STIP as STIPB1, STIPPostProcess as STIPB1PostProcess, STIPCriterion as STIPB1Criterion
from .stip_b2_svrm import STIP as STIPB2SVRM, STIPPostProcess as STIPB2SVRMPostProcess, STIPCriterion as STIPB2SVRMCriterion
from .stip_b3_vet import STIP as STIPB3VET, STIPPostProcess as STIPB3VETPostProcess, STIPCriterion as STIPB3VETCriterion
from .stip_b4_vet_lpt import STIP as STIPB4VETLPT, STIPPostProcess as STIPB4VETLPTPostProcess, STIPCriterion as STIPB4VETLPTCriterion
from .stip_b5_setm_ablation import STIP as STIPB5SETMAblation, STIPPostProcess as STIPB5SETMAblationPostProcess, STIPCriterion as STIPB5SETMAblationCriterion
from .stip_vet_internal_ablation import STIP as STIPVETInternalAblation, STIPPostProcess as STIPVETInternalAblationPostProcess, STIPCriterion as STIPVETInternalAblationCriterion
from .stip_ossm_box import (
    STIP as STIPOSSMBox,
    STIPPostProcess as STIPOSSMBoxPostProcess,
    STIPCriterion as STIPOSSMBoxCriterion,
)

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    detr = DETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality'] if args.frozen_weights is None else []

    if args.HOIDet and args.STIP_relation_head:
        stip_model_variant = getattr(args, 'model_variant', None) or getattr(args, 'stip_model_variant', 'default')
        if stip_model_variant == 'ossm_box':
            if args.dataset_file not in {'hico-det', 'vcoco'}:
                raise NotImplementedError(
                    "The migrated ossm_box variant supports HICO-DET and V-COCO only"
                )
            if not args.ossm_blip_pretrained:
                raise ValueError(
                    "--stip_model_variant ossm_box requires --ossm_blip_pretrained. "
                    "Use the old BLIP-L retrieval checkpoint for the first comparison."
                )
            blip_pretrained = args.ossm_blip_pretrained
            blip_vit = args.ossm_blip_vit
            blip_image_size = args.ossm_region_image_size
        else:
            blip_pretrained = args.blip_pretrained
            blip_vit = args.blip_vit
            blip_image_size = args.blip_image_size
        blip = blip_itm(
            pretrained=blip_pretrained,
            image_size=blip_image_size,
            vit=blip_vit,
        )
        trace_variants = {'evidence_trace', 'stip_evidence_trace'}
        if getattr(args, 'graph_export_evidence_trace', False) and stip_model_variant not in trace_variants:
            raise ValueError(
                "Evidence trace export requires --stip_model_variant evidence_trace. "
                "The default stip.py entry remains unchanged by design."
            )
        if stip_model_variant in trace_variants:
            model = STIPEvidenceTrace(args, detr=detr, blip=blip, detr_matcher=matcher)
            criterion = STIPEvidenceTraceCriterion(args, matcher)
            postprocessors = {'hoi': STIPEvidenceTracePostProcess(args, model)}
        elif stip_model_variant == 'b1_blip_detr':
            model = STIPB1(args, detr=detr, blip=blip, detr_matcher=matcher)
            criterion = STIPB1Criterion(args, matcher)
            postprocessors = {'hoi': STIPB1PostProcess(args, model)}
        elif stip_model_variant == 'b2_svrm':
            model = STIPB2SVRM(args, detr=detr, blip=blip, detr_matcher=matcher)
            criterion = STIPB2SVRMCriterion(args, matcher)
            postprocessors = {'hoi': STIPB2SVRMPostProcess(args, model)}
        elif stip_model_variant in {'b3_vet', 'stip_b3_vet'}:
            model = STIPB3VET(args, detr=detr, blip=blip, detr_matcher=matcher)
            criterion = STIPB3VETCriterion(args, matcher)
            postprocessors = {'hoi': STIPB3VETPostProcess(args, model)}
        elif stip_model_variant in {'b4_vet_lpt', 'stip_b4_vet_lpt'}:
            model = STIPB4VETLPT(args, detr=detr, blip=blip, detr_matcher=matcher)
            criterion = STIPB4VETLPTCriterion(args, matcher)
            postprocessors = {'hoi': STIPB4VETLPTPostProcess(args, model)}
        elif stip_model_variant in {'b5_setm_ablation', 'stip_b5_setm_ablation'}:
            model = STIPB5SETMAblation(args, detr=detr, blip=blip, detr_matcher=matcher)
            criterion = STIPB5SETMAblationCriterion(args, matcher)
            postprocessors = {'hoi': STIPB5SETMAblationPostProcess(args, model)}
        elif stip_model_variant == 'vet_internal_ablation':
            model = STIPVETInternalAblation(args, detr=detr, blip=blip, detr_matcher=matcher)
            criterion = STIPVETInternalAblationCriterion(args, matcher)
            postprocessors = {'hoi': STIPVETInternalAblationPostProcess(args, model)}
        elif stip_model_variant == 'ossm_box':
            model = STIPOSSMBox(args, detr=detr, blip=blip, detr_matcher=matcher)
            criterion = STIPOSSMBoxCriterion(args, matcher)
            postprocessors = {'hoi': STIPOSSMBoxPostProcess(args, model)}
        else:
            model = STIP(args, detr=detr, blip=blip, detr_matcher=matcher)
            criterion = STIPCriterion(args, matcher)
            postprocessors = {'hoi': STIPPostProcess(args, model)}
    elif args.HOIDet:
        hoi_matcher = build_hoi_matcher(args)
        hoi_losses = []
        hoi_losses.append('pair_labels')
        hoi_losses.append('pair_actions')
        if args.dataset_file == 'hico-det': hoi_losses.append('pair_targets')
        
        hoi_weight_dict={}
        hoi_weight_dict['loss_hidx'] = args.hoi_idx_loss_coef
        hoi_weight_dict['loss_oidx'] = args.hoi_idx_loss_coef
        hoi_weight_dict['loss_act'] = args.hoi_act_loss_coef
        if args.dataset_file == 'hico-det': hoi_weight_dict['loss_tgt'] = args.hoi_tgt_loss_coef
        if args.hoi_aux_loss:
            hoi_aux_weight_dict = {}
            for i in range(args.hoi_dec_layers):
                hoi_aux_weight_dict.update({k + f'_{i}': v for k, v in hoi_weight_dict.items()})
            hoi_weight_dict.update(hoi_aux_weight_dict)

        criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=hoi_weight_dict,
                                 eos_coef=args.eos_coef, losses=losses, num_actions=args.num_actions,
                                 HOI_losses=hoi_losses, HOI_matcher=hoi_matcher, args=args)

        interaction_transformer = build_hoi_transformer(args) # if (args.share_enc and args.pretrained_dec) else None

        kwargs = {}
        if args.dataset_file == 'hico-det': kwargs['return_obj_class'] = args.valid_obj_ids
        model = HOTR(
            detr=model,
            num_hoi_queries=args.num_hoi_queries,
            num_actions=args.num_actions,
            interaction_transformer=interaction_transformer,
            freeze_detr=(args.frozen_weights is not None),
            share_enc=args.share_enc,
            pretrained_dec=args.pretrained_dec,
            temperature=args.temperature,
            hoi_aux_loss=args.hoi_aux_loss,
            **kwargs # only return verb class for HICO-DET dataset
        )
        postprocessors = {'hoi': PostProcess(args.HOIDet)}
    else:
        criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses, num_actions=args.num_actions, args=args)
        postprocessors = {'bbox': PostProcess(args.HOIDet)}
    criterion.to(device)

    return model, criterion, postprocessors
