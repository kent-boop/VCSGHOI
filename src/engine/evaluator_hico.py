import math
import os
import sys
from typing import Iterable
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch

import src.util.misc as utils
import src.util.logger as loggers
from src.data.evaluators.hico_eval import HICOEvaluator
from src.models.evidence_trace_io import EvidenceTraceWriter, build_run_metadata, merge_manifests
from src.models.stip_utils import check_annotation, plot_cross_attention, plot_hoi_results

@torch.no_grad()
def hico_evaluate(model, postprocessors, data_loader, device, thr, args):
    model.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (HICO-DET)'

    preds = []
    gts = []
    indices = []
    hoi_recognition_time = []
    topk_records = []
    topk_recorder = None
    if getattr(args, 'ossm_topk_gt_stats', False):
        from src.engine.topk_gt_stats import TopKGTStats
        dataset = data_loader.dataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        topk_recorder = TopKGTStats(dataset.get_valid_obj_ids())
    trace_writer = None
    trace_dir = getattr(args, 'graph_export_trace_dir', '')
    if getattr(args, 'graph_export_evidence_trace', False):
        if not trace_dir:
            trace_dir = os.path.join(getattr(args, 'output_dir', '') or '.', 'evidence_traces')
        trace_writer = EvidenceTraceWriter(
            trace_dir,
            rank=utils.get_rank(),
            overwrite=getattr(args, 'graph_export_trace_overwrite', False),
        )
        trace_writer.write_run_metadata(build_run_metadata(
            args,
            dataset_split='test' if getattr(args, 'eval', False) else 'val',
            rank=utils.get_rank(),
            world_size=utils.get_world_size(),
        ))

    for samples, targets, images in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: (v.to(device) if hasattr(v, 'to') else v) for k, v in t.items()} for t in targets]

        # # register hooks to obtain intermediate outputs
        # dec_selfattn_weights, dec_crossattn_weights = [], []
        # if 'HOTR' in type(model).__name__:
        #     hook_self = model.interaction_transformer.decoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: dec_selfattn_weights.append(output[1]))
        #     hook_cross = model.interaction_transformer.decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_crossattn_weights.append(output[1]))
        # else:
        #     hook_self = model.interaction_decoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: dec_selfattn_weights.append(output[1]))
        #     hook_cross = model.interaction_decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_crossattn_weights.append(output[1]))

        outputs = model(samples, images, targets)
        if trace_writer is not None:
            trace_writer.write_outputs(outputs, expected_count=len(targets))
        evaluator_outputs = dict(outputs)
        evaluator_outputs.pop("evidence_traces", None)
        selected_indices = evaluator_outputs.pop("ossm_topk_indices", None)
        if topk_recorder is not None:
            if selected_indices is None or len(selected_indices) != len(targets):
                raise RuntimeError("Missing OSSM Top-k indices for this evaluation batch")
            batch_records = [topk_recorder.record(t, selected)
                             for t, selected in zip(targets, selected_indices)]
            topk_records.extend(itertools.chain.from_iterable(utils.all_gather(batch_records)))
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](evaluator_outputs, orig_target_sizes, threshold=thr, dataset='hico-det')
        hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

        # # visualize
        # if targets[0]['id'] in [57]: # [47, 57, 81, 30, 46, 97]: # 30, 46, 97
        #     # check_annotation(samples, targets, mode='eval', rel_num=20)
        #
        #     # visualize cross-attentioa
        #     if 'HOTR' in type(model).__name__:
        #         outputs['pred_actions'] = outputs['pred_actions'][:, :, :args.num_actions]
        #         outputs['pred_rel_pairs'] = [x.cpu() for x in torch.stack([outputs['pred_hidx'].argmax(-1), outputs['pred_oidx'].argmax(-1)], dim=-1)]
        #     topk_qids, q_name_list = plot_hoi_results(samples, outputs, targets, args=args)
        #     plot_cross_attention(samples, outputs, targets, dec_crossattn_weights, topk_qids=topk_qids)
        #     print(f"image_id={targets[0]['id']}")
        #
        #     # visualize self attention
        #     print('visualize self-attention')
        #     q_num = len(dec_selfattn_weights[0][0])
        #     plt.figure(figsize=(10,4))
        #     plt.imshow(dec_selfattn_weights[0][0].cpu().numpy(), vmin=0, vmax=0.4)
        #     plt.xticks(np.arange(q_num), [f"{i}" for i in range(q_num)], rotation=90, fontsize=12)
        #     plt.yticks(np.arange(q_num), [f"({q_name_list[i]})={i}" for i in range(q_num)], fontsize=12)
        #     plt.gca().xaxis.set_ticks_position('top')
        #     plt.grid(alpha=0.4, linestyle=':')
        #     plt.show()
        # hook_self.remove(); hook_cross.remove()

        # Pickled CUDA tensors restore onto their original device on every rank.
        # Gather CPU snapshots so evaluation neither opens peer-device contexts
        # nor retains the full validation set in GPU memory.
        results_cpu = [{k: v.detach().cpu() if torch.is_tensor(v) else v
                        for k, v in result.items()} for result in results]
        targets_cpu = [{k: v.detach().cpu() if torch.is_tensor(v) else v
                        for k, v in target.items()} for target in targets]
        preds.extend(itertools.chain.from_iterable(utils.all_gather(results_cpu)))
        gts.extend(itertools.chain.from_iterable(utils.all_gather(targets_cpu)))


    print(f"[stats] HOI Recognition Time (avg) : {sum(hoi_recognition_time)/len(hoi_recognition_time):.4f} ms")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]
    
    if topk_recorder is not None and utils.get_rank() == 0:
        topk_recorder.save(topk_records, args.output_dir or '.', args)

    evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat)

    stats = evaluator.evaluate()

    if trace_writer is not None:
        if utils.is_dist_avail_and_initialized():
            torch.distributed.barrier()
        if utils.get_rank() == 0:
            merge_manifests(trace_dir)

    return stats
