"""
HOI Detection Performance Bottleneck Analysis Script
=====================================================
分析 ResNet101 + BLIP + SAM 模型在 HICO-DET 上的性能瓶颈，
区分目标检测问题 vs 动作分类问题，找出最差的动作/物体类别。

使用方式:
    bash run_analysis.sh
"""

import argparse
import json
import os
import sys
import time
import copy
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---- project imports ----
import src.data.datasets as datasets
import src.util.misc as utils
from src.engine.arg_parser import get_args_parser
from src.data.datasets import build_dataset
from src.models import build_model
from src.util import box_ops
from src.data.evaluators.hico_eval import HICOEvaluator


# ===================== COCO 80 class names (index 0~79) =====================
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_iou_matrix(gt_boxes, pred_boxes):
    """Compute IoU matrix between GT and pred boxes (xyxy format)."""
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    iou_mat = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            iou_mat[i, j] = compute_iou(gt_boxes[i], pred_boxes[j])
    return iou_mat


@torch.no_grad()
def run_analysis(model, postprocessors, data_loader, device, args, output_dir):
    """Run full bottleneck analysis."""
    model.eval()
    
    # ==================== Phase 1: Collect predictions & GTs ====================
    is_main = utils.get_rank() == 0
    if is_main:
        print("\n" + "="*70)
        print("Phase 1: Running inference and collecting predictions...")
        print("="*70)
    
    all_preds = []
    all_gts = []
    all_det_results = []  # Store raw detection info for analysis
    
    total_images = 0
    total_batches = len(data_loader)
    phase1_start = time.time()
    for batch_idx, (samples, targets, images) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: (v.to(device) if k != 'id' else v) for k, v in t.items()} for t in targets]
        
        outputs = model(samples, images, targets)
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes, threshold=0, dataset='hico-det')
        
        # Store detection-level info for analysis
        bs = len(targets)
        for imgid in range(bs):
            out_logits = outputs['pred_logits'][imgid]  # [num_queries, num_classes+1]
            out_boxes = outputs['pred_boxes'][imgid]     # [num_queries, 4] cxcywh
            
            # Convert to xyxy and scale
            xyxy_boxes = box_ops.box_cxcywh_to_xyxy(out_boxes)
            img_h, img_w = targets[imgid]['orig_size']
            scale = torch.tensor([img_w, img_h, img_w, img_h], device=device, dtype=torch.float32)
            xyxy_boxes_scaled = xyxy_boxes * scale
            
            # Get class predictions (must match STIPPostProcess logic)
            # Both branches extract valid_obj_ids columns first → contiguous 0-79 labels
            _valid_obj_ids = data_loader.dataset._valid_obj_ids
            if out_logits.shape[-1] == 91:
                # H-DETR: sigmoid on valid columns
                obj_logits = out_logits[:, _valid_obj_ids]  # [num_queries, 80]
                probs = obj_logits.sigmoid()
                scores, labels = probs.max(-1)  # labels: contiguous 0-79
            else:
                # Standard DETR (92 classes): softmax on valid columns + background
                _ids_with_bg = _valid_obj_ids + [_valid_obj_ids[-1] + 1]
                obj_logits = out_logits[:, _ids_with_bg]  # [num_queries, 81]
                probs = F.softmax(obj_logits, -1)
                scores, labels = probs[:, :-1].max(-1)  # labels: contiguous 0-79
            
            # GT info
            gt_boxes = targets[imgid]['boxes'].cpu().numpy()  # xyxy
            gt_labels = targets[imgid]['labels'].cpu().numpy()
            gt_hois = targets[imgid]['hois'].cpu().numpy()  # [sub_id, obj_id, verb_id]
            
            all_det_results.append({
                'pred_boxes': xyxy_boxes_scaled.cpu().numpy(),
                'pred_scores': scores.cpu().numpy(),
                'pred_labels': labels.cpu().numpy(),
                'gt_boxes': gt_boxes,
                'gt_labels': gt_labels,
                'gt_hois': gt_hois,
                'pred_rel_pairs': outputs['pred_rel_pairs'][imgid].cpu().numpy() if imgid < len(outputs['pred_rel_pairs']) else None,
            })
        
        all_preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        cpu_targets = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in copy.deepcopy(targets)]
        all_gts.extend(list(itertools.chain.from_iterable(utils.all_gather(cpu_targets))))
        
        total_images += bs
        if is_main and (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - phase1_start
            speed = total_images / elapsed
            eta = (total_batches - batch_idx - 1) * (elapsed / (batch_idx + 1))
            print(f"  [{batch_idx+1}/{total_batches}] {total_images} images | "
                  f"{speed:.1f} img/s | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")
    
    # De-duplicate (for distributed)
    img_ids = [img_gts['id'] for img_gts in all_gts]
    _, indices = np.unique(img_ids, return_index=True)
    all_preds = [all_preds[i] for i in range(len(all_preds)) if i in indices]
    all_gts = [all_gts[i] for i in range(len(all_gts)) if i in indices]
    
    if is_main:
        print(f"\n  Total unique images: {len(all_gts)}")
    
    # Only run analysis on rank 0
    if not is_main:
        return {}
    
    # ==================== Phase 2: Standard mAP (sanity check) ====================
    phases = args.analysis_phases if hasattr(args, 'analysis_phases') else 'all'
    if phases == 'all':
        run_phases = {1, 2, 3, 4, 5}
    else:
        run_phases = set(int(x.strip()) for x in phases.split(','))
    
    print("\n" + "="*70)
    print("Phase 2: Computing standard mAP (sanity check)...")
    print("="*70)
    
    evaluator = HICOEvaluator(
        all_preds, all_gts,
        data_loader.dataset.rare_triplets,
        data_loader.dataset.non_rare_triplets,
        data_loader.dataset.correct_mat
    )
    stats = evaluator.evaluate()
    print(f"\n  ✅ mAP (full):     {stats['mAP']:.2f}")
    print(f"  ✅ mAP (rare):     {stats['mAP rare']:.2f}")
    print(f"  ✅ mAP (non-rare): {stats['mAP non-rare']:.2f}")
    print(f"  ✅ mean max Recall: {stats['mean max recall']:.2f}")
    
    # ==================== Phase 3: Object Detection Quality ====================
    det_analysis = {}
    if 3 in run_phases:
        print("\n" + "="*70)
        print("Phase 3: Analyzing Object Detection Quality...")
        print("="*70)
        
        valid_obj_ids = data_loader.dataset._valid_obj_ids
        det_analysis = analyze_detection_quality(all_det_results, valid_obj_ids)
    
    # ==================== Phase 4: Per-Verb AP Analysis ====================
    verb_analysis = {}
    if 4 in run_phases:
        print("\n" + "="*70)
        print("Phase 4: Analyzing Per-Verb AP...")
        print("="*70)
        
        verb_names = data_loader.dataset.get_actions()
        verb_analysis = analyze_per_verb_ap(evaluator, verb_names)
    
    # ==================== Phase 5: Per-Object AP + Error Analysis ====================
    obj_analysis = {}
    error_analysis = {}
    if 5 in run_phases:
        print("\n" + "="*70)
        print("Phase 5: Analyzing Per-Object AP...")
        print("="*70)
        
        valid_obj_ids = data_loader.dataset._valid_obj_ids
        obj_analysis = analyze_per_object_ap(evaluator, valid_obj_ids)
        
        print("\n" + "="*70)
        print("Phase 5b: Error Type Breakdown...")
        print("="*70)
        
        error_analysis = analyze_error_types(evaluator)
    
    # Ensure these are always available for plots/reports
    valid_obj_ids = data_loader.dataset._valid_obj_ids
    verb_names = data_loader.dataset.get_actions()
    
    # ==================== Save Results ====================
    print("\n" + "="*70)
    print("Saving results and generating plots...")
    print("="*70)
    
    all_results = {
        'mAP_stats': stats,
        'detection_quality': det_analysis,
        'per_verb_ap': verb_analysis,
        'per_object_ap': obj_analysis,
        'error_breakdown': error_analysis,
    }
    
    # Save JSON
    json_path = os.path.join(output_dir, 'bottleneck_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved JSON: {json_path}")
    
    # Generate plots
    generate_plots(all_results, verb_names, valid_obj_ids, output_dir)
    
    # Print summary report
    print_summary_report(all_results, verb_names, valid_obj_ids)
    
    return all_results


def analyze_detection_quality(det_results, valid_obj_ids):
    """Analyze object detection recall at different IoU thresholds."""
    iou_thresholds = [0.5, 0.3, 0.7]
    
    # Per-class statistics
    per_class_gt_count = defaultdict(int)
    per_class_matched = defaultdict(lambda: defaultdict(int))  # {class_id: {iou_thr: count}}
    
    total_gt_humans = 0
    total_gt_objects = 0
    human_matched = defaultdict(int)
    object_matched = defaultdict(int)
    
    score_thresholds = [0.0, 0.1, 0.3, 0.5]
    
    # Per score-threshold analysis
    recall_at_score = {}
    for score_thr in score_thresholds:
        h_total, h_matched, o_total, o_matched = 0, 0, 0, 0
        for det in det_results:
            gt_boxes = det['gt_boxes']
            gt_labels = det['gt_labels']
            pred_boxes = det['pred_boxes']
            pred_scores = det['pred_scores']
            pred_labels = det['pred_labels']
            
            # Filter by score threshold
            keep = pred_scores >= score_thr
            filtered_boxes = pred_boxes[keep]
            filtered_labels = pred_labels[keep]
            
            if len(gt_boxes) == 0 or len(filtered_boxes) == 0:
                for gl in gt_labels:
                    if gl == 1:
                        h_total += 1
                    else:
                        o_total += 1
                continue
            
            iou_mat = compute_iou_matrix(gt_boxes, filtered_boxes)
            
            for gt_idx in range(len(gt_boxes)):
                gt_label = gt_labels[gt_idx]
                is_human = (gt_label == 1)  # COCO person class
                
                if is_human:
                    h_total += 1
                else:
                    o_total += 1
                
                # Find best matching pred with same class
                best_iou = 0
                for pred_idx in range(len(filtered_boxes)):
                    # Match by category (convert DETR contiguous to COCO)
                    pred_coco_label = valid_obj_ids[filtered_labels[pred_idx]] if filtered_labels[pred_idx] < len(valid_obj_ids) else -1
                    if pred_coco_label == gt_label and iou_mat[gt_idx, pred_idx] > best_iou:
                        best_iou = iou_mat[gt_idx, pred_idx]
                
                if best_iou >= 0.5:
                    if is_human:
                        h_matched += 1
                    else:
                        o_matched += 1
        
        recall_at_score[f'score_{score_thr}'] = {
            'human_recall': h_matched / max(h_total, 1) * 100,
            'object_recall': o_matched / max(o_total, 1) * 100,
            'human_total': h_total,
            'human_matched': h_matched,
            'object_total': o_total,
            'object_matched': o_matched,
        }
        print(f"  Score ≥ {score_thr:.1f}: Human Recall = {h_matched}/{h_total} ({h_matched/max(h_total,1)*100:.1f}%), "
              f"Object Recall = {o_matched}/{o_total} ({o_matched/max(o_total,1)*100:.1f}%)")
    
    # Per-class detection recall (at IoU=0.5, score > 0)
    per_class_recall = {}
    for det in det_results:
        gt_boxes = det['gt_boxes']
        gt_labels = det['gt_labels']
        pred_boxes = det['pred_boxes']
        pred_scores = det['pred_scores']
        pred_labels = det['pred_labels']
        
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            for gl in gt_labels:
                cls_name = get_coco_name(gl, valid_obj_ids)
                per_class_gt_count[cls_name] += 1
            continue
        
        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
        
        for gt_idx in range(len(gt_boxes)):
            gt_label = gt_labels[gt_idx]
            cls_name = get_coco_name(gt_label, valid_obj_ids)
            per_class_gt_count[cls_name] += 1
            
            best_iou = 0
            for pred_idx in range(len(pred_boxes)):
                pred_coco_label = valid_obj_ids[pred_labels[pred_idx]] if pred_labels[pred_idx] < len(valid_obj_ids) else -1
                if pred_coco_label == gt_label and iou_mat[gt_idx, pred_idx] > best_iou:
                    best_iou = iou_mat[gt_idx, pred_idx]
            
            if best_iou >= 0.5:
                per_class_matched[cls_name] = per_class_matched.get(cls_name, 0) + 1
    
    for cls_name in per_class_gt_count:
        matched = per_class_matched.get(cls_name, 0)
        total = per_class_gt_count[cls_name]
        per_class_recall[cls_name] = {
            'recall': matched / max(total, 1) * 100,
            'matched': matched,
            'total': total,
        }
    
    # Sort by recall
    sorted_recall = sorted(per_class_recall.items(), key=lambda x: x[1]['recall'])
    print(f"\n  Bottom-10 detection recall by class (IoU≥0.5):")
    for name, info in sorted_recall[:10]:
        print(f"    {name:<20s}: {info['recall']:.1f}% ({info['matched']}/{info['total']})")
    
    return {
        'recall_at_score': recall_at_score,
        'per_class_recall': per_class_recall,
    }


def analyze_per_verb_ap(evaluator, verb_names):
    """Compute AP aggregated by verb (action)."""
    verb_ap = defaultdict(list)
    verb_recall = defaultdict(list)
    verb_gt_count = defaultdict(int)
    
    for triplet in evaluator.gt_triplets:
        subj_cls, obj_cls, verb_id = triplet
        sum_gts = evaluator.sum_gts[triplet]
        if sum_gts == 0:
            continue
        
        tp = np.array(evaluator.tp.get(triplet, []))
        fp = np.array(evaluator.fp.get(triplet, []))
        
        if len(tp) == 0:
            ap = 0
            max_rec = 0
        else:
            score = np.array(evaluator.score[triplet])
            sort_inds = np.argsort(-score)
            tp_sorted = tp[sort_inds]
            fp_sorted = fp[sort_inds]
            tp_cum = np.cumsum(tp_sorted)
            fp_cum = np.cumsum(fp_sorted)
            rec = tp_cum / sum_gts
            prec = tp_cum / (fp_cum + tp_cum)
            ap = evaluator.voc_ap(rec, prec)
            max_rec = np.amax(rec)
        
        verb_name = verb_names[verb_id] if verb_id < len(verb_names) else f"verb_{verb_id}"
        verb_ap[verb_name].append(ap * 100)
        verb_recall[verb_name].append(max_rec * 100)
        verb_gt_count[verb_name] += sum_gts
    
    # Compute mean AP per verb
    verb_mean_ap = {}
    for v_name in verb_ap:
        verb_mean_ap[v_name] = {
            'mean_ap': float(np.mean(verb_ap[v_name])),
            'num_triplets': len(verb_ap[v_name]),
            'gt_count': verb_gt_count[v_name],
            'mean_recall': float(np.mean(verb_recall[v_name])),
            'individual_aps': [float(x) for x in verb_ap[v_name]],
        }
    
    sorted_verbs = sorted(verb_mean_ap.items(), key=lambda x: x[1]['mean_ap'])
    
    print(f"\n  Bottom-15 verbs by mean AP:")
    for name, info in sorted_verbs[:15]:
        print(f"    {name:<25s}: AP={info['mean_ap']:.2f}%, Recall={info['mean_recall']:.1f}%, "
              f"#triplets={info['num_triplets']}, #GT={info['gt_count']}")
    
    print(f"\n  Top-10 verbs by mean AP:")
    for name, info in sorted_verbs[-10:]:
        print(f"    {name:<25s}: AP={info['mean_ap']:.2f}%, Recall={info['mean_recall']:.1f}%, "
              f"#triplets={info['num_triplets']}, #GT={info['gt_count']}")
    
    # Identify "no_interaction" verb
    if 'no_interaction' in verb_mean_ap:
        ni = verb_mean_ap['no_interaction']
        print(f"\n  ⚠️ no_interaction AP: {ni['mean_ap']:.2f}% (often excluded from eval)")
    
    return verb_mean_ap


def analyze_per_object_ap(evaluator, valid_obj_ids):
    """Compute AP aggregated by object class."""
    obj_ap = defaultdict(list)
    obj_recall = defaultdict(list)
    obj_gt_count = defaultdict(int)
    
    for triplet in evaluator.gt_triplets:
        subj_cls, obj_cls, verb_id = triplet
        sum_gts = evaluator.sum_gts[triplet]
        if sum_gts == 0:
            continue
        
        tp = np.array(evaluator.tp.get(triplet, []))
        fp = np.array(evaluator.fp.get(triplet, []))
        
        if len(tp) == 0:
            ap = 0
            max_rec = 0
        else:
            score = np.array(evaluator.score[triplet])
            sort_inds = np.argsort(-score)
            tp_sorted = tp[sort_inds]
            fp_sorted = fp[sort_inds]
            tp_cum = np.cumsum(tp_sorted)
            fp_cum = np.cumsum(fp_sorted)
            rec = tp_cum / sum_gts
            prec = tp_cum / (fp_cum + tp_cum)
            ap = evaluator.voc_ap(rec, prec)
            max_rec = np.amax(rec)
        
        obj_name = get_coco_name(valid_obj_ids[obj_cls], valid_obj_ids)
        obj_ap[obj_name].append(ap * 100)
        obj_recall[obj_name].append(max_rec * 100)
        obj_gt_count[obj_name] += sum_gts
    
    obj_mean_ap = {}
    for o_name in obj_ap:
        obj_mean_ap[o_name] = {
            'mean_ap': float(np.mean(obj_ap[o_name])),
            'num_triplets': len(obj_ap[o_name]),
            'gt_count': obj_gt_count[o_name],
            'mean_recall': float(np.mean(obj_recall[o_name])),
        }
    
    sorted_objs = sorted(obj_mean_ap.items(), key=lambda x: x[1]['mean_ap'])
    
    print(f"\n  Bottom-10 objects by mean AP:")
    for name, info in sorted_objs[:10]:
        print(f"    {name:<20s}: AP={info['mean_ap']:.2f}%, Recall={info['mean_recall']:.1f}%, "
              f"#triplets={info['num_triplets']}, #GT={info['gt_count']}")
    
    print(f"\n  Top-10 objects by mean AP:")
    for name, info in sorted_objs[-10:]:
        print(f"    {name:<20s}: AP={info['mean_ap']:.2f}%, Recall={info['mean_recall']:.1f}%, "
              f"#triplets={info['num_triplets']}, #GT={info['gt_count']}")
    
    return obj_mean_ap


def analyze_error_types(evaluator):
    """Classify errors into detection errors vs classification errors."""
    # We analyze the FP/TP patterns from the evaluator
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    
    # For each triplet, count TP/FP/FN
    triplet_stats = {}
    for triplet in evaluator.gt_triplets:
        sum_gts = evaluator.sum_gts[triplet]
        tp_arr = np.array(evaluator.tp.get(triplet, []))
        fp_arr = np.array(evaluator.fp.get(triplet, []))
        
        n_tp = int(tp_arr.sum()) if len(tp_arr) > 0 else 0
        n_fp = int(fp_arr.sum()) if len(fp_arr) > 0 else 0
        n_fn = sum_gts - n_tp  # missed GTs
        
        total_tp += n_tp
        total_fp += n_fp
        total_fn += n_fn
        total_gt += sum_gts
    
    # Precision and Recall
    precision = total_tp / max(total_tp + total_fp, 1) * 100
    recall = total_tp / max(total_gt, 1) * 100
    
    # Categorize triplets by their bottleneck
    # Low recall = detection problem (can't find pairs)
    # Low precision = classification problem (wrong action)
    zero_recall_triplets = 0
    low_recall_triplets = 0  # recall < 20%
    high_recall_low_ap_triplets = 0  # recall > 50% but AP < 30%
    good_triplets = 0
    
    for triplet in evaluator.gt_triplets:
        sum_gts = evaluator.sum_gts[triplet]
        if sum_gts == 0:
            continue
        
        tp_arr = np.array(evaluator.tp.get(triplet, []))
        if len(tp_arr) == 0:
            zero_recall_triplets += 1
            continue
        
        score_arr = np.array(evaluator.score[triplet])
        sort_inds = np.argsort(-score_arr)
        tp_sorted = tp_arr[sort_inds]
        fp_sorted = np.array(evaluator.fp.get(triplet, []))[sort_inds]
        
        tp_cum = np.cumsum(tp_sorted)
        rec = tp_cum / sum_gts
        max_recall = np.amax(rec)
        
        prec = tp_cum / (np.cumsum(fp_sorted) + tp_cum)
        ap = evaluator.voc_ap(rec, prec) * 100
        
        if max_recall < 0.2:
            low_recall_triplets += 1
        elif max_recall >= 0.5 and ap < 30:
            high_recall_low_ap_triplets += 1
        elif ap >= 30:
            good_triplets += 1
        else:
            low_recall_triplets += 1
    
    total_triplets = len(evaluator.gt_triplets)
    
    error_info = {
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_gt': total_gt,
        'precision': precision,
        'recall': recall,
        'total_triplets': total_triplets,
        'zero_recall_triplets': zero_recall_triplets,
        'low_recall_triplets': low_recall_triplets,
        'high_recall_low_ap_triplets': high_recall_low_ap_triplets,
        'good_triplets': good_triplets,
    }
    
    print(f"\n  Overall Statistics:")
    print(f"    Total GT interactions:    {total_gt}")
    print(f"    Total TP:                 {total_tp}")
    print(f"    Total FP:                 {total_fp}")
    print(f"    Total FN (missed):        {total_fn}")
    print(f"    Precision:                {precision:.1f}%")
    print(f"    Recall:                   {recall:.1f}%")
    
    print(f"\n  Triplet Error Distribution ({total_triplets} unique triplets):")
    print(f"    Zero recall (完全未检出):  {zero_recall_triplets} ({zero_recall_triplets/max(total_triplets,1)*100:.1f}%)")
    print(f"    Low recall < 20%:        {low_recall_triplets} ({low_recall_triplets/max(total_triplets,1)*100:.1f}%)")
    print(f"    High recall, low AP:     {high_recall_low_ap_triplets} ({high_recall_low_ap_triplets/max(total_triplets,1)*100:.1f}%)")
    print(f"    Good (AP ≥ 30%):         {good_triplets} ({good_triplets/max(total_triplets,1)*100:.1f}%)")
    
    # The key insight:
    # zero_recall + low_recall → likely detection problem (can't detect human/object)
    # high_recall_low_ap → classification problem (detecting pairs but wrong action)
    detection_problem_pct = (zero_recall_triplets + low_recall_triplets) / max(total_triplets, 1) * 100
    classification_problem_pct = high_recall_low_ap_triplets / max(total_triplets, 1) * 100
    
    print(f"\n  🔍 Bottleneck Diagnosis:")
    print(f"    Detection-related issues:      {detection_problem_pct:.1f}% of triplets")
    print(f"    Classification-related issues:  {classification_problem_pct:.1f}% of triplets")
    print(f"    Well-performing:               {good_triplets/max(total_triplets,1)*100:.1f}% of triplets")
    
    error_info['detection_problem_pct'] = detection_problem_pct
    error_info['classification_problem_pct'] = classification_problem_pct
    
    return error_info


def get_coco_name(coco_id, valid_obj_ids):
    """Get COCO class name from COCO category ID."""
    # valid_obj_ids maps contiguous index → COCO id
    # COCO_CLASSES is indexed by contiguous position
    if coco_id in valid_obj_ids:
        idx = valid_obj_ids.index(coco_id)
        if idx < len(COCO_CLASSES):
            return COCO_CLASSES[idx]
    return f"class_{coco_id}"


def generate_plots(results, verb_names, valid_obj_ids, output_dir):
    """Generate visualization plots."""
    plt.rcParams.update({'font.size': 10})
    
    # ---- Plot 1: Per-verb AP bar chart ----
    verb_ap_data = results['per_verb_ap']
    if verb_ap_data:
        sorted_verbs = sorted(verb_ap_data.items(), key=lambda x: x[1]['mean_ap'], reverse=True)
        names = [v[0] for v in sorted_verbs]
        aps = [v[1]['mean_ap'] for v in sorted_verbs]
        
        fig, ax = plt.subplots(figsize=(16, max(8, len(names) * 0.22)))
        colors = ['#2ecc71' if a > 50 else '#f39c12' if a > 20 else '#e74c3c' for a in aps]
        bars = ax.barh(range(len(names)), aps, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Mean AP (%)')
        ax.set_title('Per-Verb Mean AP on HICO-DET (ResNet101 + BLIP + SAM)')
        ax.invert_yaxis()
        ax.axvline(x=np.mean(aps), color='blue', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(aps):.1f}%')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, 'per_verb_ap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    
    # ---- Plot 2: Per-object AP bar chart ----
    obj_ap_data = results['per_object_ap']
    if obj_ap_data:
        sorted_objs = sorted(obj_ap_data.items(), key=lambda x: x[1]['mean_ap'], reverse=True)
        names = [v[0] for v in sorted_objs]
        aps = [v[1]['mean_ap'] for v in sorted_objs]
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(names) * 0.25)))
        colors = ['#2ecc71' if a > 50 else '#f39c12' if a > 20 else '#e74c3c' for a in aps]
        bars = ax.barh(range(len(names)), aps, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Mean AP (%)')
        ax.set_title('Per-Object Mean AP on HICO-DET (ResNet101 + BLIP + SAM)')
        ax.invert_yaxis()
        ax.axvline(x=np.mean(aps), color='blue', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(aps):.1f}%')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, 'per_object_ap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    
    # ---- Plot 3: Error type distribution pie chart ----
    err = results.get('error_breakdown')
    if err and isinstance(err, dict) and 'zero_recall_triplets' in err:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie 1: Triplet categories
        labels1 = ['Zero Recall\n(完全未检出)', 'Low Recall\n(<20%)', 'High Recall,\nLow AP', 'Good\n(AP≥30%)']
        sizes1 = [err['zero_recall_triplets'], err['low_recall_triplets'],
                  err['high_recall_low_ap_triplets'], err['good_triplets']]
        colors1 = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        axes[0].pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
        axes[0].set_title('Triplet Error Distribution', fontsize=12)
        
        # Pie 2: TP/FP/FN
        labels2 = ['True Positive', 'False Positive', 'False Negative\n(Missed)']
        sizes2 = [err['total_tp'], err['total_fp'], err['total_fn']]
        colors2 = ['#2ecc71', '#e74c3c', '#95a5a6']
        axes[1].pie(sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
        axes[1].set_title('Prediction Distribution', fontsize=12)
        
        plt.suptitle('Error Analysis: ResNet101 + BLIP + SAM on HICO-DET', fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, 'error_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    else:
        print("  Skipped error_distribution.png (Phase 5 not run)")
    
    # ---- Plot 4: Detection recall per class ----
    det_recall = results['detection_quality'].get('per_class_recall', {})
    if det_recall:
        sorted_det = sorted(det_recall.items(), key=lambda x: x[1]['recall'])
        names = [v[0] for v in sorted_det]
        recalls = [v[1]['recall'] for v in sorted_det]
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(names) * 0.25)))
        colors = ['#2ecc71' if r > 80 else '#f39c12' if r > 50 else '#e74c3c' for r in recalls]
        ax.barh(range(len(names)), recalls, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Detection Recall @ IoU≥0.5 (%)')
        ax.set_title('Per-Class Detection Recall (DETR ResNet101)')
        ax.axvline(x=np.mean(recalls), color='blue', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(recalls):.1f}%')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, 'detection_recall_per_class.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def print_summary_report(results, verb_names, valid_obj_ids):
    """Print a comprehensive summary report."""
    print("\n" + "="*70)
    print("                    📊 BOTTLENECK ANALYSIS SUMMARY 📊")
    print("="*70)
    
    stats = results.get('mAP_stats')
    err = results.get('error_breakdown')
    det = results.get('detection_quality')
    
    if stats and isinstance(stats, dict) and 'mAP' in stats:
        print(f"\n1. Overall Performance:")
        print(f"   mAP: {stats['mAP']:.2f} | Rare: {stats['mAP rare']:.2f} | Non-rare: {stats['mAP non-rare']:.2f}")
        if err and isinstance(err, dict) and 'precision' in err:
            print(f"   Precision: {err['precision']:.1f}% | Recall: {err['recall']:.1f}%")
    
    if det and isinstance(det, dict) and 'recall_at_score' in det:
        print(f"\n2. Detection Quality (DETR @IoU≥0.5):")
        r = det['recall_at_score']
        for key in sorted(r.keys()):
            info = r[key]
            print(f"   {key}: Human={info['human_recall']:.1f}%, Object={info['object_recall']:.1f}%")
    
    if err and isinstance(err, dict) and 'detection_problem_pct' in err:
        print(f"\n3. Bottleneck Diagnosis:")
        print(f"   Detection-related:      {err['detection_problem_pct']:.1f}% triplets have recall < 20%")
        print(f"   Classification-related: {err['classification_problem_pct']:.1f}% triplets have high recall but low AP")
        
        if err['detection_problem_pct'] > 60:
            print(f"\n   ⚠️ 主要瓶颈: 目标检测！大量 triplet 的 recall 很低，说明 DETR 未能提供足够准确的框。")
            print(f"   建议: 改进 DETR 检测器 (更好的预训练、fine-tune DETR backbone、增大 num_queries)")
        elif err['classification_problem_pct'] > 30:
            print(f"\n   ⚠️ 主要瓶颈: 动作分类！检测框质量尚可，但动作识别准确率低。")
            print(f"   建议: 改进 HOI 分类头 (更好的特征融合、增强类别平衡)")
        else:
            print(f"\n   ℹ️ 检测和分类问题并存，需要两方面同时改进。")
    
    # Top worst verbs
    verb_data = results.get('per_verb_ap')
    if verb_data and isinstance(verb_data, dict) and len(verb_data) > 0:
        first_val = next(iter(verb_data.values()))
        if isinstance(first_val, dict) and 'mean_ap' in first_val:
            sorted_verbs = sorted(verb_data.items(), key=lambda x: x[1]['mean_ap'])
            print(f"\n4. Worst 10 Actions (需要重点改进):")
            for name, info in sorted_verbs[:10]:
                print(f"   {name:<25s}: AP={info['mean_ap']:.2f}%, Recall={info['mean_recall']:.1f}%, GT={info['gt_count']}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        'HOI Detection Bottleneck Analysis',
        parents=[get_args_parser()]
    )
    # Same args as STIP_main.py
    parser.add_argument('--detr_weights', default=None, type=str)
    parser.add_argument('--train_detr', action='store_true', default=False)
    parser.add_argument('--finetune_detr_weight', default=0.1, type=float)
    parser.add_argument('--lr_detr', default=1e-5, type=float)
    parser.add_argument('--reduce_lr_on_plateau_patience', default=2, type=int)
    parser.add_argument('--reduce_lr_on_plateau_factor', default=0.1, type=float)
    parser.add_argument('--lr_scheduler', default='cosine_warm_restart', type=str,
                        choices=['cosine_warm_restart', 'cosine_fixed'])
    parser.add_argument('--lr_t0', default=10, type=int)
    
    # Loss args (needed for build_model)
    parser.add_argument('--proposal_focal_loss_alpha', default=0.75, type=float)
    parser.add_argument('--action_focal_loss_alpha', default=0.5, type=float)
    parser.add_argument('--proposal_focal_loss_gamma', default=2, type=float)
    parser.add_argument('--action_focal_loss_gamma', default=2, type=float)
    parser.add_argument('--proposal_loss_coef', default=1, type=float)
    parser.add_argument('--action_loss_coef', default=1, type=float)
    parser.add_argument('--action_sampled_loss_coef', default=0.0, type=float)
    parser.add_argument('--action_fore_loss_coef', default=1.0, type=float)
    parser.add_argument('--global_action_fore_loss_coef', default=1.0, type=float)
    parser.add_argument('--action_fea_loss_coef', default=1.0, type=float)
    
    # Ablation args
    parser.add_argument('--no_hard_mining_for_relation_discovery', dest='use_hard_mining_for_relation_discovery', action='store_false', default=True)
    parser.add_argument('--no_relation_dependency_encoding', dest='use_relation_dependency_encoding', action='store_false', default=True)
    parser.add_argument('--no_memory_layout_encoding', dest='use_memory_layout_encoding', action='store_false', default=True)
    parser.add_argument('--no_nms_on_detr', dest='apply_nms_on_detr', action='store_false', default=True)
    parser.add_argument('--no_tail_semantic_feature', dest='use_tail_semantic_feature', action='store_false', default=True)
    parser.add_argument('--no_spatial_feature', dest='use_spatial_feature', action='store_false', default=True)
    parser.add_argument('--no_interaction_decoder', action='store_true', default=False)
    parser.add_argument('--use_memory_union_mask', action='store_true', default=False)
    parser.add_argument('--use_union_feature', action='store_true', default=False)
    parser.add_argument('--adaptive_relation_query_num', action='store_true', default=False)
    parser.add_argument('--use_relation_tgt_mask', action='store_true', default=False)
    parser.add_argument('--use_relation_tgt_mask_attend_topk', default=10, type=int)
    parser.add_argument('--use_prior_verb_label_mask', action='store_true', default=False)
    parser.add_argument('--relation_feature_map_from', default='backbone')
    parser.add_argument('--use_query_fourier_encoding', action='store_true', default=False)
    parser.add_argument('--proj_blip', action='store_true', default=True)
    parser.add_argument('--merge_blip', action='store_true', default=True)
    parser.add_argument('--only_blip', action='store_true', default=False)
    
    # MoE
    parser.add_argument('--use_moe_visual_router', action='store_true', default=True)
    parser.add_argument('--use_moe_aux_loss', action='store_true', default=True)
    parser.add_argument('--use_moe_noisy_gating', action='store_true', default=True)
    parser.add_argument('--use_hdetr', action='store_true', default=True)
    parser.add_argument('--obj_score_power', default=0.5, type=float)
    parser.add_argument('--use_score_calibration', action='store_true', default=True)
    parser.add_argument('--use_dynamic_text', action='store_true', default=False)
    
    # HSC ablation
    parser.add_argument('--no_topk_text', dest='use_topk_text', action='store_false', default=True)
    parser.add_argument('--no_text_adapter', dest='use_text_adapter', action='store_false', default=True)
    parser.add_argument('--k_scene', default=3, type=int)
    parser.add_argument('--k_instance', default=10, type=int)
    parser.add_argument('--k_action', default=3, type=int)
    
    # IDCR ablation
    parser.add_argument('--no_scene_optimizer', dest='use_scene_optimizer', action='store_false', default=True)
    parser.add_argument('--no_instance_optimizer', dest='use_instance_optimizer', action='store_false', default=True)
    parser.add_argument('--no_blip_pos_embed', dest='use_blip_pos_embed', action='store_false', default=True)
    parser.add_argument('--no_memory_text_attn', dest='use_memory_text_attn', action='store_false', default=True)
    parser.add_argument('--no_action_cross_attn', dest='use_action_cross_attn', action='store_false', default=True)
    
    # SAQA ablation
    parser.add_argument('--no_saqa', dest='use_saqa', action='store_false', default=True)
    parser.add_argument('--saqa_fusion_mode', default='moe', type=str, choices=['moe', 'avg'])
    
    # Analysis specific
    parser.add_argument('--analysis_output_dir', default='analysis_results', type=str)
    parser.add_argument('--analysis_phases', default='all', type=str,
                        help='Comma-separated phase numbers to run (1-5), or "all". '
                             'Phase 1: Inference, Phase 2: mAP, Phase 3: Detection Quality, '
                             'Phase 4: Per-Verb AP, Phase 5: Per-Object AP + Error Analysis')
    
    args = parser.parse_args()
    args.STIP_relation_head = True
    args.seed = 3407
    
    # Force eval mode
    args.eval = True
    
    # Initialize distributed mode (works for both single-GPU and multi-GPU)
    utils.init_distributed_mode(args)
    
    device = torch.device(args.device)
    
    # Build dataset (test set)
    dataset_val = build_dataset(image_set='test', args=args)
    args.num_classes = dataset_val.num_category()
    args.num_actions = dataset_val.num_action()
    args.action_names = dataset_val.get_actions()
    if args.dataset_file == 'hico-det':
        args.valid_obj_ids = dataset_val.get_valid_obj_ids()
        args.correct_mat = torch.tensor(dataset_val.correct_mat).to(device)
    
    if args.share_enc: args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.hoi_dec_layers = args.dec_layers
    
    # Use DistributedSampler for multi-GPU, SequentialSampler for single-GPU
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    
    # Build model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # Load full model weights
    if args.resume:
        print(f"Loading model weights from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Wrap model in DDP for multi-GPU
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # Create output directory
    output_dir = args.analysis_output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    if utils.get_rank() == 0:
        print("\n" + "#"*70)
        print("  HOI Detection Bottleneck Analysis")
        print(f"  Backbone: {args.backbone}")
        print(f"  Checkpoint: {args.resume}")
        print(f"  Dataset: {args.dataset_file}")
        print(f"  GPUs: {utils.get_world_size()}")
        print(f"  Output: {output_dir}")
        print("#"*70)
    
    results = run_analysis(model, postprocessors, data_loader_val, device, args, output_dir)
    
    if utils.get_rank() == 0:
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
