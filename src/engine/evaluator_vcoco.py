# ------------------------------------------------------------------------
# HOTR official code : src/engine/evaluator_vcoco.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import torch
import time
import datetime

import src.util.misc as utils
import src.util.logger as loggers
from src.data.evaluators.vcoco_eval import VCocoEvaluator
from src.util.box_ops import rescale_bboxes, rescale_pairs
from src.models.evidence_trace_io import EvidenceTraceWriter, build_run_metadata, merge_manifests
from src.models.stip_utils import check_annotation, plot_cross_attention
from pyheatmap.heatmap import HeatMap

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import nn
import torch.nn.functional as F

import wandb

def visualize(model, samples, targets):
    param_dict = {}
    for name, parameters in model.named_parameters():
        param_dict.update({name: parameters})
        if name == 'interaction_decoder.layers.5.multihead_attn.in_proj_weight':
            in_proj_weight = parameters
        elif name == 'interaction_decoder.layers.5.multihead_attn.in_proj_bias':
            in_proj_bias = parameters
    outputs = model(samples)
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    cq = []
    pk = []
    memory = []
    hooks = [
        # 获取resnet最后一层特征图
        model.detr.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        # 获取encoder的图像特征图memory
        model.detr.transformer.encoder.register_forward_hook(
            lambda self, input, output: memory.append(output)
        ),
        # 获取encoder的最后一层layer的self-attn weights
        model.detr.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        # 获取decoder的最后一层layer中交叉注意力的 weights
        model.interaction_decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
        # 获取decoder最后一层self-attn的输出cq
        model.interaction_decoder.layers[-1].norm1.register_forward_hook(
            lambda self, input, output: cq.append(output)
        ),
        # 获取图像特征图的位置编码pk
        model.detr.backbone[-1].register_forward_hook(
            lambda self, input, output: pk.append(output)
        ),
    ]
    outputs = model(samples)
    for hook in hooks:
        hook.remove()

    conv_features = conv_features[0] # [1,2048,25,34]
    enc_attn_weights = enc_attn_weights[0]  # [1,850,850]   : [N,L,S]
    dec_attn_weights = dec_attn_weights[0]  # [1,100,850]   : [N,L,S] --> [batch, tgt_len, src_len]
    memory = memory[0]

    h, w = conv_features['0'].tensors.shape[-2:]
    img_id = str(int(targets[0]['image_id']))
    img_name = 'COCO_val2014_' + img_id.zfill(12) + '.jpg'
    img_path = 'v-coco/coco/images/val2014/' + img_name
    img = Image.open(img_path)
    scale = 16
    # -------------heatmap---------------
    atten_weight = dec_attn_weights.sum(dim=1)[0].view(h, w)
    sorted_weight = atten_weight.reshape(-1).sort()
    thresh = sorted_weight.values[-100]
    more_than_idxs = torch.nonzero((atten_weight > thresh))
    more_than_values = atten_weight[more_than_idxs[:, 0], more_than_idxs[:, 1]].unsqueeze(1)
    more_than_idxs *= scale
    more_than_values = (more_than_values*1000).int()
    points = torch.cat([more_than_idxs.int(), more_than_values], dim=1).cpu().numpy().tolist()
    image1 = cv2.imread(img_path)
    image1 = cv2.resize(image1, (w*scale, h*scale))
    background = Image.new('RGB', (image1.shape[1], image1.shape[0]), color=0)
    hm = HeatMap(points)
    hit_img = hm.heatmap(base=background, r=40)
    hit_img = cv2.cvtColor(np.asarray(hit_img),cv2.COLOR_RGB2BGR)
    overlay = image1.copy()
    alpha = 0.3
    cv2.rectangle(overlay, (0,0), (w*scale, h*scale), (255, 0, 0), -1)
    image2 = cv2.addWeighted(overlay, alpha, image1, 1-alpha, 0)
    image3 = cv2.addWeighted(hit_img, alpha, image2, 1-alpha, 0)
    cv2.imwrite(f'visual/attention_pics_hm/{img_name}', image3)

    # -------------heatmap---------------

    # --------------interpolate-------------
    # atten_weight = dec_attn_weights.sum(dim=1)[0].view(h, w).unsqueeze(0).unsqueeze(0)
    # atten_weight = F.interpolate(atten_weight, scale_factor=16, mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
    # img_plt = img.resize((atten_weight.size()[1], atten_weight.size()[0]), Image.ANTIALIAS)
    # atten_weight = np.array(atten_weight.cpu())
    # plt.imshow(img_plt, alpha=0.6)
    # plt.imshow(atten_weight, alpha=0.4)
    # plt.savefig(f'visual/attention_pics_fusion/{img_name}')
    # plt.clf()
    # --------------interpolate-------------

    # ---------------1 colum----------------
    # fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10, 10))
    # ax = axs[0]
    # atten_weight = dec_attn_weights.sum(dim=1)[0].view(h, w).cpu()
    #
    # ax.imshow(np.array(atten_weight))
    # ax = axs[1]
    # ax.imshow(img)
    # fig.tight_layout()
    # plt.savefig(f'visual/attention_pics/{img_name}')
    # ---------------1 colum----------------

    # ---------------33 colums---------------
    # fig, axs = plt.subplots(ncols=dec_attn_weights.size()[1]+1, nrows=2, figsize=(132, 8))
    # for query_idx, ax_i in zip(range(dec_attn_weights.size()[1] + 1), axs.T):
    #     ax = ax_i[0]
    #     if query_idx == 32:
    #         atten_weight = dec_attn_weights.sum(dim=1)[0].view(h, w).cpu()
    #     else:
    #         atten_weight = dec_attn_weights[0, query_idx].view(h, w).cpu()
    #     weight_img = ax.imshow(np.array(atten_weight))
    #     ax = ax_i[1]
    #     ax.imshow(img)

    # fig.tight_layout()
    # plt.savefig(f'visual/attention_pics/{img_name}')
    # plt.show()
    # ---------------33 colums---------------
def visual_obj_det(outputs, results, targets, data_loader):
    img_id = str(int(targets[0]['image_id']))
    img_name = 'COCO_val2014_' + img_id.zfill(12) + '.jpg'
    img_path = 'v-coco/coco/images/val2014/' + img_name
    img = cv2.imread(img_path)

    rel_acts = outputs['pred_actions'][0]
    rel_pairs = outputs['pred_rel_pairs'][0].to(rel_acts.device)
    keep_rels = torch.nonzero(rel_acts >= 0.8)

    obj_logits = outputs['pred_logits']
    prob = F.softmax(obj_logits, -1)
    scores, labels = prob.max(-1)
    hum_idxs = rel_pairs[:, 0][keep_rels[:, 0]]
    obj_idxs = rel_pairs[:, 1][keep_rels[:, 0]]
    hum_scores = scores.squeeze(0)[hum_idxs]
    obj_scores = scores.squeeze(0)[obj_idxs]
    hum_boxes = results[0]['boxes'][hum_idxs]
    obj_boxes = results[0]['boxes'][obj_idxs]
    obj_labels = labels.squeeze(0)[obj_idxs]

    keep_hums = hum_scores > 0.7
    keep_objs1 = obj_scores > 0.7
    keep_objs2 = obj_labels != 91
    keep_pairs = torch.nonzero(keep_hums * keep_objs1 * keep_objs2).squeeze(-1)

    keep_rels = keep_rels[keep_pairs]
    selected_acts = keep_rels[:, 1].cpu().numpy()
    selected_hum_boxes = hum_boxes[keep_pairs].cpu().numpy()
    selected_obj_boxes = obj_boxes[keep_pairs].cpu().numpy()
    selected_obj_labels = obj_labels[keep_pairs].cpu().numpy()

    coco_classes = data_loader.dataset.COCO_CLASSES
    act_classes = data_loader.dataset.CLASSES

    num = len(keep_pairs)
    last_act_ct = (0, 0)
    for i in range(num):
        hum_pt1 = (int(selected_hum_boxes[i][0]), int(selected_hum_boxes[i][1]))
        hum_pt2 = (int(selected_hum_boxes[i][2]), int(selected_hum_boxes[i][3]))
        cv2.rectangle(img, hum_pt1, hum_pt2, (255, 0, 0), 2)

        obj_pt1 = (int(selected_obj_boxes[i][0]), int(selected_obj_boxes[i][1]))
        obj_pt2 = (int(selected_obj_boxes[i][2]), int(selected_obj_boxes[i][3]))
        cv2.rectangle(img, obj_pt1, obj_pt2, (0, 255, 0), 2)
        # print(selected_obj_labels[i])
        cls = coco_classes[selected_obj_labels[i]]
        cv2.putText(img, cls, obj_pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        hum_ct = (int((hum_pt1[0] + hum_pt2[0]) / 2), int((hum_pt1[1] + hum_pt2[1]) / 2))
        obj_ct = (int((obj_pt1[0] + obj_pt2[0]) / 2), int((obj_pt1[1] + obj_pt2[1]) / 2))
        cv2.line(img, hum_ct, obj_ct, (0, 0, 255), 2, 4)

        act_ct = (int((hum_ct[0] + obj_ct[0])/2), int((hum_ct[1] + obj_ct[1])/2))
        if act_ct == last_act_ct:
            act_ct = (act_ct[0], act_ct[1]+30)
        last_act_ct = act_ct
        act_cls = act_classes[selected_acts[i]]
        cv2.putText(img, act_cls, act_ct, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(f'visual/detection/detr/{img_name}', img)

    return

def visual_target(outputs, results, targets, data_loader):
    img_id = str(int(targets[0]['image_id']))
    img_name = 'COCO_val2014_' + img_id.zfill(12) + '.jpg'
    img_path = 'v-coco/coco/images/val2014/' + img_name
    img = cv2.imread(img_path)

    boxes = targets[0]['boxes'].cpu().numpy()
    obj_labels = targets[0]['labels'].cpu().numpy()
    paired_boxes = targets[0]['pair_boxes'].cpu().numpy()
    paired_actions = targets[0]['pair_actions']

    for i, box in enumerate(boxes):
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        coco_classes = data_loader.dataset.COCO_CLASSES
        act_classes = data_loader.dataset.CLASSES
        cls = coco_classes[obj_labels[i]]
        color = (0, 255, 0)
        if obj_labels[i] == 1:
            color = (255, 0, 0)
        cv2.rectangle(img, pt1, pt2, color, 2)
        cv2.putText(img, cls, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    for i, pair_box in enumerate(paired_boxes):
        if (pair_box[-1] == -1):
            act_ct = (int((pair_box[0]+pair_box[2])/2), int((pair_box[1]+pair_box[3])/2))

        else:
            pt1 = (int((pair_box[0]+pair_box[2])/2), int((pair_box[1]+pair_box[3])/2))
            pt2 = (int((pair_box[4]+pair_box[6])/2), int((pair_box[5]+pair_box[7])/2))
            cv2.line(img, pt1, pt2, (0, 0, 255), 2, 4)
            act_ct = (int((pt1[0]+pt2[0])/2), int((pt1[1]+pt2[1])/2))
        acts = torch.nonzero(paired_actions[i]).squeeze(-1).cpu().numpy()
        for act in acts:
            act_cls = act_classes[act]
            cv2.putText(img, act_cls, act_ct, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            act_ct = (act_ct[0], act_ct[1]+30)

    cv2.imwrite(f'visual/detection/gt/{img_name}', img)

    return



@torch.no_grad()
def vcoco_evaluate(model, criterion, postprocessors, data_loader, device, output_dir, thr):
    model.eval()
    criterion.eval()
    args = getattr(criterion, 'args', None)

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (V-COCO)'

    print_freq = int(len(data_loader) // 5) # len(data_loader)
    res = {}
    hoi_recognition_time = []
    num = 0
    trace_writer = None
    trace_dir = ''
    if args is not None and getattr(args, 'graph_export_evidence_trace', False):
        trace_dir = getattr(args, 'graph_export_trace_dir', '')
        if not trace_dir:
            trace_dir = os.path.join(output_dir or '.', 'evidence_traces')
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

    for samples, targets, images in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # visualize(model, samples, targets)
        # ---------------------------------
        outputs = model(samples, images, targets)
        if trace_writer is not None:
            trace_writer.write_outputs(outputs, expected_count=len(targets))
        evaluator_outputs = dict(outputs)
        evaluator_outputs.pop("evidence_traces", None)
        loss_dict = criterion(evaluator_outputs, targets)
        loss_dict_reduced = utils.reduce_dict(loss_dict) # ddp gathering

        loss_dict_reduced_scaled = {k: v * criterion.weight_dict[k] for k, v in loss_dict_reduced.items() if k in criterion.weight_dict}
        loss_value = sum(loss_dict_reduced_scaled.values()).item()

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](evaluator_outputs, orig_target_sizes, threshold=thr, dataset='vcoco')
        targets = process_target(targets, orig_target_sizes)
        # visual_obj_det(outputs, results, targets, data_loader)
        # visual_target(outputs, results, targets, data_loader)
        # hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)
        hoi_recognition_time.append(evaluator_outputs['hoi_recognition_time'] * 1000)

        # check_annotation(samples, targets, mode='eval', rel_num=20, dataset='vcoco')
        # plot_cross_attention(samples, outputs, targets, dec_crossattn_weights, dataset='vcoco'); hook.remove()

        res.update(
            {target['image_id'].item():\
                {'target': target, 'prediction': output} for target, output in zip(targets, results)
            }
        )
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # -----------------------------------
        # if len(res) > 10: break

    print(f"[stats] HOI Recognition Time (avg) : {sum(hoi_recognition_time)/len(hoi_recognition_time):.4f} ms")
    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)

    start_time = time.time()
    torch.cuda.empty_cache()
    gather_res = utils.all_gather(res)
    total_res = {}
    for dist_res in gather_res:
        total_res.update(dist_res)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"[stats] Distributed Gathering Time : {total_time_str}")

    if trace_writer is not None:
        if utils.is_dist_avail_and_initialized():
            torch.distributed.barrier()
        if utils.get_rank() == 0:
            merge_manifests(trace_dir)

    return total_res

def vcoco_accumulate(total_res, args, print_results, wandb_log):
    vcoco_evaluator = VCocoEvaluator(args)
    vcoco_evaluator.update(total_res)    
    print(f"[stats] Score Matrix Generation completed!!          ")

    scenario1 = vcoco_evaluator.role_eval1.evaluate(print_results)
    scenario2 = vcoco_evaluator.role_eval2.evaluate(print_results)

    if wandb_log:
        wandb.log({
            'scenario1': scenario1,
            'scenario2': scenario2
        })

    return scenario1, scenario2

def process_target(targets, target_sizes):
    for idx, (target, target_size) in enumerate(zip(targets, target_sizes)):
        labels = target['labels']
        valid_boxes_inds = (labels > 0)

        targets[idx]['boxes'] = rescale_bboxes(target['boxes'], target_size) # boxes
        targets[idx]['pair_boxes'] = rescale_pairs(target['pair_boxes'], target_size) # pairs

    return targets
