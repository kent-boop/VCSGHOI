import argparse
import datetime
import json
import random
import time
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import src.data.datasets as datasets
import src.util.misc as utils
from src.engine.arg_parser import get_args_parser
from src.data.datasets import build_dataset, get_coco_api_from_dataset
from src.engine.trainer import train_one_epoch
from src.engine import hoi_evaluator, hoi_accumulator
from src.models import build_model
import wandb
from src.engine.evaluator_coco import coco_evaluate

from src.util.logger import print_params, print_args
from src.util.ossm_checkpoint import format_ossm_load_report, load_ossm_initialization
from collections import OrderedDict

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1,2,3'
import gc


class AttributeSubset(torch.utils.data.Subset):
    """Subset that preserves dataset metadata used by HOI evaluators."""

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def load_eval_image_ids(path):
    """Load ordered, unique evaluation dataset indices from JSON or text."""
    source = Path(path)
    raw = source.read_text(encoding='utf-8').strip()
    if not raw:
        raise ValueError(f"Evaluation image-ID file is empty: {source}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = [token for token in raw.replace(',', ' ').split() if token]
    if isinstance(payload, dict):
        if 'image_ids' not in payload:
            raise ValueError(
                f"JSON image-ID dict must contain an 'image_ids' list: {source}"
            )
        payload = payload['image_ids']
    if not isinstance(payload, list):
        raise ValueError(f"Evaluation image IDs must be a list: {source}")
    ordered = []
    seen = set()
    for value in payload:
        image_id = int(value)
        if image_id not in seen:
            seen.add(image_id)
            ordered.append(image_id)
    if not ordered:
        raise ValueError(f"No evaluation image IDs found in: {source}")
    return ordered


def smart_memory_guard(reserve_buffer_mb=500):
    """
    智能显存防御：
    1. 自动检测当前剩余显存。
    2. 尝试占用 (剩余显存 - reserve_buffer_mb) 的大小。
    3. 如果失败，自动缩减申请量，保证不报错。
    
    :param reserve_buffer_mb: 留给系统和显存碎片的“安全气垫”，建议 500MB - 1024MB
    """
    device = torch.device('cuda')
    
    # 1. 获取当前显存状态 (free, total) 单位是字节
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
    free_mb = free_bytes / 1024 / 1024
    
    print(f"【显存防御】检测到当前空闲显存: {free_mb:.2f} MB")

    # 2. 计算目标占用量：剩余显存 - 安全缓冲
    target_mb = free_mb - reserve_buffer_mb
    
    if target_mb <= 0:
        print("【显存防御】警告：剩余显存过少，放弃占位以保命！")
        return None

    # 3. 尝试分配
    # 1 MB = 1024 * 1024 bytes
    # float32 = 4 bytes
    # 元素个数 = (MB * 1024 * 1024) / 4
    num_elements = int(target_mb * (1024 * 1024) / 4)
    
    try:
        # 使用 empty 分配，极快且无开销
        dummy = torch.empty(num_elements, dtype=torch.float32, device=device)
        print(f"【显存防御】成功动态占用 {target_mb:.2f} MB。剩余缓冲: {reserve_buffer_mb} MB")
        return dummy
    
    except RuntimeError as e:
        # 如果还是OOM（可能是碎片化导致），尝试申请一半
        print(f"【显存防御】全额申请失败，尝试降级申请一半...")
        torch.cuda.empty_cache() # 清理一下碎片
        try:
            num_elements = num_elements // 2
            dummy = torch.empty(num_elements, dtype=torch.float32, device=device)
            print(f"【显存防御】降级占用成功: {target_mb/2:.2f} MB")
            return dummy
        except RuntimeError:
            print("【显存防御】最终失败，放弃防御，裸奔运行。")
            return None

def save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename):
    # save_ckpt: function for saving checkpoints
    output_dir = Path(args.output_dir)
    if args.output_dir:
        checkpoint_path = output_dir / f'{filename}.pth'
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)

def main(args):
    # guard_tensor = smart_memory_guard(reserve_buffer_mb=800)
    utils.init_distributed_mode(args)
    if args.init_from_ossm and args.resume:
        raise ValueError("--init_from_ossm and --resume are mutually exclusive")
    selected_variant = getattr(args, 'model_variant', None) or getattr(args, 'stip_model_variant', 'default')
    if getattr(args, 'ossm_topk_gt_stats', False) and (
        selected_variant != 'ossm_box' or args.dataset_file != 'hico-det' or not args.eval
    ):
        raise ValueError('--ossm_topk_gt_stats requires --validation, --dataset_file hico-det and --stip_model_variant ossm_box')
    if args.init_from_ossm and selected_variant != 'ossm_box':
        raise ValueError("--init_from_ossm is only valid with --stip_model_variant ossm_box")

    if not args.train_detr is not None: # pretrained DETR
        print("Freeze weights for detector")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    # print(torch.rand(1))
    torch.cuda.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)

    # Data Setup
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val' if not args.eval else 'test', args=args)
    assert dataset_train.num_action() == dataset_val.num_action(), "Number of actions should be the same between splits"
    args.num_classes = dataset_train.num_category()
    args.num_actions = dataset_train.num_action()
    args.action_names = dataset_train.get_actions()
    if args.share_enc: args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.hoi_dec_layers = args.dec_layers
    if args.dataset_file == 'vcoco':
        # Save V-COCO dataset statistics
        args.valid_ids = np.array(dataset_train.get_object_label_idx()).nonzero()[0]
        args.invalid_ids = np.argwhere(np.array(dataset_train.get_object_label_idx()) == 0).squeeze(1)
        args.human_actions = dataset_train.get_human_action()
        args.object_actions = dataset_train.get_object_action()
        args.num_human_act = dataset_train.num_human_act()
    elif args.dataset_file == 'hico-det':
        args.valid_obj_ids = dataset_train.get_valid_obj_ids()
        args.correct_mat = torch.tensor(dataset_val.correct_mat).to(device)
    if args.eval and args.eval_image_ids_file:
        if args.eval_max_samples > 0:
            raise ValueError(
                "--eval_image_ids_file and --eval_max_samples are mutually exclusive"
            )
        image_ids = load_eval_image_ids(args.eval_image_ids_file)
        invalid = [index for index in image_ids if index < 0 or index >= len(dataset_val)]
        if invalid:
            raise IndexError(
                f"Evaluation image IDs outside 0..{len(dataset_val) - 1}: {invalid[:10]}"
            )
        dataset_val = AttributeSubset(dataset_val, image_ids)
        if utils.get_rank() == 0:
            print(
                f"[Trace subset] Loaded {len(image_ids)} exact evaluation images "
                f"from {args.eval_image_ids_file}"
            )
    if args.eval and args.eval_max_samples > 0:
        sample_count = min(int(args.eval_max_samples), len(dataset_val))
        dataset_val = AttributeSubset(dataset_val, range(sample_count))
        if utils.get_rank() == 0:
            print(f"[Trace smoke] Limiting evaluation split to {sample_count} samples")
    print_args(args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)



    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_val = DataLoader(dataset_val, 4, sampler=sampler_val,
    #                             drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, 1, drop_last=True)
    #
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=0)
    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=0)


    # Model Setup
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = print_params(model)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "detr" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if ("detr" in n and 'backbone' not in n) and p.requires_grad],
            "lr": args.lr * 0.1,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if ("detr" in n and 'backbone' in n) and p.requires_grad],
            "lr": args.lr * 0.01,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.reduce_lr_on_plateau_factor, patience=args.reduce_lr_on_plateau_patience, verbose=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=2)

    # Weight Setup
    if args.detr_weights is not None:
        print(f"Loading detr weights from args.detr_weights={args.detr_weights}")
        if args.detr_weights.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.detr_weights, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.detr_weights, map_location='cpu')
        if 'hico_ft_q16.pth' in args.detr_weights: # hack: for loading hico fine-tuned detr
            mapped_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                if k.startswith('detr.'):
                    mapped_state_dict[k.replace('detr.', '')] = v
            model_without_ddp.detr.load_state_dict(mapped_state_dict)
        else:
            model_without_ddp.detr.load_state_dict(checkpoint['model']) # STIP
            # model_without_ddp.detr.load_state_dict(checkpoint['model_state_dict']) # UPT
    if args.init_from_ossm:
        if args.detr_weights is not None:
            print("[OSSM init] --detr_weights was loaded first; compatible old OSSM keys now take precedence.")
        report = load_ossm_initialization(model_without_ddp, args.init_from_ossm)
        if utils.is_main_process():
            print(format_ossm_load_report(report))
        del report
        gc.collect()
    if args.resume:
        print(f"Loading model weights from args.resume={args.resume}")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        # 1. 加载模型权重
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        # # ================= 🚀 强制初始化手术 =================
        # print("\n[DEBUG] Performing MoE Weight Reset Surgery...")
        # with torch.no_grad():
        #     # 1. 打印旧值（确认它是不是负数）
        #     old_val = model_without_ddp.moe_loss_log_var.data.item()
        #     print(f"  > Old Log Var Value: {old_val} (Probably Negative)")
            
        #     # 2. 强制设为 2.0
        #     #    Softplus(2.0) ≈ 2.12
        #     #    Weight = exp(-2.12) ≈ 0.12
        #     #    这是一个非常温和、安全的初始权重
        #     model_without_ddp.moe_loss_log_var.data.fill_(2.0)
            
        #     print(f"  > New Log Var Value: {model_without_ddp.moe_loss_log_var.data.item()}")
        #     print("  > Expected Aux Weight: ~0.12 (Safe Start)\n")
        # # ====================================================
        # ================= 🚀 插入这一段代码 🚀 =================
        # 强制重置 MoE Router，打破之前 checkpoint 里的“平均分配”惯性
        # 只有当模型里定义了这个方法时才调用
        # if hasattr(model_without_ddp, '_reset_router_parameters'):
        #     print(f"\n[Main] Detected resume. Manually resetting MoE Router for fresh start...")
        #     model_without_ddp._reset_router_parameters()
        # else:
        #     print("\n[Main Warning] _reset_router_parameters method not found in model!")
        # ========================================================
        # 2. 恢复训练进度。epoch 不应该依赖 optimizer 是否能成功恢复；
        #    否则模型权重加载成功但 optimizer 参数组不匹配时，会错误地从第 0 轮重新训练。
        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {args.start_epoch}")

        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            if not args.eval: # 如果只是 eval 测试，也没必要加载 optimizer
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    print(f"Resumed optimizer and lr_scheduler state from epoch {checkpoint.get('epoch', 'unknown')}")
                except ValueError as e:
                    print(f"[Warning] Failed to load optimizer state: {e}")
                    print("[Warning] Optimizer 参数组和当前模型不匹配，将使用新的 optimizer，但继续从 checkpoint epoch 后训练。")

    if args.eval:
        # test only mode
        if args.HOIDet:
            if args.dataset_file == 'vcoco':
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)
                sc1, sc2 = hoi_accumulator(args, total_res, True, False)
            elif args.dataset_file == 'hico-det':
                test_stats = hoi_evaluator(args, model, None, postprocessors, data_loader_val, device)
                print(f'| mAP (full)\t\t: {test_stats["mAP"]:.2f}')
                print(f'| mAP (rare)\t\t: {test_stats["mAP rare"]:.2f}')
                print(f'| mAP (non-rare)\t: {test_stats["mAP non-rare"]:.2f}')
            else: raise ValueError(f'dataset {args.dataset_file} is not supported.')
            return
        else:
            # check original detr code
            base_ds = get_coco_api_from_dataset(data_loader_val)
            test_stats, coco_evaluator = coco_evaluate(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args.output_dir)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, args.output_dir / "eval.pth")
            return

    # stats
    scenario1, scenario2 = 0, 0
    best_mAP, best_rare, best_non_rare = 0, 0, 0

    # add argparse
    if args.wandb and utils.get_rank() == 0:
        wandb.init(
            project=args.project_name,
            group=args.group_name,
            name=args.run_name,
            config=args
        )
        wandb.watch(model)

    # Training starts here!
    start_time = time.time()
    day = time.strftime('%Y-%m-%d-%H %M:%S', time.localtime()).split(' ')[0]
    for epoch in range(args.start_epoch, args.epochs):
        # if args.validate:
        #     print('-'*100)
        #     if args.dataset_file == 'vcoco':
        #         torch.cuda.empty_cache()
        #         total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)
        #         if utils.get_rank() == 0:
        #             torch.cuda.empty_cache()
        #             sc1, sc2 = hoi_accumulator(args, total_res, False, args.wandb)
        #             if sc1 > scenario1:
        #                 scenario1 = sc1
        #                 scenario2 = sc2
        #                 save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
        #             print(f'| Scenario #1 mAP : {sc1:.2f} ({scenario1:.2f})')
        #             print(f'| Scenario #2 mAP : {sc2:.2f} ({scenario2:.2f})')
        #             if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): lr_scheduler.step(sc1)
        #     elif args.dataset_file == 'hico-det':
        #         test_stats = hoi_evaluator(args, model, None, postprocessors, data_loader_val, device)
        #         if utils.get_rank() == 0:
        #             if test_stats['mAP'] > best_mAP:
        #                 best_mAP = test_stats['mAP']
        #                 best_rare = test_stats['mAP rare']
        #                 best_non_rare = test_stats['mAP non-rare']
        #                 save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
        #             print(f'| mAP (full)\t\t: {test_stats["mAP"]:.2f} ({best_mAP:.2f})')
        #             print(f'| mAP (rare)\t\t: {test_stats["mAP rare"]:.2f} ({best_rare:.2f})')
        #             print(f'| mAP (non-rare)\t: {test_stats["mAP non-rare"]:.2f} ({best_non_rare:.2f})')
        #             if args.wandb and utils.get_rank() == 0:
        #                 wandb.log({
        #                     'mAP': test_stats['mAP']
        #                 })
        #             if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): lr_scheduler.step(test_stats['mAP'])
        #     print('-'*100)

        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.epochs,
            args.clip_max_norm, dataset_file=args.dataset_file, log=args.wandb, scheduler=lr_scheduler)
        lr_scheduler.step()
        print(optimizer.param_groups[0]['lr'])

        if utils.get_rank() == 0:
            save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='checkpoint')

        # Validation
        if args.validate:
            print('-'*100)
            if args.dataset_file == 'vcoco':
                torch.cuda.empty_cache()
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)
                if utils.get_rank() == 0:
                    torch.cuda.empty_cache()
                    sc1, sc2 = hoi_accumulator(args, total_res, False, args.wandb)
                    if sc1 > scenario1:
                        scenario1 = sc1
                        scenario2 = sc2
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
                    print(f'| Scenario #1 mAP : {sc1:.2f} ({scenario1:.2f})')
                    print(f'| Scenario #2 mAP : {sc2:.2f} ({scenario2:.2f})')
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): lr_scheduler.step(sc1)
            elif args.dataset_file == 'hico-det':
                test_stats = hoi_evaluator(args, model, None, postprocessors, data_loader_val, device)
                if utils.get_rank() == 0:
                    if test_stats['mAP'] > best_mAP:
                        best_mAP = test_stats['mAP']
                        best_rare = test_stats['mAP rare']
                        best_non_rare = test_stats['mAP non-rare']
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
                    print(f'| mAP (full)\t\t: {test_stats["mAP"]:.2f} ({best_mAP:.2f})')
                    print(f'| mAP (rare)\t\t: {test_stats["mAP rare"]:.2f} ({best_rare:.2f})')
                    print(f'| mAP (non-rare)\t: {test_stats["mAP non-rare"]:.2f} ({best_non_rare:.2f})')
                    if args.wandb and utils.get_rank() == 0:
                        wandb.log({
                            'mAP': test_stats['mAP']
                        })
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): lr_scheduler.step(test_stats['mAP'])
            print('-'*100)
            torch.cuda.empty_cache()

        if args.dataset_file == 'vcoco':
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'Scenario1': scenario1,
                         'Scenario2': scenario2,
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        elif args.dataset_file == 'hico-det':
            log_stats = {
                'mAP_full': best_mAP, 
                'mAP_rare': best_rare, 
                'mAP_non_rare': best_non_rare,
                'epoch': epoch
            }

        if utils.get_rank() == 0:
            with open(args.output_dir + f'{day}-log.txt', 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.dataset_file == 'vcoco':
        print(f'| Scenario #1 mAP : {scenario1:.2f}')
        print(f'| Scenario #2 mAP : {scenario2:.2f}')
    elif args.dataset_file == 'hico-det':
        print(f'| mAP (full)\t\t: {best_mAP:.2f}')
        print(f'| mAP (rare)\t\t: {best_rare:.2f}')
        print(f'| mAP (non-rare)\t: {best_non_rare:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'End-to-End Human Object Interaction training and evaluation script',
        parents=[get_args_parser()]
    )
    # training
    parser.add_argument('--detr_weights', default=None, type=str)
    parser.add_argument('--train_detr', action='store_true', default=False)
    parser.add_argument('--finetune_detr_weight', default=0.1, type=float)
    parser.add_argument('--lr_detr', default=1e-5, type=float)
    parser.add_argument('--reduce_lr_on_plateau_patience', default=2, type=int)
    parser.add_argument('--reduce_lr_on_plateau_factor', default=0.1, type=float)

    # loss
    parser.add_argument('--proposal_focal_loss_alpha', default=0.75, type=float) # large alpha for high recall
    parser.add_argument('--action_focal_loss_alpha', default=0.5, type=float)
    parser.add_argument('--proposal_focal_loss_gamma', default=2, type=float)
    parser.add_argument('--action_focal_loss_gamma', default=2, type=float)
    parser.add_argument('--proposal_loss_coef', default=1, type=float)
    parser.add_argument('--action_loss_coef', default=1, type=float)
    parser.add_argument('--action_sampled_loss_coef', default=1.0, type=float)
    parser.add_argument('--action_fore_loss_coef', default=1.0, type=float)
    parser.add_argument('--global_action_fore_loss_coef', default=1.0, type=float)
    parser.add_argument('--action_fea_loss_coef', default=1.0, type=float)

    # ablations
    parser.add_argument('--no_hard_mining_for_relation_discovery', dest='use_hard_mining_for_relation_discovery', action='store_false', default=True)
    parser.add_argument('--no_relation_dependency_encoding', dest='use_relation_dependency_encoding', action='store_false', default=True)
    parser.add_argument('--no_memory_layout_encoding', dest='use_memory_layout_encoding', action='store_false', default=True, help='layout encodings')
    parser.add_argument('--no_nms_on_detr', dest='apply_nms_on_detr', action='store_false', default=True)
    parser.add_argument('--no_tail_semantic_feature', dest='use_tail_semantic_feature', action='store_false', default=True)
    parser.add_argument('--no_spatial_feature', dest='use_spatial_feature', action='store_false', default=True)
    parser.add_argument('--no_interaction_decoder', action='store_true', default=False)

    # not sensitive or effective
    parser.add_argument('--use_memory_union_mask', action='store_true', default=False)
    parser.add_argument('--use_union_feature', action='store_true', default=False)
    parser.add_argument('--adaptive_relation_query_num', action='store_true', default=False)
    parser.add_argument('--use_relation_tgt_mask', action='store_true', default=False)
    parser.add_argument('--use_relation_tgt_mask_attend_topk', default=10, type=int)
    parser.add_argument('--use_prior_verb_label_mask', action='store_true', default=False)
    parser.add_argument('--relation_feature_map_from', default='backbone', help='backbone | detr_encoder')
    parser.add_argument('--use_query_fourier_encoding', action='store_true', default=False)
    parser.add_argument('--proj_blip', action='store_true', default=True)
    parser.add_argument('--merge_blip', action='store_true', default=True)
    parser.add_argument('--only_blip', action='store_true', default=False)
    parser.add_argument('--zero_shot_type', default='default', type=str, 
                        choices=['default', 'RF', 'NF'], 
                        help='Zero-shot setting: default (full), RF (Rare First), NF (Non-rare First)')

    args = parser.parse_args()
    args.STIP_relation_head = True
    args.seed = 3407

    if args.output_dir:
        args.output_dir += f"/{args.group_name}/{args.run_name}/"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
