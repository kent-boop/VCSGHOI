import argparse
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from src.util.misc import nested_tensor_from_tensor_list


BLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
BLIP_STD = torch.tensor([0.26862854, 0.26130258, 0.27577711]).view(3, 1, 1)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
HICO_VALID_OBJ_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def load_get_args_parser():
    arg_parser_path = Path(__file__).resolve().parent / "src" / "engine" / "arg_parser.py"
    spec = importlib.util.spec_from_file_location("stip_arg_parser", arg_parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_args_parser


def build_parser():
    parser = argparse.ArgumentParser(
        "Visualize whether DETR boxes align with STIP BLIP inputs",
        parents=[load_get_args_parser()()],
        conflict_handler="resolve",
    )
    parser.add_argument("--image", required=True, help="Image path to visualize")
    parser.add_argument("--checkpoint", default="", help="Optional STIP checkpoint path")
    parser.add_argument("--detr_weights", default="", help="Optional DETR checkpoint path")
    parser.add_argument("--save_dir", default="vis_blip_boxes", help="Directory for visualizations")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--resize", default=800, type=int, help="Square resize used to build the debug sample")
    parser.add_argument("--max_pairs", default=12, type=int, help="Maximum proposal pairs to visualize")
    parser.add_argument("--score_threshold", default=0.0, type=float, help="Only draw pairs above this proposal logit threshold")
    parser.add_argument("--dataset_file", default="hico-det")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--num_hoi_queries", default=32, type=int)
    parser.add_argument("--HOIDet", action="store_true", default=True)
    parser.add_argument("--STIP_relation_head", action="store_true", default=True)
    parser.add_argument("--no_aux_loss", dest="aux_loss", action="store_false", default=False)
    parser.add_argument("--hoi_aux_loss", action="store_true", default=True)
    parser.add_argument("--use_visual_scene_graph", action="store_true", default=True)
    parser.add_argument("--train_detr", action="store_true", default=False)
    parser.add_argument("--finetune_detr_weight", default=0.1, type=float)
    parser.add_argument("--proposal_focal_loss_gamma", default=2, type=float)
    parser.add_argument("--action_focal_loss_gamma", default=2, type=float)
    parser.add_argument("--proposal_loss_coef", default=1, type=float)
    parser.add_argument("--action_loss_coef", default=1, type=float)
    parser.add_argument("--action_sampled_loss_coef", default=1.0, type=float)
    parser.add_argument("--action_fore_loss_coef", default=1.0, type=float)
    parser.add_argument("--global_action_fore_loss_coef", default=1.0, type=float)
    parser.add_argument("--action_fea_loss_coef", default=1.0, type=float)
    parser.add_argument("--use_hard_mining_for_relation_discovery", action="store_true", default=True)
    parser.add_argument("--use_relation_dependency_encoding", action="store_true", default=True)
    parser.add_argument("--use_memory_layout_encoding", action="store_true", default=True)
    parser.add_argument("--apply_nms_on_detr", action="store_true", default=True)
    parser.add_argument("--use_tail_semantic_feature", action="store_true", default=True)
    parser.add_argument("--use_spatial_feature", action="store_true", default=True)
    parser.add_argument("--no_interaction_decoder", action="store_true", default=False)
    parser.add_argument("--use_memory_union_mask", action="store_true", default=False)
    parser.add_argument("--use_union_feature", action="store_true", default=False)
    parser.add_argument("--adaptive_relation_query_num", action="store_true", default=False)
    parser.add_argument("--use_relation_tgt_mask", action="store_true", default=False)
    parser.add_argument("--use_relation_tgt_mask_attend_topk", default=10, type=int)
    parser.add_argument("--use_prior_verb_label_mask", action="store_true", default=False)
    parser.add_argument("--relation_feature_map_from", default="backbone")
    parser.add_argument("--use_query_fourier_encoding", action="store_true", default=False)
    return parser


def complete_args(args):
    args.num_classes = 91
    args.num_actions = 117
    args.valid_obj_ids = HICO_VALID_OBJ_IDS
    args.correct_mat = torch.ones(args.num_actions, len(args.valid_obj_ids))
    args.invalid_ids = []
    args.valid_ids = list(range(args.num_actions))
    args.action_names = [str(i) for i in range(args.num_actions)]
    args.share_enc = getattr(args, "share_enc", False)
    args.pretrained_dec = getattr(args, "pretrained_dec", False)
    args.finetune_detr = getattr(args, "finetune_detr", False)
    args.resume = args.checkpoint
    return args


def load_state(model, checkpoint_path, device, label):
    if not checkpoint_path:
        return
    checkpoint_path = str(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)} tensors from {label}: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def load_detr_state(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DETR checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    if any(k.startswith("detr.") for k in state_dict):
        state_dict = {k.replace("detr.", "", 1): v for k, v in state_dict.items() if k.startswith("detr.")}
    model.detr.load_state_dict(state_dict, strict=False)
    print(f"Loaded DETR weights: {checkpoint_path}")


def preprocess_image(image_path, resize, device):
    raw = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(raw).to(device)
    resized = raw.resize((resize, resize), Image.BICUBIC)
    return raw, resized, tensor


def tensor_to_pil(tensor, mean, std):
    cpu = tensor.detach().cpu()
    mean = mean.to(cpu)
    std = std.to(cpu)
    img = (cpu * std + mean).clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1).clamp(0, 1)


def scale_box(box, width, height):
    x1, y1, x2, y2 = box.tolist()
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def clipped_int_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(1, min(width, int(np.ceil(x2))))
    y2 = max(1, min(height, int(np.ceil(y2))))
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2


def draw_box(draw, box, color, label, width=3):
    draw.rectangle(box, outline=color, width=width)
    text_xy = (box[0] + 2, max(0, box[1] - 12))
    draw.text(text_xy, label, fill=color)


def box_inside_crop(box, crop_box, output_size=384):
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop_box
    crop_w = max(cx2 - cx1, 1)
    crop_h = max(cy2 - cy1, 1)
    return [
        (x1 - cx1) / crop_w * output_size,
        (y1 - cy1) / crop_h * output_size,
        (x2 - cx1) / crop_w * output_size,
        (y2 - cy1) / crop_h * output_size,
    ]


def make_pair_diagnostic_page(blip_image, crop_image, sub_box, obj_box, union_box, crop_box, rank, pair_idx, score):
    panel_w, panel_h = 384, 384
    left = blip_image.resize((panel_w, panel_h), Image.BICUBIC)
    right = crop_image.resize((panel_w, panel_h), Image.BICUBIC)

    left_draw = ImageDraw.Draw(left)
    sx = panel_w / blip_image.width
    sy = panel_h / blip_image.height
    scale = lambda b: [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
    draw_box(left_draw, scale(sub_box), "lime", "human", width=3)
    draw_box(left_draw, scale(obj_box), "deepskyblue", "object", width=3)
    draw_box(left_draw, scale(union_box), "red", "union", width=4)
    left_draw.text((6, 6), f"pair {rank} / query {pair_idx}\nscore={score:.3f}\nfull BLIP input", fill="yellow")

    right_draw = ImageDraw.Draw(right)
    draw_box(right_draw, box_inside_crop(sub_box, crop_box, panel_w), "lime", "human in crop", width=3)
    draw_box(right_draw, box_inside_crop(obj_box, crop_box, panel_w), "deepskyblue", "object in crop", width=3)
    right_draw.text((6, 6), "union crop resized to 384\nboxes remapped inside crop", fill="yellow")

    page = Image.new("RGB", (panel_w * 2, panel_h), "white")
    page.paste(left, (0, 0))
    page.paste(right, (panel_w, 0))
    return page


def make_manual_union_crops(blip_image_01, union_boxes, image_size=384):
    _, height, width = blip_image_01.shape
    crops = []
    crop_boxes = []
    for box in union_boxes:
        pixel_box = scale_box(box, width, height)
        x1, y1, x2, y2 = clipped_int_box(pixel_box, width, height)
        crop = blip_image_01[:, y1:y2, x1:x2]
        crop = F.interpolate(crop.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
        crops.append(crop)
        crop_boxes.append((x1, y1, x2, y2))
    if not crops:
        return torch.empty(0, 3, image_size, image_size), crop_boxes
    return torch.stack(crops), crop_boxes


def save_pair_visualizations(model, outputs, sample_tensor, sample_mask, save_dir, max_pairs, score_threshold):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    graph_input = outputs["graph_inputs"][0]
    pred_boxes = outputs["pred_boxes"][0].detach().cpu()
    pairs = outputs["pred_rel_pairs"][0].detach().cpu()
    proposal_scores = graph_input["proposal_scores"].detach().cpu()
    pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes)

    blip_tensor, valid_size = model.build_blip_tensor_from_sample(sample_tensor, sample_mask)
    blip_image = tensor_to_pil(blip_tensor, BLIP_MEAN, BLIP_STD)
    blip_image_01 = (blip_tensor.detach().cpu() * BLIP_STD + BLIP_MEAN).clamp(0, 1)
    blip_image.save(save_dir / "blip_input_from_sample.jpg")

    sample_valid = sample_tensor[:, : valid_size[0], : valid_size[1]].detach().cpu()
    sample_valid_pil = tensor_to_pil(sample_valid, IMAGENET_MEAN, IMAGENET_STD)
    sample_valid_pil.save(save_dir / "detr_sample_valid_region.jpg")

    keep = torch.arange(pairs.shape[0])
    if score_threshold > 0:
        keep = keep[proposal_scores > score_threshold]
    keep = keep[:max_pairs]
    if keep.numel() == 0:
        print("No proposal pairs selected for visualization.")
        return

    width, height = blip_image.size
    overlay = blip_image.copy()
    draw = ImageDraw.Draw(overlay)
    selected_union_boxes = []
    selected_sub_boxes = []
    selected_obj_boxes = []
    selected_pair_indices = []

    for rank, pair_idx in enumerate(keep.tolist()):
        sub_idx = int(pairs[pair_idx, 0])
        obj_idx = int(pairs[pair_idx, 1])
        sub_box = pred_boxes_xyxy[sub_idx]
        obj_box = pred_boxes_xyxy[obj_idx]
        union_box = torch.cat([torch.minimum(sub_box[:2], obj_box[:2]), torch.maximum(sub_box[2:], obj_box[2:])])
        selected_union_boxes.append(union_box)
        selected_sub_boxes.append(sub_box)
        selected_obj_boxes.append(obj_box)
        selected_pair_indices.append(pair_idx)

        sub_pixel = scale_box(sub_box, width, height)
        obj_pixel = scale_box(obj_box, width, height)
        union_pixel = scale_box(union_box, width, height)
        draw_box(draw, sub_pixel, "lime", f"p{rank} H q{sub_idx}", width=2)
        draw_box(draw, obj_pixel, "deepskyblue", f"p{rank} O q{obj_idx}", width=2)
        draw_box(draw, union_pixel, "red", f"p{rank} U score={float(proposal_scores[pair_idx]):.2f}", width=3)

    overlay.save(save_dir / "proposal_pairs_on_blip_input.jpg")

    union_boxes = torch.stack(selected_union_boxes)
    manual_crops_01, crop_boxes = make_manual_union_crops(blip_image_01, union_boxes)
    internal_crops = model.make_cropped_region_imgs_from_boxes(union_boxes.to(sample_tensor.device), blip_tensor.to(sample_tensor.device))
    internal_crops_01 = (internal_crops.detach().cpu() * BLIP_STD + BLIP_MEAN).clamp(0, 1)

    for rank in range(union_boxes.shape[0]):
        manual = Image.fromarray((manual_crops_01[rank].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        internal = Image.fromarray((internal_crops_01[rank].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        manual.save(save_dir / f"pair_{rank:02d}_manual_union_crop.jpg")
        internal.save(save_dir / f"pair_{rank:02d}_model_union_crop.jpg")

        sub_pixel = scale_box(selected_sub_boxes[rank], width, height)
        obj_pixel = scale_box(selected_obj_boxes[rank], width, height)
        union_pixel = scale_box(union_boxes[rank], width, height)
        page = make_pair_diagnostic_page(
            blip_image=blip_image,
            crop_image=internal,
            sub_box=sub_pixel,
            obj_box=obj_pixel,
            union_box=union_pixel,
            crop_box=crop_boxes[rank],
            rank=rank,
            pair_idx=selected_pair_indices[rank],
            score=float(proposal_scores[selected_pair_indices[rank]]),
        )
        page.save(save_dir / f"pair_{rank:02d}_diagnostic.jpg")

        compare = Image.new("RGB", (manual.width * 2, manual.height), "white")
        compare.paste(manual, (0, 0))
        compare.paste(internal, (manual.width, 0))
        compare_draw = ImageDraw.Draw(compare)
        compare_draw.text((8, 8), "manual crop", fill="yellow")
        compare_draw.text((manual.width + 8, 8), "model crop", fill="yellow")
        compare.save(save_dir / f"pair_{rank:02d}_crop_compare.jpg")

    diag_paths = [save_dir / f"pair_{rank:02d}_diagnostic.jpg" for rank in range(union_boxes.shape[0])]
    thumbs = [Image.open(path).resize((384, 192), Image.BICUBIC) for path in diag_paths]
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 384, rows * 192), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 384, (idx // cols) * 192))
    sheet.save(save_dir / "pair_diagnostic_contact_sheet.jpg")

    max_abs_diff = (manual_crops_01 - internal_crops_01).abs().max().item()
    print(f"Saved visualizations to: {save_dir}")
    print(f"BLIP valid region from sample: {valid_size}")
    print(f"Selected proposal pairs: {keep.tolist()}")
    print(f"Union crop pixel boxes on 384x384 BLIP input: {crop_boxes}")
    print(f"Max absolute difference between manual crop and model crop: {max_abs_diff:.8f}")


def main():
    args = complete_args(build_parser().parse_args())
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.device = str(device)

    from src.models import build_model

    model, _, _ = build_model(args)
    model.to(device)
    model.eval()
    load_detr_state(model, args.detr_weights, device)
    load_state(model, args.checkpoint, device, "STIP")

    raw_img, resized_img, sample_tensor = preprocess_image(args.image, args.resize, device)
    samples = nested_tensor_from_tensor_list([sample_tensor])
    targets = [{"orig_size": torch.as_tensor([args.resize, args.resize], device=device), "image_id": torch.as_tensor(0, device=device)}]

    with torch.no_grad():
        outputs = model(samples, [resized_img], targets=targets)

    save_pair_visualizations(
        model=model,
        outputs=outputs,
        sample_tensor=samples.tensors[0],
        sample_mask=samples.mask[0],
        save_dir=args.save_dir,
        max_pairs=args.max_pairs,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
