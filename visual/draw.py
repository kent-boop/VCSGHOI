import cv2
import os
import json
import torch
import numpy as np

source_path = 'visual/compare_new.txt'
anno_path = 'visual/vcoco_test_line.odgt'

with open('visual/vcoco_verb_names.json', 'r') as f:
    for verb_list in f.readlines():
        item = json.loads(verb_list)
        hoi_interaction_names = item['verb_names']

coco_verbs_ID = ["__background__", "surf", "ski", "cut", "walk", "cut", "ride", "talk_on", "kick", "work_on", "eat",
                "sit", "surf", "lay", "drink", "carry", "throw", "eat", "smile", "look", "hit", "hit", "snowboard", "run", "point", # surf was jump
                "read", "hold", "skateboard", "stand", "catch"]
coco_classes_originID = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90,
}

def convert_xywh2x1y1x2y2(box, shape, flip):
    ih, iw = shape[:2]
    x, y, w, h = box
    if flip == 1:
        x1_org = x
        x2_org = x + w - 1
        x2 = iw - 1 - x1_org
        x1 = iw - 1 - x2_org
    else:
        x1 = x
        x2 = x + w - 1
    x1 = max(x1, 0)
    x2 = min(x2, iw-1)
    y1 = max(y, 0)
    y2 = min(y + h - 1, ih-1)
    return [x1, y1, x2, y2]

def get_interaction_box(human_box, object_box, hoi_id):
    hx1, hy1, hx2, hy2, hid = human_box
    ox1, oy1, ox2, oy2, oid = object_box
    xx1, yy1, xx2, yy2 = min(hx1, ox1), min(hy1, oy1), max(hx2, ox2), max(hy2, oy2)
    return [xx1, yy1, xx2, yy2, hoi_id]

def get_det_annotation_from_odgt(item, shape, flip, gt_size_min=1):
    total_boxes, gt_boxes, ignored_boxes = [], [], []
    for annot in item['gtboxes']:
        box = convert_xywh2x1y1x2y2(annot['box'], shape, flip)
        x1, y1, x2, y2 = box
        cls_id = coco_classes_originID[annot['tag']]
        total_boxes.append([x1, y1, x2, y2, cls_id, ])
        if annot['tag'] not in coco_classes_originID:
            continue
        if annot.get('extra', {}).get('ignore', 0) == 1:
            ignored_boxes.append(box)
            continue
        if (x2 - x1 + 1) * (y2 - y1 + 1) < gt_size_min ** 2:
            ignored_boxes.append(box)
            continue
        if x2 <= x1 or y2 <= y1:
            ignored_boxes.append(box)
            continue
        gt_boxes.append([x1, y1, x2, y2, cls_id, ])
    return gt_boxes, ignored_boxes, total_boxes

def get_hoi_annotation_from_odgt(item, total_boxes, scale):
    human_boxes, object_boxes, action_boxes = [], [], []
    human_labels, object_labels, action_labels = [], [], []
    img_hh, img_ww = item['height'], item['width']
    for hoi in item.get('hoi', []):
        x1, y1, x2, y2, cls_id = list(map(int, total_boxes[hoi['subject_id']]))
        human_box = x1 // scale, y1 // scale, x2 // scale, y2 // scale, cls_id
        if cls_id == -1 or x1 >= x2 or y1 >= y2:
            continue
        x1, y1, x2, y2, cls_id = list(map(int, total_boxes[hoi['object_id']]))
        object_box = x1 // scale, y1 // scale, x2 // scale, y2 // scale, cls_id
        if cls_id == -1 or x1 >= x2 or y1 >= y2:
            continue
        hoi_id = hoi_interaction_names.index(hoi['interaction'])
        hoi_box = get_interaction_box(human_box=human_box, object_box=object_box, hoi_id=hoi_id)

        human_boxes.append(human_box[0:4])
        object_boxes.append(object_box[0:4])
        action_boxes.append(hoi_box[0:4])
        human_labels.append(human_box[4])
        object_labels.append(object_box[4])
        action_labels.append(hoi_box[4])
    return dict(
        human_boxes=torch.from_numpy(np.array(human_boxes).astype(np.float32)),
        human_labels=torch.from_numpy(np.array(human_labels)),
        object_boxes=torch.from_numpy(np.array(object_boxes).astype(np.float32)),
        object_labels=torch.from_numpy(np.array(object_labels)),
        action_boxes=torch.from_numpy(np.array(action_boxes).astype(np.float32)),
        action_labels=torch.from_numpy(np.array(action_labels)),
        image_id=item['file_name'],
        org_size=torch.as_tensor([int(img_hh), int(img_ww)]),
    )

def main():
    annos_dict = {}
    with open(anno_path, 'r') as f:
        for anno_line in f.readlines():
            item = json.loads(anno_line)
            img_name = item['file_name']
            img_shape = item['height'], item['width']
            gt_boxes, ignored_boxes, total_boxes = get_det_annotation_from_odgt(item, img_shape, flip=0)
            interaction_boxes = get_hoi_annotation_from_odgt(item, total_boxes, 1)
            annos_dict.update({img_name: interaction_boxes})

    with open(source_path, 'r') as f:
        for img_data in f.readlines():
            img_data = img_data.strip('\n')
            img_id = img_data.split(':')[0]
            img_name = 'COCO_val2014_' + img_id.zfill(12) + '.jpg'
            img_path = 'v-coco/coco/images/val2014/' + img_name
            IATN_score = eval(img_data.split(',')[0].split(': ')[2])
            STIP_score = eval(img_data.split('STIP: ')[1])
            if IATN_score-STIP_score > 0:
                try:
                    anno = annos_dict[img_name]
                    img_src = cv2.imread(img_path)
                    human_boxes = np.array(anno['human_boxes']).astype(int)
                    object_boxes = np.array(anno['object_boxes']).astype(int)
                    object_labels = np.array(anno['object_labels']).astype(int)
                    action_labels = np.array(anno['action_labels']).astype(int)
                    text_T_list = {}
                    for num in range(len(human_boxes)):
                        T_flag = True
                        human_box_ptLT = human_boxes[num][:2]
                        human_box_ptRB = human_boxes[num][2:]
                        object_box_ptLT = object_boxes[num][:2]
                        object_box_ptBB = object_boxes[num][2:]
                        human_point = (int((human_box_ptLT[0]+human_box_ptRB[0])/2), int((human_box_ptLT[1]+human_box_ptRB[1])/2))
                        object_point = (int((object_box_ptLT[0]+object_box_ptBB[0])/2), int((object_box_ptLT[1]+object_box_ptBB[1])/2))
                        action_point = (int((human_point[0]+object_point[0])/2), int((human_point[1]+object_point[1])/2))
                        action_name = coco_verbs_ID[action_labels[num]]
                        text_size = cv2.getTextSize(action_name, cv2.FONT_HERSHEY_PLAIN, 4, 2)[0]
                        text_LT = (int(action_point[0]-text_size[0]/2), action_point[1]-5-text_size[1])
                        if text_LT[1] in text_T_list:
                            continue # for compare only
                            text_LT = (int(action_point[0]-text_size[0]/2), action_point[1]-12-2*text_size[1])
                            T_flag = False
                        text_T_list.update({text_LT[1]: 0})
                        text_RB = (text_LT[0]+text_size[0], text_LT[1]+text_size[1])

                        human_color = (196, 114, 68)
                        object_color = (0, 0, 255)
                        action_color = (49, 125, 237)
                        background_color = (240, 240, 240)
                        thickness = 3
                        lineType = 4
                        point_size = 1
                        cv2.rectangle(img_src, tuple(human_box_ptLT), tuple(human_box_ptRB), human_color, thickness, lineType)
                        cv2.rectangle(img_src, tuple(object_box_ptLT), tuple(object_box_ptBB), object_color, thickness, lineType)
                        cv2.rectangle(img_src, text_LT, text_RB, background_color, -1)
                        cv2.circle(img_src, human_point, point_size, human_color, thickness)
                        cv2.circle(img_src, object_point, point_size, object_color, thickness)
                        cv2.circle(img_src, action_point, point_size, action_color, thickness)
                        cv2.line(img_src, human_point, action_point, human_color, thickness)
                        cv2.line(img_src, action_point, object_point, object_color, thickness)
                        if T_flag:
                            text_LT = (int(action_point[0]-text_size[0]/2), action_point[1] - 6)
                        else:
                            text_LT = (int(action_point[0]-text_size[0]/2), action_point[1]-12-1*text_size[1])
                        cv2.putText(img_src, action_name, text_LT, cv2.FONT_HERSHEY_PLAIN, 4, action_color, 2)


                    cv2.imwrite(f'visual/drawed_pics_compare/{img_name}', img_src)
                except Exception as e:
                    continue

if __name__ == '__main__':
    main()