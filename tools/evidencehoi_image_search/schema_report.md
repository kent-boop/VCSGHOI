# HOI Annotation Schema Report

Stage 1 only: candidate discovery and sampled schema inspection. No scoring, filtering, contact sheets, or visualization logic is implemented here.

## Scan Configuration

- Dataset root: `/share/share/Datasets/hico_20160224_det`
- Project root: `/wzj/zijian/OSSM-HOI-main`
- HICO-only mode: `True`
- Candidate files discovered: `16`

## Discovered Candidate Files

- `/share/share/Datasets/hico_20160224_det/anno.mat`
- `/share/share/Datasets/hico_20160224_det/anno_bbox.mat`
- `/share/share/Datasets/hico_20160224_det/annotations/corre_hico.npy`
- `/share/share/Datasets/hico_20160224_det/annotations/test_hico.json`
- `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico.json`
- `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico_15%.json`
- `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico_25%.json`
- `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico_5%.json`
- `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico_50%.json`
- `/share/share/Datasets/hico_20160224_det/hico_list_vb.txt`
- `/share/share/Datasets/hico_20160224_det/list_action.txt`
- `/wzj/zijian/OSSM-HOI-main/hico_20160224_det/hico_list_vb.txt`
- `/wzj/zijian/OSSM-HOI-main/hico_lib.py`
- `/wzj/zijian/OSSM-HOI-main/hico_lib.txt`
- `/wzj/zijian/OSSM-HOI-main/hico_text_descriptions.json`
- `/wzj/zijian/OSSM-HOI-main/hico_text_label.py`

## Per-File Schema Inspection

### `/share/share/Datasets/hico_20160224_det/anno.mat`

- Role guess: binary metadata / original annotation (listed only)
- Size: `2116152` bytes
- Top-level type: `mat binary`
- Image record keys: not found in samples
- BBox field names: not found in samples
- Object category field names: not found in samples
- HOI relation field names: not found in samples
- Subject id field names: not found in samples
- Object id field names: not found in samples
- Verb/action/category field names: not found in samples
- Image file name field names: not found in samples
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `not attempted`
- Warnings:
  - Not parsed: scipy is intentionally not required for Stage 1 inspection.
- Sample preview: none

### `/share/share/Datasets/hico_20160224_det/anno_bbox.mat`

- Role guess: binary metadata / original annotation (listed only)
- Size: `8697167` bytes
- Top-level type: `mat binary`
- Image record keys: not found in samples
- BBox field names: not found in samples
- Object category field names: not found in samples
- HOI relation field names: not found in samples
- Subject id field names: not found in samples
- Object id field names: not found in samples
- Verb/action/category field names: not found in samples
- Image file name field names: not found in samples
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `not attempted`
- Warnings:
  - Not parsed: scipy is intentionally not required for Stage 1 inspection.
- Sample preview: none

### `/share/share/Datasets/hico_20160224_det/annotations/corre_hico.npy`

- Role guess: binary metadata / original annotation (listed only)
- Size: `75008` bytes
- Top-level type: `npy binary`
- Image record keys: not found in samples
- BBox field names: not found in samples
- Object category field names: not found in samples
- HOI relation field names: not found in samples
- Subject id field names: not found in samples
- Object id field names: not found in samples
- Verb/action/category field names: not found in samples
- Image file name field names: not found in samples
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `not attempted`
- Warnings:
  - Not parsed: numpy is intentionally not required for Stage 1 inspection.
- Sample preview: none

### `/share/share/Datasets/hico_20160224_det/annotations/test_hico.json`

- Role guess: hico test annotation
- Size: `5409909` bytes
- Top-level type: `list`
- Image record keys: `annotations`, `file_name`, `hoi_annotation`
- BBox field names: `bbox`
- Object category field names: `category_id`
- HOI relation field names: `hoi_annotation`
- Subject id field names: `subject_id`
- Object id field names: `object_id`
- Verb/action/category field names: `category_id`
- Image file name field names: `file_name`
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `yes: opened `/share/share/Datasets/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg` -> width=640, height=427`
- Warnings: none
- Sample preview:
  - sample 1: `{'annotations': [{'bbox': [320, 306, 359, 349], 'category_id': 1}, {'bbox': [148, 345, 376, 414], 'category_id': 15}, {'bbox': [270, 303, 311, 350], 'category_id': 1}], 'file_name': 'HICO_test2015_00000001.jpg', 'hoi_annotation': [{'category_id': 88, 'object_id': 1, 'subject_id': 0}, {'category_id': 88, 'object_id': 1, 'subject_id': 2}]}`
  - sample 2: `{'annotations': [{'bbox': [226, 18, 340, 210], 'category_id': 1}, {'bbox': [174, 65, 393, 440], 'category_id': 19}, {'bbox': [230, 19, 356, 212], 'category_id': 1}, {'bbox': [186, 62, 392, 438], 'category_id': 19}, {'bbox': [234, 13, 342, 202], 'category_id': 1}, ...], 'file_name': 'HICO_test2015_00000002.jpg', 'hoi_annotation': [{'category_id': 37, 'object_id': 1, 'subject_id': 0}, {'category_id': 77, 'object_id': 3, 'subject_id': 2}, {'category_id': 111, 'object_id': 5, 'subject_id': 4}]}`
  - sample 3: `{'annotations': [{'bbox': [302, 182, 322, 236], 'category_id': 1}, {'bbox': [234, 85, 444, 298], 'category_id': 9}], 'file_name': 'HICO_test2015_00000003.jpg', 'hoi_annotation': [{'category_id': 77, 'object_id': 1, 'subject_id': 0}]}`
  - sample 4: `{'annotations': [{'bbox': [411, 83, 481, 186], 'category_id': 1}, {'bbox': [338, 109, 574, 247], 'category_id': 4}, {'bbox': [464, 118, 547, 221], 'category_id': 1}, {'bbox': [322, 171, 451, 331], 'category_id': 1}, {'bbox': [110, 183, 501, 364], 'category_id': 4}, ...], 'file_name': 'HICO_test2015_00000004.jpg', 'hoi_annotation': [{'category_id': 77, 'object_id': 1, 'subject_id': 0}, {'category_id': 77, 'object_id': 1, 'subject_id': 2}, {'category_id': 77, 'object_id': 4, 'subject_id': 3}, {'category_id': 77, 'object_id': 4, 'subject_id': 5}, {'category_id': 88, 'object_id': 7, 'subject_id': 6}, ...]}`
  - sample 5: `{'annotations': [{'bbox': [5, 62, 409, 638], 'category_id': 1}, {'bbox': [197, 167, 466, 638], 'category_id': 27}, {'bbox': [2, 60, 415, 637], 'category_id': 1}, {'bbox': [327, 165, 470, 524], 'category_id': 27}], 'file_name': 'HICO_test2015_00000005.jpg', 'hoi_annotation': [{'category_id': 9, 'object_id': 1, 'subject_id': 0}, {'category_id': 115, 'object_id': 3, 'subject_id': 2}]}`

### `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico.json`

- Role guess: hico trainval annotation
- Size: `20510344` bytes
- Top-level type: `list`
- Image record keys: `annotations`, `file_name`, `hoi_annotation`, `img_id`
- BBox field names: `bbox`
- Object category field names: `category_id`, `hoi_category_id`
- HOI relation field names: `hoi_annotation`
- Subject id field names: `subject_id`
- Object id field names: `object_id`
- Verb/action/category field names: `category_id`, `hoi_category_id`
- Image file name field names: `file_name`
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `yes: opened `/share/share/Datasets/hico_20160224_det/images/train2015/HICO_train2015_00000001.jpg` -> width=640, height=480`
- Warnings: none
- Sample preview:
  - sample 1: `{'annotations': [{'bbox': [207, 32, 426, 299], 'category_id': 1}, {'bbox': [58, 97, 571, 404], 'category_id': 4}], 'file_name': 'HICO_train2015_00000001.jpg', 'hoi_annotation': [{'category_id': 73, 'hoi_category_id': 153, 'object_id': 1, 'subject_id': 0}, {'category_id': 77, 'hoi_category_id': 154, 'object_id': 1, 'subject_id': 0}, {'category_id': 88, 'hoi_category_id': 155, 'object_id': 1, 'subject_id': 0}, {'category_id': 99, 'hoi_category_id': 156, 'object_id': 1, 'subject_id': 0}], 'img_id': 1}`
  - sample 2: `{'annotations': [{'bbox': [261, 0, 637, 137], 'category_id': 1}, {'bbox': [161, 130, 257, 164], 'category_id': 52}], 'file_name': 'HICO_train2015_00000002.jpg', 'hoi_annotation': [{'category_id': 58, 'hoi_category_id': 224, 'object_id': 1, 'subject_id': 0}], 'img_id': 2}`
  - sample 3: `{'annotations': [{'bbox': [566, 360, 579, 389], 'category_id': 1}, {'bbox': [73, 51, 509, 483], 'category_id': 6}], 'file_name': 'HICO_train2015_00000003.jpg', 'hoi_annotation': [{'category_id': 58, 'hoi_category_id': 65, 'object_id': 1, 'subject_id': 0}], 'img_id': 3}`
  - sample 4: `{'annotations': [{'bbox': [222, 28, 483, 367], 'category_id': 1}, {'bbox': [268, 88, 362, 121], 'category_id': 44}], 'file_name': 'HICO_train2015_00000004.jpg', 'hoi_annotation': [{'category_id': 21, 'hoi_category_id': 48, 'object_id': 1, 'subject_id': 0}, {'category_id': 37, 'hoi_category_id': 49, 'object_id': 1, 'subject_id': 0}], 'img_id': 4}`
  - sample 5: `{'annotations': [{'bbox': [81, 153, 202, 271], 'category_id': 1}, {'bbox': [0, 175, 312, 421], 'category_id': 65}], 'file_name': 'HICO_train2015_00000005.jpg', 'hoi_annotation': [{'category_id': 88, 'hoi_category_id': 242, 'object_id': 1, 'subject_id': 0}], 'img_id': 5}`

### `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico_15%.json`

- Role guess: hico trainval annotation
- Size: `5353566` bytes
- Top-level type: `list`
- Image record keys: `annotations`, `file_name`, `hoi_annotation`, `img_id`
- BBox field names: `bbox`
- Object category field names: `category_id`, `hoi_category_id`
- HOI relation field names: `hoi_annotation`
- Subject id field names: `subject_id`
- Object id field names: `object_id`
- Verb/action/category field names: `category_id`, `hoi_category_id`
- Image file name field names: `file_name`
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `yes: opened `/share/share/Datasets/hico_20160224_det/images/train2015/HICO_train2015_00033167.jpg` -> width=500, height=378`
- Warnings: none
- Sample preview:
  - sample 1: `{'annotations': [{'bbox': [118, 261, 133, 289], 'category_id': 1}, {'bbox': [66, 56, 263, 317], 'category_id': 9}, {'bbox': [91, 257, 106, 290], 'category_id': 1}, {'bbox': [114, 254, 148, 296], 'category_id': 1}, {'bbox': [52, 45, 268, 325], 'category_id': 9}, ...], 'file_name': 'HICO_train2015_00033167.jpg', 'hoi_annotation': [{'category_id': 77, 'hoi_category_id': 39, 'object_id': 1, 'subject_id': 0}, {'category_id': 77, 'hoi_category_id': 39, 'object_id': 1, 'subject_id': 2}, {'category_id': 94, 'hoi_category_id': 43, 'object_id': 1, 'subject_id': 2}, {'category_id': 80, 'hoi_category_id': 41, 'object_id': 4, 'subject_id': 3}, {'category_id': 94, 'hoi_category_id': 43, 'object_id': 6, 'subject_id': 5}], 'img_id': 33167}`
  - sample 2: `{'annotations': [{'bbox': [117, 282, 260, 637], 'category_id': 1}, {'bbox': [1, 5, 394, 585], 'category_id': 25}], 'file_name': 'HICO_train2015_00033173.jpg', 'hoi_annotation': [{'category_id': 27, 'hoi_category_id': 343, 'object_id': 1, 'subject_id': 0}, {'category_id': 113, 'hoi_category_id': 347, 'object_id': 1, 'subject_id': 0}], 'img_id': 33173}`
  - sample 3: `{'annotations': [{'bbox': [214, 27, 407, 329], 'category_id': 1}, {'bbox': [131, 266, 405, 370], 'category_id': 35}], 'file_name': 'HICO_train2015_00033174.jpg', 'hoi_annotation': [{'category_id': 44, 'hoi_category_id': 468, 'object_id': 1, 'subject_id': 0}, {'category_id': 77, 'hoi_category_id': 471, 'object_id': 1, 'subject_id': 0}, {'category_id': 94, 'hoi_category_id': 472, 'object_id': 1, 'subject_id': 0}, {'category_id': 115, 'hoi_category_id': 473, 'object_id': 1, 'subject_id': 0}], 'img_id': 33174}`
  - sample 4: `{'annotations': [{'bbox': [1, 1, 359, 255], 'category_id': 1}, {'bbox': [2, 71, 579, 261], 'category_id': 49}], 'file_name': 'HICO_train2015_00000021.jpg', 'hoi_annotation': [{'category_id': 17, 'hoi_category_id': 377, 'object_id': 1, 'subject_id': 0}, {'category_id': 37, 'hoi_category_id': 378, 'object_id': 1, 'subject_id': 0}, {'category_id': 116, 'hoi_category_id': 381, 'object_id': 1, 'subject_id': 0}], 'img_id': 21}`
  - sample 5: `{'annotations': [{'bbox': [326, 67, 431, 264], 'category_id': 1}, {'bbox': [404, 265, 481, 288], 'category_id': 35}, {'bbox': [316, 252, 400, 273], 'category_id': 35}, {'bbox': [317, 65, 435, 280], 'category_id': 1}, {'bbox': [312, 249, 482, 287], 'category_id': 35}], 'file_name': 'HICO_train2015_00000022.jpg', 'hoi_annotation': [{'category_id': 44, 'hoi_category_id': 468, 'object_id': 1, 'subject_id': 0}, {'category_id': 44, 'hoi_category_id': 468, 'object_id': 2, 'subject_id': 0}, {'category_id': 77, 'hoi_category_id': 471, 'object_id': 4, 'subject_id': 3}, {'category_id': 94, 'hoi_category_id': 472, 'object_id': 4, 'subject_id': 3}, {'category_id': 115, 'hoi_category_id': 473, 'object_id': 4, 'subject_id': 3}], 'img_id': 22}`

### `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico_25%.json`

- Role guess: hico trainval annotation
- Size: `8004015` bytes
- Top-level type: `list`
- Image record keys: `annotations`, `file_name`, `hoi_annotation`, `img_id`
- BBox field names: `bbox`
- Object category field names: `category_id`, `hoi_category_id`
- HOI relation field names: `hoi_annotation`
- Subject id field names: `subject_id`
- Object id field names: `object_id`
- Verb/action/category field names: `category_id`, `hoi_category_id`
- Image file name field names: `file_name`
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `yes: opened `/share/share/Datasets/hico_20160224_det/images/train2015/HICO_train2015_00033156.jpg` -> width=576, height=576`
- Warnings: none
- Sample preview:
  - sample 1: `{'annotations': [{'bbox': [196, 223, 287, 453], 'category_id': 1}, {'bbox': [17, 83, 370, 448], 'category_id': 28}, {'bbox': [190, 221, 292, 375], 'category_id': 1}, {'bbox': [10, 79, 426, 455], 'category_id': 28}], 'file_name': 'HICO_train2015_00033156.jpg', 'hoi_annotation': [{'category_id': 9, 'hoi_category_id': 577, 'object_id': 1, 'subject_id': 0}, {'category_id': 59, 'hoi_category_id': 580, 'object_id': 1, 'subject_id': 0}, {'category_id': 37, 'hoi_category_id': 578, 'object_id': 3, 'subject_id': 2}], 'img_id': 33156}`
  - sample 2: `{'annotations': [{'bbox': [104, 98, 151, 198], 'category_id': 1}, {'bbox': [106, 144, 123, 160], 'category_id': 58}, {'bbox': [164, 191, 250, 379], 'category_id': 1}, {'bbox': [227, 266, 249, 307], 'category_id': 58}, {'bbox': [440, 89, 630, 367], 'category_id': 1}, ...], 'file_name': 'HICO_train2015_00033159.jpg', 'hoi_annotation': [{'category_id': 24, 'hoi_category_id': 360, 'object_id': 1, 'subject_id': 0}, {'category_id': 24, 'hoi_category_id': 360, 'object_id': 3, 'subject_id': 2}, {'category_id': 37, 'hoi_category_id': 361, 'object_id': 5, 'subject_id': 4}], 'img_id': 33159}`
  - sample 3: `{'annotations': [{'bbox': [81, 153, 202, 271], 'category_id': 1}, {'bbox': [0, 175, 312, 421], 'category_id': 65}], 'file_name': 'HICO_train2015_00000005.jpg', 'hoi_annotation': [{'category_id': 88, 'hoi_category_id': 242, 'object_id': 1, 'subject_id': 0}], 'img_id': 5}`
  - sample 4: `{'annotations': [{'bbox': [168, 13, 419, 277], 'category_id': 1}, {'bbox': [195, 137, 399, 271], 'category_id': 84}, {'bbox': [191, 0, 412, 293], 'category_id': 1}, {'bbox': [212, 157, 381, 284], 'category_id': 84}], 'file_name': 'HICO_train2015_00033160.jpg', 'hoi_annotation': [{'category_id': 37, 'hoi_category_id': 249, 'object_id': 1, 'subject_id': 0}, {'category_id': 59, 'hoi_category_id': 250, 'object_id': 3, 'subject_id': 2}, {'category_id': 74, 'hoi_category_id': 251, 'object_id': 3, 'subject_id': 2}], 'img_id': 33160}`
  - sample 5: `{'annotations': [{'bbox': [2, 1, 109, 315], 'category_id': 1}, {'bbox': [0, 6, 488, 475], 'category_id': 15}, {'bbox': [500, 0, 636, 449], 'category_id': 1}], 'file_name': 'HICO_train2015_00033162.jpg', 'hoi_annotation': [{'category_id': 42, 'hoi_category_id': 244, 'object_id': 1, 'subject_id': 0}, {'category_id': 42, 'hoi_category_id': 244, 'object_id': 1, 'subject_id': 2}], 'img_id': 33162}`

### `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico_5%.json`

- Role guess: hico trainval annotation
- Size: `1673706` bytes
- Top-level type: `list`
- Image record keys: `annotations`, `file_name`, `hoi_annotation`, `img_id`
- BBox field names: `bbox`
- Object category field names: `category_id`, `hoi_category_id`
- HOI relation field names: `hoi_annotation`
- Subject id field names: `subject_id`
- Object id field names: `object_id`
- Verb/action/category field names: `category_id`, `hoi_category_id`
- Image file name field names: `file_name`
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `yes: opened `/share/share/Datasets/hico_20160224_det/images/train2015/HICO_train2015_00016609.jpg` -> width=480, height=640`
- Warnings: none
- Sample preview:
  - sample 1: `{'annotations': [{'bbox': [53, 9, 173, 403], 'category_id': 1}, {'bbox': [50, 256, 440, 636], 'category_id': 15}], 'file_name': 'HICO_train2015_00016609.jpg', 'hoi_annotation': [{'category_id': 42, 'hoi_category_id': 244, 'object_id': 1, 'subject_id': 0}], 'img_id': 16609}`
  - sample 2: `{'annotations': [{'bbox': [214, 52, 439, 258], 'category_id': 1}, {'bbox': [58, 278, 142, 320], 'category_id': 73}, {'bbox': [212, 45, 444, 260], 'category_id': 1}, {'bbox': [65, 326, 111, 341], 'category_id': 75}], 'file_name': 'HICO_train2015_00016612.jpg', 'hoi_annotation': [{'category_id': 58, 'hoi_category_id': 389, 'object_id': 1, 'subject_id': 0}, {'category_id': 58, 'hoi_category_id': 438, 'object_id': 3, 'subject_id': 2}], 'img_id': 16612}`
  - sample 3: `{'annotations': [{'bbox': [2, 2, 597, 420], 'category_id': 1}, {'bbox': [8, 261, 208, 422], 'category_id': 90}], 'file_name': 'HICO_train2015_00000008.jpg', 'hoi_annotation': [{'category_id': 7, 'hoi_category_id': 559, 'object_id': 1, 'subject_id': 0}, {'category_id': 37, 'hoi_category_id': 560, 'object_id': 1, 'subject_id': 0}], 'img_id': 8}`
  - sample 4: `{'annotations': [{'bbox': [61, 84, 242, 302], 'category_id': 1}, {'bbox': [125, 130, 293, 306], 'category_id': 4}, {'bbox': [230, 125, 433, 371], 'category_id': 1}, {'bbox': [311, 166, 551, 374], 'category_id': 4}], 'file_name': 'HICO_train2015_00016620.jpg', 'hoi_annotation': [{'category_id': 73, 'hoi_category_id': 153, 'object_id': 1, 'subject_id': 0}, {'category_id': 77, 'hoi_category_id': 154, 'object_id': 1, 'subject_id': 0}, {'category_id': 88, 'hoi_category_id': 155, 'object_id': 1, 'subject_id': 0}, {'category_id': 99, 'hoi_category_id': 156, 'object_id': 1, 'subject_id': 0}, {'category_id': 109, 'hoi_category_id': 157, 'object_id': 1, 'subject_id': 0}, ...], 'img_id': 16620}`
  - sample 5: `{'annotations': [{'bbox': [81, 125, 260, 305], 'category_id': 1}, {'bbox': [0, 230, 637, 478], 'category_id': 67}, {'bbox': [133, 113, 399, 349], 'category_id': 1}, {'bbox': [300, 105, 557, 395], 'category_id': 1}, {'bbox': [550, 240, 638, 397], 'category_id': 1}, ...], 'file_name': 'HICO_train2015_00000017.jpg', 'hoi_annotation': [{'category_id': 25, 'hoi_category_id': 109, 'object_id': 1, 'subject_id': 0}, {'category_id': 87, 'hoi_category_id': 110, 'object_id': 1, 'subject_id': 0}, {'category_id': 25, 'hoi_category_id': 109, 'object_id': 1, 'subject_id': 2}, {'category_id': 87, 'hoi_category_id': 110, 'object_id': 1, 'subject_id': 2}, {'category_id': 25, 'hoi_category_id': 109, 'object_id': 1, 'subject_id': 3}, ...], 'img_id': 17}`

### `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico_50%.json`

- Role guess: hico trainval annotation
- Size: `13070691` bytes
- Top-level type: `list`
- Image record keys: `annotations`, `file_name`, `hoi_annotation`, `img_id`
- BBox field names: `bbox`
- Object category field names: `category_id`, `hoi_category_id`
- HOI relation field names: `hoi_annotation`
- Subject id field names: `subject_id`
- Object id field names: `object_id`
- Verb/action/category field names: `category_id`, `hoi_category_id`
- Image file name field names: `file_name`
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `yes: opened `/share/share/Datasets/hico_20160224_det/images/train2015/HICO_train2015_00000001.jpg` -> width=640, height=480`
- Warnings: none
- Sample preview:
  - sample 1: `{'annotations': [{'bbox': [207, 32, 426, 299], 'category_id': 1}, {'bbox': [58, 97, 571, 404], 'category_id': 4}], 'file_name': 'HICO_train2015_00000001.jpg', 'hoi_annotation': [{'category_id': 73, 'hoi_category_id': 153, 'object_id': 1, 'subject_id': 0}, {'category_id': 77, 'hoi_category_id': 154, 'object_id': 1, 'subject_id': 0}, {'category_id': 88, 'hoi_category_id': 155, 'object_id': 1, 'subject_id': 0}, {'category_id': 99, 'hoi_category_id': 156, 'object_id': 1, 'subject_id': 0}], 'img_id': 1}`
  - sample 2: `{'annotations': [{'bbox': [87, 92, 355, 226], 'category_id': 1}, {'bbox': [219, 140, 357, 226], 'category_id': 41}], 'file_name': 'HICO_train2015_00033157.jpg', 'hoi_annotation': [{'category_id': 77, 'hoi_category_id': 460, 'object_id': 1, 'subject_id': 0}, {'category_id': 94, 'hoi_category_id': 462, 'object_id': 1, 'subject_id': 0}], 'img_id': 33157}`
  - sample 3: `{'annotations': [{'bbox': [402, 137, 447, 173], 'category_id': 1}, {'bbox': [72, 119, 614, 338], 'category_id': 3}, {'bbox': [447, 129, 515, 194], 'category_id': 1}, {'bbox': [398, 132, 460, 184], 'category_id': 1}, {'bbox': [75, 121, 611, 344], 'category_id': 3}, ...], 'file_name': 'HICO_train2015_00033158.jpg', 'hoi_annotation': [{'category_id': 22, 'hoi_category_id': 68, 'object_id': 1, 'subject_id': 0}, {'category_id': 22, 'hoi_category_id': 68, 'object_id': 1, 'subject_id': 2}, {'category_id': 77, 'hoi_category_id': 74, 'object_id': 4, 'subject_id': 3}, {'category_id': 77, 'hoi_category_id': 74, 'object_id': 4, 'subject_id': 5}], 'img_id': 33158}`
  - sample 4: `{'annotations': [{'bbox': [222, 28, 483, 367], 'category_id': 1}, {'bbox': [268, 88, 362, 121], 'category_id': 44}], 'file_name': 'HICO_train2015_00000004.jpg', 'hoi_annotation': [{'category_id': 21, 'hoi_category_id': 48, 'object_id': 1, 'subject_id': 0}, {'category_id': 37, 'hoi_category_id': 49, 'object_id': 1, 'subject_id': 0}], 'img_id': 4}`
  - sample 5: `{'annotations': [{'bbox': [322, 3, 412, 254], 'category_id': 1}, {'bbox': [148, 97, 509, 373], 'category_id': 19}, {'bbox': [417, 51, 475, 165], 'category_id': 1}, {'bbox': [259, 117, 620, 375], 'category_id': 19}, {'bbox': [331, 4, 429, 299], 'category_id': 1}, ...], 'file_name': 'HICO_train2015_00033161.jpg', 'hoi_annotation': [{'category_id': 77, 'hoi_category_id': 140, 'object_id': 1, 'subject_id': 0}, {'category_id': 99, 'hoi_category_id': 142, 'object_id': 3, 'subject_id': 2}, {'category_id': 99, 'hoi_category_id': 142, 'object_id': 5, 'subject_id': 4}], 'img_id': 33161}`

### `/share/share/Datasets/hico_20160224_det/hico_list_vb.txt`

- Role guess: hico label/name metadata
- Size: `2379` bytes
- Top-level type: `text`
- Image record keys: not found in samples
- BBox field names: not found in samples
- Object category field names: not found in samples
- HOI relation field names: not found in samples
- Subject id field names: not found in samples
- Object id field names: not found in samples
- Verb/action/category field names: not found in samples
- Image file name field names: not found in samples
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `not attempted`
- Warnings: none
- Sample preview:
  - sample 1: `'id   verb          '`
  - sample 2: `'-------------------'`
  - sample 3: `'001  adjust        '`
  - sample 4: `'002  assemble      '`
  - sample 5: `'003  block         '`

### `/share/share/Datasets/hico_20160224_det/list_action.txt`

- Role guess: hico label/name metadata
- Size: `2379` bytes
- Top-level type: `text`
- Image record keys: not found in samples
- BBox field names: not found in samples
- Object category field names: not found in samples
- HOI relation field names: not found in samples
- Subject id field names: not found in samples
- Object id field names: not found in samples
- Verb/action/category field names: not found in samples
- Image file name field names: not found in samples
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `not attempted`
- Warnings: none
- Sample preview:
  - sample 1: `'id   verb          '`
  - sample 2: `'-------------------'`
  - sample 3: `'001  adjust        '`
  - sample 4: `'002  assemble      '`
  - sample 5: `'003  block         '`

### `/wzj/zijian/OSSM-HOI-main/hico_20160224_det/hico_list_vb.txt`

- Role guess: hico label/name metadata
- Size: `2379` bytes
- Top-level type: `text`
- Image record keys: not found in samples
- BBox field names: not found in samples
- Object category field names: not found in samples
- HOI relation field names: not found in samples
- Subject id field names: not found in samples
- Object id field names: not found in samples
- Verb/action/category field names: not found in samples
- Image file name field names: not found in samples
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `not attempted`
- Warnings: none
- Sample preview:
  - sample 1: `'id   verb          '`
  - sample 2: `'-------------------'`
  - sample 3: `'001  adjust        '`
  - sample 4: `'002  assemble      '`
  - sample 5: `'003  block         '`

### `/wzj/zijian/OSSM-HOI-main/hico_lib.py`

- Role guess: candidate
- Size: `58627` bytes
- Top-level type: `python metadata`
- Image record keys: `assignment`, `items`, `type`
- BBox field names: not found in samples
- Object category field names: not found in samples
- HOI relation field names: not found in samples
- Subject id field names: not found in samples
- Object id field names: not found in samples
- Verb/action/category field names: not found in samples
- Image file name field names: not found in samples
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `not needed or no image file field found in sampled records`
- Warnings: none
- Sample preview:
  - sample 1: `{'assignment': 'hico_lib', 'items': {(4, 4): ['a person boarding an airplane', 'an airplane near a person ', 'a person with a bent leg '], (17, 4): ['a person directing an airplane', 'a person near an airplane', 'a person directing an airplane in road '], (25, 4): ['a person exiting an airplane', 'a person body off an airplane', 'a person near an airplane'], (30, 4): ['a person flying an airplane', 'an airplane in air ', 'a person sitting in an airplane'], (41, 4): ['a person inspecting an airplane', 'a person hand on an airplane', 'a person looking at an airplane']}, 'type': 'dict'}`

### `/wzj/zijian/OSSM-HOI-main/hico_lib.txt`

- Role guess: hico label/name metadata
- Size: `10177585` bytes
- Top-level type: `text`
- Image record keys: not found in samples
- BBox field names: not found in samples
- Object category field names: not found in samples
- HOI relation field names: not found in samples
- Subject id field names: not found in samples
- Object id field names: not found in samples
- Verb/action/category field names: not found in samples
- Image file name field names: not found in samples
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `not attempted`
- Warnings: none
- Sample preview:
  - sample 1: `'-3.267727419734001160e-02,-1.111887767910957336e-01,7.774376776069402695e-03,3.11793759...1363220e-02,4.950321838259696960e-02,-4.967294633388519287e-02,-3.769907727837562561e-02'`
  - sample 2: `'1.531997788697481155e-02,-3.502087667584419250e-02,-3.365091979503631592e-02,4.66880276...1118622e-03,4.717968869954347610e-03,-1.548361126333475113e-02,-4.542430490255355835e-02'`
  - sample 3: `'-2.261216007173061371e-02,1.133783161640167236e-02,-9.462771937251091003e-03,4.62239608...3559570e-02,6.787642091512680054e-02,-4.637322854250669479e-03,-3.994899615645408630e-02'`
  - sample 4: `'1.412946917116641998e-02,-7.187303900718688965e-02,-7.854485511779785156e-02,7.34131634...6094589e-03,4.904643073678016663e-02,-1.749442517757415771e-02,-8.869620412588119507e-02'`
  - sample 5: `'-1.233038259670138359e-03,-4.994899034500122070e-02,-4.753134399652481079e-02,5.3983572...8962555e-03,1.926359022036194801e-03,-1.802733354270458221e-02,-3.761829063296318054e-02'`

### `/wzj/zijian/OSSM-HOI-main/hico_text_descriptions.json`

- Role guess: hico label/name metadata
- Size: `181426` bytes
- Top-level type: `dict`
- Image record keys: `background`, `hoi_id`, `human_object`, `obj_phrase`, `verb_phrase`
- BBox field names: not found in samples
- Object category field names: not found in samples
- HOI relation field names: not found in samples
- Subject id field names: not found in samples
- Object id field names: not found in samples
- Verb/action/category field names: not found in samples
- Image file name field names: not found in samples
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `not needed or no image file field found in sampled records`
- Warnings: none
- Sample preview:
  - sample 1: `{'background': 'Airport tarmac / Boarding gate / Airplane cabin', 'hoi_id': '4_4', 'human_object': 'Person standing upright, one hand gripping suitcase handle, other hand free.', 'obj_phrase': 'an airplane', 'verb_phrase': 'boarding'}`
  - sample 2: `{'background': 'Airport tarmac', 'hoi_id': '17_4', 'human_object': 'Arms extended, palms facing outward, signaling with hands; plane stationary', 'obj_phrase': 'an airplane', 'verb_phrase': 'directing'}`
  - sample 3: `{'background': 'Airport tarmac / Boarding gate area / Airplane door', 'hoi_id': '25_4', 'human_object': 'Person standing upright, one hand gripping suitcase handle', 'obj_phrase': 'an airplane', 'verb_phrase': 'exiting'}`
  - sample 4: `{'background': 'Sky with clouds / Airfield runway / Mountain landscape', 'hoi_id': '30_4', 'human_object': 'Hands gripping control yoke, arms extended forward, seated upright', 'obj_phrase': 'an airplane', 'verb_phrase': 'flying'}`
  - sample 5: `{'background': 'Airport tarmac / Hangar interior / Maintenance facility', 'hoi_id': '41_4', 'human_object': 'Person standing upright, hands flat against airplane fuselage, fingers spread wide.', 'obj_phrase': 'an airplane', 'verb_phrase': 'inspecting'}`

### `/wzj/zijian/OSSM-HOI-main/hico_text_label.py`

- Role guess: hico label/name metadata
- Size: `47746` bytes
- Top-level type: `python metadata`
- Image record keys: `assignment`, `items`, `type`
- BBox field names: not found in samples
- Object category field names: not found in samples
- HOI relation field names: not found in samples
- Subject id field names: not found in samples
- Object id field names: not found in samples
- Verb/action/category field names: not found in samples
- Image file name field names: not found in samples
- Width field names: not found in samples
- Height field names: not found in samples
- Need PIL for width/height: `not needed or no image file field found in sampled records`
- Warnings: none
- Sample preview:
  - sample 1: `{'assignment': 'hico_text_label', 'items': {(4, 4): 'a photo of a person boarding an airplane', (17, 4): 'a photo of a person directing an airplane', (25, 4): 'a photo of a person exiting an airplane', (30, 4): 'a photo of a person flying an airplane', (41, 4): 'a photo of a person inspecting an airplane'}, 'type': 'dict'}`
  - sample 2: `{'assignment': 'hico_obj_text_label', 'items': [(0, 'a photo of a person'), (1, 'a photo of a bicycle'), (2, 'a photo of a car'), (3, 'a photo of a motorcycle'), (4, 'a photo of an airplane')], 'type': 'list'}`
  - sample 3: `{'assignment': 'hico_unseen_index', 'items': {'default': [], 'non_rare_first': [38, 41, 20, 18, 245, ...], 'rare_first': [509, 279, 280, 402, 504, ...], 'unseen_object': [111, 112, 113, 114, 115, ...], 'unseen_verb': [4, 6, 12, 15, 18, ...]}, 'type': 'dict'}`

## Recommended Inputs for the Unified Loader

- HICO trainval annotation: `/share/share/Datasets/hico_20160224_det/annotations/trainval_hico.json`
- HICO test annotation: `/share/share/Datasets/hico_20160224_det/annotations/test_hico.json`
- HICO verb names: `/share/share/Datasets/hico_20160224_det/list_action.txt`
- Alternate HICO verb names: `/share/share/Datasets/hico_20160224_det/hico_list_vb.txt`
- HICO text label metadata: `/wzj/zijian/OSSM-HOI-main/hico_text_label.py`
- HICO compatibility metadata: `/share/share/Datasets/hico_20160224_det/annotations/corre_hico.npy`
- V-COCO train/test inputs: intentionally not scanned in this run per current scope.

## Fields Still Uncertain

- `trainval_hico.json` includes `hoi_category_id`; `test_hico.json` samples do not, so test HOI triplet labels may require verb id plus object category mapping.
- JSON image records do not include width/height in sampled HICO records; geometry-aware scoring should open images with PIL when needed.
- `.mat` and `.npy` metadata files are discovered but not parsed in this Stage 1 script because only standard library, pickle, json, pathlib, and PIL are used.
- Fractional HICO annotation files appear to share the trainval schema but should not be used as the default full-data source unless explicitly requested.
