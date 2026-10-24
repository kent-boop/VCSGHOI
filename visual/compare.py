import json
ap_path1 = 'visual/recall_AP1.json'
ap_path2 = '/home/mahanwei/Documents/STIP-main/visual/recall_AP1.json'

with open(ap_path1, 'r') as f:
    ap_IATN = json.load(f)
with open(ap_path2, 'r') as f:
    ap_STIP = json.load(f)

image_ids = list(ap_IATN.keys())
IATN_superiority = []
STIP_superiority = []
for num_id, img_id in enumerate(image_ids):
    print(num_id, img_id)
    try:
        if ap_IATN[img_id] > ap_STIP[img_id]:
            IATN_superiority.append(img_id)
        elif ap_IATN[img_id] < ap_STIP[img_id]:
            STIP_superiority.append(img_id)
    except Exception as e:
        pass

with open('visual/compare.txt', 'w') as f:
    f.write('IATN_superiority')
    f.write('\n')
    for IATN_id in IATN_superiority:
        f.write(f'{IATN_id}: IATN: {ap_IATN[IATN_id]}, STIP: {ap_STIP[IATN_id]}')
        f.write('\n')
    f.write('STIP_superiority')
    for STIP_id in STIP_superiority:
        f.write(f'{STIP_id}: STIP: {ap_STIP[STIP_id]}, IATN: {ap_IATN[STIP_id]}')
        f.write('\n')



