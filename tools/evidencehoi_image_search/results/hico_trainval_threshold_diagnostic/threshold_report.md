# EvidenceHOI Threshold Diagnostics

## Run Summary

- records: `tools/evidencehoi_image_search/cache/hico_trainval_records.jsonl`
- image_root: `/share/share/Datasets/hico_20160224_det/images/train2015`
- output_dir: `tools/evidencehoi_image_search/results/hico_trainval_threshold_diagnostic`
- total_images: `37633`
- strict_figure1_retained: `0`
- target_min_candidates: `100`
- target_max_candidates: `1000`
- status: `too_few_candidates`

## Strict Figure 1 Thresholds Used

- num_unique_action_categories >= `3`
- max_unique_person_object_matching >= `3`
- primary_pair_context_count >= `2`
- min_valid_pairs >= `3`
- max_valid_pairs <= `8`
- min_mean_pair_context_count_top3 >= `1.0`
- context_expand_scale = `1.75`

## Per-Rule Filtering Counts

| Rule | Images Failing Rule | Only This Rule Failed |
| --- | ---: | ---: |
| `num_persons outside range` (`reject_num_persons`) | 23367 | 0 |
| `num_valid_hoi_pairs outside range` (`reject_num_valid_hoi_pairs`) | 30950 | 0 |
| `total_gt_boxes outside range` (`reject_total_gt_boxes`) | 31835 | 0 |
| `num_unique_object_categories too low` (`reject_unique_object_categories`) | 37625 | 0 |
| `num_unique_action_categories too low` (`reject_unique_action_categories`) | 29075 | 0 |
| `max_unique_person_object_matching too low` (`reject_matching`) | 32946 | 0 |
| `primary pair visibility too low` (`reject_main_person_area`) | 6953 | 0 |
| `secondary pair visibility too low` (`reject_secondary_person_area`) | 3550 | 0 |
| `primary_pair_context_count too low` (`reject_primary_pair_context`) | 23886 | 0 |
| `mean_pair_context_count_top3 too low` (`reject_mean_context_top3`) | 22129 | 0 |
| `human-human overlap too high` (`reject_human_overlap`) | 9335 | 2 |
| `dominant_action_group_ratio too high` (`reject_action_homogeneity`) | 19771 | 0 |
| `dominant_object_category_ratio too high` (`reject_object_homogeneity`) | 36651 | 0 |
| `sports scene dominance too high` (`reject_severe_sports`) | 6212 | 0 |

## Most Common Reject Reasons

- `reject_unique_object_categories`: `37625`
- `reject_object_homogeneity`: `36651`
- `reject_matching`: `32946`
- `reject_total_gt_boxes`: `31835`
- `reject_num_valid_hoi_pairs`: `30950`
- `reject_unique_action_categories`: `29075`
- `reject_primary_pair_context`: `23886`
- `reject_num_persons`: `23367`
- `reject_mean_context_top3`: `22129`
- `reject_action_homogeneity`: `19771`
- `reject_human_overlap`: `9335`
- `reject_main_person_area`: `6953`
- `reject_severe_sports`: `6212`
- `reject_secondary_person_area`: `3550`

## Most Common Reject Reason Combinations

- `reject_num_persons,reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_unique_action_categories,reject_matching,reject_primary_pair_context,reject_mean_context_top3,reject_action_homogeneity,reject_object_homogeneity`: `10477`
- `reject_num_persons,reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_unique_action_categories,reject_matching,reject_primary_pair_context,reject_mean_context_top3,reject_object_homogeneity`: `2826`
- `reject_num_persons,reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_matching,reject_primary_pair_context,reject_mean_context_top3,reject_object_homogeneity`: `2008`
- `reject_num_persons,reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_unique_action_categories,reject_matching,reject_primary_pair_context,reject_mean_context_top3,reject_action_homogeneity,reject_object_homogeneity,reject_severe_sports`: `1462`
- `reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_unique_action_categories,reject_matching,reject_human_overlap,reject_object_homogeneity`: `1258`
- `reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_matching,reject_human_overlap,reject_object_homogeneity`: `992`
- `reject_unique_object_categories,reject_human_overlap,reject_object_homogeneity`: `859`
- `reject_num_persons,reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_unique_action_categories,reject_matching,reject_primary_pair_context,reject_mean_context_top3,reject_object_homogeneity,reject_severe_sports`: `855`
- `reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_unique_action_categories,reject_matching,reject_primary_pair_context,reject_action_homogeneity,reject_object_homogeneity`: `783`
- `reject_num_persons,reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_unique_action_categories,reject_matching,reject_main_person_area,reject_secondary_person_area,reject_primary_pair_context,reject_mean_context_top3,reject_action_homogeneity,reject_object_homogeneity`: `748`
- `reject_num_persons,reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_matching,reject_primary_pair_context,reject_mean_context_top3,reject_object_homogeneity,reject_severe_sports`: `702`
- `reject_num_persons,reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_unique_action_categories,reject_matching,reject_main_person_area,reject_primary_pair_context,reject_mean_context_top3,reject_action_homogeneity,reject_object_homogeneity`: `633`
- `reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_unique_action_categories,reject_matching,reject_main_person_area,reject_secondary_person_area,reject_primary_pair_context,reject_mean_context_top3`: `527`
- `reject_num_valid_hoi_pairs,reject_total_gt_boxes,reject_unique_object_categories,reject_matching,reject_human_overlap,reject_object_homogeneity,reject_severe_sports`: `519`
- `reject_total_gt_boxes,reject_unique_object_categories,reject_unique_action_categories,reject_matching,reject_action_homogeneity,reject_object_homogeneity`: `514`

## Top Retained Object And Action Patterns

- No retained candidates; object/action patterns are unavailable.

## Sports Scene Check

- sports scenes appear frequent enough to monitor; severe_sports_rejects=6212/37633 (16.5%), retained_sports=0/0 (0.0%)

## Recommendation

- Candidate count is below target_min_candidates; relax action diversity, matching, and primary context.
- Try context expansion scale in the 1.5 to 2.0 range; the command below uses 2.0 for recall.

## Recommended Next Command

```bash
python \
  tools/evidencehoi_image_search/filter_candidates.py \
  --records \
  tools/evidencehoi_image_search/cache/hico_trainval_records.jsonl \
  --image_root \
  /share/share/Datasets/hico_20160224_det/images/train2015 \
  --output_dir \
  tools/evidencehoi_image_search/results/hico_trainval_threshold_diagnostic \
  --mode \
  relaxed \
  --min_unique_action_categories \
  2 \
  --min_matching \
  2 \
  --min_primary_pair_context_count \
  1 \
  --context_expand_scale \
  2.0 \
  --auto_threshold_report
```

## Relaxed Settings To Try

- num_unique_action_categories >= `2`
- max_unique_person_object_matching >= `2`
- primary_pair_context_count >= `1`
- context_expand_scale from `1.5` to `2.0`
