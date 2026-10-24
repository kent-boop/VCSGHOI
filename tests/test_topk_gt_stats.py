from types import SimpleNamespace
import json

from src.engine.topk_gt_stats import TopKGTStats
from src.engine.arg_parser import get_args_parser


def test_flags_default_off_and_validation_alias():
    parser = get_args_parser()
    assert not parser.parse_args([]).ossm_topk_gt_stats
    args = parser.parse_args(['--validation', '--ossm_topk_gt_stats'])
    assert args.eval and args.ossm_topk_gt_stats


def test_sparse_object_mapping_duplicates_missing_and_empty_gt(tmp_path):
    # COCO sparse label 5 is contiguous index 4 (airplane).
    recorder = TopKGTStats([1, 2, 3, 4, 5])
    target = {'id': 9, 'file_name': 'test.jpg', 'labels': [1, 5],
              'hois': [[0, 1, 4], [0, 1, 4], [0, 1, 17], [0, 1, 57]]}
    selected = {'scene': [1], 'instance': [0, 1], 'action': [0]}
    record = recorder.record(target, selected)
    assert record['gt'][0]['text_row'] == 0
    assert record['gt'][0]['rank_in_topk']['action'] == 1
    assert record['gt'][2]['hit']['action'] is False
    assert record['gt'][3]['in_library'] is False
    empty = recorder.record(dict(target, id=10, hois=[]), selected)
    summary = recorder.save([record, record, empty], tmp_path, SimpleNamespace())
    assert summary['num_images'] == 2
    assert summary['excluded_no_interaction_instances'] == 1
    action = summary['branches']['action']
    assert action['image_class_recall'] == 0.5
    assert action['instance_recall'] == 2 / 3
    assert action['eligible_images'] == 1
    assert action['all_hit_image_rate'] == 0
    assert summary['branches']['instance']['all_hit_image_rate'] == 1
    assert len((tmp_path / 'topk_gt_stats/per_image.jsonl').read_text().splitlines()) == 2
    assert json.loads((tmp_path / 'topk_gt_stats/summary.json').read_text()) == summary


def test_no_eligible_gt_has_no_artificial_perfect_recall(tmp_path):
    recorder = TopKGTStats([1, 2, 3, 4, 5])
    target = {'id': 1, 'labels': [1, 5], 'hois': [[0, 1, 57]]}
    record = recorder.record(target, {'scene': [0], 'instance': [0], 'action': [0]})
    summary = recorder.save([record], tmp_path, SimpleNamespace())
    assert summary['branches']['action']['image_class_recall'] is None
    assert summary['branches']['action']['all_hit_images'] == 0
