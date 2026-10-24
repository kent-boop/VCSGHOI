"""Image-level CLS text retrieval diagnostics; no detector matching is involved."""
import json
from pathlib import Path

from hico_lib import hico_lib
from hico_text_label import hico_text_label


class TopKGTStats:
    def __init__(self, object_ids):
        # The embedding-generation loop iterates hico_lib in insertion order.
        self.keys = list(hico_lib)
        if len(self.keys) != 520 or any(verb == 57 for verb, _ in self.keys):
            raise ValueError('Expected the original 520-row HICO library without no_interaction')
        self.row_by_key = {key: row for row, key in enumerate(self.keys)}
        self.object_index = {int(label): i for i, label in enumerate(object_ids)}

    def describe(self, row):
        key = self.keys[row]
        return {'text_row': row, 'verb_index': key[0], 'object_index': key[1],
                'hoi_text': hico_text_label[key], 'library_texts': hico_lib[key]}

    def record(self, target, selected):
        labels = target['labels']
        hois = target['hois']
        # Accept both evaluator tensors and already serialized CPU records.
        if hasattr(labels, 'detach'):
            labels = labels.detach().cpu().tolist()
        if hasattr(hois, 'detach'):
            hois = hois.detach().cpu().reshape(-1, 3).tolist()
        gt = []
        for subject, obj, verb in hois:
            key = (verb, self.object_index[int(labels[obj])])
            if key not in hico_text_label:
                raise ValueError(f'Unknown HICO GT verb/object pair: {key}')
            row = self.row_by_key.get(key)
            if row is None and verb != 57:
                raise ValueError(f'Interactive GT missing from text library: {key}')
            ranks = {name: (ids.index(row) + 1 if row in ids else None)
                     for name, ids in selected.items()}
            gt.append({'subject_id': subject, 'object_id': obj,
                       'verb_index': verb, 'object_index': key[1],
                       'hoi_text': hico_text_label[key], 'text_row': row,
                       'in_library': row is not None, 'rank_in_topk': ranks,
                       'hit': {name: rank is not None for name, rank in ranks.items()}})
        return {'image_id': int(target['id']), 'file_name': target.get('file_name', ''),
                'topk': {name: [self.describe(row) for row in ids]
                         for name, ids in selected.items()}, 'gt': gt}

    def save(self, records, output_dir, args):
        # DistributedSampler may pad the last batch with repeated images.
        records = list({r['image_id']: r for r in records}.values())
        records.sort(key=lambda r: r['image_id'])
        branches = ('scene', 'instance', 'action')
        counts = {name: {'gt_instances': 0, 'hit_instances': 0,
                         'gt_image_classes': 0, 'hit_image_classes': 0,
                         'eligible_images': 0, 'any_hit_images': 0,
                         'all_hit_images': 0} for name in branches}
        excluded = 0
        for record in records:
            eligible = [gt for gt in record['gt'] if gt['in_library']]
            excluded += len(record['gt']) - len(eligible)
            unique = {gt['text_row']: gt for gt in eligible}.values()
            for name in branches:
                c = counts[name]
                c['gt_instances'] += len(eligible)
                c['hit_instances'] += sum(gt['hit'][name] for gt in eligible)
                c['gt_image_classes'] += len(unique)
                c['hit_image_classes'] += sum(gt['hit'][name] for gt in unique)
                c['eligible_images'] += bool(eligible)
                c['any_hit_images'] += any(gt['hit'][name] for gt in eligible)
                c['all_hit_images'] += bool(eligible) and all(gt['hit'][name] for gt in eligible)
        for name, c in counts.items():
            c['k'] = len(records[0]['topk'][name]) if records else None
            for metric, numerator, denominator in (
                ('instance_recall', 'hit_instances', 'gt_instances'),
                ('image_class_recall', 'hit_image_classes', 'gt_image_classes'),
                ('any_hit_image_rate', 'any_hit_images', 'eligible_images'),
                ('all_hit_image_rate', 'all_hit_images', 'eligible_images'),
            ):
                c[metric] = c[numerator] / c[denominator] if c[denominator] else None
        summary = {'num_images': len(records), 'excluded_no_interaction_instances': excluded,
                   'checkpoint': getattr(args, 'resume', ''),
                   'text_library': getattr(args, 'ossm_text_library', ''),
                   'mapping': 'hico_lib.py insertion order; text_row/verb_index/object_index are zero-based; ranks are one-based',
                   'scope': 'Whole-image CLS retrieval; no box matching. Recall excludes no_interaction. image_class_recall deduplicates each HOI class within each image.',
                   'branches': counts}
        folder = Path(output_dir) / 'topk_gt_stats'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'per_image.jsonl').open('w', encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        for name, c in counts.items():
            recall = c['image_class_recall']
            value = f'{recall:.2%}' if recall is not None else 'N/A'
            print(f"[Top-k GT] {name} Top-{c['k']}: image-class recall={value} "
                  f"({c['hit_image_classes']}/{c['gt_image_classes']})")
        print(f'[Top-k GT] Results: {folder}')
        return summary
