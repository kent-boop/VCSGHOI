from __future__ import annotations

import unittest

from tools.evidencehoi_image_search_v2.canonicalize import SourceInstance, canonicalize_instances


def source(source_id: int, role: str, category: int, box: tuple[float, float, float, float]) -> SourceInstance:
    return SourceInstance(source_id, role, category, str(category), box, {"source": source_id})


class CanonicalizationTests(unittest.TestCase):
    def test_exact_duplicate_bbox_is_counted_once(self) -> None:
        canonical, mapping = canonicalize_instances(
            "image", [source(0, "object", 2, (10, 10, 30, 30)), source(1, "object", 2, (10, 10, 30, 30))]
        )
        self.assertEqual(len(canonical), 1)
        self.assertEqual(mapping[0], mapping[1])
        self.assertEqual(canonical[0].merge_method, "exact_bbox")

    def test_nearby_distinct_instances_are_not_merged(self) -> None:
        canonical, mapping = canonicalize_instances(
            "image", [source(0, "object", 2, (10, 10, 30, 30)), source(1, "object", 2, (14, 10, 34, 30))]
        )
        self.assertEqual(len(canonical), 2)
        self.assertNotEqual(mapping[0], mapping[1])

    def test_different_object_categories_never_merge(self) -> None:
        canonical, mapping = canonicalize_instances(
            "image", [source(0, "object", 2, (10, 10, 30, 30)), source(1, "object", 4, (10, 10, 30, 30))]
        )
        self.assertEqual(len(canonical), 2)
        self.assertNotEqual(mapping[0], mapping[1])

    def test_human_and_object_never_merge(self) -> None:
        canonical, mapping = canonicalize_instances(
            "image", [source(0, "human", 1, (10, 10, 30, 30)), source(1, "object", 1, (10, 10, 30, 30))]
        )
        self.assertEqual(len(canonical), 2)
        self.assertNotEqual(mapping[0], mapping[1])

    def test_high_iou_complete_link_merge_records_confidence(self) -> None:
        canonical, _ = canonicalize_instances(
            "image", [source(0, "human", 1, (10, 10, 110, 110)), source(1, "human", 1, (11, 10, 111, 110))]
        )
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0].merge_method, "high_iou_complete_link")
        self.assertGreaterEqual(canonical[0].merge_confidence, 0.92)

    def test_pair_provenance_remains_source_addressable(self) -> None:
        canonical, mapping = canonicalize_instances(
            "image", [source(3, "human", 1, (0, 0, 20, 40)), source(7, "object", 2, (10, 20, 30, 40))]
        )
        self.assertEqual(len(canonical), 2)
        self.assertIn(3, mapping)
        self.assertIn(7, mapping)


if __name__ == "__main__":
    unittest.main()
