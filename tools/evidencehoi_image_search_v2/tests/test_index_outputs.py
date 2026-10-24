from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


INDEX_ROOT = Path("/wzj/zijian/OSSM-HOI-main/outputs/figure1_search_v2/index")


class IndexOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = [
            INDEX_ROOT / "instances.parquet",
            INDEX_ROOT / "pairs.parquet",
            INDEX_ROOT / "image_summary.parquet",
            INDEX_ROOT / "index_statistics.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest(f"Stage 2 index outputs are not present: {missing}")
        cls.instances = pd.read_parquet(INDEX_ROOT / "instances.parquet")
        cls.pairs = pd.read_parquet(INDEX_ROOT / "pairs.parquet")
        cls.images = pd.read_parquet(INDEX_ROOT / "image_summary.parquet")
        cls.stats = json.loads((INDEX_ROOT / "index_statistics.json").read_text())

    def test_all_pairs_trace_to_real_connection(self) -> None:
        self.assertTrue(self.stats["all_pairs_trace_to_real_connection"])
        self.assertTrue((self.pairs["raw_relation_count"] >= 1).all())
        provenance_counts = self.pairs["raw_annotation_provenance_json"].map(lambda value: len(json.loads(value)))
        self.assertTrue((provenance_counts == self.pairs["raw_relation_count"]).all())

    def test_multiple_hoi_same_pair_are_aggregated_not_duplicate_pairs(self) -> None:
        key_columns = ["image_key", "canonical_human_id", "canonical_object_id"]
        self.assertFalse(self.pairs.duplicated(key_columns).any())
        self.assertLess(len(self.pairs), int(self.pairs["raw_relation_count"].sum()))

    def test_canonical_instance_ids_are_unique(self) -> None:
        self.assertTrue(self.instances["canonical_instance_id"].is_unique)
        self.assertTrue(self.pairs["pair_id"].is_unique)

    def test_legacy_exclusions_are_indexed_but_ineligible(self) -> None:
        old = {
            "HICO_train2015_00003946.jpg",
            "HICO_train2015_00024103.jpg",
            "HICO_train2015_00006309.jpg",
            "HICO_train2015_00018460.jpg",
            "HICO_train2015_00019218.jpg",
        }
        rows = self.images[self.images["file_name"].isin(old)]
        self.assertEqual(set(rows["file_name"]), old)
        self.assertFalse(rows["eligible_for_final"].any())
        self.assertTrue(rows["legacy_negative_example"].all())


if __name__ == "__main__":
    unittest.main()
