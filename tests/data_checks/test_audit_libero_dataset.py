from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "data_checks"
    / "audit_libero_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("audit_libero_dataset", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def episode(length: int = 3):
    return [
        {
            "image": {"path": f"main_{frame}.png", "bytes": None},
            "wrist_image": {"path": f"wrist_{frame}.png", "bytes": None},
            "state": [0.1] * 8,
            "actions": [0.1] * 6 + [-1.0],
            "timestamp": frame / 10,
            "frame_index": frame,
            "episode_index": 4,
            "task_index": 2,
        }
        for frame in range(length)
    ]


class DatasetAuditTest(unittest.TestCase):
    def setUp(self):
        self.config = AUDIT.AuditConfig(min_episode_length=1)
        self.tasks = {2: "put the object in the container"}

    def audit(self, rows):
        return AUDIT.validate_episode_rows(
            rows,
            episode_file="episode_000004.parquet",
            tasks=self.tasks,
            config=self.config,
        )

    def test_accepts_valid_episode(self):
        report = self.audit(episode())
        self.assertTrue(report["accepted"])
        self.assertEqual(report["flags"], [])
        self.assertEqual(report["action_dim"], 7)

    def test_rejects_non_finite_action(self):
        rows = episode()
        rows[1]["actions"][0] = math.nan
        report = self.audit(rows)
        self.assertFalse(report["accepted"])
        self.assertIn("non_finite_action", report["flags"])

    def test_rejects_dirty_action_outlier(self):
        rows = episode()
        rows[1]["actions"][0] = 3.0
        report = self.audit(rows)
        self.assertIn("action_out_of_range", report["flags"])

    def test_rejects_missing_wrist_image(self):
        rows = episode()
        rows[0]["wrist_image"] = None
        report = self.audit(rows)
        self.assertIn("missing_wrist_image", report["flags"])

    def test_rejects_unmapped_task_and_frame_gap(self):
        rows = episode()
        rows[1]["frame_index"] = 5
        report = AUDIT.validate_episode_rows(
            rows,
            episode_file="episode_000004.parquet",
            tasks={},
            config=self.config,
        )
        self.assertIn("unmapped_task", report["flags"])
        self.assertIn("non_contiguous_frame_index", report["flags"])


if __name__ == "__main__":
    unittest.main()
