#!/usr/bin/env python3
"""CPU-only tests for the clean-L89 local manifest resolver."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from research.tempo_20260728 import resolve_l89_clean_manifests as target


def make_frame(folder: str, event: str, row_id: int) -> pd.DataFrame:
    row = {
        "id": row_id,
        "label": row_id % 2,
        "plume_id": f"{event}-A",
        "event_group_id": event,
        "path": f"/unread/source/{folder}",
    }
    row.update({column: "/ignored/remote/value.tif" for column in target.ROLE_FILES})
    return pd.DataFrame([row])


def materialize(root: Path, folder: str, roles: tuple[str, ...]) -> None:
    destination = root / folder
    destination.mkdir(parents=True, exist_ok=True)
    for role in roles:
        (destination / target.ROLE_FILES[role][1]).write_bytes(b"tif")


class ResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="l89_clean_unit_")
        self.root = Path(self.temporary.name)
        self.old_three = self.root / "old_three"
        self.old_extra = self.root / "old_extra"
        self.staging = self.root / "staging"
        for path in (self.old_three, self.old_extra, self.staging):
            path.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_historical_priority_and_staging_fallback(self) -> None:
        train_folder = "00000001"
        dev_folder = "00000002"
        materialize(
            self.old_three,
            train_folder,
            ("path_t0", "path_prev1", "path_seasonal"),
        )
        materialize(
            self.old_extra,
            train_folder,
            ("path_prev2", "path_prev3", "path_year"),
        )
        materialize(self.staging, train_folder, tuple(target.ROLE_FILES))
        materialize(self.staging, dev_folder, tuple(target.ROLE_FILES))
        frames = {
            "train": make_frame(train_folder, "event_train", 1),
            "dev": make_frame(dev_folder, "event_dev", 2),
        }
        resolved, audit = target.resolve_frames(
            frames,
            old_3time=self.old_three,
            old_extra=self.old_extra,
            staging_root=self.staging,
            workers=2,
            enforce_expected_counts=False,
        )
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["event_overlap"], 0)
        self.assertTrue(
            resolved["train"]["path_t0"].iloc[0].startswith(str(self.old_three))
        )
        self.assertTrue(
            resolved["train"]["path_year"].iloc[0].startswith(str(self.old_extra))
        )
        self.assertTrue(
            resolved["dev"]["path_t0"].iloc[0].startswith(str(self.staging))
        )
        self.assertEqual(
            audit["role_source_counts"]["train"]["path_t0"]["historical"], 1
        )
        self.assertEqual(
            audit["role_source_counts"]["dev"]["path_t0"]["staged"], 1
        )

    def test_missing_file_fails_closed(self) -> None:
        materialize(self.staging, "00000001", tuple(target.ROLE_FILES)[:-1])
        materialize(self.staging, "00000002", tuple(target.ROLE_FILES))
        with self.assertRaises(target.IncompleteLocalStaging):
            target.resolve_frames(
                {
                    "train": make_frame("00000001", "event_train", 1),
                    "dev": make_frame("00000002", "event_dev", 2),
                },
                old_3time=self.old_three,
                old_extra=self.old_extra,
                staging_root=self.staging,
                workers=2,
                enforce_expected_counts=False,
            )

    def test_smoke_selects_only_complete_rows(self) -> None:
        materialize(self.staging, "00000001", tuple(target.ROLE_FILES))
        materialize(self.staging, "00000002", tuple(target.ROLE_FILES)[:-1])
        materialize(self.staging, "00000003", tuple(target.ROLE_FILES))
        train = pd.concat(
            [
                make_frame("00000001", "event_train_a", 1),
                make_frame("00000002", "event_train_b", 2),
            ],
            ignore_index=True,
        )
        resolved, audit = target.resolve_frames(
            {
                "train": train,
                "dev": make_frame("00000003", "event_dev", 3),
            },
            old_3time=self.old_three,
            old_extra=self.old_extra,
            staging_root=self.staging,
            workers=2,
            enforce_expected_counts=False,
            smoke_rows=1,
            smoke_seed=17,
        )
        self.assertEqual(resolved["train"]["id"].astype(int).tolist(), [1])
        self.assertEqual(resolved["dev"]["id"].astype(int).tolist(), [3])
        self.assertFalse(audit["formal_full_manifest"])
        self.assertTrue(audit["smoke"]["every_selected_role_file_exists"])
        self.assertGreater(audit["missing_role_files"], 0)

    def test_event_overlap_is_rejected(self) -> None:
        for folder in ("00000001", "00000002"):
            materialize(self.staging, folder, tuple(target.ROLE_FILES))
        with self.assertRaisesRegex(ValueError, "overlap"):
            target.resolve_frames(
                {
                    "train": make_frame("00000001", "same_event", 1),
                    "dev": make_frame("00000002", "same_event", 2),
                },
                old_3time=self.old_three,
                old_extra=self.old_extra,
                staging_root=self.staging,
                workers=2,
                enforce_expected_counts=False,
            )

    def test_held_out_marker_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "held-out"):
            target.safe_path(
                self.root / "outer_eval" / "data.csv", purpose="unit input"
            )

    def test_folder_must_be_eight_digits(self) -> None:
        frame = make_frame("not_a_folder", "event_train", 1)
        with self.assertRaisesRegex(ValueError, "eight digits"):
            target.folder_names(frame)

    def test_frozen_split_lock_rejects_different_split(self) -> None:
        input_root = self.root / "frozen_input"
        input_root.mkdir()
        (input_root / "train.csv").write_bytes(b"different train split")
        (input_root / "dev.csv").write_bytes(b"different dev split")
        (input_root / target.READINESS_FILENAME).write_bytes(b"different receipt")
        with self.assertRaisesRegex(ValueError, "train SHA-256 mismatch"):
            target.validate_frozen_split_inputs(input_root)

    def test_resolver_audit_records_exact_frozen_split_lock(self) -> None:
        input_root = self.root / "frozen_input"
        output_root = self.root / "formal_manifests"
        input_root.mkdir()
        materialize(self.staging, "00000001", tuple(target.ROLE_FILES))
        materialize(self.staging, "00000002", tuple(target.ROLE_FILES))
        train = make_frame("00000001", "event_train", 1)
        dev = make_frame("00000002", "event_dev", 2)
        train_path = input_root / "train.csv"
        dev_path = input_root / "dev.csv"
        readiness_path = input_root / target.READINESS_FILENAME
        train.to_csv(train_path, index=False)
        dev.to_csv(dev_path, index=False)
        readiness_path.write_bytes(b'{"unit_test":true}\n')
        expected_inputs = {
            "train": target.sha256_file(train_path),
            "dev": target.sha256_file(dev_path),
        }
        expected_readiness = target.sha256_file(readiness_path)
        args = target.build_parser().parse_args(
            [
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--old-3time-root",
                str(self.old_three),
                "--old-extra-root",
                str(self.old_extra),
                "--staging-root",
                str(self.staging),
                "--workers",
                "2",
                "--no-frozen-count-check",
            ]
        )
        with mock.patch.object(
            target, "EXPECTED_FROZEN_INPUT_SHA256", expected_inputs
        ), mock.patch.object(
            target, "EXPECTED_READINESS_SHA256", expected_readiness
        ):
            audit = target.run(args)
        lock = audit["frozen_split_lock"]
        self.assertTrue(lock["enforced"])
        self.assertTrue(lock["different_split_rejected"])
        self.assertTrue(lock["files"]["train"]["exact_sha_match"])
        self.assertTrue(lock["files"]["dev"]["exact_sha_match"])
        self.assertTrue(lock["files"]["readiness"]["exact_sha_match"])
        self.assertEqual(
            lock["files"]["train"]["sha256"], expected_inputs["train"]
        )


if __name__ == "__main__":
    unittest.main()
