from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

import shard_and_merge_feature_cache as utility


def _manifest(path: Path, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in (
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
        "l89_0_path",
        "l89_90_path",
        "l89_360_path",
        "emit_0_path",
        "emit_90_path",
        "emit_360_path",
        "s5p_0_path",
    ):
        if column not in frame:
            frame[column] = ""
    frame.to_csv(path, index=False)
    return frame


def _row(
    index: int,
    *,
    event: str,
    plume: str,
    signature: str = "s2",
) -> dict:
    result = {
        "id": str(index),
        "plume_id": plume,
        "event_id": event,
        "label": index % 2,
        "query360_index": index,
        "availability_signature": signature,
    }
    if "s2" in signature.split("+"):
        result.update(
            {
                "s2_0_path": f"/remote/{index}/s2_0.tif",
                "s2_90_path": f"/remote/{index}/s2_90.tif",
                "s2_360_path": f"/remote/{index}/s2_360.tif",
            }
        )
    if "l89" in signature.split("+"):
        result.update(
            {
                "l89_0_path": f"/remote/{index}/l89_0.tif",
                "l89_90_path": f"/remote/{index}/l89_90.tif",
                "l89_360_path": f"/remote/{index}/l89_360.tif",
            }
        )
    if "emit" in signature.split("+"):
        result.update(
            {
                "emit_0_path": f"/remote/{index}/emit_0.tif",
                "emit_90_path": f"/remote/{index}/emit_90.tif",
                "emit_360_path": f"/remote/{index}/emit_360.tif",
            }
        )
    if "s5p" in signature.split("+"):
        result["s5p_0_path"] = f"/remote/{index}/s5p_0.npz"
    return result


def _cache(path: Path, manifest_path: Path) -> None:
    frame = pd.read_csv(
        manifest_path, dtype=str, keep_default_na=False, low_memory=False
    )
    rows = len(frame)
    features = torch.arange(rows * 4 * 3 * 2, dtype=torch.float16).reshape(
        rows, 4, 3, 2
    )
    valid = torch.ones(rows, 4, 3, dtype=torch.bool)
    sensor_logits = torch.zeros(rows, 4)
    sensor_valid = torch.ones(rows, 4, dtype=torch.bool)
    hybrid = torch.linspace(-1, 1, rows)
    payload = {
        "schema_version": utility.FEATURE_SCHEMA,
        "script_version": "query360-two-axis-full-legacy-v2",
        "split": "train_core",
        "features": features,
        "features_hybrid": features,
        "features_universal": features + 1,
        "valid_mask": valid,
        "base_sensor_logits": sensor_logits,
        "base_sensor_valid": sensor_valid,
        "base_sensor_logits_hybrid": sensor_logits,
        "base_sensor_valid_hybrid": sensor_valid,
        "base_sensor_logits_universal": sensor_logits + 1,
        "base_sensor_valid_universal": sensor_valid,
        "base_fused_logits": hybrid,
        "base_hybrid_logits": hybrid,
        "base_universal_logits": hybrid + 1,
        "base_definitions": {"hybrid": "smoke", "universal": "smoke"},
        "labels": torch.tensor(frame["label"].astype(int).tolist()),
        "ids": frame["id"].astype(str).tolist(),
        "plume_ids": frame["plume_id"].astype(str).tolist(),
        "event_ids": frame["event_id"].astype(str).tolist(),
        "availability_signatures": frame[
            "availability_signature"
        ].astype(str).tolist(),
        "sensor_names": list(utility.SENSOR_NAMES),
        "manifest": {
            "path": str(manifest_path),
            "sha256": utility.sha256_file(manifest_path),
            "rows": rows,
        },
        "encoder": {
            "state_sha256": "same-encoder",
            "provenance": {"source": "smoke"},
        },
        "extraction": {
            "observations": rows * 12,
            "elapsed_seconds": float(rows),
        },
        "sealed_test_read": False,
    }
    torch.save(payload, path)


def test_shard_and_merge_smoke(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    dev_path = tmp_path / "dev.csv"
    train_rows = [
        _row(1, event="e1", plume="p1", signature="s2"),
        _row(2, event="e1", plume="p1", signature="s2+l89"),
        _row(3, event="e2", plume="p2", signature="emit"),
        _row(4, event="e3", plume="p3", signature="s5p"),
        _row(5, event="e4", plume="p4", signature="s2"),
        _row(6, event="e5", plume="p5", signature="l89"),
    ]
    _manifest(train_path, train_rows)
    _manifest(
        dev_path,
        [_row(100, event="dev-event", plume="dev-plume", signature="s2")],
    )
    utility.command_shard_manifest(
        SimpleNamespace(
            train_manifest=str(train_path),
            dev_manifest=str(dev_path),
            output_dir=str(tmp_path / "shards"),
            prefix="train_core",
            shard0="",
            shard1="",
            audit="",
            overwrite=False,
        )
    )
    shard0 = tmp_path / "shards" / "train_core_gpu0.csv"
    shard1 = tmp_path / "shards" / "train_core_gpu1.csv"
    frame0 = pd.read_csv(shard0, dtype=str, keep_default_na=False)
    frame1 = pd.read_csv(shard1, dtype=str, keep_default_na=False)
    assert set(frame0["event_id"]).isdisjoint(set(frame1["event_id"]))
    assert sorted(
        frame0["query360_index"].astype(int).tolist()
        + frame1["query360_index"].astype(int).tolist()
    ) == [1, 2, 3, 4, 5, 6]

    cache0 = tmp_path / "cache0.pt"
    cache1 = tmp_path / "cache1.pt"
    _cache(cache0, shard0)
    _cache(cache1, shard1)
    output = tmp_path / "merged.pt"
    utility.command_merge_cache(
        SimpleNamespace(
            input_cache=[str(cache0), str(cache1)],
            output_cache=str(output),
            output_manifest="",
            audit="",
            overwrite=False,
        )
    )
    merged = torch.load(output, map_location="cpu", weights_only=False)
    assert merged["split"] == "train_core"
    assert merged["features"].shape == (6, 4, 3, 2)
    assert merged["features_hybrid"] is merged["features"]
    assert merged["base_fused_logits"] is merged["base_hybrid_logits"]
    assert len(set(merged["ids"])) == 6
    assert len(set(merged["query360_indices"])) == 6
    audit = output.with_suffix(".pt.audit.json")
    sidecar = audit.with_suffix(".json.sha256")
    expected = sidecar.read_text().split()[0]
    assert expected == hashlib.sha256(audit.read_bytes()).hexdigest()


def test_merge_rejects_cross_shard_plume_overlap(tmp_path: Path) -> None:
    manifest0 = tmp_path / "m0.csv"
    manifest1 = tmp_path / "m1.csv"
    _manifest(manifest0, [_row(1, event="e", plume="same")])
    _manifest(manifest1, [_row(2, event="e", plume="same")])
    cache0 = tmp_path / "c0.pt"
    cache1 = tmp_path / "c1.pt"
    _cache(cache0, manifest0)
    _cache(cache1, manifest1)
    try:
        utility.command_merge_cache(
            SimpleNamespace(
                input_cache=[str(cache0), str(cache1)],
                output_cache=str(tmp_path / "bad.pt"),
                output_manifest="",
                audit="",
                overwrite=False,
            )
        )
    except ValueError as error:
        assert "plume overlap" in str(error)
    else:
        raise AssertionError("cross-shard plume overlap was not rejected")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="legacy360-shard-smoke-") as root:
        root_path = Path(root)
        positive = root_path / "positive"
        negative = root_path / "negative"
        positive.mkdir()
        negative.mkdir()
        test_shard_and_merge_smoke(positive)
        test_merge_rejects_cross_shard_plume_overlap(negative)
    print("2 smoke tests passed")
