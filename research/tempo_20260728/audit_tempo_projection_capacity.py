#!/usr/bin/env python3
"""Verify 64/128-D projected caches and record the capacity comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tempo_l89_patch as tempo


FORMAT_VERSION = "tempo-l89-projection-capacity-audit-v1"


def _cache_record(root: Path, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "cache" / split / "manifest.json"
    manifest = tempo.load_manifest(manifest_path, expected_split=split)
    shard_bytes = sum(
        (manifest_path.parent / record["file"]).stat().st_size
        for record in manifest["shards"]
    )
    rows = int(manifest["rows"])
    visits = int(manifest["configuration"]["timepoints"])
    patches = int(manifest["grid_shape"][0]) * int(manifest["grid_shape"][1])
    dimension = int(manifest["configuration"]["projection"]["output_dim"])
    projected_tensor_bytes = rows * visits * patches * dimension * 2
    return manifest, {
        "manifest": str(manifest_path),
        "manifest_sha256": tempo.sha256_file(manifest_path),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "rows": rows,
        "shards": len(manifest["shards"]),
        "grid_shape": manifest["grid_shape"],
        "projection": manifest["configuration"]["projection"],
        "projected_tensor_bytes": projected_tensor_bytes,
        "actual_shard_file_bytes": shard_bytes,
        "raw_768d_patch_tokens_persisted": False,
        "all_shard_file_and_tensor_sha_verified": True,
    }


def _head_parameters(feature_dim: int) -> dict[str, int]:
    model = tempo.TempoPatchHead(
        feature_dim,
        match_rank=16,
        value_dim=32,
        hidden_dim=64,
        radius=1,
        temperature=0.10,
        topk_fraction=0.10,
        normality_scale=1.0,
        use_normality_features=True,
        residual_cap=1.5,
    )
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-64", required=True)
    parser.add_argument("--root-128", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root64 = Path(args.root_64).expanduser().resolve()
    root128 = Path(args.root_128).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path, purpose in (
        (root64, "64-D root"),
        (root128, "128-D root"),
        (output, "capacity audit"),
    ):
        tempo.assert_development_path(path, purpose=purpose)

    records: dict[str, dict[str, Any]] = {}
    manifests: dict[tuple[str, str], dict[str, Any]] = {}
    for name, root in (("dim64", root64), ("dim128", root128)):
        records[name] = {}
        for split in ("train", "val"):
            manifest, record = _cache_record(root, split)
            manifests[(name, split)] = manifest
            records[name][split] = record
        tempo.validate_cache_pair(
            root / "cache" / "train" / "manifest.json",
            root / "cache" / "val" / "manifest.json",
        )

    for split in ("train", "val"):
        left = manifests[("dim64", split)]
        right = manifests[("dim128", split)]
        if left["identity"] != right["identity"]:
            raise ValueError(f"{split}: 64/128 identities differ")
        for key in (
            "weights_sha256",
            "base_head_sha256",
            "timepoints",
            "t0_index",
            "comparable_input_contract_sha256",
            "projection_storage_dtype",
            "amp_dtype",
        ):
            if left["configuration"][key] != right["configuration"][key]:
                raise ValueError(f"{split}: cache contract differs for {key}")

    parameters64 = _head_parameters(64)
    parameters128 = _head_parameters(128)
    audit = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "caches": records,
        "same_ordered_identity_across_projection_dimensions": True,
        "same_train_val_input_contract_across_projection_dimensions": True,
        "event_overlap_within_each_dimension": 0,
        "locked_head": {
            "configuration": {
                "match_rank": 16,
                "value_dim": 32,
                "hidden_dim": 64,
                "radius": 1,
                "topk_fraction": 0.10,
                "normality_features": True,
                "normality_scale": 1.0,
                "residual_cap": 1.5,
            },
            "parameters_dim64": parameters64,
            "parameters_dim128": parameters128,
            "parameter_ratio_128_over_64": (
                parameters128["total"] / parameters64["total"]
            ),
        },
        "storage_ratio_128_over_64": {
            split: (
                records["dim128"][split]["actual_shard_file_bytes"]
                / records["dim64"][split]["actual_shard_file_bytes"]
            )
            for split in ("train", "val")
        },
        "interpretation": (
            "Projection width changes both retained frozen-token information "
            "and the small Q/K/V input parameter count; the audit records both "
            "so a gain is not attributed to information width alone."
        ),
        "test_or_sealed_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tempo.atomic_json(output, audit)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": tempo.sha256_file(output),
                "parameter_ratio": audit["locked_head"][
                    "parameter_ratio_128_over_64"
                ],
                "storage_ratio": audit["storage_ratio_128_over_64"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
