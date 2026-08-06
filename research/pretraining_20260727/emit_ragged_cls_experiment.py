#!/usr/bin/env python3
"""Leakage-safe frozen-CLS correspondence screen for three-timepoint EMIT32.

Stage ``cache`` encodes t0, prev1, and seasonal independently with the
official frozen Panopticon backbone.  EMIT TIFFs are read through the existing
HWC-aware 32-band reader and carry their physical wavelength IDs.  Normalizing
statistics must come from an ``emit`` block cryptographically bound to the
declared training manifest.

Stage ``train-heads`` delegates to the already tested, parameter-matched
ragged temporal heads.  It accepts only caches stamped as EMIT32 and retains
the fail-closed canonical-event train/validation overlap check.  Paths with a
``test`` or ``sealed`` component are refused by both stages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgraded_dataset import (  # noqa: E402
    dino_classifier_head_emit32_temporal_satmae as emit_module,
)
from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as common,
)


SCRIPT_VERSION = "emit-ragged-cls-v1"
SENSOR_NAME = "emit32"
CACHE_FORMAT_VERSION = common.CACHE_FORMAT_VERSION
ARM_NAMES = common.ARM_NAMES

PATH_COLUMNS = ("path_t0", "path_prev1", "path_seasonal")
TIME_COLUMNS = (
    "t0_image_time",
    "prev1_image_time",
    "seasonal_image_time",
)
ROLE_NAMES = ("t0", "prev1", "seasonal")
BAND_INDICES = tuple(range(32))
ID_COLUMN = "sample_id"
LABEL_COLUMN = "label"
PLUME_ID_COLUMN = "plume_id"
EVENT_COLUMN = "event_group_id"

DEFAULT_PROTOCOL_TRAIN = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests_protocol/"
    "four_sensor_global_val_purge_v1/emit_train_global_val_purged.csv"
)
DEFAULT_INNER_VAL = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/manifests_staged/"
    "emit_3time/val.csv"
)
DEFAULT_STATS_JSON = Path(
    "/diniuvol/yuyao/methanefuse_research_20260727/shared_residual/"
    "four_sensor_shared_mae_globalpurge_e1_seed20260727/"
    "normalization_stats.json"
)
EXPECTED_PROTOCOL_TRAIN_SHA256 = (
    "d2b9ccbfc021fad8508a96ffb216ed503a68fbc19afeb7dd6d54d7ef4ee65a39"
)
EXPECTED_STATS_JSON_SHA256 = (
    "033ab9523146c00647f8b1e32a694dca4238b5ffa3d9b71ca571774db89a2697"
)


def _same_resolved_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def _validate_float_vector(
    value: Any,
    *,
    name: str,
    length: int,
    strictly_positive: bool,
) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"EMIT normalization {name} must be a numeric sequence.")
    result = [float(item) for item in value]
    if len(result) != length:
        raise ValueError(
            f"EMIT normalization {name} must have {length} values, "
            f"got {len(result)}."
        )
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"EMIT normalization {name} contains non-finite values.")
    if strictly_positive and any(item <= 0 for item in result):
        raise ValueError(f"EMIT normalization {name} must be strictly positive.")
    return result


def load_sha_bound_emit_stats(
    stats_json: Path,
    normalization_train_csv: Path,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Load only a nested EMIT block bound to the exact declared train CSV."""

    stats_json = stats_json.expanduser().resolve()
    normalization_train_csv = normalization_train_csv.expanduser().resolve()
    common.assert_not_sealed_path(stats_json, purpose="normalization JSON")
    common.assert_not_sealed_path(
        normalization_train_csv, purpose="normalization train CSV"
    )
    if not stats_json.is_file():
        raise FileNotFoundError(stats_json)
    if not normalization_train_csv.is_file():
        raise FileNotFoundError(normalization_train_csv)

    stats_file_sha = common.sha256_file(stats_json)
    train_file_sha = common.sha256_file(normalization_train_csv)
    if _same_resolved_path(stats_json, DEFAULT_STATS_JSON):
        if stats_file_sha != EXPECTED_STATS_JSON_SHA256:
            raise ValueError(
                "The default normalization JSON changed: "
                f"{stats_file_sha} != {EXPECTED_STATS_JSON_SHA256}."
            )
    if _same_resolved_path(normalization_train_csv, DEFAULT_PROTOCOL_TRAIN):
        if train_file_sha != EXPECTED_PROTOCOL_TRAIN_SHA256:
            raise ValueError(
                "The default protocol train CSV changed: "
                f"{train_file_sha} != {EXPECTED_PROTOCOL_TRAIN_SHA256}."
            )

    payload = json.loads(stats_json.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Normalization JSON must contain a mapping.")
    sensors = payload.get("sensors")
    if not isinstance(sensors, Mapping) or not isinstance(
        sensors.get("emit"), Mapping
    ):
        raise ValueError(
            "Normalization JSON must contain a nested sensors.emit block."
        )
    block = dict(sensors["emit"])
    if str(block.get("train_csv_sha256", "")) != train_file_sha:
        raise ValueError(
            "EMIT normalization block is not bound to the declared train CSV: "
            f"recorded={block.get('train_csv_sha256')!r}, actual={train_file_sha}."
        )
    recorded_train = block.get("train_csv")
    if not isinstance(recorded_train, str) or not _same_resolved_path(
        Path(recorded_train), normalization_train_csv
    ):
        raise ValueError(
            "EMIT normalization block train_csv path differs from the declared "
            "normalization train CSV."
        )
    if int(block.get("frames_per_row", -1)) != len(PATH_COLUMNS):
        raise ValueError(
            f"EMIT normalization must use {len(PATH_COLUMNS)} frames per row."
        )
    if block.get("zero_is_nodata") is not True:
        raise ValueError("EMIT normalization must explicitly record zero_is_nodata=true.")
    sampled_rows = int(block.get("sampled_rows", 0))
    requested_rows = int(block.get("requested_sample_rows", 0))
    if sampled_rows <= 0 or requested_rows <= 0 or sampled_rows > requested_rows:
        raise ValueError("EMIT normalization sample-row provenance is invalid.")

    mean = _validate_float_vector(
        block.get("mean"),
        name="mean",
        length=len(BAND_INDICES),
        strictly_positive=False,
    )
    std = _validate_float_vector(
        block.get("std"),
        name="std",
        length=len(BAND_INDICES),
        strictly_positive=True,
    )
    provenance = {
        "stats_json": str(stats_json),
        "stats_json_sha256": stats_file_sha,
        "normalization_train_csv": str(normalization_train_csv),
        "normalization_train_csv_sha256": train_file_sha,
        "stats_sensor_block": "sensors.emit",
        "frames_per_row": int(block["frames_per_row"]),
        "sampled_rows": sampled_rows,
        "requested_sample_rows": requested_rows,
        "sample_seed": int(block["sample_seed"]),
        "zero_is_nodata": True,
    }
    return mean, std, provenance


class EmitFrameCacheDataset(Dataset):
    """Strict EMIT HWC32 reader that never substitutes another row."""

    STATUS_MISSING = 0
    STATUS_USABLE = 1
    STATUS_LOW_COVERAGE = 2
    STATUS_READ_ERROR = 3

    def __init__(
        self,
        csv_path: Path,
        frame: pd.DataFrame,
        *,
        mean: Sequence[float],
        std: Sequence[float],
        image_size: int,
        min_valid_fraction: float,
        validity_band_index: int,
        local_file_cache: Optional[Any],
        local_cache_bypass_root: Optional[Path],
        zero_invalid_pixels: bool,
    ):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.image_size = int(image_size)
        self.min_valid_fraction = float(min_valid_fraction)
        self.validity_band_index = int(validity_band_index)
        self.local_file_cache = local_file_cache
        self.local_cache_bypass_root = (
            local_cache_bypass_root.expanduser().resolve()
            if local_cache_bypass_root is not None
            else None
        )
        self.zero_invalid_pixels = bool(zero_invalid_pixels)
        if self.validity_band_index not in BAND_INDICES:
            raise ValueError(
                f"validity_band_index={self.validity_band_index} is out of range."
            )

        # This reader is required: it transposes pixel-interleaved HWC EMIT
        # TIFFs to CHW and overrides the generic dataset wavelengths with the
        # 32 physical EMIT wavelengths.
        self.reader = emit_module.Emit32CsvDataset(
            csv_path=str(csv_path),
            path_column=PATH_COLUMNS[0],
            normalize_stats=None,
            scale_to_unit=False,
            compute_stats=False,
            pad_to_multiple=None,
            skip_invalid_samples=False,
            path_columns_for_validation=PATH_COLUMNS,
        )
        channel_ids = torch.as_tensor(
            self.reader.chn_ids, dtype=torch.float32
        ).contiguous()
        expected_ids = torch.tensor(
            emit_module.EMIT32_WAVELENGTHS_NM, dtype=torch.float32
        )
        if channel_ids.shape != (len(BAND_INDICES),) or not torch.equal(
            channel_ids, expected_ids
        ):
            raise ValueError(
                "Emit32CsvDataset did not expose the expected 32 physical "
                "wavelength IDs."
            )
        self.channel_ids = channel_ids
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        if self.mean.shape != (32, 1, 1) or self.std.shape != (32, 1, 1):
            raise ValueError("EMIT normalization tensors must be shaped (32,1,1).")

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_path(self, value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        path = value.strip()
        if self.local_cache_bypass_root is not None:
            expanded = Path(os.path.abspath(os.path.expanduser(path)))
            try:
                expanded.relative_to(self.local_cache_bypass_root)
                return path
            except ValueError:
                pass
        if self.local_file_cache is not None:
            path = self.local_file_cache.ensure_local(path)
        return path

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        timepoints = len(PATH_COLUMNS)
        images = torch.zeros(
            timepoints,
            len(BAND_INDICES),
            self.image_size,
            self.image_size,
            dtype=torch.float32,
        )
        image_valid = torch.zeros(timepoints, dtype=torch.bool)
        valid_fraction = torch.zeros(timepoints, dtype=torch.float32)
        status = torch.full(
            (timepoints,), self.STATUS_MISSING, dtype=torch.int8
        )
        for time_index, column in enumerate(PATH_COLUMNS):
            path = self._resolve_path(row[column])
            if path is None:
                continue
            try:
                raw = self.reader._read_image_raw(path)
                if raw.shape[0] != len(BAND_INDICES) or raw.ndim != 3:
                    raise ValueError(
                        f"Expected CHW EMIT32 image, got {tuple(raw.shape)}."
                    )
                native_valid = torch.isfinite(raw) & raw.ne(0)
                fraction = native_valid[
                    self.validity_band_index
                ].float().mean()
                valid_fraction[time_index] = fraction
                if float(fraction) < self.min_valid_fraction:
                    status[time_index] = self.STATUS_LOW_COVERAGE
                    continue
                clean = torch.nan_to_num(
                    raw, nan=0.0, posinf=0.0, neginf=0.0
                )
                normalized = (clean - self.mean) / self.std
                if self.zero_invalid_pixels:
                    normalized = torch.where(
                        native_valid,
                        normalized,
                        torch.zeros((), dtype=normalized.dtype),
                    )
                normalized = F.interpolate(
                    normalized.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                images[time_index] = normalized
                image_valid[time_index] = True
                status[time_index] = self.STATUS_USABLE
            except Exception:
                status[time_index] = self.STATUS_READ_ERROR
        return (
            torch.tensor(index, dtype=torch.long),
            images,
            image_valid,
            valid_fraction,
            status,
        )


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def cache_features(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv).expanduser().resolve()
    output_path = Path(args.output_cache).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    stats_path = Path(args.stats_json).expanduser().resolve()
    normalization_train_path = (
        Path(args.normalization_train_csv).expanduser().resolve()
    )
    common.assert_not_sealed_path(csv_path, purpose="CSV")
    common.assert_not_sealed_path(output_path, purpose="cache")
    common.assert_not_sealed_path(weights_path, purpose="weights")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"A real frozen Panopticon checkpoint is required: {weights_path}"
        )
    if args.split == "train" and not _same_resolved_path(
        csv_path, normalization_train_path
    ):
        raise ValueError(
            "A train cache must use the same CSV to which normalization is bound."
        )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Cache already exists: {output_path}; pass --overwrite to replace it."
        )

    csv_sha_before = common.sha256_file(csv_path)
    weights_sha_before = common.sha256_file(weights_path)
    mean, std, stats_provenance = load_sha_bound_emit_stats(
        stats_path, normalization_train_path
    )
    frame = pd.read_csv(csv_path, low_memory=False)
    source_rows = len(frame)
    frame = common.deterministic_stratified_limit(
        frame,
        label_column=LABEL_COLUMN,
        maximum_rows=args.max_rows,
        seed=args.row_selection_seed,
    )
    if frame.empty:
        raise ValueError(f"No rows selected from {csv_path}.")
    required = (
        set(PATH_COLUMNS)
        | set(TIME_COLUMNS)
        | {ID_COLUMN, LABEL_COLUMN, PLUME_ID_COLUMN}
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required EMIT columns: {missing}.")

    temporal = common.build_temporal_metadata(
        frame, PATH_COLUMNS, TIME_COLUMNS
    )
    if tuple(temporal.role_names) != ROLE_NAMES or temporal.t0_index != 0:
        raise RuntimeError("The fixed EMIT role contract was not preserved.")
    labels = common.validate_binary_labels(frame, LABEL_COLUMN)
    plume_ids = common.string_column(frame, PLUME_ID_COLUMN)
    ids = common.string_column(frame, ID_COLUMN)
    event_ids, event_rule = common.canonical_event_ids(
        frame,
        event_column=EVENT_COLUMN,
        plume_id_column=PLUME_ID_COLUMN,
    )

    cache_object = None
    if args.local_cache_mode != "off":
        cache_object = emit_module.StaticAnchoredCache(
            args.local_cache_dir,
            min_free_gb=args.local_cache_min_free_gb,
            async_mode=args.local_cache_mode == "async",
            max_workers=args.local_cache_workers,
        )
    dataset = EmitFrameCacheDataset(
        csv_path,
        frame,
        mean=mean,
        std=std,
        image_size=args.image_size,
        min_valid_fraction=args.min_valid_fraction,
        validity_band_index=args.validity_band_index,
        local_file_cache=cache_object,
        local_cache_bypass_root=(
            Path(args.local_cache_bypass_root)
            if args.local_cache_bypass_root
            else None
        ),
        zero_invalid_pixels=args.zero_invalid_pixels,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers
    loader = DataLoader(dataset, **loader_kwargs)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.cuda.set_device(device)
    backbone = emit_module.load_backbone(
        str(weights_path), device=device, debug=args.debug
    ).to(device)
    backbone.requires_grad_(False)
    backbone.eval()

    rows = len(frame)
    timepoints = len(PATH_COLUMNS)
    feature_dim = int(backbone.embed_dim)
    storage_dtype = (
        torch.float16 if args.storage_dtype == "float16" else torch.float32
    )
    features = torch.zeros(
        rows, timepoints, feature_dim, dtype=storage_dtype
    )
    image_valid_mask = torch.zeros(rows, timepoints, dtype=torch.bool)
    valid_fraction = torch.zeros(rows, timepoints, dtype=torch.float32)
    load_status = torch.zeros(rows, timepoints, dtype=torch.int8)
    seen = torch.zeros(rows, dtype=torch.bool)
    channel_ids = dataset.channel_ids

    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            indices, images, batch_image_valid, batch_fraction, batch_status = (
                batch
            )
            (
                batch_size,
                batch_timepoints,
                channels,
                height,
                width,
            ) = images.shape
            if channels != 32 or batch_timepoints != 3:
                raise RuntimeError(
                    "EMIT cache loader changed its fixed Bx3x32 input contract."
                )
            flat_images = images.reshape(
                batch_size * batch_timepoints, channels, height, width
            ).to(device, non_blocking=True)
            flat_channel_ids = (
                channel_ids.view(1, -1)
                .expand(batch_size * batch_timepoints, -1)
                .clone()
                .to(device, non_blocking=True)
            )
            with autocast_context(device, args.amp_dtype):
                output = backbone.forward_features(
                    {"imgs": flat_images, "chn_ids": flat_channel_ids}
                )
                batch_features = output["x_norm_clstoken"].reshape(
                    batch_size, batch_timepoints, feature_dim
                )
            batch_features = batch_features.float().cpu()
            batch_features[~batch_image_valid] = 0.0
            indices = indices.long()
            if seen[indices].any():
                raise RuntimeError("A cache row was emitted more than once.")
            seen[indices] = True
            features[indices] = batch_features.to(storage_dtype)
            image_valid_mask[indices] = batch_image_valid
            valid_fraction[indices] = batch_fraction
            load_status[indices] = batch_status
            if (
                batch_index % max(1, args.log_interval) == 0
                or batch_index == len(loader)
            ):
                print(
                    f"[emit-cache] batches={batch_index}/{len(loader)} "
                    f"rows={int(seen.sum())}/{rows} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    if not seen.all():
        missing_rows = torch.nonzero(
            ~seen, as_tuple=False
        ).flatten().tolist()[:20]
        raise RuntimeError(f"Feature cache missed rows: {missing_rows}.")

    valid_mask = image_valid_mask & temporal.timestamp_valid_mask
    (
        duplicate_mask,
        unique_mask,
        duplicate_group_mask,
    ) = common.compute_duplicate_unique_masks(
        temporal.timestamps_utc_ns,
        valid_mask,
        t0_index=temporal.t0_index,
    )
    features[~unique_mask] = 0.0
    t0_invalid = int((~unique_mask[:, temporal.t0_index]).sum())
    read_errors = int(
        load_status.eq(EmitFrameCacheDataset.STATUS_READ_ERROR).sum()
    )
    if t0_invalid > int(args.max_invalid_t0):
        raise RuntimeError(
            f"{t0_invalid} rows have unusable t0 evidence, exceeding "
            f"--max-invalid-t0={args.max_invalid_t0}; no cache was written."
        )
    if read_errors > int(args.max_read_errors):
        raise RuntimeError(
            f"{read_errors} sources failed to read, exceeding "
            f"--max-read-errors={args.max_read_errors}; no cache was written."
        )

    csv_sha = common.sha256_file(csv_path)
    weights_sha = common.sha256_file(weights_path)
    if csv_sha != csv_sha_before:
        raise RuntimeError("CSV contents changed during feature extraction.")
    if weights_sha != weights_sha_before:
        raise RuntimeError("Panopticon checkpoint changed during extraction.")
    table_columns = (
        ID_COLUMN,
        LABEL_COLUMN,
        PLUME_ID_COLUMN,
        *PATH_COLUMNS,
        *TIME_COLUMNS,
    )
    table_sha = common.input_table_sha256(frame, table_columns)
    input_contract = {
        "script_version": SCRIPT_VERSION,
        "sensor": SENSOR_NAME,
        "reader": "Emit32CsvDataset:HWC-or-CHW-to-CHW-v1",
        "csv_sha256": csv_sha,
        "weights_sha256": weights_sha,
        "input_table_sha256": table_sha,
        "path_columns": list(PATH_COLUMNS),
        "time_columns": list(TIME_COLUMNS),
        "role_names": list(ROLE_NAMES),
        "band_indices": list(BAND_INDICES),
        "channel_ids": channel_ids.tolist(),
        "channel_id_semantics": "physical-wavelength-nanometers",
        "normalization_mean": mean,
        "normalization_std": std,
        "normalization_source": stats_provenance,
        "image_size": int(args.image_size),
        "min_valid_fraction": float(args.min_valid_fraction),
        "validity_band_index": int(args.validity_band_index),
        "zero_invalid_pixels": bool(args.zero_invalid_pixels),
        "local_cache_bypass_root": (
            str(Path(args.local_cache_bypass_root).expanduser().resolve())
            if args.local_cache_bypass_root
            else ""
        ),
        "duplicate_rule": "exact-utc-ns-valid-evidence-t0-then-role-order-v1",
        "canonical_event_rule": event_rule,
        "source_rows": int(source_rows),
        "selected_rows": int(len(frame)),
        "row_selection": (
            "all"
            if not args.max_rows or int(args.max_rows) >= source_rows
            else "deterministic-label-stratified-v1"
        ),
        "row_selection_seed": int(args.row_selection_seed),
    }
    input_contract_sha = common.sha256_bytes(
        common.canonical_json_bytes(input_contract)
    )
    payload: dict[str, Any] = {
        "format_version": CACHE_FORMAT_VERSION,
        "script_version": SCRIPT_VERSION,
        "sensor": SENSOR_NAME,
        "split": args.split,
        "features": features,
        "labels": labels,
        "ids": ids,
        "plume_ids": plume_ids,
        "event_ids": event_ids,
        "event_id_rule": event_rule,
        "timestamps_utc_ns": temporal.timestamps_utc_ns,
        "timestamps_utc_iso": temporal.timestamps_utc_iso,
        "timestamp_valid_mask": temporal.timestamp_valid_mask,
        "delta_days": temporal.delta_days,
        "role_names": list(ROLE_NAMES),
        "role_index": temporal.role_index,
        "t0_index": int(temporal.t0_index),
        "image_valid_mask": image_valid_mask,
        "valid_mask": valid_mask,
        "duplicate_mask": duplicate_mask,
        "duplicate_group_mask": duplicate_group_mask,
        "unique_mask": unique_mask,
        "valid_fraction": valid_fraction,
        "load_status": load_status,
        "path_columns": list(PATH_COLUMNS),
        "time_columns": list(TIME_COLUMNS),
        "input_contract": input_contract,
        "input_contract_sha256": input_contract_sha,
        "csv_path": str(csv_path),
        "csv_sha256": csv_sha,
        "weights_path": str(weights_path),
        "weights_sha256": weights_sha,
        "input_table_sha256": table_sha,
        "feature_sha256": common.tensor_sha256(features),
        "label_sha256": common.tensor_sha256(labels),
        "timestamp_sha256": common.tensor_sha256(
            temporal.timestamps_utc_ns
        ),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": float(time.monotonic() - started),
    }
    common.atomic_torch_save(output_path, payload)
    cache_sha = common.sha256_file(output_path)
    summary = {
        "cache_path": str(output_path),
        "cache_sha256": cache_sha,
        "sensor": SENSOR_NAME,
        "script_version": SCRIPT_VERSION,
        "split": args.split,
        "rows": rows,
        "timepoints": timepoints,
        "feature_dim": feature_dim,
        "storage_dtype": str(features.dtype),
        "usable_t0_rows": int(unique_mask[:, temporal.t0_index].sum()),
        "invalid_t0_rows": t0_invalid,
        "usable_observations": int(valid_mask.sum()),
        "unique_observations": int(unique_mask.sum()),
        "suppressed_duplicate_observations": int(duplicate_mask.sum()),
        "read_errors": read_errors,
        "input_contract_sha256": input_contract_sha,
        "csv_sha256": csv_sha,
        "weights_sha256": weights_sha,
        "feature_sha256": payload["feature_sha256"],
        "normalization_train_csv_sha256": stats_provenance[
            "normalization_train_csv_sha256"
        ],
        "stats_json_sha256": stats_provenance["stats_json_sha256"],
    }
    common.atomic_json_write(
        output_path.with_suffix(output_path.suffix + ".json"), summary
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def validate_emit_cache_identity(path: Path, expected_split: str) -> None:
    common.assert_not_sealed_path(path, purpose=f"{expected_split} cache")
    payload = common.torch_load_trusted(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a cache mapping.")
    if payload.get("sensor") != SENSOR_NAME:
        raise ValueError(
            f"{path} sensor={payload.get('sensor')!r}; expected {SENSOR_NAME!r}."
        )
    contract = payload.get("input_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{path} lacks an input contract.")
    if contract.get("sensor") != SENSOR_NAME:
        raise ValueError(f"{path} input contract is not stamped EMIT32.")
    expected_ids = [float(value) for value in emit_module.EMIT32_WAVELENGTHS_NM]
    if contract.get("channel_ids") != expected_ids:
        raise ValueError(f"{path} physical wavelength IDs changed.")
    if contract.get("path_columns") != list(PATH_COLUMNS):
        raise ValueError(f"{path} does not use the fixed three EMIT roles.")
    common.validate_cache_payload(
        payload, path=path, expected_split=expected_split
    )


@contextmanager
def _common_emit_context():
    previous_version = common.SCRIPT_VERSION
    previous_loader = common.load_cache_pair

    def emit_load_cache_pair(train_path: Path, val_path: Path):
        train, val, audit = previous_loader(train_path, val_path)
        if train.get("sensor") != SENSOR_NAME or val.get("sensor") != SENSOR_NAME:
            raise ValueError("Matched-head cache pair is not entirely EMIT32.")
        audit = {
            **audit,
            "sensor": SENSOR_NAME,
            "cache_script_version": SCRIPT_VERSION,
        }
        return train, val, audit

    common.SCRIPT_VERSION = SCRIPT_VERSION
    common.load_cache_pair = emit_load_cache_pair
    try:
        yield
    finally:
        common.load_cache_pair = previous_loader
        common.SCRIPT_VERSION = previous_version


def train_heads(args: argparse.Namespace) -> None:
    train_path = Path(args.train_cache).expanduser().resolve()
    val_path = Path(args.val_cache).expanduser().resolve()
    validate_emit_cache_identity(train_path, "train")
    validate_emit_cache_identity(val_path, "val")
    with _common_emit_context():
        common.train_heads(args)


def add_boolean_pair(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name}", dest=destination, action="store_true", help=help_text
    )
    group.add_argument(
        f"--no-{name}",
        dest=destination,
        action="store_false",
        help=f"Disable: {help_text}",
    )
    parser.set_defaults(**{destination: default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser(
        "cache", help="Create one strict frozen EMIT32 per-frame CLS cache."
    )
    cache_parser.add_argument("--csv", required=True)
    cache_parser.add_argument(
        "--split", choices=("train", "val"), required=True
    )
    cache_parser.add_argument("--output-cache", required=True)
    cache_parser.add_argument(
        "--weights", default="weights/panopticon_vitb14_teacher.pth"
    )
    cache_parser.add_argument(
        "--stats-json", default=str(DEFAULT_STATS_JSON)
    )
    cache_parser.add_argument(
        "--normalization-train-csv",
        default=str(DEFAULT_PROTOCOL_TRAIN),
        help=(
            "Exact train CSV whose SHA must match the nested sensors.emit "
            "normalization block."
        ),
    )
    cache_parser.add_argument("--image-size", type=int, default=224)
    cache_parser.add_argument(
        "--min-valid-fraction", type=float, default=0.75
    )
    cache_parser.add_argument(
        "--validity-band-index", type=int, default=0
    )
    add_boolean_pair(
        cache_parser,
        "zero-invalid-pixels",
        default=True,
        help_text="Zero native invalid pixels after normalization.",
    )
    cache_parser.add_argument("--batch-size", type=int, default=8)
    cache_parser.add_argument("--num-workers", type=int, default=6)
    cache_parser.add_argument("--prefetch-factor", type=int, default=2)
    add_boolean_pair(
        cache_parser,
        "persistent-workers",
        default=True,
        help_text="Keep DataLoader workers alive for the cache pass.",
    )
    cache_parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    cache_parser.add_argument(
        "--amp-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    cache_parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    cache_parser.add_argument(
        "--local-cache-dir",
        default="/diniuvol/yuyao/emit_ragged_cls_file_cache",
    )
    cache_parser.add_argument(
        "--local-cache-bypass-root", default="/diniuvol/yuyao"
    )
    cache_parser.add_argument(
        "--local-cache-mode",
        choices=("off", "sync", "async"),
        default="off",
    )
    cache_parser.add_argument("--local-cache-workers", type=int, default=2)
    cache_parser.add_argument(
        "--local-cache-min-free-gb", type=float, default=20.0
    )
    cache_parser.add_argument("--max-rows", type=int, default=0)
    cache_parser.add_argument(
        "--row-selection-seed", type=int, default=20260727
    )
    cache_parser.add_argument("--max-invalid-t0", type=int, default=0)
    cache_parser.add_argument("--max-read-errors", type=int, default=0)
    cache_parser.add_argument("--log-interval", type=int, default=20)
    cache_parser.add_argument("--overwrite", action="store_true")
    cache_parser.add_argument("--debug", action="store_true")
    cache_parser.set_defaults(handler=cache_features)

    head_parser = subparsers.add_parser(
        "train-heads",
        help="Train the four parameter-matched correspondence heads.",
    )
    head_parser.add_argument("--train-cache", required=True)
    head_parser.add_argument("--val-cache", required=True)
    head_parser.add_argument("--output-dir", required=True)
    head_parser.add_argument("--arms", default=",".join(ARM_NAMES))
    head_parser.add_argument("--epochs", type=int, default=3)
    head_parser.add_argument("--batch-size", type=int, default=256)
    head_parser.add_argument("--eval-batch-size", type=int, default=512)
    head_parser.add_argument("--learning-rate", type=float, default=3e-4)
    head_parser.add_argument("--weight-decay", type=float, default=0.05)
    head_parser.add_argument("--grad-clip", type=float, default=1.0)
    head_parser.add_argument("--model-dim", type=int, default=256)
    head_parser.add_argument("--num-heads", type=int, default=8)
    head_parser.add_argument("--mlp-ratio", type=float, default=2.0)
    head_parser.add_argument("--dropout", type=float, default=0.1)
    head_parser.add_argument(
        "--delta-periods", default="1,3,7,30,90,365"
    )
    head_parser.add_argument("--max-train-steps", type=int, default=0)
    head_parser.add_argument("--seed", type=int, default=20260727)
    head_parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    head_parser.add_argument("--overwrite", action="store_true")
    head_parser.set_defaults(handler=train_heads)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "cache":
        if args.image_size <= 0 or args.batch_size <= 0:
            raise ValueError("Image and batch sizes must be positive.")
        if args.num_workers < 0 or args.prefetch_factor <= 0:
            raise ValueError("Worker count/prefetch factor is invalid.")
        if not 0 <= args.validity_band_index < 32:
            raise ValueError("--validity-band-index must be in [0,31].")
        if not 0 <= args.min_valid_fraction <= 1:
            raise ValueError("--min-valid-fraction must be in [0,1].")
        if args.max_rows < 0:
            raise ValueError("--max-rows cannot be negative.")
    else:
        if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
            raise ValueError("Epoch and batch sizes must be positive.")
        if args.model_dim <= 0 or args.num_heads <= 0:
            raise ValueError("Model dimension and head count must be positive.")
        if args.model_dim % args.num_heads:
            raise ValueError("--model-dim must be divisible by --num-heads.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    args.handler(args)


if __name__ == "__main__":
    main()
