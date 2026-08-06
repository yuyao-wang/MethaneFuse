#!/usr/bin/env python3
"""Fixed-threshold cross-event history-shuffle audit for TEMPO patch heads.

The audit loads the complete development patch cache into host memory once.
For every row it keeps t0, validity, time lag, quality and the frozen base
logit, but replaces historical patch tokens with a deterministic donor row
from another canonical event.  The donor mapping is shared by every
checkpoint, original checkpoints are replay-verified, seed logits are
equal-averaged, and every original threshold remains fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

import tempo_l89_global as global_tempo
import tempo_l89_patch as tempo
from research.pretraining_20260727 import l89_ragged_cls_experiment as l89


FORMAT_VERSION = "tempo-l89-patch-history-shuffle-audit-v2"


def _read_prediction(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tempo.assert_development_path(path, purpose=f"{name} predictions")
    frame = pd.read_csv(path, low_memory=False)
    required = ("id", "plume_id", "event_id", "label", "logit")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    identity = manifest["identity"]
    expected = {
        "id": [str(value) for value in identity["ids"]],
        "plume_id": [str(value) for value in identity["plume_ids"]],
        "event_id": [str(value) for value in identity["event_ids"]],
        "label": [int(value) for value in identity["labels"]],
    }
    observed = {
        "id": frame["id"].astype(str).tolist(),
        "plume_id": frame["plume_id"].astype(str).tolist(),
        "event_id": frame["event_id"].astype(str).tolist(),
        "label": frame["label"].astype(int).tolist(),
    }
    for column in expected:
        if observed[column] != expected[column]:
            raise ValueError(f"{name}: ordered {column} differs")
    logits = frame["logit"].to_numpy(dtype=np.float64)
    if logits.shape != (int(manifest["rows"]),) or not np.isfinite(
        logits
    ).all():
        raise ValueError(f"{name}: invalid saved logits")
    return frame, {
        "path": str(path),
        "sha256": tempo.sha256_file(path),
    }


def _load_dense_cache(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, torch.Tensor]:
    rows = int(manifest["rows"])
    visits = int(manifest["configuration"]["timepoints"])
    height, width = (int(value) for value in manifest["grid_shape"])
    patches = height * width
    dimension = int(manifest["configuration"]["projection"]["output_dim"])
    dense = {
        "patch_tokens": torch.empty(
            rows, visits, patches, dimension, dtype=torch.float16
        ),
        "unique_mask": torch.empty(rows, visits, dtype=torch.bool),
        "delta_days": torch.empty(rows, visits, dtype=torch.float32),
        "quality": torch.empty(rows, visits, dtype=torch.float16),
    }
    seen = torch.zeros(rows, dtype=torch.bool)
    for record in sorted(
        manifest["shards"], key=lambda value: int(value["shard_index"])
    ):
        payload = tempo._load_shard(
            manifest_path.parent / str(record["file"])
        )
        indices = payload["row_indices"].long()
        expected = torch.arange(
            int(record["row_start"]), int(record["row_stop"])
        )
        if not torch.equal(indices, expected):
            raise ValueError("history-shuffle cache rows are not contiguous")
        if seen[indices].any():
            raise ValueError("history-shuffle cache repeats a row")
        seen[indices] = True
        for key in dense:
            dense[key][indices] = payload[key]
    if not seen.all():
        raise ValueError("history-shuffle cache has missing rows")
    return dense


def _build_model(
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> tempo.TempoPatchHead:
    config = checkpoint["configuration"]
    model = tempo.TempoPatchHead(
        int(config["feature_dim"]),
        match_rank=int(config["match_rank"]),
        value_dim=int(config["value_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        radius=int(config["radius"]),
        temperature=float(config["temperature"]),
        topk_fraction=float(config["topk_fraction"]),
        normality_scale=float(config["normality_scale"]),
        use_normality_features=bool(config["use_normality_features"]),
        residual_cap=float(config["residual_cap"]),
        zero_init_mode=str(config.get("zero_init_mode", "scalar")),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.requires_grad_(False).eval().to(device)


def apply_patch_history_donors(
    patch_tokens: torch.Tensor,
    donor_indices: torch.Tensor,
    *,
    t0_index: int,
    target_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace only historical patch content for selected target rows."""

    if patch_tokens.ndim != 4:
        raise ValueError("patch_tokens must have shape [N,T,P,D]")
    rows, visits = patch_tokens.shape[:2]
    if donor_indices.shape != (rows,):
        raise ValueError("donor_indices must have one entry per cache row")
    if donor_indices.numel() and (
        int(donor_indices.min()) < 0 or int(donor_indices.max()) >= rows
    ):
        raise ValueError("donor_indices contains an out-of-range row")
    if not 0 <= int(t0_index) < visits:
        raise ValueError("t0_index is out of range")
    if target_indices is None:
        target_indices = torch.arange(rows)
    target_indices = target_indices.long()
    output = patch_tokens[target_indices].clone()
    history = [index for index in range(visits) if index != int(t0_index)]
    output[:, history] = patch_tokens[donor_indices[target_indices]][
        :, history
    ]
    return output


def build_unique_mask_matched_cross_event_donors(
    unique_mask: torch.Tensor,
    event_ids: Sequence[str],
    *,
    t0_index: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build cross-event donors within complete availability-mask strata.

    A stratum with fewer than two canonical events is left bit-exact and
    marked ineligible. This prevents zero-filled invalid cache entries from
    replacing target-valid historical tokens.
    """

    if unique_mask.ndim != 2:
        raise ValueError("unique_mask must have shape [N,T]")
    rows, visits = unique_mask.shape
    if len(event_ids) != rows:
        raise ValueError("event_ids and unique_mask row counts differ")
    if not 0 <= int(t0_index) < visits:
        raise ValueError("t0_index is out of range")
    mask = unique_mask.detach().cpu().bool()
    patterns: dict[tuple[bool, ...], list[int]] = {}
    for row in range(rows):
        pattern = tuple(bool(value) for value in mask[row].tolist())
        patterns.setdefault(pattern, []).append(row)

    donors = torch.arange(rows, dtype=torch.long)
    eligible = torch.zeros(rows, dtype=torch.bool)
    pattern_receipts: list[dict[str, Any]] = []
    for pattern_index, pattern in enumerate(sorted(patterns)):
        target_rows = patterns[pattern]
        group_events = [str(event_ids[index]) for index in target_rows]
        event_count = len(set(group_events))
        pattern_seed = int(seed) + 1_000_003 * (pattern_index + 1)
        if event_count >= 2:
            local_donors = l89.build_cross_event_donor_indices(
                group_events, seed=pattern_seed
            )
            for local_target, local_donor in enumerate(
                local_donors.tolist()
            ):
                donors[target_rows[local_target]] = target_rows[local_donor]
            eligible[target_rows] = True
        pattern_receipts.append(
            {
                "pattern": [bool(value) for value in pattern],
                "rows": len(target_rows),
                "canonical_events": event_count,
                "eligible": event_count >= 2,
                "pattern_seed": pattern_seed if event_count >= 2 else None,
            }
        )

    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    full_mismatch = 0
    history_mismatch = 0
    valid_to_invalid = 0
    if eligible_indices.numel():
        donor_indices = donors[eligible_indices]
        if any(
            str(event_ids[target]) == str(event_ids[donor])
            for target, donor in zip(
                eligible_indices.tolist(), donor_indices.tolist()
            )
        ):
            raise RuntimeError("mask-matched donor shares target event")
        target_masks = mask[eligible_indices]
        donor_masks = mask[donor_indices]
        full_mismatch = int((target_masks != donor_masks).sum().item())
        history = [
            index for index in range(visits) if index != int(t0_index)
        ]
        history_mismatch = int(
            (target_masks[:, history] != donor_masks[:, history])
            .sum()
            .item()
        )
        valid_to_invalid = int(
            (target_masks[:, history] & ~donor_masks[:, history])
            .sum()
            .item()
        )
    if full_mismatch or history_mismatch or valid_to_invalid:
        raise RuntimeError("mask-matched donor availability contract failed")

    excluded_indices = torch.nonzero(~eligible, as_tuple=False).flatten()
    receipt = {
        "strategy": "full_unique_mask_pattern",
        "patterns": pattern_receipts,
        "pattern_count": len(pattern_receipts),
        "eligible_rows": int(eligible.sum().item()),
        "excluded_rows": int((~eligible).sum().item()),
        "excluded_row_indices": excluded_indices.tolist(),
        "excluded_rows_left_bit_exact": True,
        "donor_full_mask_target_mask_mismatch_slots": full_mismatch,
        "donor_history_mask_target_mask_mismatch_slots": history_mismatch,
        "target_valid_to_donor_invalid_history_slots": valid_to_invalid,
    }
    return donors, eligible, receipt


def donor_availability_receipt(
    unique_mask: torch.Tensor,
    donors: torch.Tensor,
    *,
    t0_index: int,
) -> dict[str, int]:
    """Count availability mismatches for a complete donor intervention."""

    mask = unique_mask.detach().cpu().bool()
    donor_mask = mask[donors.long()]
    history = [
        index for index in range(mask.shape[1]) if index != int(t0_index)
    ]
    return {
        "donor_full_mask_target_mask_mismatch_slots": int(
            (mask != donor_mask).sum().item()
        ),
        "donor_history_mask_target_mask_mismatch_slots": int(
            (mask[:, history] != donor_mask[:, history]).sum().item()
        ),
        "target_valid_to_donor_invalid_history_slots": int(
            (mask[:, history] & ~donor_mask[:, history]).sum().item()
        ),
    }


def _predict(
    model: tempo.TempoPatchHead,
    dense: Mapping[str, torch.Tensor],
    base_logits: torch.Tensor,
    manifest: Mapping[str, Any],
    *,
    donors: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows = int(manifest["rows"])
    t0_index = int(manifest["configuration"]["t0_index"])
    grid_shape = tuple(int(value) for value in manifest["grid_shape"])
    logits = torch.empty(rows, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, rows, int(batch_size)):
            indices = torch.arange(start, min(rows, start + int(batch_size)))
            patch_tokens = dense["patch_tokens"][indices]
            unique_mask = dense["unique_mask"][indices]
            delta_days = dense["delta_days"][indices]
            quality = dense["quality"][indices]
            if donors is not None:
                patch_tokens = apply_patch_history_donors(
                    dense["patch_tokens"],
                    donors,
                    t0_index=t0_index,
                    target_indices=indices,
                )
            output = model(
                base_logits[indices].to(device),
                patch_tokens.to(device),
                unique_mask.to(device),
                delta_days.to(device),
                quality.to(device),
                t0_index=t0_index,
                grid_shape=grid_shape,
            )
            logits[indices] = output.logits.cpu()
    return logits.numpy()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--p0-overlay", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--donor-seed", type=int, default=20260728)
    parser.add_argument(
        "--donor-strategy",
        choices=("unmatched_cross_event", "full_unique_mask_pattern"),
        default="unmatched_cross_event",
    )
    parser.add_argument("--replay-tolerance", type=float, default=2e-6)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.predictions):
        raise ValueError("checkpoint and prediction counts differ")
    if len(args.checkpoints) < 2:
        raise ValueError("history-shuffle ensemble needs at least two seeds")

    manifest_path = Path(args.val_manifest).expanduser().resolve()
    overlay_path = Path(args.p0_overlay).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path, purpose in (
        (manifest_path, "validation manifest"),
        (overlay_path, "P0 overlay"),
        (output_dir, "audit output"),
    ):
        tempo.assert_development_path(path, purpose=purpose)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = tempo.load_manifest(manifest_path, expected_split="val")
    base_logits, overlay_audit = tempo.load_base_overlay(
        overlay_path, manifest, expected_split="val"
    )
    if "p0" not in str(overlay_audit["family"]).lower():
        raise ValueError("history-shuffle audit requires the P0 base overlay")

    prediction_frames: list[pd.DataFrame] = []
    prediction_sources: list[dict[str, Any]] = []
    for index, raw_path in enumerate(args.predictions):
        frame, source = _read_prediction(
            Path(raw_path).expanduser().resolve(),
            manifest,
            name=f"seed {index}",
        )
        prediction_frames.append(frame)
        prediction_sources.append(source)

    checkpoint_payloads: list[dict[str, Any]] = []
    checkpoint_sources: list[dict[str, Any]] = []
    for raw_path in args.checkpoints:
        path = Path(raw_path).expanduser().resolve()
        tempo.assert_development_path(path, purpose="patch checkpoint")
        payload = dict(tempo._load_torch_payload(path))
        if bool(payload.get("test_or_sealed_read", True)):
            raise ValueError("checkpoint violates development-only contract")
        config = payload["configuration"]
        if config["val_manifest_sha256"] != tempo.sha256_file(manifest_path):
            raise ValueError("checkpoint validation manifest differs")
        if config["val_base_overlay_sha256"] != tempo.sha256_file(
            overlay_path
        ):
            raise ValueError("checkpoint P0 overlay differs")
        checkpoint_payloads.append(payload)
        checkpoint_sources.append(
            {
                "path": str(path),
                "sha256": tempo.sha256_file(path),
                "seed": int(config["seed"]),
                "selected_epoch": int(payload["epoch"]),
            }
        )

    dense = _load_dense_cache(manifest, manifest_path)
    event_ids = [str(value) for value in manifest["identity"]["event_ids"]]
    t0_index = int(manifest["configuration"]["t0_index"])
    if args.donor_strategy == "full_unique_mask_pattern":
        donors, eligible, donor_receipt = (
            build_unique_mask_matched_cross_event_donors(
                dense["unique_mask"],
                event_ids,
                t0_index=t0_index,
                seed=int(args.donor_seed),
            )
        )
    else:
        donors = l89.build_cross_event_donor_indices(
            event_ids, seed=int(args.donor_seed)
        )
        eligible = torch.ones(len(event_ids), dtype=torch.bool)
        donor_receipt = {
            "strategy": "unmatched_cross_event",
            "eligible_rows": len(event_ids),
            "excluded_rows": 0,
            "excluded_row_indices": [],
            **donor_availability_receipt(
                dense["unique_mask"],
                donors,
                t0_index=t0_index,
            ),
        }
    if any(
        event_ids[index] == event_ids[donor]
        for index, donor in enumerate(donors.tolist())
        if bool(eligible[index])
    ):
        raise RuntimeError("history donor shares the target event")
    device = torch.device(args.device)
    original_logits: list[np.ndarray] = []
    shuffled_logits: list[np.ndarray] = []
    replay_errors: list[float] = []
    for checkpoint, frame in zip(
        checkpoint_payloads, prediction_frames
    ):
        model = _build_model(checkpoint, device=device)
        replay = _predict(
            model,
            dense,
            base_logits,
            manifest,
            donors=None,
            batch_size=int(args.batch_size),
            device=device,
        )
        saved = frame["logit"].to_numpy(dtype=np.float64)
        error = float(np.max(np.abs(replay.astype(np.float64) - saved)))
        if not np.isfinite(error) or error > float(args.replay_tolerance):
            raise AssertionError(
                f"original checkpoint replay error {error} exceeds tolerance"
            )
        replay_errors.append(error)
        original_logits.append(saved)
        shuffled_logits.append(
            _predict(
                model,
                dense,
                base_logits,
                manifest,
                donors=donors,
                batch_size=int(args.batch_size),
                device=device,
            ).astype(np.float64)
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    original_ensemble_logit = np.mean(
        np.stack(original_logits, axis=0), axis=0
    )
    shuffled_ensemble_logit = np.mean(
        np.stack(shuffled_logits, axis=0), axis=0
    )
    labels = np.asarray(manifest["identity"]["labels"], dtype=np.int64)
    original_probability = _sigmoid(original_ensemble_logit)
    shuffled_probability = _sigmoid(shuffled_ensemble_logit)
    original_metrics = global_tempo.metric_bundle(
        labels, original_probability, event_ids
    )
    fixed_threshold = float(original_metrics["selected_threshold"])
    shuffled_metrics = global_tempo.metric_bundle(
        labels,
        shuffled_probability,
        event_ids,
        threshold=fixed_threshold,
    )
    delta_keys = (
        "event_balanced_ap",
        "event_balanced_auc",
        "event_balanced_positive_f1_selected",
        "event_balanced_macro_f1_selected",
        "all_negative_fp_mass",
    )
    deltas = {
        key: float(shuffled_metrics[key] - original_metrics[key])
        for key in delta_keys
    }
    deltas["mean_absolute_probability_delta"] = float(
        np.mean(np.abs(shuffled_probability - original_probability))
    )
    per_seed: list[dict[str, Any]] = []
    for source, original_logit, shuffled_logit, replay_error in zip(
        checkpoint_sources,
        original_logits,
        shuffled_logits,
        replay_errors,
    ):
        original_seed_probability = _sigmoid(original_logit)
        shuffled_seed_probability = _sigmoid(shuffled_logit)
        original_seed_metrics = global_tempo.metric_bundle(
            labels, original_seed_probability, event_ids
        )
        seed_threshold = float(original_seed_metrics["selected_threshold"])
        shuffled_seed_metrics = global_tempo.metric_bundle(
            labels,
            shuffled_seed_probability,
            event_ids,
            threshold=seed_threshold,
        )
        seed_delta = {
            key: float(
                shuffled_seed_metrics[key] - original_seed_metrics[key]
            )
            for key in delta_keys
        }
        seed_delta["mean_absolute_probability_delta"] = float(
            np.mean(
                np.abs(
                    shuffled_seed_probability - original_seed_probability
                )
            )
        )
        per_seed.append(
            {
                **source,
                "original_checkpoint_replay_max_abs_error": replay_error,
                "fixed_original_threshold": seed_threshold,
                "original_metrics": original_seed_metrics,
                "history_shuffle_fixed_model_and_threshold": (
                    shuffled_seed_metrics
                ),
                "history_shuffle_delta": seed_delta,
            }
        )

    prediction_output = pd.DataFrame(
        {
            "id": manifest["identity"]["ids"],
            "plume_id": manifest["identity"]["plume_ids"],
            "event_id": event_ids,
            "label": labels,
            "intervention_eligible": eligible.numpy(),
            "donor_row_index": donors.numpy(),
            "donor_id": [
                manifest["identity"]["ids"][index]
                for index in donors.tolist()
            ],
            "donor_event_id": [
                event_ids[index] for index in donors.tolist()
            ],
            "original_ensemble_logit": original_ensemble_logit.astype(
                np.float32
            ),
            "shuffled_ensemble_logit": shuffled_ensemble_logit.astype(
                np.float32
            ),
            "original_ensemble_probability": original_probability,
            "shuffled_ensemble_probability": shuffled_probability,
        }
    )
    prediction_path = output_dir / "history_shuffle_predictions.csv"
    l89.atomic_csv_write(prediction_path, prediction_output)
    audit = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "rows": len(labels),
        "events": len(set(event_ids)),
        "seeds": len(checkpoint_payloads),
        "original_checkpoint_replay_max_abs_errors": replay_errors,
        "required_replay_tolerance": float(args.replay_tolerance),
        "per_seed": per_seed,
        "original_metrics": original_metrics,
        "history_shuffle_fixed_model_and_threshold": shuffled_metrics,
        "history_shuffle_delta": deltas,
        "history_shuffle_contract": {
            "donor_seed": int(args.donor_seed),
            "donor_strategy": str(args.donor_strategy),
            "same_donor_mapping_for_all_checkpoints": True,
            "eligible_donor_from_different_canonical_event": True,
            "preserved_current_t0": True,
            "preserved_frozen_p0_base_logit": True,
            "replaced_history_patch_tokens": True,
            "preserved_target_unique_mask": True,
            "preserved_target_delta_days": True,
            "preserved_target_quality": True,
            "donor_availability_receipt": donor_receipt,
            "original_ensemble_threshold_fixed": fixed_threshold,
            "threshold_refit_after_shuffle": False,
        },
        "sources": {
            "val_manifest": str(manifest_path),
            "val_manifest_sha256": tempo.sha256_file(manifest_path),
            "p0_overlay": str(overlay_path),
            "p0_overlay_sha256": tempo.sha256_file(overlay_path),
            "p0_overlay_audit": overlay_audit,
            "checkpoints": checkpoint_sources,
            "predictions": prediction_sources,
        },
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": tempo.sha256_file(prediction_path),
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": tempo.sha256_file(Path(__file__).resolve()),
        "test_or_sealed_read": False,
    }
    output_path = output_dir / "history_shuffle_audit.json"
    tempo.atomic_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
