#!/usr/bin/env python3
"""Inference script for overlap rows with 4 single-sensor models + logical OR fusion.

Given a wide-table CSV (s2/l89/emit or wv3/s5p columns), this script:
1) runs sensor-specific single models independently;
2) aggregates predictions by row id;
3) applies logical OR alarm rule (any sensor alarm => row alarm).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make repository root importable when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hubconf import _panopticon_vitb14
from universal_models_fusion.multi_sensor_panopticon_4_nonlinear_loraadapter_sensor_specific_earlystopping import (
    DEFAULT_WV3_BANDS,
    ConcatTemporalDataset,
    StaticAnchoredCache,
    TriSensorTemporalCsvDataset,
    compute_binary_metrics,
    custom_collate_fn,
    load_wv3_channel_ids_from_srf,
    recursive_to_device,
)


SENSOR_ORDER = ("s2", "l89", "s5p", "wv3")


class CLSHead(nn.Module):
    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(self.norm(cls))


@dataclass
class SensorModel:
    name: str
    backbone: nn.Module
    head: nn.Module

    @torch.no_grad()
    def predict_logits(self, x_dict: MutableMapping[str, torch.Tensor]) -> torch.Tensor:
        feats = self.backbone(x_dict, is_training=True)
        cls_token = feats["x_norm_clstoken"]
        logits = self.head(cls_token)
        return torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return dict(state_dict)
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _resolve_checkpoint_path(path_like: str) -> Path:
    path = Path(str(path_like).strip()).expanduser()
    if str(path).endswith("/"):
        path = Path(str(path).rstrip("/"))
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.is_dir():
        candidate = path / "ckpt_best_test.pth"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"Checkpoint path points to a directory without ckpt_best_test.pth: {path}"
        )
    return path


def _extract_backbone_head_state(payload: Mapping[str, Any], ckpt_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    backbone_state = None
    head_state = None
    if isinstance(payload.get("backbone"), Mapping):
        backbone_state = payload["backbone"]
    elif isinstance(payload.get("backbone_state_dict"), Mapping):
        backbone_state = payload["backbone_state_dict"]

    if isinstance(payload.get("head"), Mapping):
        head_state = payload["head"]
    elif isinstance(payload.get("head_state_dict"), Mapping):
        head_state = payload["head_state_dict"]

    if backbone_state is None or head_state is None:
        raise ValueError(
            f"Unsupported checkpoint format in {ckpt_path}. "
            "Expected keys (backbone/head) or (backbone_state_dict/head_state_dict)."
        )
    return _strip_module_prefix(backbone_state), _strip_module_prefix(head_state)


def _infer_num_classes(head_state: Mapping[str, torch.Tensor]) -> int:
    for key, tensor in head_state.items():
        if key.endswith("fc.weight") and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.shape[0])
        if key.endswith("weight") and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.shape[0])
    return 2


def _load_sensor_model(sensor: str, ckpt_path: str, device: torch.device) -> SensorModel:
    resolved = _resolve_checkpoint_path(ckpt_path)
    payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must be a mapping/dict: {resolved}")

    backbone_state, head_state = _extract_backbone_head_state(payload, resolved)
    backbone = _panopticon_vitb14()
    backbone.load_state_dict(backbone_state, strict=True)
    num_classes = _infer_num_classes(head_state)
    head = CLSHead(embed_dim=int(getattr(backbone, "embed_dim", 768)), num_classes=num_classes)
    head.load_state_dict(head_state, strict=True)

    backbone = backbone.to(device).eval()
    head = head.to(device).eval()
    return SensorModel(name=sensor, backbone=backbone, head=head)


def _slice_x_dict(x_dict: MutableMapping[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v.index_select(0, idx) for k, v in x_dict.items()}


def _as_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _safe_artifact_name(name: str, *, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(name).strip())
    cleaned = cleaned.strip("-.")
    if not cleaned:
        cleaned = "single4_or_infer"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-.")
    if not cleaned:
        cleaned = "single4_or_infer"
    return cleaned


def _try_parse_binary_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return None
    try:
        num = float(text)
    except Exception:
        return None
    if num != num:  # NaN
        return None
    return 1 if num >= 0.5 else 0


def _autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def compute_split_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    pos_scores: Sequence[float],
) -> Dict[str, float]:
    count = int(len(labels))
    out: Dict[str, float] = {
        "count": float(count),
        "acc": float("nan"),
        "fpr": float("nan"),
        "recall": float("nan"),
        "auroc": float("nan"),
    }
    if count == 0 or len(preds) != count:
        return out
    labels_np = np.asarray(labels, dtype=np.int64)
    preds_np = np.asarray(preds, dtype=np.int64)
    out["acc"] = float((labels_np == preds_np).mean())
    if len(pos_scores) == count:
        out.update(
            compute_binary_metrics(
                labels=labels_np,
                preds=preds_np,
                pos_scores=np.asarray(pos_scores, dtype=np.float64),
            )
        )
    return out


def _compute_eval_split_panels(
    labels: np.ndarray,
    preds: np.ndarray,
    pos_scores: np.ndarray,
    overlap_flags: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    if not (labels.shape == preds.shape == pos_scores.shape == overlap_flags.shape):
        raise ValueError("labels/preds/pos_scores/overlap_flags must have identical shape")
    split_masks = {
        "ALL": np.ones(labels.shape[0], dtype=np.bool_),
        "OVERLAP": overlap_flags.astype(np.bool_, copy=False),
        "SINGLE": (~overlap_flags.astype(np.bool_, copy=False)),
    }
    panel: Dict[str, Dict[str, float]] = {}
    for split_name, mask in split_masks.items():
        idx = np.nonzero(mask)[0]
        panel[split_name] = compute_split_metrics(
            labels=labels[idx],
            preds=preds[idx],
            pos_scores=pos_scores[idx],
        )
    return panel


def _print_eval_split_panel(strategy_name: str, panel: Mapping[str, Mapping[str, float]]) -> None:
    for split_name in ("ALL", "OVERLAP", "SINGLE"):
        m = panel[split_name]
        print(
            f"[EvalSplit][{strategy_name}][{split_name}][OVERALL] count={int(m['count'])} "
            f"acc={m['acc']:.4f} fpr={m['fpr']:.4f} recall={m['recall']:.4f} auroc={m['auroc']:.4f}",
            flush=True,
        )


def _build_dataset(args) -> TriSensorTemporalCsvDataset:
    wv3_bands = [b.strip() for b in str(args.wv3_bands).split(",") if b.strip()]
    if not wv3_bands:
        raise ValueError("--wv3_bands must provide at least one band name")
    wv3_chn_ids = load_wv3_channel_ids_from_srf(args.wv3_srf_csv, wv3_bands).unsqueeze(-1)

    cache_obj = None
    if args.local_cache_dir:
        cache_obj = StaticAnchoredCache(args.local_cache_dir, min_free_gb=args.local_cache_min_free_gb)

    ds = TriSensorTemporalCsvDataset(
        csv_path=args.csv_path,
        fusion_group_column=args.group_column,
        local_file_cache=cache_obj,
        s5p_data_key=args.s5p_data_key,
        s5p_chn_ids_key=args.s5p_chn_ids_key,
        s5p_channels_last=args.s5p_channels_last,
        align_l89_to_s2=args.align_l89_to_s2,
        wv3_chn_ids=wv3_chn_ids,
        pad_to_multiple=14,
    )
    if bool(args.overlap_only):
        if "overlap_mode" not in ds.df.columns:
            raise ValueError("CSV has no 'overlap_mode' column, cannot apply --overlap_only")
        mask = ds.df["overlap_mode"].apply(_as_bool)
        ds.df = ds.df[mask].reset_index(drop=True)
    if int(args.max_rows) > 0:
        ds.df = ds.df.iloc[: int(args.max_rows)].reset_index(drop=True)
    return ds


def parse_args():
    parser = argparse.ArgumentParser(description="4 single-model inference + logical OR fusion (wide overlap CSV).")
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Input wide-table CSV path.",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Output CSV path. If empty, auto-generates next to input CSV.",
    )
    parser.add_argument("--group_column", default="id")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--overlap_only", action="store_true")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Optional row cap after filtering. <=0 means all rows.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--sensor_sub_batch_size",
        type=int,
        default=8,
        help="Micro-batch size per sensor on GPU. <=0 means no sub-batching within a loader batch.",
    )
    parser.add_argument(
        "--model_resident",
        choices=("one_by_one", "all"),
        default="one_by_one",
        help="one_by_one keeps only one sensor model on GPU at a time (lower memory).",
    )
    parser.add_argument(
        "--amp_dtype",
        choices=("none", "fp16", "bf16"),
        default="fp16",
        help="Autocast dtype for inference on CUDA.",
    )
    parser.add_argument("--alarm_threshold", type=float, default=0.5)

    parser.add_argument(
        "--s2_ckpt",
        default="/transferdiniu2/yuyao/checkpoints/s2/ckpt_best_test.pth",
    )
    parser.add_argument(
        "--l89_ckpt",
        default="/transferdiniu2/yuyao/checkpoints/l89/ckpt_best_test.pth",
    )
    parser.add_argument(
        "--s5p_ckpt",
        default="/transferdiniu2/yuyao/checkpoints/s5p",
    )
    parser.add_argument(
        "--wv3_ckpt",
        default="checkpoints/manifest_multisensor_crop_scheme2_train__manifest_multisensor_crop_scheme2_test__ft/ckpt_best_test.pth",
    )

    parser.add_argument(
        "--s5p_data_key",
        default="ch4",
        help="NPZ key for S5P arrays.",
    )
    parser.add_argument("--s5p_chn_ids_key", default="chn_ids")
    parser.add_argument("--s5p_channels_last", action="store_true")
    parser.add_argument("--align_l89_to_s2", action="store_true")
    parser.add_argument(
        "--wv3_srf_csv",
        default=str(Path(__file__).resolve().parents[1] / "WV3_VNIR_SWIR_response.csv"),
    )
    parser.add_argument(
        "--wv3_bands",
        default=",".join(DEFAULT_WV3_BANDS),
    )
    parser.add_argument("--local_cache_dir", default=None)
    parser.add_argument("--local_cache_min_free_gb", type=float, default=20.0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="panopticon-multisensor")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument(
        "--wandb_log_table_rows",
        type=int,
        default=2000,
        help="How many rows to log in wandb table preview. <=0 disables table logging.",
    )
    parser.add_argument(
        "--wandb_artifact_name",
        default="",
        help="Optional artifact name. Defaults to single4_or_infer_<input_stem>.",
    )
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if not (0.0 <= float(args.alarm_threshold) <= 1.0):
        raise ValueError(f"--alarm_threshold must be in [0,1], got {args.alarm_threshold}")

    wandb_run = None
    wandb_mod = None
    if bool(args.use_wandb):
        try:
            import wandb as _wandb
        except Exception as exc:
            raise RuntimeError(
                "--use_wandb is set but wandb import failed. Install wandb in the running environment."
            ) from exc
        wandb_mod = _wandb
        wandb_run = wandb_mod.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity if args.wandb_entity else None,
            config=vars(args),
        )

    try:
        dataset = _build_dataset(args)
        if len(dataset.df) == 0:
            raise RuntimeError("No rows available for inference after filtering.")

        infer_ds = ConcatTemporalDataset(dataset)
        loader = DataLoader(
            infer_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=custom_collate_fn,
        )

        sensor_ckpts = {
            "s2": args.s2_ckpt,
            "l89": args.l89_ckpt,
            "s5p": args.s5p_ckpt,
            "wv3": args.wv3_ckpt,
        }
        results_by_id: Dict[str, Dict[str, Tuple[float, int]]] = {}
        sensor_sample_records: Dict[str, list[Tuple[str, int, int, float]]] = {
            sensor_name: [] for sensor_name in SENSOR_ORDER
        }
        threshold = float(args.alarm_threshold)

        def infer_one_sensor(sensor_name: str, model: SensorModel) -> None:
            sub_batch_size = int(args.sensor_sub_batch_size)
            for x_dict_cpu, labels_batch, sensors, group_ids in loader:
                batch_size = len(sensors)
                if len(group_ids) != batch_size:
                    raise RuntimeError("group_ids size mismatch with batch size")

                indices = [i for i, s in enumerate(sensors) if s == sensor_name]
                if not indices:
                    continue
                step = max(1, sub_batch_size) if sub_batch_size > 0 else len(indices)

                for start in range(0, len(indices), step):
                    sub_indices = indices[start:start + step]
                    idx_tensor = torch.tensor(sub_indices, dtype=torch.long)
                    x_sub_cpu = _slice_x_dict(x_dict_cpu, idx_tensor)
                    x_sub = recursive_to_device(x_sub_cpu, device)
                    try:
                        with _autocast_context(device, args.amp_dtype):
                            logits = model.predict_logits(x_sub)
                        probs = torch.softmax(logits.float(), dim=-1)
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            suggested_sub_bs = max(1, step // 2)
                            raise RuntimeError(
                                "CUDA OOM during inference. "
                                f"sensor={sensor_name}, batch_size={args.batch_size}, "
                                f"sensor_sub_batch_size={step}. "
                                "Try lowering --sensor_sub_batch_size and/or --batch_size. "
                                f"Suggested next try: --sensor_sub_batch_size {suggested_sub_bs} --batch_size {max(1, int(args.batch_size) // 2)}"
                            ) from exc
                        raise
                    if probs.shape[-1] < 2:
                        raise RuntimeError(f"Expected binary head for sensor={sensor_name}, got shape={tuple(probs.shape)}")
                    prob1 = probs[:, 1].detach().to("cpu").tolist()
                    pred = [1 if p >= threshold else 0 for p in prob1]

                    for local_i, global_i in enumerate(sub_indices):
                        gid = str(group_ids[global_i])
                        if gid not in results_by_id:
                            results_by_id[gid] = {}
                        results_by_id[gid][sensor_name] = (float(prob1[local_i]), int(pred[local_i]))
                        if isinstance(labels_batch, torch.Tensor):
                            y_true = int(labels_batch[global_i].item())
                        else:
                            y_true = int(labels_batch[global_i])
                        sensor_sample_records[sensor_name].append(
                            (gid, int(y_true), int(pred[local_i]), float(prob1[local_i]))
                        )
                    del x_sub, logits, probs

        if args.model_resident == "all":
            sensor_models = {
                "s2": _load_sensor_model("s2", sensor_ckpts["s2"], device),
                "l89": _load_sensor_model("l89", sensor_ckpts["l89"], device),
                "s5p": _load_sensor_model("s5p", sensor_ckpts["s5p"], device),
                "wv3": _load_sensor_model("wv3", sensor_ckpts["wv3"], device),
            }
            for sensor_name in SENSOR_ORDER:
                infer_one_sensor(sensor_name, sensor_models[sensor_name])
        else:
            for sensor_name in SENSOR_ORDER:
                print(f"[Infer] Loading sensor model: {sensor_name}", flush=True)
                model = _load_sensor_model(sensor_name, sensor_ckpts[sensor_name], device)
                infer_one_sensor(sensor_name, model)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        out_df = dataset.df.copy()
        for sensor_name in SENSOR_ORDER:
            out_df[f"pred_prob1_{sensor_name}_single"] = np.nan
            out_df[f"pred_label_{sensor_name}_single"] = np.nan
        out_df["pred_alarm_or_single4"] = 0
        out_df["pred_prob1_or_single4_max"] = 0.0
        out_df["pred_alarm_softvote_single4"] = 0
        out_df["pred_prob1_softvote_single4_mean"] = 0.0
        out_df["available_sensor_count_single4"] = 0

        for row_idx, row in out_df.iterrows():
            gid = str(row[args.group_column])
            sensor_res = results_by_id.get(gid, {})
            alarms = []
            probs = []
            for sensor_name in SENSOR_ORDER:
                value = sensor_res.get(sensor_name)
                if value is None:
                    continue
                prob1, pred = value
                out_df.at[row_idx, f"pred_prob1_{sensor_name}_single"] = float(prob1)
                out_df.at[row_idx, f"pred_label_{sensor_name}_single"] = int(pred)
                alarms.append(int(pred))
                probs.append(float(prob1))
            out_df.at[row_idx, "available_sensor_count_single4"] = int(len(sensor_res))
            out_df.at[row_idx, "pred_alarm_or_single4"] = int(max(alarms) if alarms else 0)
            out_df.at[row_idx, "pred_prob1_or_single4_max"] = float(max(probs) if probs else 0.0)
            soft_prob = float(sum(probs) / len(probs)) if probs else 0.0
            out_df.at[row_idx, "pred_prob1_softvote_single4_mean"] = soft_prob
            out_df.at[row_idx, "pred_alarm_softvote_single4"] = int(1 if soft_prob >= threshold else 0)

        input_path = Path(args.csv_path).expanduser()
        if args.output_csv and str(args.output_csv).strip():
            output_path = Path(args.output_csv).expanduser()
        else:
            suffix = ".overlap_only" if args.overlap_only else ".all_rows"
            output_path = input_path.with_name(input_path.stem + f".single4_or_infer{suffix}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)

        total_rows = int(len(out_df))
        alarm_rows = int((out_df["pred_alarm_or_single4"] == 1).sum())
        softvote_alarm_rows = int((out_df["pred_alarm_softvote_single4"] == 1).sum())
        overlap_text = "overlap-only" if args.overlap_only else "all rows"
        alarm_ratio = 100.0 * alarm_rows / max(1, total_rows)
        softvote_alarm_ratio = 100.0 * softvote_alarm_rows / max(1, total_rows)

        # Sample-level split metrics (same counting unit as training eval loop).
        group_to_sensors_seen: Dict[str, set[str]] = {}
        for sensor_name, records in sensor_sample_records.items():
            for gid, _, _, _ in records:
                if gid not in group_to_sensors_seen:
                    group_to_sensors_seen[gid] = set()
                group_to_sensors_seen[gid].add(sensor_name)
        overlap_groups = {gid for gid, sensors_seen in group_to_sensors_seen.items() if len(sensors_seen) >= 2}
        sensor_sample_split_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for sensor_name in SENSOR_ORDER:
            records = sensor_sample_records.get(sensor_name, [])
            if len(records) == 0:
                empty = compute_split_metrics(labels=[], preds=[], pos_scores=[])
                sensor_sample_split_metrics[sensor_name] = {
                    "ALL": dict(empty),
                    "OVERLAP": dict(empty),
                    "SINGLE": dict(empty),
                }
                continue
            labels_np = np.asarray([int(r[1]) for r in records], dtype=np.int64)
            preds_np = np.asarray([int(r[2]) for r in records], dtype=np.int64)
            scores_np = np.asarray([float(r[3]) for r in records], dtype=np.float64)
            overlap_flags_np = np.asarray([str(r[0]) in overlap_groups for r in records], dtype=np.bool_)
            sensor_sample_split_metrics[sensor_name] = _compute_eval_split_panels(
                labels=labels_np,
                preds=preds_np,
                pos_scores=scores_np,
                overlap_flags=overlap_flags_np,
            )

        or_metrics: Dict[str, float] = {}
        softvote_metrics: Dict[str, float] = {}
        or_split_metrics: Dict[str, Dict[str, float]] = {}
        softvote_split_metrics: Dict[str, Dict[str, float]] = {}
        valid_eval_rows = 0
        overlap_column_exists = "overlap_mode" in out_df.columns
        available_sensor_count_exists = "available_sensor_count_single4" in out_df.columns
        if args.label_column in out_df.columns:
            eval_labels = []
            eval_preds_or = []
            eval_scores_or = []
            eval_preds_softvote = []
            eval_scores_softvote = []
            eval_row_indices = []
            for row_idx, row in out_df.iterrows():
                y = _try_parse_binary_label(row[args.label_column])
                if y is None:
                    continue
                eval_labels.append(int(y))
                eval_preds_or.append(int(row["pred_alarm_or_single4"]))
                eval_scores_or.append(float(row["pred_prob1_or_single4_max"]))
                eval_preds_softvote.append(int(row["pred_alarm_softvote_single4"]))
                eval_scores_softvote.append(float(row["pred_prob1_softvote_single4_mean"]))
                eval_row_indices.append(int(row_idx))
            valid_eval_rows = len(eval_labels)
            if valid_eval_rows > 0:
                y_np = np.asarray(eval_labels, dtype=np.int64)
                p_or = np.asarray(eval_preds_or, dtype=np.int64)
                s_or = np.asarray(eval_scores_or, dtype=np.float64)
                p_soft = np.asarray(eval_preds_softvote, dtype=np.int64)
                s_soft = np.asarray(eval_scores_softvote, dtype=np.float64)
                split_source = "fallback_all_overlap"
                if available_sensor_count_exists:
                    sensor_count_all = np.asarray(
                        out_df["available_sensor_count_single4"].to_numpy(),
                        dtype=np.int64,
                    )
                    valid_sensor_counts = sensor_count_all[np.asarray(eval_row_indices, dtype=np.int64)]
                    # Align with training eval: OVERLAP means >=2 sensors available on that row.
                    valid_overlap_flags = valid_sensor_counts >= 2
                    split_source = "available_sensor_count_single4"
                elif overlap_column_exists:
                    overlap_flags_all = np.asarray(
                        [_as_bool(v) for v in out_df["overlap_mode"].tolist()],
                        dtype=np.bool_,
                    )
                    valid_overlap_flags = overlap_flags_all[np.asarray(eval_row_indices, dtype=np.int64)]
                    split_source = "overlap_mode"
                else:
                    valid_overlap_flags = np.ones(valid_eval_rows, dtype=np.bool_)
                or_split_metrics = _compute_eval_split_panels(
                    labels=y_np,
                    preds=p_or,
                    pos_scores=s_or,
                    overlap_flags=valid_overlap_flags,
                )
                softvote_split_metrics = _compute_eval_split_panels(
                    labels=y_np,
                    preds=p_soft,
                    pos_scores=s_soft,
                    overlap_flags=valid_overlap_flags,
                )
                print(
                    "[EvalSplit] split source: "
                    f"{split_source} (OVERLAP={int(valid_overlap_flags.sum())}, "
                    f"SINGLE={int(valid_eval_rows - int(valid_overlap_flags.sum()))})",
                    flush=True,
                )
                all_or = or_split_metrics["ALL"]
                all_soft = softvote_split_metrics["ALL"]
                or_metrics = {
                    "or_acc": float(all_or["acc"]),
                    "or_fpr": float(all_or["fpr"]),
                    "or_auroc": float(all_or["auroc"]),
                    "or_recall": float(all_or["recall"]),
                }
                softvote_metrics = {
                    "softvote_acc": float(all_soft["acc"]),
                    "softvote_fpr": float(all_soft["fpr"]),
                    "softvote_auroc": float(all_soft["auroc"]),
                    "softvote_recall": float(all_soft["recall"]),
                }
            else:
                print(
                    f"[Warn] label column '{args.label_column}' exists but has no valid numeric labels for OR/SoftVote metrics.",
                    flush=True,
                )
        else:
            print(
                f"[Warn] label column '{args.label_column}' not found; skip OR/SoftVote acc/fpr/auroc/recall.",
                flush=True,
            )

        print(
            f"[Done] Wrote {total_rows} rows ({overlap_text}) to: {output_path}\n"
            f"[Done] OR alarms: {alarm_rows}/{total_rows} ({alarm_ratio:.2f}%)\n"
            f"[Done] SoftVote alarms: {softvote_alarm_rows}/{total_rows} ({softvote_alarm_ratio:.2f}%)",
            flush=True,
        )
        if or_metrics:
            print(
                "[Done] OR metrics "
                f"(n={valid_eval_rows}): acc={or_metrics['or_acc']:.4f}, "
                f"fpr={or_metrics['or_fpr']:.4f}, auroc={or_metrics['or_auroc']:.4f}, "
                f"recall={or_metrics['or_recall']:.4f}",
                flush=True,
            )
        if softvote_metrics:
            print(
                "[Done] SoftVote metrics "
                f"(n={valid_eval_rows}): acc={softvote_metrics['softvote_acc']:.4f}, "
                f"fpr={softvote_metrics['softvote_fpr']:.4f}, auroc={softvote_metrics['softvote_auroc']:.4f}, "
                f"recall={softvote_metrics['softvote_recall']:.4f}",
                flush=True,
            )
        if valid_eval_rows > 0:
            if (not available_sensor_count_exists) and (not overlap_column_exists):
                print(
                    "[Warn] columns 'available_sensor_count_single4' and 'overlap_mode' not found; "
                    "treating all valid rows as OVERLAP for split panel.",
                    flush=True,
                )
            if softvote_split_metrics:
                _print_eval_split_panel("SOFTVOTE", softvote_split_metrics)
            if or_split_metrics:
                _print_eval_split_panel("LOGICAL_OR", or_split_metrics)
            for sensor_name in SENSOR_ORDER:
                panel = sensor_sample_split_metrics.get(sensor_name, None)
                if panel is None:
                    continue
                for split_name in ("ALL", "OVERLAP", "SINGLE"):
                    m = panel[split_name]
                    print(
                        f"[EvalSplit][SINGLE_SENSOR][{split_name}][SENSOR_{sensor_name.upper()}] count={int(m['count'])} "
                        f"acc={m['acc']:.4f} fpr={m['fpr']:.4f} recall={m['recall']:.4f} auroc={m['auroc']:.4f}",
                        flush=True,
                    )

        if wandb_run is not None and wandb_mod is not None:
            metrics = {
                "rows_total": total_rows,
                "rows_alarm_or_single4": alarm_rows,
                "alarm_or_single4_ratio": float(alarm_ratio / 100.0),
                "rows_alarm_softvote_single4": softvote_alarm_rows,
                "alarm_softvote_single4_ratio": float(softvote_alarm_ratio / 100.0),
                "rows_with_valid_label_for_or_metrics": int(valid_eval_rows),
            }
            metrics.update(or_metrics)
            metrics.update(softvote_metrics)
            for strategy_key, panel in (("softvote", softvote_split_metrics), ("logical_or", or_split_metrics)):
                for split_name in ("ALL", "OVERLAP", "SINGLE"):
                    split = panel.get(split_name)
                    if split is None:
                        continue
                    split_key = split_name.lower()
                    metrics[f"{strategy_key}_{split_key}_count"] = int(split["count"])
                    metrics[f"{strategy_key}_{split_key}_acc"] = float(split["acc"])
                    metrics[f"{strategy_key}_{split_key}_fpr"] = float(split["fpr"])
                    metrics[f"{strategy_key}_{split_key}_recall"] = float(split["recall"])
                    metrics[f"{strategy_key}_{split_key}_auroc"] = float(split["auroc"])
            for sensor_name, panel in sensor_sample_split_metrics.items():
                sensor_key = sensor_name.lower()
                for split_name in ("ALL", "OVERLAP", "SINGLE"):
                    split = panel.get(split_name)
                    if split is None:
                        continue
                    split_key = split_name.lower()
                    metrics[f"single_sensor_{split_key}_{sensor_key}_count"] = int(split["count"])
                    metrics[f"single_sensor_{split_key}_{sensor_key}_acc"] = float(split["acc"])
                    metrics[f"single_sensor_{split_key}_{sensor_key}_fpr"] = float(split["fpr"])
                    metrics[f"single_sensor_{split_key}_{sensor_key}_recall"] = float(split["recall"])
                    metrics[f"single_sensor_{split_key}_{sensor_key}_auroc"] = float(split["auroc"])
            for sensor_name in SENSOR_ORDER:
                col = f"pred_label_{sensor_name}_single"
                vals = []
                for v in out_df[col].tolist():
                    parsed = _try_parse_binary_label(v)
                    if parsed is None:
                        continue
                    vals.append(int(parsed))
                metrics[f"rows_with_{sensor_name}"] = int(len(vals))
                metrics[f"alarm_rate_{sensor_name}"] = float(sum(vals) / max(1, len(vals))) if vals else 0.0
            wandb_run.log(metrics)
            for k, v in metrics.items():
                wandb_run.summary[k] = v
            wandb_run.summary["output_csv"] = str(output_path)

            artifact_name_raw = (
                str(args.wandb_artifact_name).strip()
                if str(args.wandb_artifact_name).strip()
                else f"single4_or_infer_{input_path.stem}"
            )
            artifact_name = _safe_artifact_name(artifact_name_raw)
            artifact = wandb_mod.Artifact(name=artifact_name, type="inference")
            artifact.add_file(str(output_path), name=output_path.name)
            wandb_run.log_artifact(artifact)

            preview_rows = int(args.wandb_log_table_rows)
            if preview_rows > 0:
                table_df = out_df.head(preview_rows).copy()
                table_df = table_df.fillna("").astype(str)
                try:
                    wandb_run.log({"inference_preview": wandb_mod.Table(dataframe=table_df)})
                except Exception as exc:
                    print(f"[Warn] W&B table logging skipped due to type mismatch: {exc}", flush=True)
            print(f"[W&B] Logged metrics + artifact '{artifact_name}'", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
