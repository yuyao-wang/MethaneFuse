#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import baselines.per_vit.s5p_temporal as s5p_mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run S5P inference from an existing checkpoint, tag correct rows, and analyze error factors."
    )
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint saved by dino_classifier_head_s5p_temporal_one_block.py")
    p.add_argument("--test_csv", required=True, help="Full test CSV (may include multi-sensor rows).")
    p.add_argument("--out_csv", required=True, help="Output CSV with prediction columns appended.")
    p.add_argument("--out_report_json", required=True, help="Output analysis JSON report.")
    p.add_argument("--train_csv_override", default=None, help="Optional override for train CSV used to compute normalization stats.")
    p.add_argument("--device", default="cuda", help="cuda or cpu.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def parse_float_list(text: Optional[str]) -> Optional[List[float]]:
    if text is None:
        return None
    txt = str(text).strip()
    if txt == "":
        return None
    return [float(x.strip()) for x in txt.split(",") if x.strip() != ""]


def load_norm_stats_from_ckpt_args(args_dict: Dict) -> Optional[Tuple[Sequence[float], Sequence[float]]]:
    stats_npz = args_dict.get("normalize_stats_npz")
    if stats_npz:
        with np.load(stats_npz) as npz:
            if "mean" not in npz or "std" not in npz:
                raise KeyError(f"{stats_npz} must contain keys 'mean' and 'std'.")
            return npz["mean"].tolist(), npz["std"].tolist()

    mean_list = parse_float_list(args_dict.get("normalize_mean"))
    std_list = parse_float_list(args_dict.get("normalize_std"))
    if mean_list is not None and std_list is not None:
        return mean_list, std_list
    # Match training script behavior: default to built-in PRECOMPUTED_STATS.
    return s5p_mod.PRECOMPUTED_STATS


def valid_path_mask(s: pd.Series) -> pd.Series:
    txt = s.astype("string").str.strip().str.lower()
    return (~s.isna()) & (txt != "") & (txt != "nan") & (txt != "none") & (txt != "null")


def choose_sensor_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("sensor", "anchor_sensor"):
        if c in df.columns:
            return c
    return None


def build_s5p_subset_with_orig_idx(full_df: pd.DataFrame, path_col: str = "s5p_0_path") -> pd.DataFrame:
    df = full_df.copy()
    sensor_col = choose_sensor_col(df)
    if sensor_col is not None:
        sensor_vals = df[sensor_col].astype("string").str.strip().str.lower()
        df = df[sensor_vals == "s5p"].copy()
    if path_col not in df.columns:
        raise KeyError(f"Missing required column in test CSV: {path_col}")
    df = df[valid_path_mask(df[path_col])].copy()
    df["__orig_idx"] = df.index.astype(int)
    return df.reset_index(drop=True)


def prepare_datasets(
    ckpt_args: Dict,
    train_csv: str,
    test_s5p_csv: str,
    device: torch.device,
):
    requested_cols = (
        ckpt_args.get("t0_col", "path_t0"),
        ckpt_args.get("t90_col", "path_t90"),
        ckpt_args.get("t360_col", "path_t360"),
    )
    train_path_cols = s5p_mod.resolve_s5p_path_columns(train_csv, requested_cols)
    test_path_cols = s5p_mod.resolve_s5p_path_columns(test_s5p_csv, requested_cols)

    norm_stats = load_norm_stats_from_ckpt_args(ckpt_args)
    compute_stats = bool(ckpt_args.get("compute_stats", True)) and norm_stats is None
    compute_stats_subset = ckpt_args.get("compute_stats_subset", None)
    compute_stats_subset = s5p_mod.parse_subset_value(compute_stats_subset)
    chn_ids = s5p_mod.parse_comma_separated_floats(ckpt_args.get("chn_ids"))

    common_kwargs = dict(
        pad_to_multiple=ckpt_args.get("pad_to_multiple", None),
        pad_value=float(ckpt_args.get("pad_value", 0.0)),
        default_chn_id_value=float(ckpt_args.get("default_chn_id_value", 0.0)),
        local_file_cache=None,
        nan_to_num=float(ckpt_args.get("nan_to_num", 0.0)),
        data_key=ckpt_args.get("data_key", "ch4"),
        chn_ids_key=ckpt_args.get("chn_ids_key", "chn_ids"),
        channel_last=bool(ckpt_args.get("channel_last", False)),
        allow_pickle=bool(ckpt_args.get("allow_pickle", False)),
        scale_to_unit=bool(ckpt_args.get("scale_to_unit", False)),
        scale_value=float(ckpt_args.get("scale_value", 65535.0)),
    )

    # Build first without compute_stats, because this wide-table CSV may include empty temporal columns.
    # We must filter to valid S5P rows before computing stats.
    base_train = s5p_mod.S5pTemporalTiffDataset(
        csv_path=train_csv,
        path_columns=train_path_cols,
        normalize_stats=norm_stats,
        chn_ids=chn_ids,
        compute_stats=False,
        compute_stats_subset=compute_stats_subset,
        **common_kwargs,
    )
    train_sensor_col = "anchor_sensor" if "anchor_sensor" in base_train.df.columns else "sensor"
    s5p_mod.filter_dataset_to_s5p_only(base_train, split_name="train", sensor_column=train_sensor_col)
    print(
        f"[Stage] train S5P rows after filtering: {len(base_train.df)} (sensor_column={train_sensor_col})",
        flush=True,
    )
    if compute_stats:
        print("[Stage] computing normalization stats on filtered train S5P rows (CPU stage; GPU may be idle)...", flush=True)
        mean_t, std_t = base_train._compute_dataset_stats(subset=compute_stats_subset)  # noqa: SLF001
        mean_l = mean_t.detach().cpu().tolist()
        std_l = torch.clamp(std_t, min=1e-6).detach().cpu().tolist()
        base_train._mean = torch.as_tensor(mean_l, dtype=torch.float32).view(3, 1, 1)  # noqa: SLF001
        base_train._std = torch.as_tensor(std_l, dtype=torch.float32).view(3, 1, 1)  # noqa: SLF001
        base_train._stats_source = "computed"  # noqa: SLF001
        resolved_stats = (mean_l, std_l)
    else:
        resolved_stats = base_train.get_normalize_stats()
        if resolved_stats is None:
            resolved_stats = norm_stats

    base_test = s5p_mod.S5pTemporalTiffDataset(
        csv_path=test_s5p_csv,
        path_columns=test_path_cols,
        normalize_stats=resolved_stats,
        chn_ids=chn_ids,
        compute_stats=False,
        **common_kwargs,
    )
    test_sensor_col = "anchor_sensor" if "anchor_sensor" in base_test.df.columns else "sensor"
    s5p_mod.filter_dataset_to_s5p_only(base_test, split_name="test", sensor_column=test_sensor_col)
    print(
        f"[Stage] test S5P rows after filtering: {len(base_test.df)} (sensor_column={test_sensor_col})",
        flush=True,
    )

    resize_size = int(ckpt_args.get("resize_size", 224))
    test_ds = s5p_mod.S5pSimpleNpzDataset(base_test, resize_to=resize_size)
    return base_test, test_ds


def run_inference(
    ckpt: Dict,
    ckpt_args: Dict,
    test_ds,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    backbone = s5p_mod.load_backbone(ckpt_args.get("weights", "weights/panopticon_vitb14_teacher.pth"), device=device, debug=False)
    head = s5p_mod.CLSHead(embed_dim=int(ckpt_args.get("embed_dim", 768)), num_classes=2)
    backbone.load_state_dict(ckpt["backbone_state_dict"], strict=True)
    head.load_state_dict(ckpt["head_state_dict"], strict=True)
    backbone = backbone.to(device).eval()
    head = head.to(device).eval()

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    preds: List[int] = []
    probs1: List[float] = []
    labels_all: List[int] = []

    print("[Stage] start model inference (GPU stage).", flush=True)
    with torch.no_grad():
        for x_dict, labels in loader:
            x_dict = s5p_mod.recursive_to_device(x_dict, device)
            labels = labels.to(device)
            imgs = x_dict.get("imgs")
            if isinstance(imgs, torch.Tensor) and not torch.isfinite(imgs).all():
                x_dict["imgs"] = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)
            feats = backbone(x_dict, is_training=True)
            cls_token = feats["x_norm_clstoken"]
            logits = head(cls_token)
            prob1 = F.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)

            preds.extend(pred.detach().cpu().tolist())
            probs1.extend(prob1.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())

    print("[Stage] inference finished.", flush=True)
    return np.array(preds, dtype=int), np.array(probs1, dtype=float), np.array(labels_all, dtype=int)


def region_name(lat: float, lon: float) -> str:
    if -170 <= lon <= -50 and 5 <= lat <= 83:
        return "NorthAmerica"
    if -90 <= lon <= -30 and -60 <= lat < 15:
        return "SouthAmerica"
    if -30 <= lon <= 60 and 35 <= lat <= 72:
        return "Europe"
    if -20 <= lon <= 55 and -35 <= lat < 35:
        return "Africa"
    if 55 < lon <= 180 and 5 <= lat <= 80:
        return "Asia"
    if 110 <= lon <= 180 and -50 <= lat < 5:
        return "Oceania"
    return "Other"


def load_arr_for_metrics(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if "ch4" in z.files:
            a = np.asarray(z["ch4"])
        else:
            a = np.asarray(z[next(iter(z.files))])
    if a.ndim == 2:
        a = a[None, ...]
    elif a.ndim == 3 and a.shape[0] > 16 and a.shape[-1] <= 16:
        a = np.transpose(a, (2, 0, 1))
    if a.shape[0] >= 3:
        a = a[:3]
    else:
        rep = int(math.ceil(3 / max(1, a.shape[0])))
        a = np.repeat(a, rep, axis=0)[:3]
    return a.astype(np.float32, copy=False)


def nanmean_safe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else 0.0


def compute_row_metrics(path: str) -> Tuple[float, float]:
    a = load_arr_for_metrics(path)
    a0 = a[0]
    tex_std = float(np.nanstd(np.where(np.isfinite(a0), a0, np.nan)))
    d01 = nanmean_safe(np.abs(a[0] - a[1]))
    d12 = nanmean_safe(np.abs(a[1] - a[2]))
    d02 = nanmean_safe(np.abs(a[0] - a[2]))
    temp_absdiff_sum = d01 + d12 + d02
    return tex_std, temp_absdiff_sum


def analyze(eval_df: pd.DataFrame) -> Dict:
    eval_df = eval_df.copy()
    eval_df["pred_correct_s5p"] = pd.to_numeric(eval_df["pred_correct_s5p"], errors="coerce").fillna(0).astype(int)
    eval_df["label"] = pd.to_numeric(eval_df["label"], errors="coerce").fillna(0).astype(int)
    eval_df["pred_label_s5p"] = pd.to_numeric(eval_df["pred_label_s5p"], errors="coerce").fillna(0).astype(int)

    report: Dict = {}
    report["n_s5p_eval_rows"] = int(len(eval_df))
    report["accuracy"] = float((eval_df["pred_correct_s5p"] == 1).mean()) if len(eval_df) else 0.0

    pos = eval_df[eval_df["label"] == 1]
    neg = eval_df[eval_df["label"] == 0]
    report["pos_acc"] = float((pos["pred_correct_s5p"] == 1).mean()) if len(pos) else None
    report["neg_acc"] = float((neg["pred_correct_s5p"] == 1).mean()) if len(neg) else None

    tp = int(((eval_df["label"] == 1) & (eval_df["pred_label_s5p"] == 1)).sum())
    fp = int(((eval_df["label"] == 0) & (eval_df["pred_label_s5p"] == 1)).sum())
    fn = int(((eval_df["label"] == 1) & (eval_df["pred_label_s5p"] == 0)).sum())
    tn = int(((eval_df["label"] == 0) & (eval_df["pred_label_s5p"] == 0)).sum())
    report["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    if "datetime" in eval_df.columns:
        years = pd.to_datetime(eval_df["datetime"], utc=True, errors="coerce").dt.year
        eval_df["year"] = years
        year_tbl = (
            eval_df.groupby("year", dropna=True)
            .agg(n=("pred_correct_s5p", "size"), acc=("pred_correct_s5p", "mean"))
            .reset_index()
        )
        report["by_year"] = year_tbl.to_dict(orient="records")

    if "latitude" in eval_df.columns and "longitude" in eval_df.columns:
        eval_df["region"] = [
            region_name(float(la), float(lo))
            for la, lo in zip(eval_df["latitude"], eval_df["longitude"])
        ]
        reg_tbl = (
            eval_df.groupby("region")
            .agg(n=("pred_correct_s5p", "size"), acc=("pred_correct_s5p", "mean"))
            .sort_values("n", ascending=False)
            .reset_index()
        )
        report["by_region"] = reg_tbl.to_dict(orient="records")

    tex_vals = []
    temp_vals = []
    for p in eval_df["s5p_0_path"].astype(str).tolist():
        try:
            tex, tmp = compute_row_metrics(p)
        except Exception:
            tex, tmp = float("nan"), float("nan")
        tex_vals.append(tex)
        temp_vals.append(tmp)
    eval_df["tex_std"] = tex_vals
    eval_df["temp_absdiff_sum"] = temp_vals

    metric_summary = {}
    for metric in ("tex_std", "temp_absdiff_sum"):
        metric_summary[metric] = {
            "correct_mean": float(eval_df.loc[eval_df["pred_correct_s5p"] == 1, metric].mean()),
            "wrong_mean": float(eval_df.loc[eval_df["pred_correct_s5p"] == 0, metric].mean()),
            "pos_correct_mean": float(eval_df.loc[(eval_df["label"] == 1) & (eval_df["pred_correct_s5p"] == 1), metric].mean()),
            "pos_wrong_mean": float(eval_df.loc[(eval_df["label"] == 1) & (eval_df["pred_correct_s5p"] == 0), metric].mean()),
            "neg_correct_mean": float(eval_df.loc[(eval_df["label"] == 0) & (eval_df["pred_correct_s5p"] == 1), metric].mean()),
            "neg_wrong_mean": float(eval_df.loc[(eval_df["label"] == 0) & (eval_df["pred_correct_s5p"] == 0), metric].mean()),
        }
    report["feature_gap"] = metric_summary

    if "plume_id" in eval_df.columns:
        plume_tbl = (
            eval_df.groupby(eval_df["plume_id"].astype(str))
            .agg(n=("pred_correct_s5p", "size"), acc=("pred_correct_s5p", "mean"))
            .reset_index()
            .rename(columns={"plume_id": "plume_id"})
        )
        hardest = plume_tbl.sort_values(["acc", "n"], ascending=[True, False]).head(20)
        report["hardest_plumes_top20"] = hardest.to_dict(orient="records")

    return report


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unexpected checkpoint object type: {type(ckpt)}")
    ckpt_args = dict(ckpt.get("args", {}))

    if "backbone_state_dict" not in ckpt or "head_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain backbone_state_dict and head_state_dict.")

    if args.train_csv_override:
        train_csv = args.train_csv_override
    else:
        train_csv = ckpt_args.get("train_csv")
    if not train_csv:
        raise ValueError("train_csv not found in checkpoint args; pass --train_csv_override.")

    device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    full_test_df = pd.read_csv(args.test_csv, low_memory=False)
    s5p_test_df = build_s5p_subset_with_orig_idx(full_test_df, path_col="s5p_0_path")
    if len(s5p_test_df) == 0:
        raise ValueError("No valid S5P rows found in test CSV.")

    with tempfile.TemporaryDirectory(prefix="s5p_infer_") as td:
        tmp_test_csv = str(Path(td) / "test_s5p_only.csv")
        s5p_test_df.to_csv(tmp_test_csv, index=False)

        base_test, test_ds = prepare_datasets(
            ckpt_args=ckpt_args,
            train_csv=train_csv,
            test_s5p_csv=tmp_test_csv,
            device=device,
        )

        preds, probs1, labels = run_inference(
            ckpt=ckpt,
            ckpt_args=ckpt_args,
            test_ds=test_ds,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        test_meta = base_test.df.reset_index(drop=True)
        if "__orig_idx" not in test_meta.columns:
            raise KeyError("Internal error: __orig_idx missing after test filtering.")
        if len(test_meta) != len(preds):
            raise RuntimeError(f"Prediction length mismatch: meta={len(test_meta)} preds={len(preds)}")

        pred_correct = (preds == labels).astype(int)

        out_df = full_test_df.copy()
        out_df["pred_label_s5p"] = np.nan
        out_df["pred_prob1_s5p"] = np.nan
        out_df["pred_correct_s5p"] = np.nan

        for i in range(len(test_meta)):
            orig_idx = int(test_meta.iloc[i]["__orig_idx"])
            out_df.at[orig_idx, "pred_label_s5p"] = int(preds[i])
            out_df.at[orig_idx, "pred_prob1_s5p"] = float(probs1[i])
            out_df.at[orig_idx, "pred_correct_s5p"] = int(pred_correct[i])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    eval_df = out_df[out_df["pred_correct_s5p"].notna()].copy()
    report = analyze(eval_df)
    report["checkpoint"] = str(ckpt_path)
    report["train_csv"] = train_csv
    report["test_csv"] = args.test_csv
    report["out_csv"] = args.out_csv
    report["best_metric_name"] = ckpt.get("best_metric_name")
    report["best_metric_value"] = ckpt.get("best_metric_value")

    Path(args.out_report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
