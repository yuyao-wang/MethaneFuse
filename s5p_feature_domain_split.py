#!/usr/bin/env python3
"""
Split S5P test samples into ID/OOD using feature-space similarity, not geo distance.

Core idea:
1) Build feature vectors from S5P NPZ patches (spectral stats + temporal deltas + texture + seasonality).
2) Fit train feature distribution (kNN distance in standardized feature space).
3) Mark test sample as OOD if its kNN score is above train-quantile threshold.

Outputs:
- test_s5p_id.csv
- test_s5p_ood.csv
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SENSOR_CANDIDATES = ("sensor", "anchor_sensor")


@dataclass
class DomainResult:
    threshold: float
    train_scores: np.ndarray
    test_scores: np.ndarray
    test_is_id: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature-domain split for S5P test set.")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--out_id_csv", required=True, help="Output CSV for S5P in-domain test rows.")
    p.add_argument("--out_ood_csv", required=True, help="Output CSV for S5P out-of-domain test rows.")
    p.add_argument("--out_report_json", default=None)
    p.add_argument("--path_col", default="s5p_0_path")
    p.add_argument("--id_col", default="plume_id")
    p.add_argument("--datetime_col", default="datetime")
    p.add_argument(
        "--sensor_col",
        default="auto",
        help="Sensor column name. 'auto' tries common candidates and keeps only s5p rows if found.",
    )
    p.add_argument("--sensor_name", default="s5p")
    p.add_argument("--knn_k", type=int, default=5)
    p.add_argument(
        "--train_quantile",
        type=float,
        default=0.95,
        help="Threshold quantile of train self-kNN score. Higher => fewer test rows marked OOD.",
    )
    p.add_argument("--allow_pickle", action="store_true")
    p.add_argument("--verbose_every", type=int, default=300)
    return p.parse_args()


def _valid_path_mask(s: pd.Series) -> pd.Series:
    text = s.astype("string").str.strip().str.lower()
    return (~s.isna()) & (text != "") & (text != "nan") & (text != "none") & (text != "null")


def _resolve_sensor_col(df: pd.DataFrame, sensor_col_arg: str) -> Optional[str]:
    if sensor_col_arg != "auto":
        return sensor_col_arg if sensor_col_arg in df.columns else None
    for cand in SENSOR_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


def _select_s5p_rows(
    df: pd.DataFrame, *, path_col: str, sensor_col: Optional[str], sensor_name: str
) -> pd.DataFrame:
    out = df
    if sensor_col is not None and sensor_col in out.columns:
        sensor_vals = out[sensor_col].astype("string").str.strip().str.lower()
        out = out[sensor_vals == sensor_name.lower()]
    if path_col not in out.columns:
        raise ValueError(f"Missing path column: {path_col}")
    out = out[_valid_path_mask(out[path_col])].copy()
    return out


def _infer_npz_array(np_obj: np.lib.npyio.NpzFile) -> np.ndarray:
    preferred_keys = ("ch4", "image", "imgs", "arr_0", "data")
    for key in preferred_keys:
        if key in np_obj:
            arr = np.asarray(np_obj[key])
            if arr.ndim in (2, 3):
                return arr
    for key in np_obj.files:
        key_l = key.lower()
        if key_l in {"meta", "metadata", "chn_ids"}:
            continue
        arr = np.asarray(np_obj[key])
        if arr.ndim in (2, 3) and arr.dtype.kind in {"f", "i", "u", "b"}:
            return arr
    raise ValueError(f"Cannot infer image-like array from keys={list(np_obj.files)}")


def _to_chw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    c_first, c_last = arr.shape[0], arr.shape[-1]
    if c_first <= 16:
        return arr
    if c_last <= 16:
        return np.transpose(arr, (2, 0, 1))
    return arr


def _robust_stats(x: np.ndarray) -> List[float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [0.0] * 6
    q10, q50, q90 = np.percentile(x, [10, 50, 90])
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(q10),
        float(q50),
        float(q90),
        float(q90 - q10),
    ]


def _seasonal_features(dt: str) -> Tuple[float, float]:
    try:
        t = datetime.strptime(str(dt), "%Y-%m-%dT%H:%M:%SZ")
        doy = t.timetuple().tm_yday
        return float(math.sin(2 * math.pi * doy / 365.25)), float(math.cos(2 * math.pi * doy / 365.25))
    except Exception:
        return 0.0, 0.0


def extract_feature_vector(npz_path: str, dt: str, allow_pickle: bool) -> np.ndarray:
    with np.load(npz_path, allow_pickle=allow_pickle) as np_obj:
        arr = _infer_npz_array(np_obj)

    arr = _to_chw(arr).astype(np.float32, copy=False)
    if arr.shape[0] >= 3:
        arr = arr[:3]
    else:
        rep = int(math.ceil(3 / max(1, arr.shape[0])))
        arr = np.repeat(arr, repeats=rep, axis=0)[:3]

    feats: List[float] = []
    for c in range(3):
        feats.extend(_robust_stats(arr[c].reshape(-1)))

    d01 = (arr[0] - arr[1]).reshape(-1)
    d12 = (arr[1] - arr[2]).reshape(-1)
    d02 = (arr[0] - arr[2]).reshape(-1)
    feats.extend(_robust_stats(d01)[:4])
    feats.extend(_robust_stats(d12)[:4])
    feats.extend(_robust_stats(d02)[:4])

    # Basic texture proxy on latest frame.
    a0 = arr[0]
    gx = np.diff(a0, axis=1)[:-1, :]
    gy = np.diff(a0, axis=0)[:, :-1]
    grad_mag = np.sqrt(gx * gx + gy * gy).reshape(-1)
    feats.extend(_robust_stats(grad_mag)[:4])

    sin_doy, cos_doy = _seasonal_features(dt)
    feats.extend([sin_doy, cos_doy])

    return np.asarray(feats, dtype=np.float32)


def dedupe_units(df: pd.DataFrame, id_col: str, path_col: str) -> pd.DataFrame:
    if id_col in df.columns:
        return df.drop_duplicates(subset=[id_col], keep="first").copy()
    return df.drop_duplicates(subset=[path_col], keep="first").copy()


def build_matrix(
    df_units: pd.DataFrame,
    *,
    path_col: str,
    datetime_col: str,
    allow_pickle: bool,
    verbose_every: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    feats: List[Optional[np.ndarray]] = []
    for i, (_, row) in enumerate(df_units.iterrows(), start=1):
        path = str(row[path_col]).strip()
        dt = str(row[datetime_col]) if datetime_col in df_units.columns else ""
        try:
            feats.append(extract_feature_vector(path, dt, allow_pickle=allow_pickle))
        except Exception:
            feats.append(None)
        if verbose_every > 0 and i % verbose_every == 0:
            print(f"[Feature] processed {i}/{len(df_units)}", flush=True)

    ok_mask = np.array([x is not None for x in feats], dtype=bool)
    df_ok = df_units.loc[ok_mask].reset_index(drop=True)
    x_ok = np.stack([x for x in feats if x is not None], axis=0)
    return df_ok, x_ok


def compute_domain_split(
    x_train: np.ndarray, x_test: np.ndarray, *, knn_k: int, train_quantile: float
) -> DomainResult:
    if x_train.shape[0] < 2:
        raise ValueError("Need at least 2 train samples for kNN domain scoring.")

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    k = max(1, int(knn_k))
    self_k = min(k + 1, x_train_s.shape[0])
    nn_self = NearestNeighbors(n_neighbors=self_k, metric="euclidean")
    nn_self.fit(x_train_s)
    d_self, _ = nn_self.kneighbors(x_train_s)
    train_scores = d_self[:, 1:].mean(axis=1) if d_self.shape[1] > 1 else d_self[:, 0]

    threshold = float(np.quantile(train_scores, train_quantile))

    cross_k = min(k, x_train_s.shape[0])
    nn_cross = NearestNeighbors(n_neighbors=cross_k, metric="euclidean")
    nn_cross.fit(x_train_s)
    d_test, _ = nn_cross.kneighbors(x_test_s)
    test_scores = d_test.mean(axis=1)
    test_is_id = test_scores <= threshold

    return DomainResult(
        threshold=threshold,
        train_scores=train_scores,
        test_scores=test_scores,
        test_is_id=test_is_id,
    )


def _safe_pos_rate(df: pd.DataFrame) -> Optional[float]:
    if "label" not in df.columns or len(df) == 0:
        return None
    return float(pd.to_numeric(df["label"], errors="coerce").fillna(0).mean())


def main() -> None:
    args = parse_args()

    train_df_all = pd.read_csv(args.train_csv)
    test_df_all = pd.read_csv(args.test_csv)

    sensor_col = _resolve_sensor_col(train_df_all, args.sensor_col)
    if sensor_col is None:
        sensor_col = _resolve_sensor_col(test_df_all, args.sensor_col)

    train_s5p = _select_s5p_rows(
        train_df_all, path_col=args.path_col, sensor_col=sensor_col, sensor_name=args.sensor_name
    )
    test_s5p = _select_s5p_rows(
        test_df_all, path_col=args.path_col, sensor_col=sensor_col, sensor_name=args.sensor_name
    )

    train_units = dedupe_units(train_s5p, id_col=args.id_col, path_col=args.path_col)
    test_units = dedupe_units(test_s5p, id_col=args.id_col, path_col=args.path_col)

    print(f"[Data] train_s5p_rows={len(train_s5p)} test_s5p_rows={len(test_s5p)}", flush=True)
    print(f"[Data] train_units={len(train_units)} test_units={len(test_units)}", flush=True)

    train_units_ok, x_train = build_matrix(
        train_units,
        path_col=args.path_col,
        datetime_col=args.datetime_col,
        allow_pickle=args.allow_pickle,
        verbose_every=args.verbose_every,
    )
    test_units_ok, x_test = build_matrix(
        test_units,
        path_col=args.path_col,
        datetime_col=args.datetime_col,
        allow_pickle=args.allow_pickle,
        verbose_every=args.verbose_every,
    )
    print(f"[Feature] x_train={x_train.shape} x_test={x_test.shape}", flush=True)

    domain = compute_domain_split(
        x_train, x_test, knn_k=args.knn_k, train_quantile=args.train_quantile
    )

    if args.id_col in test_units_ok.columns:
        id_keys = set(test_units_ok.loc[domain.test_is_id, args.id_col].astype(str))
        ood_keys = set(test_units_ok.loc[~domain.test_is_id, args.id_col].astype(str))
        row_keys = test_s5p[args.id_col].astype(str)
    else:
        id_keys = set(test_units_ok.loc[domain.test_is_id, args.path_col].astype(str))
        ood_keys = set(test_units_ok.loc[~domain.test_is_id, args.path_col].astype(str))
        row_keys = test_s5p[args.path_col].astype(str)

    id_mask = row_keys.isin(id_keys)
    ood_mask = row_keys.isin(ood_keys)

    out_id = test_s5p[id_mask].copy()
    out_ood = test_s5p[ood_mask].copy()

    Path(args.out_id_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_ood_csv).parent.mkdir(parents=True, exist_ok=True)
    out_id.to_csv(args.out_id_csv, index=False)
    out_ood.to_csv(args.out_ood_csv, index=False)

    report = {
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "sensor_col": sensor_col,
        "path_col": args.path_col,
        "id_col": args.id_col,
        "knn_k": int(args.knn_k),
        "train_quantile": float(args.train_quantile),
        "threshold": float(domain.threshold),
        "train_s5p_rows": int(len(train_s5p)),
        "test_s5p_rows": int(len(test_s5p)),
        "train_units_used": int(x_train.shape[0]),
        "test_units_used": int(x_test.shape[0]),
        "test_id_rows": int(len(out_id)),
        "test_ood_rows": int(len(out_ood)),
        "test_id_ratio": float(len(out_id) / max(1, len(test_s5p))),
        "test_ood_ratio": float(len(out_ood) / max(1, len(test_s5p))),
        "test_s5p_pos_rate": _safe_pos_rate(test_s5p),
        "test_id_pos_rate": _safe_pos_rate(out_id),
        "test_ood_pos_rate": _safe_pos_rate(out_ood),
    }

    print("[Result] threshold={:.6f}".format(report["threshold"]), flush=True)
    print(
        "[Result] id_rows={} ({:.2%}), ood_rows={} ({:.2%})".format(
            report["test_id_rows"],
            report["test_id_ratio"],
            report["test_ood_rows"],
            report["test_ood_ratio"],
        ),
        flush=True,
    )
    print(
        "[Result] pos_rate: full={} id={} ood={}".format(
            report["test_s5p_pos_rate"],
            report["test_id_pos_rate"],
            report["test_ood_pos_rate"],
        ),
        flush=True,
    )

    if args.out_report_json:
        Path(args.out_report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Result] report -> {args.out_report_json}", flush=True)

    print(f"[Result] ID CSV  -> {args.out_id_csv}", flush=True)
    print(f"[Result] OOD CSV -> {args.out_ood_csv}", flush=True)


if __name__ == "__main__":
    main()
