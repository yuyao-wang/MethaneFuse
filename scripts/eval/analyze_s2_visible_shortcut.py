#!/usr/bin/env python3
"""Quantify simple visible/SWIR texture shortcuts in legacy-360 S2 crops."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FRAME_COLUMNS = (("t0", "s2_0_path"), ("t90", "s2_90_path"), ("t360", "s2_360_path"))
EXPECTED_SHAPE = (12, 224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-reference-json", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--model-predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def read_chw(path: str) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(f"{path}: expected {EXPECTED_SHAPE}, got {array.shape}")
    return array.astype(np.float32, copy=False)


def group_features(array: np.ndarray, indices: tuple[int, ...], prefix: str) -> dict[str, float]:
    values = array[list(indices)].astype(np.float64)
    std = float(values.std())
    dx = np.abs(np.diff(values, axis=2)).mean()
    dy = np.abs(np.diff(values, axis=1)).mean()
    gradient = float(0.5 * (dx + dy))
    channels, height, width = values.shape
    patch = 14
    patches = (
        values.reshape(channels, height // patch, patch, width // patch, patch)
        .transpose(0, 1, 3, 2, 4)
    )
    patch_means = patches.mean(axis=(3, 4))
    within_patch_std = patches.std(axis=(3, 4))
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": std,
        f"{prefix}_gradient": gradient,
        f"{prefix}_gradient_over_std": float(gradient / max(std, 1e-8)),
        f"{prefix}_patch_mean_std": float(patch_means.std()),
        f"{prefix}_within_patch_std": float(within_patch_std.mean()),
    }


def row_features(record: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {
        "id": str(record["id"]),
        "label": int(record["label"]),
        "source_group": str(record["source_group"]),
    }
    frame_summary: dict[str, dict[str, float]] = {}
    for frame, column in FRAME_COLUMNS:
        array = read_chw(str(record[column]))
        output.update(group_features(array, (0, 1, 2, 3), f"{frame}_visible"))
        output.update(group_features(array, (4, 5, 6, 7, 8, 9), f"{frame}_rededge_nir"))
        output.update(group_features(array, (10, 11), f"{frame}_swir"))
        b11 = array[10].astype(np.float64)
        b12 = array[11].astype(np.float64)
        ratio = (b11 - b12) / np.maximum(b11 + b12, 1.0)
        output[f"{frame}_b11_b12_ratio_mean"] = float(ratio.mean())
        output[f"{frame}_b11_b12_ratio_std"] = float(ratio.std())
        frame_summary[frame] = {
            "visible_mean": float(array[:4].mean()),
            "visible_std": float(array[:4].std()),
            "visible_gradient": float(
                0.5
                * (
                    np.abs(np.diff(array[:4], axis=2)).mean()
                    + np.abs(np.diff(array[:4], axis=1)).mean()
                )
            ),
            "swir_mean": float(array[10:12].mean()),
            "swir_std": float(array[10:12].std()),
        }
    for key in ("visible_mean", "visible_std", "visible_gradient", "swir_mean", "swir_std"):
        values = np.asarray([frame_summary[frame][key] for frame, _ in FRAME_COLUMNS])
        output[f"pooled_{key}_mean"] = float(values.mean())
        output[f"pooled_{key}_max"] = float(values.max())
        output[f"pooled_{key}_range"] = float(values.max() - values.min())
        output[f"t0_{key}_minus_history"] = float(values[0] - values[1:].mean())
    return output


def cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled <= 1e-12:
        return None
    return float((a.mean() - b.mean()) / pooled)


def main() -> None:
    args = parse_args()
    reference = json.loads(args.training_reference_json.read_text(encoding="utf-8"))
    train = pd.read_csv(reference["training_manifest"], low_memory=False)
    train["id"] = train["id"].astype(str)
    by_id = train.drop_duplicates("id").set_index("id")
    selected_parts = []
    for label in (0, 1):
        ids = [str(value) for value in reference["sampled_ids"][str(label)]]
        part = by_id.loc[ids].reset_index()
        part["source_group"] = f"train_label_{label}_reference"
        selected_parts.append(part)
    test = pd.read_csv(args.test_manifest, low_memory=False)
    test["id"] = test.get("parent_id", test["id"]).astype(str).str.replace("__scale_1p00", "", regex=False)
    test["source_group"] = np.where(test["label"].astype(int) == 1, "test_trusted_positive", "test_2018_negative")
    rows = pd.concat(
        [
            pd.concat(selected_parts, ignore_index=True),
            test,
        ],
        ignore_index=True,
        sort=False,
    )
    records = rows[["id", "label", "source_group", *(column for _, column in FRAME_COLUMNS)]].to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        feature_rows = list(executor.map(row_features, records))
    features = pd.DataFrame(feature_rows)

    feature_columns = [column for column in features.columns if column not in {"id", "label", "source_group"}]
    train_mask = features["source_group"].str.startswith("train_")
    x_train = features.loc[train_mask, feature_columns].to_numpy(dtype=float)
    y_train = features.loc[train_mask, "label"].to_numpy(dtype=int)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260731)
    oof = cross_val_predict(pipeline, x_train, y_train, cv=folds, method="predict_proba")[:, 1]
    pipeline.fit(x_train, y_train)
    all_probabilities = pipeline.predict_proba(features[feature_columns].to_numpy(dtype=float))[:, 1]
    features["visible_texture_proxy_probability"] = all_probabilities

    model_predictions = pd.read_csv(args.model_predictions, low_memory=False)
    model_predictions["id"] = model_predictions["id"].astype(str).str.replace("__original", "", regex=False)
    model_score = model_predictions.drop_duplicates("id").set_index("id")["positive_probability"]
    features["universal360_probability"] = features["id"].map(model_score)

    train_neg = features[(features["source_group"] == "train_label_0_reference")]
    train_pos = features[(features["source_group"] == "train_label_1_reference")]
    effect_sizes = []
    for column in feature_columns:
        neg = train_neg[column].to_numpy(dtype=float)
        pos = train_pos[column].to_numpy(dtype=float)
        effect_sizes.append(
            {
                "feature": column,
                "train_positive_mean": float(pos.mean()),
                "train_negative_mean": float(neg.mean()),
                "cohens_d_positive_minus_negative": cohens_d(pos, neg),
            }
        )
    effect_sizes.sort(key=lambda row: abs(row["cohens_d_positive_minus_negative"] or 0.0), reverse=True)

    test_rows = features[~train_mask].copy()
    test_y = test_rows["label"].to_numpy(dtype=int)
    test_proxy = test_rows["visible_texture_proxy_probability"].to_numpy(dtype=float)
    valid_model = test_rows["universal360_probability"].notna()
    correlation = None
    if valid_model.sum() >= 3:
        correlation = float(
            spearmanr(
                test_rows.loc[valid_model, "visible_texture_proxy_probability"],
                test_rows.loc[valid_model, "universal360_probability"],
            ).statistic
        )
    summary = {
        "rows": len(features),
        "feature_count": len(feature_columns),
        "training_reference_rows": int(train_mask.sum()),
        "test_rows": int((~train_mask).sum()),
        "training_proxy_oof": {
            "auroc": float(roc_auc_score(y_train, oof)),
            "average_precision": float(average_precision_score(y_train, oof)),
            "accuracy_at_0p5": float(accuracy_score(y_train, oof >= 0.5)),
        },
        "test_proxy": {
            "auroc": float(roc_auc_score(test_y, test_proxy)),
            "average_precision": float(average_precision_score(test_y, test_proxy)),
            "negative_mean": float(test_proxy[test_y == 0].mean()),
            "positive_mean": float(test_proxy[test_y == 1].mean()),
            "fpr_at_0p5": float(np.mean(test_proxy[test_y == 0] >= 0.5)),
            "recall_at_0p5": float(np.mean(test_proxy[test_y == 1] >= 0.5)),
        },
        "test_proxy_vs_universal360_spearman": correlation,
        "top_training_effect_sizes": effect_sizes[:20],
        "group_means": features.groupby("source_group")[feature_columns + ["visible_texture_proxy_probability", "universal360_probability"]]
        .mean(numeric_only=True)
        .to_dict(orient="index"),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("rows", "feature_count", "training_proxy_oof", "test_proxy", "test_proxy_vs_universal360_spearman")}, indent=2))


if __name__ == "__main__":
    main()
