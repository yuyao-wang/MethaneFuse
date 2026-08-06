#!/usr/bin/env python3
"""Label-free temporal correspondence pretraining for frozen L89 CLS caches.

The self-supervised objective receives the same t0 query in both classes:

* positive: the row's coherent, unique history;
* negative: the entire history from a different canonical event.

Negative donors form a deterministic cross-event bijection, so the marginal
history distribution is exactly matched.  The pretext query cannot attend to
t0 as a key/value.  All experiments use role encoding only
(``enable_delta=False``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pretraining_20260727 import (  # noqa: E402
    l89_ragged_cls_experiment as cache_runner,
)


SCRIPT_VERSION = "l89-correspondence-pretrain-v1"
FORWARD_CONTRACT_VERSION = "t0-query-history-only-kv-role-only-v1"
BASE_ARMS = (
    "scratch",
    "correspondence_pretrained",
    "permuted_pretext_labels_control",
)
PAST_PREDICTION_ARM = "past_prediction_pretrained"
CLASSIFIER_STATE_KEYS = ("classifier.bias", "classifier.weight")
PAST_PREDICTOR_ONLY_KEYS = (
    "history_query",
    "null_history_token",
    "output_projection.bias",
    "output_projection.weight",
)
CHECKPOINT_NAME = "checkpoint_best_ap.pt"
COHERENT_PREDICTION_NAME = "validation_best_ap_coherent_predictions.csv"
SHUFFLED_PREDICTION_NAME = (
    "validation_best_ap_cross_event_shuffled_predictions.csv"
)
PRETRAIN_CHECKPOINT_NAME = "checkpoint_final.pt"
FIXED_THRESHOLD = 0.5
MODEL_CONFIG_KEYS = (
    "feature_dim",
    "num_roles",
    "model_dim",
    "num_heads",
    "depth",
    "mlp_ratio",
    "dropout",
    "periods_days",
    "t0_index",
)


def parse_float_tuple(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        result = tuple(
            float(part.strip()) for part in value.split(",") if part.strip()
        )
    else:
        result = tuple(float(part) for part in value)
    if not result or not all(math.isfinite(item) and item > 0 for item in result):
        raise ValueError(f"Expected positive finite values, got {value!r}.")
    return result


def parse_fractions(value: str | Sequence[float]) -> tuple[float, ...]:
    fractions = parse_float_tuple(value)
    if any(fraction > 1.0 for fraction in fractions):
        raise ValueError("Labeled event fractions must lie in (0, 1].")
    if len(set(fractions)) != len(fractions):
        raise ValueError("Labeled event fractions must be unique.")
    return fractions


def fraction_tag(fraction: float) -> str:
    percentage = float(fraction) * 100.0
    if math.isclose(percentage, round(percentage), abs_tol=1e-10):
        value = str(int(round(percentage)))
    else:
        value = f"{percentage:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"labeled_events_{value}pct"


def set_reproducible_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    return device


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "resolved_device" or callable(value):
            continue
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, torch.device):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    # Fail before a checkpoint is written with a non-JSON argument.
    json.dumps(result, allow_nan=False)
    return result


def hash_strings(values: Sequence[str]) -> str:
    return cache_runner.sha256_bytes(
        cache_runner.canonical_json_bytes([str(value) for value in values])
    )


def hash_integer_tensor(value: torch.Tensor) -> str:
    return cache_runner.tensor_sha256(value.detach().cpu().long())


def batch_plan_sha256(
    plan: Sequence[Sequence[torch.Tensor]],
) -> str:
    serializable = [
        [indices.detach().cpu().long().tolist() for indices in epoch_batches]
        for epoch_batches in plan
    ]
    return cache_runner.sha256_bytes(
        cache_runner.canonical_json_bytes(serializable)
    )


def classifier_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    classifier = {
        key: state[key].detach().cpu() for key in CLASSIFIER_STATE_KEYS
    }
    return cache_runner.state_dict_sha256(classifier)


def normalized_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Model config is missing keys: {missing}.")
    normalized = {
        "feature_dim": int(config["feature_dim"]),
        "num_roles": int(config["num_roles"]),
        "model_dim": int(config["model_dim"]),
        "num_heads": int(config["num_heads"]),
        "depth": int(config["depth"]),
        "mlp_ratio": float(config["mlp_ratio"]),
        "dropout": float(config["dropout"]),
        "periods_days": [
            float(value) for value in config["periods_days"]
        ],
        "t0_index": int(config["t0_index"]),
    }
    if normalized["depth"] != 2:
        raise ValueError("Correspondence experiments require depth=2.")
    if normalized["feature_dim"] <= 0 or normalized["num_roles"] < 2:
        raise ValueError("Invalid feature_dim/num_roles in model config.")
    if normalized["model_dim"] % normalized["num_heads"]:
        raise ValueError("model_dim must be divisible by num_heads.")
    if not 0 <= normalized["t0_index"] < normalized["num_roles"]:
        raise ValueError("t0_index is outside the role range.")
    parse_float_tuple(normalized["periods_days"])
    return normalized


def build_model(config: Mapping[str, Any]) -> cache_runner.RaggedCurrentQueryHead:
    normalized = normalized_model_config(config)
    return cache_runner.RaggedCurrentQueryHead(
        feature_dim=normalized["feature_dim"],
        num_roles=normalized["num_roles"],
        model_dim=normalized["model_dim"],
        num_heads=normalized["num_heads"],
        depth=normalized["depth"],
        mlp_ratio=normalized["mlp_ratio"],
        dropout=normalized["dropout"],
        periods_days=normalized["periods_days"],
        t0_index=normalized["t0_index"],
    )


def build_bijective_cross_event_donors(
    event_ids: Sequence[str],
    *,
    seed: int,
) -> torch.Tensor:
    """Build a deterministic cross-event permutation over row indices."""

    events = [str(value) for value in event_ids]
    if len(events) < 2 or any(not value.strip() for value in events):
        raise ValueError("Donor construction needs nonempty IDs from >=2 rows.")
    groups: dict[str, list[int]] = {}
    for row, event_id in enumerate(events):
        groups.setdefault(event_id, []).append(row)
    if len(groups) < 2:
        raise ValueError("Cross-event donors require at least two events.")
    largest_group = max(len(rows) for rows in groups.values())
    if largest_group * 2 > len(events):
        raise ValueError(
            "An event occupies more than half of rows; a cross-event "
            "bijection is impossible."
        )
    rng = random.Random(int(seed))
    event_order = sorted(groups)
    rng.shuffle(event_order)
    flattened: list[int] = []
    for event_id in event_order:
        rows = list(groups[event_id])
        rng.shuffle(rows)
        flattened.extend(rows)
    donors = torch.empty(len(events), dtype=torch.long)
    for position, target in enumerate(flattened):
        donor = flattened[(position + largest_group) % len(flattened)]
        donors[target] = donor
    if sorted(donors.tolist()) != list(range(len(events))):
        raise RuntimeError("Donor construction did not produce a bijection.")
    for target, donor in enumerate(donors.tolist()):
        if events[target] == events[donor]:
            raise RuntimeError("A correspondence donor shares the target event.")
    return donors


def extract_unlabeled_correspondence_view(
    data: Mapping[str, Any],
    *,
    cache_row_indices: torch.Tensor,
    t0_index: int,
) -> dict[str, Any]:
    """Copy only tensors/IDs allowed to enter the self-supervised objective."""

    features = data["features"].detach().cpu().float()
    valid = data["valid_mask"].detach().cpu().bool()
    unique = data["unique_mask"].detach().cpu().bool()
    delta = data["delta_days"].detach().cpu().float()
    if features.ndim != 3:
        raise ValueError("Correspondence features must have shape (N,T,D).")
    rows, timepoints, _ = features.shape
    expected = (rows, timepoints)
    for name, tensor in (
        ("valid_mask", valid),
        ("unique_mask", unique),
        ("delta_days", delta),
    ):
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")
    if cache_row_indices.shape != (rows,):
        raise ValueError("cache_row_indices must have one entry per usable row.")
    if len(data["event_ids"]) != rows:
        raise ValueError("event_ids must have one entry per usable row.")
    if not 0 <= int(t0_index) < timepoints:
        raise ValueError("t0_index is out of range.")
    effective = valid & unique
    if not effective[:, int(t0_index)].all():
        raise ValueError("Every correspondence row must have a usable unique t0.")
    if not torch.isfinite(features).all():
        raise ValueError("Correspondence features contain non-finite values.")
    history = [
        index for index in range(timepoints) if index != int(t0_index)
    ]
    eligible = effective[:, history].any(dim=1)
    selected = torch.nonzero(eligible, as_tuple=False).flatten()
    if selected.numel() < 2:
        raise ValueError("Correspondence pretraining needs >=2 rows with history.")
    effective_history = effective[selected][:, history]
    selected_delta = delta[selected][:, history]
    if torch.any(effective_history & ~torch.isfinite(selected_delta)):
        raise ValueError("A usable correspondence history delta is non-finite.")
    if torch.any(effective_history & (selected_delta >= 0)):
        raise ValueError("Usable correspondence history must precede t0.")
    selected_events = [str(data["event_ids"][index]) for index in selected.tolist()]
    if any(not event_id.strip() for event_id in selected_events):
        raise ValueError("Correspondence event IDs must be nonempty.")
    return {
        "features": features[selected].contiguous(),
        "valid_mask": valid[selected].contiguous(),
        "unique_mask": unique[selected].contiguous(),
        "delta_days": delta[selected].contiguous(),
        "event_ids": selected_events,
        "cache_row_indices": cache_row_indices[selected].long().contiguous(),
        "t0_index": int(t0_index),
        "source_usable_rows": rows,
        "eligible_rows": int(selected.numel()),
    }


def correspondence_input_sha256(view: Mapping[str, Any]) -> str:
    evidence = {
        "features_sha256": cache_runner.tensor_sha256(view["features"]),
        "valid_mask_sha256": cache_runner.tensor_sha256(view["valid_mask"]),
        "unique_mask_sha256": cache_runner.tensor_sha256(view["unique_mask"]),
        "delta_days_sha256": cache_runner.tensor_sha256(view["delta_days"]),
        "cache_row_indices_sha256": hash_integer_tensor(
            view["cache_row_indices"]
        ),
        "event_ids_sha256": hash_strings(view["event_ids"]),
        "t0_index": int(view["t0_index"]),
    }
    return cache_runner.sha256_bytes(
        cache_runner.canonical_json_bytes(evidence)
    )


def build_correspondence_pretext(
    view: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    rows, timepoints, _ = view["features"].shape
    t0_index = int(view["t0_index"])
    history = [index for index in range(timepoints) if index != t0_index]
    donors = build_bijective_cross_event_donors(
        view["event_ids"], seed=seed
    )
    outputs: dict[str, torch.Tensor] = {}
    for key in ("features", "valid_mask", "unique_mask", "delta_days"):
        source = view[key]
        output_shape = (rows * 2,) + tuple(source.shape[1:])
        output = torch.empty(output_shape, dtype=source.dtype)
        output[0::2] = source
        output[1::2] = source
        output[1::2, history] = source[donors][:, history]
        outputs[key] = output.contiguous()
    labels = torch.empty(rows * 2, dtype=torch.float32)
    labels[0::2] = 1.0
    labels[1::2] = 0.0
    anchor_indices = torch.arange(rows).repeat_interleave(2)
    if int(labels.sum()) * 2 != labels.numel():
        raise RuntimeError("Correspondence pretext labels are not balanced.")
    return {
        **outputs,
        "labels": labels,
        "anchor_view_indices": anchor_indices,
        "donor_view_indices": donors,
        "donor_cache_row_indices": view["cache_row_indices"][donors],
        "seed": int(seed),
        "input_sha256": correspondence_input_sha256(view),
        "donor_sha256": hash_integer_tensor(donors),
        "label_sha256": cache_runner.tensor_sha256(labels),
        "positive_rows": rows,
        "negative_rows": rows,
    }


def build_permuted_pretext_labels(
    labels: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    labels = labels.detach().cpu().float()
    if labels.ndim != 1 or labels.numel() < 4:
        raise ValueError("Permutation control needs at least four examples.")
    if int(labels.sum()) * 2 != labels.numel():
        raise ValueError("Permutation control requires balanced source labels.")
    generator = torch.Generator().manual_seed(int(seed))
    permuted: Optional[torch.Tensor] = None
    permutation: Optional[torch.Tensor] = None
    for _ in range(100):
        candidate_permutation = torch.randperm(
            labels.numel(), generator=generator
        )
        candidate = labels[candidate_permutation]
        if not torch.equal(candidate, labels) and not torch.equal(
            candidate, 1.0 - labels
        ):
            permuted = candidate
            permutation = candidate_permutation
            break
    if permuted is None or permutation is None:
        raise RuntimeError("Could not construct a nontrivial label permutation.")
    if int(permuted.sum()) * 2 != permuted.numel():
        raise RuntimeError("Permuted control labels lost class balance.")
    audit = {
        "seed": int(seed),
        "permutation_sha256": hash_integer_tensor(permutation),
        "label_sha256": cache_runner.tensor_sha256(permuted),
        "balanced": True,
        "methane_labels_used": False,
    }
    return permuted.contiguous(), audit


def correspondence_forward(
    model: cache_runner.RaggedCurrentQueryHead,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    unique_mask: torch.Tensor,
    delta_days: torch.Tensor,
    role_index: torch.Tensor,
) -> torch.Tensor:
    """Run t0 as query while excluding t0 from attention keys/values."""

    if features.ndim != 3:
        raise ValueError("features must have shape (B,T,D).")
    batch_size, timepoints, _ = features.shape
    expected = (batch_size, timepoints)
    for name, tensor in (
        ("valid_mask", valid_mask),
        ("unique_mask", unique_mask),
        ("delta_days", delta_days),
    ):
        if tensor.shape != expected:
            raise ValueError(f"{name} shape does not match features.")
    if role_index.ndim == 1:
        role_index = role_index.view(1, -1).expand(batch_size, -1)
    if role_index.shape != expected:
        raise ValueError("role_index shape does not match features.")
    effective_valid = valid_mask.bool() & unique_mask.bool()
    if not effective_valid[:, model.t0_index].all():
        raise ValueError("Every correspondence query needs a valid unique t0.")
    context_valid = effective_valid.clone()
    context_valid[:, model.t0_index] = False
    if not context_valid.any(dim=1).all():
        raise ValueError("Every correspondence query needs usable history.")

    # Execute the delta encoder for architecture/compute parity, but role-only
    # correspondence explicitly disables its contribution.
    delta_embedding = model.delta_encoder(delta_days) * 0.0
    context = (
        model.input_projection(features)
        + model.role_embedding(role_index.long())
        + delta_embedding
    )
    context = model.input_norm(context)
    query = context[:, model.t0_index : model.t0_index + 1]
    for block in model.blocks:
        query = block(query, context, context_valid)
    return model.classifier(model.output_norm(query[:, 0])).squeeze(-1)


def transfer_temporal_encoder_state(
    target_model: cache_runner.RaggedCurrentQueryHead,
    *,
    source_state: Mapping[str, torch.Tensor],
    downstream_initial_state: Mapping[str, torch.Tensor],
    source_model_config: Mapping[str, Any],
    target_model_config: Mapping[str, Any],
    source_name: str,
    allowed_source_only_keys: Sequence[str],
) -> dict[str, Any]:
    """Strictly transfer the common temporal trunk and reset the classifier."""

    source_config = normalized_model_config(source_model_config)
    target_config = normalized_model_config(target_model_config)
    if source_config != target_config:
        raise ValueError(
            f"{source_name}: source/target model configs differ."
        )
    target_state = target_model.state_dict()
    if set(target_state) != set(downstream_initial_state):
        raise ValueError("Downstream initial state schema differs from target.")
    classifier_keys = set(CLASSIFIER_STATE_KEYS)
    if not classifier_keys.issubset(target_state):
        raise ValueError("Target classifier state keys are missing.")
    transfer_keys = sorted(set(target_state) - classifier_keys)
    missing = sorted(set(transfer_keys) - set(source_state))
    if missing:
        raise ValueError(
            f"{source_name}: source is missing temporal state keys: {missing}."
        )
    source_only = sorted(set(source_state) - set(transfer_keys))
    unexpected = sorted(set(source_only) - set(allowed_source_only_keys))
    if unexpected:
        raise ValueError(
            f"{source_name}: unexpected source-only state keys: {unexpected}."
        )
    combined = {
        key: value.detach().cpu().clone()
        for key, value in downstream_initial_state.items()
    }
    tensor_audit: dict[str, dict[str, Any]] = {}
    for key in transfer_keys:
        source_value = source_state[key]
        target_value = combined[key]
        if not isinstance(source_value, torch.Tensor):
            raise ValueError(f"{source_name}: state {key!r} is not a tensor.")
        source_cpu = source_value.detach().cpu()
        if (
            source_cpu.shape != target_value.shape
            or source_cpu.dtype != target_value.dtype
        ):
            raise ValueError(
                f"{source_name}: temporal state {key!r} shape/dtype differs."
            )
        if key == "delta_encoder.periods_days" and not torch.equal(
            source_cpu, target_value
        ):
            raise ValueError(
                f"{source_name}: Fourier period buffer differs from target."
            )
        combined[key] = source_cpu.clone()
        tensor_audit[key] = {
            "shape": list(source_cpu.shape),
            "dtype": str(source_cpu.dtype),
            "numel": int(source_cpu.numel()),
            "sha256": cache_runner.tensor_sha256(source_cpu),
        }
    classifier_before = classifier_state_sha256(combined)
    target_model.load_state_dict(combined, strict=True)
    observed = target_model.state_dict()
    for key in transfer_keys:
        if not torch.equal(observed[key].cpu(), source_state[key].cpu()):
            raise RuntimeError(f"{source_name}: transfer failed for {key!r}.")
    for key in CLASSIFIER_STATE_KEYS:
        if not torch.equal(
            observed[key].cpu(), downstream_initial_state[key].cpu()
        ):
            raise RuntimeError(f"{source_name}: classifier reset failed.")
    classifier_after = classifier_state_sha256(observed)
    if classifier_before != classifier_after:
        raise RuntimeError(f"{source_name}: classifier SHA changed on transfer.")
    parameter_keys = set(dict(target_model.named_parameters()))
    transferred_parameter_keys = sorted(
        key for key in transfer_keys if key in parameter_keys
    )
    return {
        "source_name": str(source_name),
        "transferred_state_keys": transfer_keys,
        "transferred_state_count_keys": len(transfer_keys),
        "transferred_parameter_keys": transferred_parameter_keys,
        "transferred_parameter_count_keys": len(transferred_parameter_keys),
        "nontransferred_state_keys": sorted(CLASSIFIER_STATE_KEYS),
        "ignored_source_only_keys": source_only,
        "transfer_key_list_sha256": hash_strings(transfer_keys),
        "tensor_audit_sha256": cache_runner.sha256_bytes(
            cache_runner.canonical_json_bytes(tensor_audit)
        ),
        "classifier_sha256_before": classifier_before,
        "classifier_sha256_after": classifier_after,
        "period_buffer_verified_equal": True,
        "source_model_config_sha256": cache_runner.sha256_bytes(
            cache_runner.canonical_json_bytes(source_config)
        ),
        "target_model_config_sha256": cache_runner.sha256_bytes(
            cache_runner.canonical_json_bytes(target_config)
        ),
    }


def select_labeled_event_rows(
    labels: torch.Tensor,
    event_ids: Sequence[str],
    *,
    fraction: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select complete events by stable hash, then require both row labels."""

    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    labels = labels.detach().cpu().long()
    events = [str(value) for value in event_ids]
    if labels.shape != (len(events),) or not events:
        raise ValueError("labels/event_ids must have the same nonzero length.")
    if any(not event_id.strip() for event_id in events):
        raise ValueError("Labeled selection event IDs must be nonempty.")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Labeled selection source must contain both classes.")
    unique_events = sorted(set(events))

    def stable_key(event_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{int(seed)}\0{event_id}".encode("utf-8")
        ).hexdigest()
        return digest, event_id

    ordered_events = sorted(unique_events, key=stable_key)
    selected_event_count = min(
        len(unique_events),
        max(1, int(math.ceil(fraction * len(unique_events)))),
    )
    selected_events = sorted(ordered_events[:selected_event_count])
    selected_set = set(selected_events)
    indices = torch.tensor(
        [
            index
            for index, event_id in enumerate(events)
            if event_id in selected_set
        ],
        dtype=torch.long,
    )
    selected_labels = labels[indices]
    if set(selected_labels.tolist()) != {0, 1}:
        raise ValueError(
            "Stable event selection did not preserve both row labels; "
            "the preregistered membership is not redrawn."
        )
    observed_events = {events[index] for index in indices.tolist()}
    if observed_events != selected_set:
        raise RuntimeError("Event-level row selection is incomplete.")
    audit = {
        "requested_fraction": fraction,
        "seed": int(seed),
        "source_events": len(unique_events),
        "source_rows": len(events),
        "selected_events": len(selected_events),
        "selected_rows": int(indices.numel()),
        "realized_event_fraction": len(selected_events) / len(unique_events),
        "realized_row_fraction": int(indices.numel()) / len(events),
        "selected_negative_rows": int((selected_labels == 0).sum()),
        "selected_positive_rows": int((selected_labels == 1).sum()),
        "selected_event_ids": selected_events,
        "selected_event_sha256": hash_strings(selected_events),
        "selected_row_index_sha256": hash_integer_tensor(indices),
        "selection_rule": "stable_sha256_seed_event_take_ceil_fraction",
        "event_complete": True,
        "both_row_labels_present": True,
    }
    return indices, audit


def subset_rows(data: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    rows = indices.tolist()
    return {
        "features": data["features"][indices].float(),
        "labels": data["labels"][indices].float(),
        "valid_mask": data["valid_mask"][indices].bool(),
        "unique_mask": data["unique_mask"][indices].bool(),
        "delta_days": data["delta_days"][indices].float(),
        "ids": [data["ids"][index] for index in rows],
        "plume_ids": [data["plume_ids"][index] for index in rows],
        "event_ids": [data["event_ids"][index] for index in rows],
    }


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    loss: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Classification metrics require both labels.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values.")
    predictions = (probabilities >= FIXED_THRESHOLD).astype(np.int64)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    return {
        "loss": float(loss),
        "ap": float(average_precision_score(labels, probabilities)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "macro_f1_at_0_5": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "pred_positive_rate_at_0_5": float(predictions.mean()),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()),
        "probability_std": float(probabilities.std()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "rows": int(labels.size),
    }


def fit_pretext_model(
    *,
    initial_state: Mapping[str, torch.Tensor],
    model_config: Mapping[str, Any],
    pretext: Mapping[str, Any],
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    max_train_steps: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    targets = targets.detach().cpu().float()
    if targets.shape != pretext["labels"].shape:
        raise ValueError("Pretext target shape differs from paired examples.")
    if int(targets.sum()) * 2 != targets.numel():
        raise ValueError("Pretext BCE targets must be exactly balanced.")
    model = build_model(model_config).to(device)
    model.load_state_dict(initial_state, strict=True)
    set_reproducible_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, Any]] = []
    total_steps = 0
    total_rows = 0
    batch_plan: list[list[torch.Tensor]] = []
    started = time.monotonic()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        batches = cache_runner.fixed_epoch_batches(
            int(targets.numel()),
            batch_size=int(batch_size),
            seed=int(seed),
            epoch=epoch,
            shuffle=True,
        )
        if int(max_train_steps):
            batches = batches[: int(max_train_steps)]
        batch_plan.append(
            [indices.detach().cpu().clone() for indices in batches]
        )
        loss_sum = 0.0
        correct = 0
        rows_seen = 0
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = correspondence_forward(
                model,
                pretext["features"][indices].to(device),
                pretext["valid_mask"][indices].to(device),
                pretext["unique_mask"][indices].to(device),
                pretext["delta_days"][indices].to(device),
                torch.arange(
                    pretext["features"].shape[1],
                    dtype=torch.long,
                    device=device,
                ),
            )
            batch_targets = targets[indices].to(device)
            loss = criterion(logits, batch_targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Pretext BCE loss became non-finite.")
            loss.backward()
            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(grad_clip)
                )
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            correct += int(
                ((torch.sigmoid(logits) >= 0.5).float() == batch_targets)
                .sum()
                .detach()
            )
            rows_seen += len(indices)
        if not batches:
            raise ValueError("Pretext training produced no optimizer steps.")
        total_steps += len(batches)
        total_rows += rows_seen
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": len(batches),
                "rows_seen": rows_seen,
                "balanced_bce_loss": loss_sum / rows_seen,
                "pretext_accuracy_at_0_5": correct / rows_seen,
                "elapsed_seconds": float(time.monotonic() - started),
            }
        )
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    summary = {
        "epochs": int(epochs),
        "optimizer_steps": total_steps,
        "train_rows_seen": total_rows,
        "positive_examples": int((targets == 1).sum()),
        "negative_examples": int((targets == 0).sum()),
        "balanced_bce": True,
        "methane_labels_used": False,
        "methane_labels_used_by_pretext_objective": False,
        "batch_plan_sha256": batch_plan_sha256(batch_plan),
        "history": history,
        "final_state_sha256": cache_runner.state_dict_sha256(state),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, summary


def prepare_role_only_inputs(
    data: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = data["features"].float()
    effective_valid = data["valid_mask"].bool() & data["unique_mask"].bool()
    delta = torch.zeros_like(data["delta_days"].float())
    return features, effective_valid, delta


def build_shuffled_history_data(
    data: Mapping[str, Any],
    *,
    donors: torch.Tensor,
    t0_index: int,
) -> dict[str, Any]:
    features, valid, unique, delta = cache_runner.apply_history_donors(
        data["features"].float(),
        data["valid_mask"].bool(),
        data["unique_mask"].bool(),
        data["delta_days"].float(),
        donors,
        t0_index=int(t0_index),
    )
    result = dict(data)
    result.update(
        {
            "features": features,
            "valid_mask": valid,
            "unique_mask": unique,
            "delta_days": delta,
        }
    )
    return result


def evaluate_downstream(
    model: cache_runner.RaggedCurrentQueryHead,
    data: Mapping[str, Any],
    *,
    role_index: torch.Tensor,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[dict[str, Any], np.ndarray]:
    features, effective_valid, delta = prepare_role_only_inputs(data)
    rows = int(features.shape[0])
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for indices in cache_runner.fixed_epoch_batches(
            rows,
            batch_size=int(batch_size),
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            labels = data["labels"][indices].float().to(device)
            logits = model(
                features[indices].to(device),
                effective_valid[indices].to(device),
                role_index.to(device),
                delta[indices].to(device),
                enable_delta=False,
            )
            loss = criterion(logits, labels)
            loss_sum += float(loss) * len(indices)
            probabilities.append(torch.sigmoid(logits).cpu())
            targets.append(labels.cpu())
    probability = torch.cat(probabilities).numpy()
    target = torch.cat(targets).numpy().astype(np.int64)
    return (
        classification_metrics(
            target, probability, loss=loss_sum / max(rows, 1)
        ),
        probability,
    )


def train_downstream(
    *,
    arm: str,
    starting_state: Mapping[str, torch.Tensor],
    model_config: Mapping[str, Any],
    train_data: Mapping[str, Any],
    validation_data: Mapping[str, Any],
    shuffled_validation_data: Mapping[str, Any],
    role_index: torch.Tensor,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    max_train_steps: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    model = build_model(model_config).to(device)
    model.load_state_dict(starting_state, strict=True)
    set_reproducible_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_labels = train_data["labels"].float()
    positives = float(train_labels.sum())
    negatives = float(train_labels.numel() - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("Supervised subset must contain both labels.")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device)
    )
    train_features, train_valid, train_delta = prepare_role_only_inputs(
        train_data
    )
    history: list[dict[str, Any]] = []
    best_record: Optional[dict[str, Any]] = None
    best_state: Optional[dict[str, torch.Tensor]] = None
    total_steps = 0
    total_rows = 0
    batch_plan: list[list[torch.Tensor]] = []
    started = time.monotonic()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        batches = cache_runner.fixed_epoch_batches(
            int(train_labels.numel()),
            batch_size=int(batch_size),
            seed=int(seed),
            epoch=epoch,
            shuffle=True,
        )
        if int(max_train_steps):
            batches = batches[: int(max_train_steps)]
        batch_plan.append(
            [indices.detach().cpu().clone() for indices in batches]
        )
        loss_sum = 0.0
        rows_seen = 0
        for indices in batches:
            optimizer.zero_grad(set_to_none=True)
            labels = train_labels[indices].to(device)
            logits = model(
                train_features[indices].to(device),
                train_valid[indices].to(device),
                role_index.to(device),
                train_delta[indices].to(device),
                enable_delta=False,
            )
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{arm}: supervised loss became non-finite."
                )
            loss.backward()
            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(grad_clip)
                )
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            rows_seen += len(indices)
        if not batches:
            raise ValueError("Supervised training produced no optimizer steps.")
        total_steps += len(batches)
        total_rows += rows_seen
        validation_metrics, _ = evaluate_downstream(
            model,
            validation_data,
            role_index=role_index,
            batch_size=eval_batch_size,
            device=device,
            criterion=criterion,
        )
        record = {
            "arm": arm,
            "epoch": epoch,
            "optimizer_steps": len(batches),
            "train_rows_seen": rows_seen,
            "train_loss": loss_sum / rows_seen,
            "coherent_validation": validation_metrics,
            "elapsed_seconds": float(time.monotonic() - started),
        }
        history.append(record)
        if (
            best_record is None
            or validation_metrics["ap"]
            > best_record["coherent_validation"]["ap"]
        ):
            best_record = copy.deepcopy(record)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_record is None or best_state is None:
        raise RuntimeError(f"{arm}: no supervised best checkpoint.")
    model.load_state_dict(best_state, strict=True)
    coherent_metrics, coherent_probability = evaluate_downstream(
        model,
        validation_data,
        role_index=role_index,
        batch_size=eval_batch_size,
        device=device,
        criterion=criterion,
    )
    shuffled_metrics, shuffled_probability = evaluate_downstream(
        model,
        shuffled_validation_data,
        role_index=role_index,
        batch_size=eval_batch_size,
        device=device,
        criterion=criterion,
    )
    result = {
        "arm": arm,
        "best_epoch": int(best_record["epoch"]),
        "selection_metric": "coherent_validation_ap",
        "coherent_validation": coherent_metrics,
        "history_shuffled_validation": {
            **shuffled_metrics,
            "used_for_selection": False,
        },
        "mechanism_coherent_minus_shuffled": {
            key: float(coherent_metrics[key] - shuffled_metrics[key])
            for key in (
                "ap",
                "auc",
                "macro_f1_at_0_5",
                "balanced_accuracy_at_0_5",
            )
        },
        "supervised_optimizer_steps": total_steps,
        "supervised_train_rows_seen": total_rows,
        "supervised_batch_plan_sha256": batch_plan_sha256(batch_plan),
        "history": history,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, best_state, coherent_probability, shuffled_probability


def prediction_frame(
    *,
    data: Mapping[str, Any],
    probabilities: np.ndarray,
    arm: str,
    fraction: float,
    epoch: int,
    evaluation_condition: str,
    history_source_event_ids: Sequence[str],
) -> pd.DataFrame:
    if len(probabilities) != len(data["ids"]):
        raise ValueError("Prediction length differs from validation rows.")
    return pd.DataFrame(
        {
            "id": data["ids"],
            "plume_id": data["plume_ids"],
            "event_id": data["event_ids"],
            "label": data["labels"].long().tolist(),
            "probability": probabilities,
            "prediction_at_0_5": (
                probabilities >= FIXED_THRESHOLD
            ).astype(np.int64),
            "arm": arm,
            "labeled_event_fraction": float(fraction),
            "epoch": int(epoch),
            "evaluation_condition": evaluation_condition,
            "history_source_event_id": [
                str(value) for value in history_source_event_ids
            ],
        }
    )


def _cache_integrity_checks(
    train_cache: Mapping[str, Any],
    validation_cache: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
) -> None:
    for name, payload in (
        ("train", train_cache),
        ("validation", validation_cache),
    ):
        if payload.get("script_version") != cache_runner.SCRIPT_VERSION:
            raise ValueError(f"{name} cache script_version is stale/mismatched.")
        events = [str(value) for value in payload["event_ids"]]
        if any(not value.strip() for value in events):
            raise ValueError(f"{name} cache contains empty event IDs.")
        timepoints = int(payload["features"].shape[1])
        if not torch.equal(
            payload["role_index"].long(), torch.arange(timepoints)
        ):
            raise ValueError(f"{name} cache role_index is stale/mismatched.")
    overlap = set(train_cache["event_ids"]) & set(validation_cache["event_ids"])
    if overlap or int(cache_audit["event_overlap"]) != 0:
        raise ValueError("Train/validation canonical event overlap is nonzero.")


def _assert_cache_files_unchanged(cache_audit: Mapping[str, Any]) -> None:
    checks = (
        ("train_cache", "train_cache_sha256"),
        ("validation_cache", "validation_cache_sha256"),
    )
    for path_key, sha_key in checks:
        path = Path(cache_audit[path_key]).expanduser().resolve()
        observed = cache_runner.sha256_file(path)
        if observed != cache_audit[sha_key]:
            raise RuntimeError(f"{path_key} changed during the experiment.")


def _pretrain_checkpoint(
    *,
    arm: str,
    state: Mapping[str, torch.Tensor],
    initial_state_sha256: str,
    parameter_signature: Mapping[str, Any],
    model_config: Mapping[str, Any],
    summary: Mapping[str, Any],
    pretext_audit: Mapping[str, Any],
    train_cache_audit: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "mode": "correspondence_pretrain",
        "objective": arm,
        "forward_contract_version": FORWARD_CONTRACT_VERSION,
        "role_only_enable_delta": False,
        "model": {
            key: value.detach().cpu() for key, value in state.items()
        },
        "model_state_sha256": cache_runner.state_dict_sha256(state),
        "initial_state_sha256": initial_state_sha256,
        "model_parameter_signature": dict(parameter_signature),
        "model_config": normalized_model_config(model_config),
        "training_summary": dict(summary),
        "pretext_audit": dict(pretext_audit),
        "train_cache_audit": dict(train_cache_audit),
        "methane_labels_used": False,
        "methane_labels_used_by_pretext_objective": False,
        "label_access_boundary": (
            "cache loader validates stored labels; pretext construction and "
            "pretext BCE do not consume methane class labels"
        ),
        "sealed_test_read": False,
        "args": serializable_args(args),
    }


def load_past_predictor_source(
    path: Path,
    *,
    cache_audit: Mapping[str, Any],
    train_event_ids: Sequence[str],
    target_model_config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    cache_runner.assert_not_sealed_path(
        path, purpose="past predictor checkpoint"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = cache_runner.torch_load_trusted(path)
    if not isinstance(checkpoint, dict):
        raise ValueError("Past predictor checkpoint must contain a dictionary.")
    if checkpoint.get("script_version") != "l89-innovation-pretrain-v1":
        raise ValueError("Past predictor script_version is incompatible.")
    if checkpoint.get("fold") != "final_all_train":
        raise ValueError("Past predictor must be the final_all_train checkpoint.")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Past predictor checkpoint has no model state.")
    observed_state_sha = cache_runner.state_dict_sha256(state)
    if observed_state_sha != checkpoint.get("model_state_sha256"):
        raise ValueError("Past predictor model state SHA is invalid.")
    source_config = normalized_model_config(checkpoint["model_config"])
    if source_config != normalized_model_config(target_model_config):
        raise ValueError("Past predictor architecture differs from target head.")
    predictor_audit = checkpoint.get("cache_audit")
    if not isinstance(predictor_audit, Mapping):
        raise ValueError("Past predictor cache audit is missing.")
    for key in ("train_cache_sha256", "weights_sha256"):
        if predictor_audit.get(key) != cache_audit.get(key):
            raise ValueError(f"Past predictor cache audit mismatch for {key}.")
    if int(predictor_audit.get("event_overlap", -1)) != 0:
        raise ValueError("Past predictor cache audit has event overlap.")
    expected_event_sha = hash_strings(sorted(set(train_event_ids)))
    if checkpoint.get("train_event_sha256") != expected_event_sha:
        raise ValueError("Past predictor train event SHA differs from this cache.")
    evidence = {
        "checkpoint": str(path),
        "checkpoint_sha256": cache_runner.sha256_file(path),
        "model_state_sha256": observed_state_sha,
        "train_event_sha256": expected_event_sha,
        "source_script_version": checkpoint["script_version"],
    }
    return (
        {key: value.detach().cpu() for key, value in state.items()},
        source_config,
        evidence,
    )


def run_experiment(args: argparse.Namespace) -> None:
    train_path = Path(args.train_cache).expanduser().resolve()
    validation_path = Path(args.val_cache).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Both train and validation caches must exist.")
    cache_runner.assert_not_sealed_path(train_path, purpose="train cache")
    cache_runner.assert_not_sealed_path(
        validation_path, purpose="validation cache"
    )
    cache_runner.assert_not_sealed_path(
        output_dir, purpose="correspondence output"
    )
    if (output_dir / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir}/summary.json exists; pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_runner.atomic_json_write(
        output_dir / "run_status.json",
        {
            "status": "running",
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "script_version": SCRIPT_VERSION,
            "sealed_test_read": False,
        },
    )
    try:
        train_cache, validation_cache, cache_audit = (
            cache_runner.load_cache_pair(train_path, validation_path)
        )
        _cache_integrity_checks(
            train_cache, validation_cache, cache_audit
        )
        train_indices = cache_runner.select_usable_rows(train_cache)
        validation_indices = cache_runner.select_usable_rows(validation_cache)
        train_data = cache_runner.take_rows(train_cache, train_indices)
        validation_data = cache_runner.take_rows(
            validation_cache, validation_indices
        )
        t0_index = int(train_cache["t0_index"])
        role_index = train_cache["role_index"].long()
        fractions = parse_fractions(args.labeled_event_fractions)
        tags = [fraction_tag(fraction) for fraction in fractions]
        if len(tags) != len(set(tags)):
            raise ValueError("Labeled fractions produce duplicate output tags.")
        device = resolve_device(args.device)
        args.resolved_device = device
        periods = parse_float_tuple(args.delta_periods)
        model_config = normalized_model_config(
            {
                "feature_dim": int(train_data["features"].shape[-1]),
                "num_roles": int(train_data["features"].shape[1]),
                "model_dim": int(args.model_dim),
                "num_heads": int(args.num_heads),
                "depth": 2,
                "mlp_ratio": float(args.mlp_ratio),
                "dropout": float(args.dropout),
                "periods_days": periods,
                "t0_index": t0_index,
            }
        )
        set_reproducible_seed(args.seed)
        prototype = build_model(model_config)
        downstream_initial_state = {
            key: value.detach().cpu().clone()
            for key, value in prototype.state_dict().items()
        }
        initial_state_sha = cache_runner.state_dict_sha256(
            downstream_initial_state
        )
        parameter_signature = cache_runner.model_parameter_signature(
            prototype
        )
        del prototype

        # This function has no methane-label argument.  It copies only feature,
        # mask, time, event, and cache-row provenance into the pretext view.
        unlabeled_view = extract_unlabeled_correspondence_view(
            train_data,
            cache_row_indices=train_indices,
            t0_index=t0_index,
        )
        pretext = build_correspondence_pretext(
            unlabeled_view, seed=int(args.seed) + 100_003
        )
        permuted_labels, permutation_audit = (
            build_permuted_pretext_labels(
                pretext["labels"], seed=int(args.seed) + 200_003
            )
        )
        donor_events = [
            unlabeled_view["event_ids"][index]
            for index in pretext["donor_view_indices"].tolist()
        ]
        for target_event, donor_event in zip(
            unlabeled_view["event_ids"], donor_events
        ):
            if target_event == donor_event:
                raise RuntimeError("Pretext donor event exclusion failed.")
        pretext_audit = {
            "forward_contract_version": FORWARD_CONTRACT_VERSION,
            "role_only_enable_delta": False,
            "t0_excluded_from_attention_keys_values": True,
            "methane_labels_used": False,
            "methane_labels_used_by_pretext_objective": False,
            "label_access_boundary": (
                "cache loader validates stored labels; the pretext view and "
                "pretext BCE receive no methane class labels"
            ),
            "source_usable_rows": int(
                unlabeled_view["source_usable_rows"]
            ),
            "eligible_rows": int(unlabeled_view["eligible_rows"]),
            "positive_rows": int(pretext["positive_rows"]),
            "negative_rows": int(pretext["negative_rows"]),
            "balanced": True,
            "input_sha256": pretext["input_sha256"],
            "donor_sha256": pretext["donor_sha256"],
            "coherent_label_sha256": pretext["label_sha256"],
            "permutation_control": permutation_audit,
            "donor_is_bijective": True,
            "donor_is_cross_canonical_event": True,
            "whole_history_replaced": True,
        }
        train_cache_audit = {
            "train_cache": cache_audit["train_cache"],
            "train_cache_sha256": cache_audit["train_cache_sha256"],
            "weights_sha256": cache_audit["weights_sha256"],
            "train_usable_rows": int(train_indices.numel()),
            "train_events": len(set(train_data["event_ids"])),
            "train_event_sha256": hash_strings(
                sorted(set(train_data["event_ids"]))
            ),
        }

        external_source = None
        external_config = None
        external_evidence = None
        active_arms = list(BASE_ARMS)
        if args.past_predictor_checkpoint:
            (
                external_source,
                external_config,
                external_evidence,
            ) = load_past_predictor_source(
                Path(args.past_predictor_checkpoint),
                cache_audit=cache_audit,
                train_event_ids=train_data["event_ids"],
                target_model_config=model_config,
            )
            active_arms.append(PAST_PREDICTION_ARM)

        run_config = {
            "script_version": SCRIPT_VERSION,
            "forward_contract_version": FORWARD_CONTRACT_VERSION,
            "arms": active_arms,
            "base_preregistered_arms": list(BASE_ARMS),
            "optional_past_prediction_arm_enabled": bool(
                args.past_predictor_checkpoint
            ),
            "seed": int(args.seed),
            "labeled_event_fractions": list(fractions),
            "model_config": model_config,
            "parameter_signature": parameter_signature,
            "initial_state_sha256": initial_state_sha,
            "classifier_state_keys": sorted(CLASSIFIER_STATE_KEYS),
            "temporal_transfer_state_keys": sorted(
                set(downstream_initial_state) - set(CLASSIFIER_STATE_KEYS)
            ),
            "pretext_audit": pretext_audit,
            "cache_audit": cache_audit,
            "external_past_predictor": external_evidence,
            "args": serializable_args(args),
            "matched_compute_contract": {
                "same_model_parameter_shapes": True,
                "same_downstream_classifier_initialization": True,
                "same_labeled_event_membership_within_fraction": True,
                "same_supervised_batches_and_optimizer_steps": True,
                "same_supervised_optimizer": True,
                "same_coherent_validation_selection": True,
                "history_shuffle_evaluated_only_after_selection": True,
                "role_only_enable_delta": False,
            },
            "sealed_test_read": False,
        }
        cache_runner.atomic_json_write(
            output_dir / "run_config.json", run_config
        )

        pretraining_states: dict[str, dict[str, torch.Tensor]] = {}
        pretraining_results: dict[str, dict[str, Any]] = {}
        pretraining_specs = (
            ("correspondence_pretrained", pretext["labels"]),
            ("permuted_pretext_labels_control", permuted_labels),
        )
        for arm, targets in pretraining_specs:
            state, result = fit_pretext_model(
                initial_state=downstream_initial_state,
                model_config=model_config,
                pretext=pretext,
                targets=targets,
                epochs=args.pretrain_epochs,
                batch_size=args.pretrain_batch_size,
                learning_rate=args.pretrain_learning_rate,
                weight_decay=args.pretrain_weight_decay,
                grad_clip=args.grad_clip,
                max_train_steps=args.max_pretrain_steps,
                seed=args.seed,
                device=device,
            )
            pretraining_states[arm] = state
            arm_dir = output_dir / "pretraining" / arm
            checkpoint_path = arm_dir / PRETRAIN_CHECKPOINT_NAME
            checkpoint = _pretrain_checkpoint(
                arm=arm,
                state=state,
                initial_state_sha256=initial_state_sha,
                parameter_signature=parameter_signature,
                model_config=model_config,
                summary=result,
                pretext_audit={
                    **pretext_audit,
                    "training_target_sha256": cache_runner.tensor_sha256(
                        targets
                    ),
                    "is_permuted_label_control": (
                        arm == "permuted_pretext_labels_control"
                    ),
                },
                train_cache_audit=train_cache_audit,
                args=args,
            )
            cache_runner.atomic_torch_save(checkpoint_path, checkpoint)
            pretraining_results[arm] = {
                **result,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": cache_runner.sha256_file(
                    checkpoint_path
                ),
                "methane_labels_used": False,
            }
        matched_pretrain = {
            (
                result["optimizer_steps"],
                result["train_rows_seen"],
                result["batch_plan_sha256"],
            )
            for result in pretraining_results.values()
        }
        if len(matched_pretrain) != 1:
            raise RuntimeError("Pretext/control compute is not matched.")

        validation_donors = build_bijective_cross_event_donors(
            validation_data["event_ids"],
            seed=int(args.seed) + 300_007,
        )
        shuffled_validation_data = build_shuffled_history_data(
            validation_data,
            donors=validation_donors,
            t0_index=t0_index,
        )
        shuffled_source_events = [
            validation_data["event_ids"][index]
            for index in validation_donors.tolist()
        ]
        for target_event, donor_event in zip(
            validation_data["event_ids"], shuffled_source_events
        ):
            if target_event == donor_event:
                raise RuntimeError("Validation history donor exclusion failed.")
        validation_shuffle_audit = {
            "seed": int(args.seed) + 300_007,
            "donor_sha256": hash_integer_tensor(validation_donors),
            "bijective": True,
            "cross_canonical_event": True,
            "whole_history_replaced": True,
            "used_for_checkpoint_selection": False,
        }

        results: dict[str, dict[str, Any]] = {}
        selection_audits: dict[str, dict[str, Any]] = {}
        for fraction in fractions:
            tag = fraction_tag(fraction)
            selected_indices, selection_audit = select_labeled_event_rows(
                train_data["labels"],
                train_data["event_ids"],
                fraction=fraction,
                seed=args.seed,
            )
            selection_audits[tag] = selection_audit
            supervised_data = subset_rows(train_data, selected_indices)
            arm_results: dict[str, Any] = {}
            initialization_by_arm: dict[
                str, tuple[dict[str, torch.Tensor], dict[str, Any]]
            ] = {}
            scratch_state = {
                key: value.clone()
                for key, value in downstream_initial_state.items()
            }
            initialization_by_arm["scratch"] = (
                scratch_state,
                {
                    "source_name": "scratch",
                    "transferred_state_keys": [],
                    "nontransferred_state_keys": sorted(
                        downstream_initial_state
                    ),
                    "classifier_sha256_before": classifier_state_sha256(
                        downstream_initial_state
                    ),
                    "classifier_sha256_after": classifier_state_sha256(
                        downstream_initial_state
                    ),
                },
            )
            for arm in (
                "correspondence_pretrained",
                "permuted_pretext_labels_control",
            ):
                transfer_model = build_model(model_config)
                transfer_audit = transfer_temporal_encoder_state(
                    transfer_model,
                    source_state=pretraining_states[arm],
                    downstream_initial_state=downstream_initial_state,
                    source_model_config=model_config,
                    target_model_config=model_config,
                    source_name=arm,
                    allowed_source_only_keys=CLASSIFIER_STATE_KEYS,
                )
                initialization_by_arm[arm] = (
                    {
                        key: value.detach().cpu().clone()
                        for key, value in transfer_model.state_dict().items()
                    },
                    transfer_audit,
                )
            if external_source is not None and external_config is not None:
                transfer_model = build_model(model_config)
                transfer_audit = transfer_temporal_encoder_state(
                    transfer_model,
                    source_state=external_source,
                    downstream_initial_state=downstream_initial_state,
                    source_model_config=external_config,
                    target_model_config=model_config,
                    source_name=PAST_PREDICTION_ARM,
                    allowed_source_only_keys=PAST_PREDICTOR_ONLY_KEYS,
                )
                transfer_audit["external_checkpoint"] = external_evidence
                initialization_by_arm[PAST_PREDICTION_ARM] = (
                    {
                        key: value.detach().cpu().clone()
                        for key, value in transfer_model.state_dict().items()
                    },
                    transfer_audit,
                )

            classifier_initial_shas = {
                classifier_state_sha256(state)
                for state, _ in initialization_by_arm.values()
            }
            if len(classifier_initial_shas) != 1:
                raise RuntimeError(
                    "Downstream classifier reset differs across arms."
                )
            signatures = []
            for arm in active_arms:
                model = build_model(model_config)
                model.load_state_dict(
                    initialization_by_arm[arm][0], strict=True
                )
                signatures.append(
                    cache_runner.model_parameter_signature(model)
                )
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise RuntimeError("Downstream parameter signatures differ.")

            for arm in active_arms:
                starting_state, transfer_audit = initialization_by_arm[arm]
                (
                    record,
                    best_state,
                    coherent_probability,
                    shuffled_probability,
                ) = train_downstream(
                    arm=arm,
                    starting_state=starting_state,
                    model_config=model_config,
                    train_data=supervised_data,
                    validation_data=validation_data,
                    shuffled_validation_data=shuffled_validation_data,
                    role_index=role_index,
                    epochs=args.supervised_epochs,
                    batch_size=args.supervised_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    learning_rate=args.supervised_learning_rate,
                    weight_decay=args.supervised_weight_decay,
                    grad_clip=args.grad_clip,
                    max_train_steps=args.max_supervised_steps,
                    seed=args.seed,
                    device=device,
                )
                arm_dir = output_dir / tag / arm
                checkpoint_path = arm_dir / CHECKPOINT_NAME
                coherent_path = arm_dir / COHERENT_PREDICTION_NAME
                shuffled_path = arm_dir / SHUFFLED_PREDICTION_NAME
                checkpoint = {
                    "script_version": SCRIPT_VERSION,
                    "arm": arm,
                    "labeled_event_fraction": float(fraction),
                    "labeled_event_selection": selection_audit,
                    "epoch": int(record["best_epoch"]),
                    "model": best_state,
                    "model_state_sha256": cache_runner.state_dict_sha256(
                        best_state
                    ),
                    "model_parameter_signature": parameter_signature,
                    "model_config": model_config,
                    "downstream_initial_state_sha256": initial_state_sha,
                    "state_transfer_audit": transfer_audit,
                    "coherent_validation": record[
                        "coherent_validation"
                    ],
                    "history_shuffled_validation": record[
                        "history_shuffled_validation"
                    ],
                    "supervised_training_history": record["history"],
                    "supervised_optimizer_steps": record[
                        "supervised_optimizer_steps"
                    ],
                    "supervised_train_rows_seen": record[
                        "supervised_train_rows_seen"
                    ],
                    "supervised_batch_plan_sha256": record[
                        "supervised_batch_plan_sha256"
                    ],
                    "selection_metric": "coherent_validation_ap",
                    "cache_audit": cache_audit,
                    "validation_shuffle_audit": validation_shuffle_audit,
                    "sealed_test_read": False,
                    "args": serializable_args(args),
                }
                cache_runner.atomic_torch_save(
                    checkpoint_path, checkpoint
                )
                cache_runner.atomic_csv_write(
                    coherent_path,
                    prediction_frame(
                        data=validation_data,
                        probabilities=coherent_probability,
                        arm=arm,
                        fraction=fraction,
                        epoch=record["best_epoch"],
                        evaluation_condition="coherent_history",
                        history_source_event_ids=validation_data[
                            "event_ids"
                        ],
                    ),
                )
                cache_runner.atomic_csv_write(
                    shuffled_path,
                    prediction_frame(
                        data=validation_data,
                        probabilities=shuffled_probability,
                        arm=arm,
                        fraction=fraction,
                        epoch=record["best_epoch"],
                        evaluation_condition=(
                            "cross_event_shuffled_history"
                        ),
                        history_source_event_ids=shuffled_source_events,
                    ),
                )
                arm_results[arm] = {
                    **record,
                    "parameter_signature_sha256": parameter_signature[
                        "shape_sha256"
                    ],
                    "state_transfer_audit": transfer_audit,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": cache_runner.sha256_file(
                        checkpoint_path
                    ),
                    "coherent_predictions": str(coherent_path),
                    "coherent_predictions_sha256": cache_runner.sha256_file(
                        coherent_path
                    ),
                    "history_shuffled_predictions": str(shuffled_path),
                    "history_shuffled_predictions_sha256": (
                        cache_runner.sha256_file(shuffled_path)
                    ),
                }
                cache_runner.atomic_json_write(
                    output_dir / "partial_results.json",
                    {
                        **results,
                        tag: arm_results,
                    },
                )
                print(
                    f"[correspondence] fraction={fraction:.3f} arm={arm} "
                    f"epoch={record['best_epoch']} "
                    f"AP={record['coherent_validation']['ap']:.6f} "
                    "shuffle_AP="
                    f"{record['history_shuffled_validation']['ap']:.6f}",
                    flush=True,
                )
            compute_signatures = {
                (
                    result["supervised_optimizer_steps"],
                    result["supervised_train_rows_seen"],
                    result["supervised_batch_plan_sha256"],
                    result["parameter_signature_sha256"],
                )
                for result in arm_results.values()
            }
            if len(compute_signatures) != 1:
                raise RuntimeError(
                    f"{tag}: supervised arm compute/signatures differ."
                )
            results[tag] = arm_results

        _assert_cache_files_unchanged(cache_audit)
        summary = {
            "script_version": SCRIPT_VERSION,
            "arms": active_arms,
            "pretraining": pretraining_results,
            "labeled_event_selection": selection_audits,
            "results": results,
            "validation_shuffle_audit": validation_shuffle_audit,
            "cache_audit": cache_audit,
            "initial_state_sha256": initial_state_sha,
            "parameter_signature": parameter_signature,
            "selection_metric": "coherent_validation_ap",
            "history_shuffle_used_for_selection": False,
            "pretext_methane_labels_used": False,
            "pretext_methane_labels_used_by_objective": False,
            "label_access_boundary": (
                "cache validation reads stored labels; correspondence example "
                "construction and pretext BCE do not"
            ),
            "role_only_enable_delta": False,
            "sealed_test_read": False,
        }
        cache_runner.atomic_json_write(
            output_dir / "summary.json", summary
        )
        cache_runner.atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "complete",
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "script_version": SCRIPT_VERSION,
                "arms": active_arms,
                "labeled_event_fractions": list(fractions),
                "sealed_test_read": False,
            },
        )
    except Exception as error:
        cache_runner.atomic_json_write(
            output_dir / "run_status.json",
            {
                "status": "failed",
                "failed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "script_version": SCRIPT_VERSION,
                "error_type": type(error).__name__,
                "error": str(error),
                "sealed_test_read": False,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Label-free coherent-vs-cross-event temporal correspondence "
            "pretraining and matched downstream L89 classification."
        )
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labeled-event-fractions", default="0.1,1.0")
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--pretrain-weight-decay", type=float, default=0.05)
    parser.add_argument("--max-pretrain-steps", type=int, default=0)
    parser.add_argument("--supervised-epochs", type=int, default=3)
    parser.add_argument("--supervised-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--supervised-learning-rate", type=float, default=3e-4
    )
    parser.add_argument(
        "--supervised-weight-decay", type=float, default=0.05
    )
    parser.add_argument("--max-supervised-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--delta-periods", default="1,3,7,30,90,365")
    parser.add_argument(
        "--past-predictor-checkpoint",
        help=(
            "Optional train-only l89-innovation final_all_train predictor; "
            "strictly transfers only common temporal-trunk keys."
        ),
    )
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(handler=run_experiment)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
