"""Train the minimal ordered five-frame GRU baseline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch  # type: ignore[import-not-found,unused-ignore]
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]
from torch import nn
from torch.nn import functional as functional  # type: ignore[import-not-found,unused-ignore]
from torch.utils.data import (  # type: ignore[import-not-found,unused-ignore]
    DataLoader,
    TensorDataset,
)

from driving_scene_data_loop.lr_baselines import BaselineWindowData

JsonObject = dict[str, object]
SEEDS = (17, 29, 43)
HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
# Warm starting from a converged checkpoint at the full rate would step straight
# off that minimum, so incremental fine-tuning drops to a tenth, as one protocol.
FINE_TUNE_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
MAX_EPOCHS = 50
PATIENCE = 5
GATE_B_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """One declared training protocol.

    The frozen v1 runs peaked on Development within 2 to 6 epochs while train
    loss fell to 0.14, and their Development curve swung about 0.036 between
    adjacent epochs -- an order of magnitude larger than the between-arm
    differences the experiment tries to measure. Because early stopping takes
    the argmax of that curve, the reported score partly measures which epoch
    happened to land high. These fields exist so a quieter protocol can be
    declared and compared, not so hyperparameters can be tuned for score.

    `smoothing_window` averages the last N Development scores before comparing,
    so a single lucky epoch cannot define the checkpoint. At 1 it is exactly the
    original argmax rule.
    """

    name: str
    hidden_size: int = HIDDEN_SIZE
    dropout: float = 0.2
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    smoothing_window: int = 1

    def to_json(self) -> JsonObject:
        return {
            "name": self.name,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "smoothing_window": self.smoothing_window,
        }


# Reproduces every run recorded before the training review, bit for bit.
FROZEN_CONFIG = TrainingConfig(name="v1-frozen")


class GruBaselineError(ValueError):
    """Raised when the GRU experiment cannot follow its frozen protocol."""


class GlobalGRU(nn.Module):  # type: ignore[misc,unused-ignore]
    """One ordered GRU followed by explicit dropout and a multi-label head."""

    def __init__(
        self,
        class_count: int,
        hidden_size: int = HIDDEN_SIZE,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=384,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, class_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(inputs)
        return self.classifier(  # type: ignore[no-any-return,unused-ignore]
            self.dropout(hidden[-1])
        )


@dataclass(frozen=True, slots=True)
class ApMetrics:
    """Threshold-free Development metrics for one prediction matrix."""

    average_precision: tuple[float, ...]
    prevalence: tuple[float, ...]

    @property
    def macro_average_precision(self) -> float:
        return float(np.mean(self.average_precision))

    def to_json(self, scenario_ids: tuple[str, ...]) -> JsonObject:
        return {
            "macro_average_precision": self.macro_average_precision,
            "classes": {
                scenario_id: {
                    "average_precision": self.average_precision[index],
                    "prevalence": self.prevalence[index],
                }
                for index, scenario_id in enumerate(scenario_ids)
            },
        }


@dataclass(frozen=True, slots=True)
class SeedRun:
    """One seed's report and Development predictions."""

    report: JsonObject
    probabilities: NDArray[np.float64]
    reversed_probabilities: NDArray[np.float64]


def build_sequence_features(
    frame_features: NDArray[np.float32],
    frame_indices: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Join five indexed frame features into `[window, time, feature]`."""

    if frame_indices.ndim != 2 or frame_indices.shape[1] != 5:
        raise GruBaselineError("frame indices must have shape [window_count,5]")
    if frame_indices.size == 0 or frame_indices.min() < 0:
        raise GruBaselineError("frame indices must not be empty or negative")
    if frame_indices.max() >= len(frame_features):
        raise GruBaselineError("frame index is outside the feature matrix")
    return np.asarray(frame_features[frame_indices], dtype=np.float32)


def calculate_pos_weight(
    labels: NDArray[np.int8],
    loss_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Compute `negative / positive` independently from valid L0 labels."""

    if labels.shape != loss_mask.shape or labels.ndim != 2:
        raise GruBaselineError("labels and loss_mask must have the same rank-two shape")
    weights: list[float] = []
    for class_index in range(labels.shape[1]):
        valid_labels = labels[loss_mask[:, class_index], class_index]
        positive = int(valid_labels.sum())
        negative = len(valid_labels) - positive
        if min(positive, negative) <= 0:
            raise GruBaselineError("each class needs positive and negative L0 labels")
        weights.append(negative / positive)
    return np.asarray(weights, dtype=np.float32)


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply BCE only to valid class labels in each window."""

    if logits.shape != targets.shape or logits.shape != loss_mask.shape:
        raise GruBaselineError("logits, targets, and loss_mask must share one shape")
    valid_count = loss_mask.sum()
    if int(valid_count.item()) == 0:
        raise GruBaselineError("a training batch has no valid labels")
    safe_targets = torch.where(loss_mask, targets, torch.zeros_like(targets))
    element_loss = functional.binary_cross_entropy_with_logits(
        logits,
        safe_targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    return (element_loss * loss_mask).sum() / valid_count


def calculate_ap_metrics(
    labels: NDArray[np.int8],
    loss_mask: NDArray[np.bool_],
    probabilities: NDArray[np.float64],
) -> ApMetrics:
    """Calculate per-class AP and prevalence on valid labels only."""

    if labels.shape != loss_mask.shape or labels.shape != probabilities.shape:
        raise GruBaselineError("labels, mask, and probabilities must share one shape")
    average_precisions: list[float] = []
    prevalences: list[float] = []
    for class_index in range(labels.shape[1]):
        valid = loss_mask[:, class_index]
        y_true = labels[valid, class_index]
        y_score = probabilities[valid, class_index]
        if len(y_true) == 0 or min(int(y_true.sum()), len(y_true) - int(y_true.sum())) <= 0:
            raise GruBaselineError("AP requires positive and negative valid labels")
        average_precisions.append(float(average_precision_score(y_true, y_score)))
        prevalences.append(float(y_true.mean()))
    return ApMetrics(tuple(average_precisions), tuple(prevalences))


def best_f1_threshold(
    labels: NDArray[np.int8],
    probabilities: NDArray[np.float64],
) -> tuple[float, float]:
    """Choose maximum F1; ascending candidates make the lowest tied threshold win."""

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.unique(probabilities):
        predicted = probabilities >= threshold
        true_positive = int(((labels == 1) & predicted).sum())
        false_positive = int(((labels == 0) & predicted).sum())
        false_negative = int(((labels == 1) & ~predicted).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = f1
    return best_threshold, best_f1


def train_gru_baselines(
    *,
    window_data: BaselineWindowData,
    frame_features: NDArray[np.float32],
    output_dir: Path,
    num_threads: int = 16,
    scenario_ids: tuple[str, ...] | None = None,
    run_name: str = "base",
    warm_start_dir: Path | None = None,
    feature_cache: JsonObject | None = None,
    config: TrainingConfig = FROZEN_CONFIG,
) -> JsonObject:
    """Train three normal-order seeds and diagnose reversed Development order.

    `warm_start_dir` selects the incremental fine-tuning protocol: each seed
    resumes from that run's own seed-matched checkpoint and trains at
    `FINE_TUNE_LEARNING_RATE`. The rate is not separately settable, because
    warm starting at the full rate is not fine-tuning.
    """

    if output_dir.exists():
        raise GruBaselineError("output directory must not already exist")
    if num_threads <= 0:
        raise GruBaselineError("num_threads must be positive")
    learning_rate = (
        config.learning_rate if warm_start_dir is None else FINE_TUNE_LEARNING_RATE
    )
    warm_start_paths: dict[int, Path] = {}
    if warm_start_dir is not None:
        for seed in SEEDS:
            path = warm_start_dir / f"gru_seed_{seed}.pt"
            if not path.is_file():
                raise GruBaselineError(f"warm start checkpoint is missing: {path}")
            warm_start_paths[seed] = path
    output_dir.mkdir(parents=True)
    torch.set_num_threads(num_threads)

    selected_scenarios = scenario_ids or window_data.scenario_ids
    if len(selected_scenarios) != len(set(selected_scenarios)):
        raise GruBaselineError("scenario IDs must be unique")
    try:
        class_indices = np.asarray(
            [window_data.scenario_ids.index(value) for value in selected_scenarios],
            dtype=np.int64,
        )
    except ValueError as error:
        raise GruBaselineError("requested scenario is not present in the window data") from error

    sequences = build_sequence_features(
        frame_features,
        window_data.frame_feature_indices,
    )
    l0_rows = np.asarray(
        [partition == "l0" for partition in window_data.partitions],
        dtype=np.bool_,
    )
    feedback_rows = np.asarray(
        [partition == "feedback" for partition in window_data.partitions],
        dtype=np.bool_,
    )
    train_rows = l0_rows | feedback_rows
    development_rows = np.asarray(
        [partition == "development" for partition in window_data.partitions],
        dtype=np.bool_,
    )
    if not train_rows.any() or not development_rows.any():
        raise GruBaselineError("both L0 and Development windows are required")

    train_sequences = sequences[train_rows]
    train_labels = window_data.labels[train_rows][:, class_indices]
    train_mask = window_data.loss_mask[train_rows][:, class_indices]
    development_sequences = sequences[development_rows]
    development_labels = window_data.labels[development_rows][:, class_indices]
    development_mask = window_data.loss_mask[development_rows][:, class_indices]
    l0_labels = window_data.labels[l0_rows][:, class_indices]
    l0_mask = window_data.loss_mask[l0_rows][:, class_indices]
    pos_weight = calculate_pos_weight(l0_labels, l0_mask)
    development_ids = tuple(
        window_id
        for window_id, is_development in zip(
            window_data.window_ids,
            development_rows,
            strict=True,
        )
        if is_development
    )
    seed_runs: dict[int, SeedRun] = {}
    for seed in SEEDS:
        seed_run = _train_one_seed(
            seed=seed,
            train_sequences=train_sequences,
            train_labels=train_labels,
            train_mask=train_mask,
            development_sequences=development_sequences,
            development_labels=development_labels,
            development_mask=development_mask,
            pos_weight=pos_weight,
            checkpoint_path=output_dir / f"gru_seed_{seed}.pt",
            warm_start_path=warm_start_paths.get(seed),
            learning_rate=learning_rate,
            config=config,
            scenario_ids=selected_scenarios,
        )
        seed_runs[seed] = seed_run
        _write_predictions(
            output_dir / f"gru_seed_{seed}_development_predictions.jsonl",
            development_ids,
            development_labels,
            development_mask,
            seed_run.probabilities,
            seed_run.reversed_probabilities,
        )

    normal_scores = np.asarray(
        [
            calculate_ap_metrics(
                development_labels,
                development_mask,
                seed_runs[seed].probabilities,
            ).average_precision
            for seed in SEEDS
        ],
        dtype=np.float64,
    )
    reversed_scores = np.asarray(
        [
            calculate_ap_metrics(
                development_labels,
                development_mask,
                seed_runs[seed].reversed_probabilities,
            ).average_precision
            for seed in SEEDS
        ],
        dtype=np.float64,
    )
    development_metrics = calculate_ap_metrics(
        development_labels,
        development_mask,
        seed_runs[SEEDS[0]].probabilities,
    )
    gate_b: dict[str, object] = {}
    retained_classes: list[str] = []
    for class_index, scenario_id in enumerate(selected_scenarios):
        mean_ap = float(normal_scores[:, class_index].mean())
        prevalence = development_metrics.prevalence[class_index]
        margin = mean_ap - prevalence
        passed = margin >= GATE_B_MARGIN
        if passed:
            retained_classes.append(scenario_id)
        gate_b[scenario_id] = {
            "mean_average_precision": mean_ap,
            "development_prevalence": prevalence,
            "margin": margin,
            "required_margin": GATE_B_MARGIN,
            "passed": passed,
        }

    base_thresholds: dict[str, object] = {}
    base_probabilities = seed_runs[17].probabilities
    for class_index, scenario_id in enumerate(selected_scenarios):
        valid = development_mask[:, class_index]
        threshold, f1 = best_f1_threshold(
            development_labels[valid, class_index],
            base_probabilities[valid, class_index],
        )
        base_thresholds[scenario_id] = {"threshold": threshold, "f1": f1}

    report: JsonObject = {
        "schema_version": "1.0",
        "model": "global_gru",
        "run_name": run_name,
        "feature_cache": feature_cache,
        "training_config": config.to_json(),
        "scenario_ids": list(selected_scenarios),
        "train_partitions": ["l0"] + (["feedback"] if feedback_rows.any() else []),
        "evaluation_partition": "development",
        "l0_window_count": int(l0_rows.sum()),
        "feedback_window_count": int(feedback_rows.sum()),
        "training_window_count": int(train_rows.sum()),
        "development_window_count": int(development_rows.sum()),
        "config": {
            "input_shape": [5, 384],
            "hidden_size": HIDDEN_SIZE,
            "layers": 1,
            "direction": "unidirectional",
            "dropout_after_final_hidden": 0.2,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "initialization": (
                "random" if warm_start_dir is None else f"warm_start:{warm_start_dir.name}"
            ),
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "seeds": list(SEEDS),
            "num_threads": num_threads,
            "checkpoint_metric": "development_macro_average_precision",
            "reversed_diagnostic": "same checkpoint, reversed Development time axis",
        },
        "pos_weight": pos_weight.tolist(),
        "pos_weight_source": "l0_only",
        "runs": {str(seed): seed_runs[seed].report for seed in SEEDS},
        "aggregate": {
            "normal_order": _aggregate_scores(
                normal_scores,
                selected_scenarios,
            ),
            "reversed_order": _aggregate_scores(
                reversed_scores,
                selected_scenarios,
            ),
        },
        "gate_b": gate_b,
        "gate_b_passed_classes": retained_classes,
        "loop_classes": list(selected_scenarios),
        "base_seed": 17,
        "base_thresholds": base_thresholds,
    }
    (output_dir / "gru_report.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _train_one_seed(
    *,
    seed: int,
    train_sequences: NDArray[np.float32],
    train_labels: NDArray[np.int8],
    train_mask: NDArray[np.bool_],
    development_sequences: NDArray[np.float32],
    development_labels: NDArray[np.int8],
    development_mask: NDArray[np.bool_],
    pos_weight: NDArray[np.float32],
    checkpoint_path: Path,
    scenario_ids: tuple[str, ...],
    warm_start_path: Path | None = None,
    learning_rate: float = LEARNING_RATE,
    config: TrainingConfig = FROZEN_CONFIG,
) -> SeedRun:
    _set_seed(seed)
    model = GlobalGRU(len(scenario_ids), config.hidden_size, config.dropout)
    if warm_start_path is not None:
        model.load_state_dict(
            torch.load(warm_start_path, map_location="cpu", weights_only=True)
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    train_inputs = torch.from_numpy(train_sequences)
    train_targets = torch.from_numpy(np.where(train_mask, train_labels, 0).astype(np.float32))
    train_masks = torch.from_numpy(train_mask)
    development_inputs = torch.from_numpy(development_sequences)
    torch_pos_weight = torch.from_numpy(pos_weight)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_inputs, train_targets, train_masks),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[JsonObject] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        valid_count = 0
        for batch_inputs, batch_targets, batch_masks in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = masked_bce_with_logits(
                logits,
                batch_targets,
                batch_masks,
                torch_pos_weight,
            )
            loss.backward()  # type: ignore[no-untyped-call,unused-ignore]
            optimizer.step()
            batch_valid = int(batch_masks.sum().item())
            loss_sum += float(loss.item()) * batch_valid
            valid_count += batch_valid

        probabilities = _predict(model, development_inputs, reversed_order=False)
        metrics = calculate_ap_metrics(
            development_labels,
            development_mask,
            probabilities,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / valid_count,
                "development_macro_average_precision": metrics.macro_average_precision,
            }
        )
        # Compare a short trailing mean rather than one epoch, so a single lucky
        # epoch on a noisy curve cannot define the checkpoint. At window 1 this
        # is exactly the original argmax rule.
        window = [
            float(cast(float, item["development_macro_average_precision"]))
            for item in history[-config.smoothing_window :]
        ]
        smoothed = sum(window) / len(window)
        if smoothed > best_score:
            best_score = smoothed
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    best_model = GlobalGRU(len(scenario_ids), config.hidden_size, config.dropout)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_model.load_state_dict(state)
    probabilities = _predict(best_model, development_inputs, reversed_order=False)
    reversed_probabilities = _predict(best_model, development_inputs, reversed_order=True)
    normal_metrics = calculate_ap_metrics(
        development_labels,
        development_mask,
        probabilities,
    )
    reversed_metrics = calculate_ap_metrics(
        development_labels,
        development_mask,
        reversed_probabilities,
    )
    # How unstable was the curve the checkpoint was chosen from, and what would
    # a selection-free estimate have been? Both quantify the measurement itself.
    curve = [
        float(cast(float, item["development_macro_average_precision"]))
        for item in history
    ]
    steps = [abs(curve[i] - curve[i - 1]) for i in range(1, len(curve))]
    return SeedRun(
        report={
            "best_epoch": best_epoch,
            "epochs_trained": len(history),
            "development_curve_mean_absolute_step": (
                sum(steps) / len(steps) if steps else 0.0
            ),
            "development_curve_range": (max(curve) - min(curve)) if curve else 0.0,
            "final_three_epoch_mean_ap": sum(curve[-3:]) / len(curve[-3:]),
            "normal_order": normal_metrics.to_json(scenario_ids),
            "reversed_order": reversed_metrics.to_json(scenario_ids),
            "history": history,
            "checkpoint": checkpoint_path.name,
        },
        probabilities=probabilities,
        reversed_probabilities=reversed_probabilities,
    )


def _predict(
    model: GlobalGRU,
    inputs: torch.Tensor,
    *,
    reversed_order: bool,
) -> NDArray[np.float64]:
    model.eval()
    model_inputs = torch.flip(inputs, dims=(1,)) if reversed_order else inputs
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(model_inputs)).cpu().numpy()
    return np.asarray(probabilities, dtype=np.float64)


def _aggregate_scores(
    scores: NDArray[np.float64],
    scenario_ids: tuple[str, ...],
) -> JsonObject:
    class_values = {
        scenario_id: {
            "mean_average_precision": float(scores[:, index].mean()),
            "standard_deviation": float(scores[:, index].std()),
        }
        for index, scenario_id in enumerate(scenario_ids)
    }
    macro_per_seed = scores.mean(axis=1)
    return {
        "macro_average_precision_mean": float(macro_per_seed.mean()),
        "macro_average_precision_standard_deviation": float(macro_per_seed.std()),
        "classes": class_values,
    }


def _write_predictions(
    path: Path,
    window_ids: tuple[str, ...],
    labels: NDArray[np.int8],
    loss_mask: NDArray[np.bool_],
    probabilities: NDArray[np.float64],
    reversed_probabilities: NDArray[np.float64],
) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row_index, window_id in enumerate(window_ids):
            destination.write(
                json.dumps(
                    {
                        "window_id": window_id,
                        "partition": "development",
                        "probabilities": probabilities[row_index].tolist(),
                        "reversed_probabilities": reversed_probabilities[row_index].tolist(),
                        "labels": [
                            int(label) if mask else None
                            for label, mask in zip(
                                labels[row_index],
                                loss_mask[row_index],
                                strict=True,
                            )
                        ],
                        "loss_mask": loss_mask[row_index].tolist(),
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
