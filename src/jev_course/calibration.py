"""Independent temperature calibration, held-out metrics and reload-and-predict.

Nothing here trains weights or changes labels. ``fit`` reads only the calibration
file it is handed and records that file's digest, so an auditor can confirm the
test split was never in the fit; ``test`` reads the temperature back and reports
the raw and calibrated held-out numbers side by side. A temperature is a
post-processing parameter, so the split used to fit it must stay separate from the
split used to judge it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jevlike.data import ChoiceExample, JsonlDataset
from jevlike.eval import metrics as upstream_metrics
from jevlike.model import load_checkpoint, select_device
from jevlike.train import move

from .dataset import describe, load_examples

BATCH_SIZE = 64
FIT_GRID = 81
FIT_RANGE = 0.7


@dataclass(frozen=True)
class Scored:
    """One held-out row: a float64 score per option plus the correct index."""

    logits: torch.Tensor
    label: int


def validate_temperature(value: object, source: str) -> float:
    """Accept a finite positive real number and nothing else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"temperature in {source} must be a number, got {value!r}")
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"temperature in {source} must be finite and positive, got {temperature!r}"
        )
    return temperature


def validate_batch_size(value: object) -> int:
    """Accept a positive integer and nothing else.

    Bools are integers in Python, so ``True`` would silently score one row per
    batch; it is refused explicitly rather than coerced.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"batch size must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"batch size must be positive, got {value!r}")
    return value


def read_temperature(path: str | Path) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "temperature" not in payload:
        raise ValueError(f"{path} must be a JSON object with a 'temperature' key")
    return validate_temperature(payload["temperature"], str(path))


def validate_options(options: object) -> tuple[str, ...]:
    """Options must be at least two distinct non-empty strings."""
    names = tuple(options)  # type: ignore[arg-type]
    if len(names) < 2:
        raise ValueError(f"need at least two options, got {len(names)}")
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"every option must be a non-empty string, got {name!r}")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"options must be distinct, got repeats {repeated}; "
            "duplicate options would silently collapse into one probability"
        )
    return names


def load_scorer(checkpoint: str | Path, device_name: str = "cpu"):
    """Reload a checkpoint from disk and put it in eval mode."""
    device = select_device(device_name)
    model, collator, config = load_checkpoint(checkpoint, device)
    model.eval()
    return model, collator, config, device


def score_rows(
    checkpoint_model,
    collator,
    path: str | Path,
    device,
    batch_size: int = BATCH_SIZE,
) -> list[Scored]:
    """Score every row of one split; the rows are validated before use.

    ``batch_size`` controls only how many rows the loader hands over at a time; the
    rows and their order are the same for every value.
    """
    batch_size = validate_batch_size(batch_size)
    scored: list[Scored] = []
    loader = DataLoader(
        load_examples(path), batch_size=batch_size, collate_fn=collator
    )
    with torch.no_grad():
        for host_batch in loader:
            batch = move(host_batch, device)
            scores = checkpoint_model(batch)
            for row, mask, label in zip(scores, batch["option_mask"], batch["labels"]):
                scored.append(Scored(row[mask].detach().cpu().double(), int(label)))
    if not scored:
        raise ValueError(f"no rows scored from {path}")
    return scored


def score_menu(model, collator, context: str, options: tuple[str, ...], device) -> torch.Tensor:
    """Score one menu of distinct options for a single context."""
    options = validate_options(options)
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string")
    batch = move(collator([ChoiceExample(context, options, 0)]), device)
    with torch.no_grad():
        logits = model(batch)[0, : len(options)].cpu().double()
    return logits


def confidence_bins(confidences: list[float]) -> list[list[int]]:
    """Ten equal-width bins with unambiguous ownership.

    Bin index is ``floor(p * 10)`` clamped to 9, so every row lands in exactly one
    bin and a confidence of exactly 1.0 is counted in the top bin.
    """
    bins: list[list[int]] = [[] for _ in range(10)]
    for index, confidence in enumerate(confidences):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence {confidence!r} outside [0, 1]")
        bins[min(9, math.floor(confidence * 10.0))].append(index)
    return bins


def metrics(scored: list[Scored], temperature: float) -> dict[str, float | int]:
    """Accuracy, NLL, Brier sum and 10-bin ECE at the given temperature."""
    temperature = validate_temperature(temperature, "metrics()")
    if not scored:
        raise ValueError("no rows to score")
    total_nll = total_brier = 0.0
    confidences: list[float] = []
    correct: list[bool] = []
    for row in scored:
        log_probabilities = torch.log_softmax(row.logits / temperature, dim=-1)
        probabilities = log_probabilities.exp()
        target = torch.zeros_like(probabilities)
        target[row.label] = 1
        total_nll -= float(log_probabilities[row.label])
        total_brier += float(((probabilities - target) ** 2).sum())
        confidences.append(float(probabilities.max()))
        correct.append(int(probabilities.argmax()) == row.label)
    count = len(scored)
    bins = confidence_bins(confidences)
    ece = 0.0
    for members in bins:
        if members:
            accuracy = sum(correct[index] for index in members) / len(members)
            mean_confidence = sum(confidences[index] for index in members) / len(members)
            ece += len(members) / count * abs(accuracy - mean_confidence)
    return {
        "examples": count,
        "accuracy": sum(correct) / count,
        "nll": total_nll / count,
        "brier_sum": total_brier / count,
        "ece_10bins": ece,
        "bins_used": sum(1 for members in bins if members),
        "binned_examples": sum(len(members) for members in bins),
    }


def fit_temperature(scored: list[Scored]) -> float:
    """Deterministic finite grid search over temperatures, always including 1.0.

    Keeping 1.0 in the candidate set makes the fitted calibration NLL on the
    calibration split no worse than the raw one. That guarantee is about the
    calibration split only and says nothing about the test split.
    """
    grid = torch.logspace(-FIT_RANGE, FIT_RANGE, FIT_GRID, dtype=torch.float64).tolist()
    candidates = sorted({1.0, *grid})
    return min(candidates, key=lambda value: metrics(scored, value)["nll"])


def fit_report(
    scorer,
    collator,
    calibration_path: str | Path,
    device,
    batch_size: int = BATCH_SIZE,
) -> dict[str, object]:
    """Fit on the calibration split and describe what the fit saw."""
    batch_size = validate_batch_size(batch_size)
    rows = load_examples(calibration_path)
    scored = score_rows(scorer, collator, calibration_path, device, batch_size)
    temperature = validate_temperature(fit_temperature(scored), "fitted grid")
    before = metrics(scored, 1.0)
    after = metrics(scored, temperature)
    if after["nll"] > before["nll"]:
        raise ValueError(
            "fitted temperature is worse than the identity on its own split: "
            f"{after['nll']} > {before['nll']}"
        )
    return {
        "temperature": temperature,
        "batch_size": batch_size,
        "fitted_on": describe(calibration_path, rows),
        "candidates": {"grid": FIT_GRID, "range_log10": FIT_RANGE, "includes_identity": True},
        "calibration_before": before,
        "calibration_after": after,
    }


def held_out_report(
    scorer,
    collator,
    test_path: str | Path,
    temperature: float,
    device,
    batch_size: int = BATCH_SIZE,
) -> dict[str, object]:
    """Report raw and calibrated numbers on the held-out split, plus a control."""
    batch_size = validate_batch_size(batch_size)
    rows = load_examples(test_path)
    scored = score_rows(scorer, collator, test_path, device, batch_size)
    loader = DataLoader(
        JsonlDataset(test_path), batch_size=batch_size, collate_fn=collator
    )
    return {
        "temperature": validate_temperature(temperature, "held-out report"),
        "batch_size": batch_size,
        "test": describe(test_path, rows),
        "test_before": metrics(scored, 1.0),
        "test_after": metrics(scored, temperature),
        "shuffled_context_control": upstream_metrics(scorer, loader, device, True),
        "raw_context": upstream_metrics(scorer, loader, device, False),
    }


def predict_report(
    scorer,
    collator,
    context: str,
    options: tuple[str, ...],
    temperature: float,
    device,
    permute_check: bool = False,
) -> dict[str, object]:
    """Reload-style single-menu prediction, optionally with an order control."""
    temperature = validate_temperature(temperature, "predict")
    options = validate_options(options)
    logits = score_menu(scorer, collator, context, options, device)
    probabilities = (logits / temperature).softmax(-1).tolist()
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    report: dict[str, object] = {
        "context": context,
        "options": list(options),
        "temperature": temperature,
        "choice": options[best],
        "probabilities": {
            option: probability for option, probability in zip(options, probabilities)
        },
        "probability_sum": sum(probabilities),
        "options_unique": len(set(options)) == len(options),
    }
    if permute_check:
        report["permutation_check"] = permuted_choice_report(
            scorer, collator, context, options, temperature, device, report
        )
    return report


def permuted_choice_report(
    scorer,
    collator,
    context: str,
    options: tuple[str, ...],
    temperature: float,
    device,
    baseline: dict[str, object],
) -> dict[str, object]:
    """Re-score the same menu in reverse order and compare by option text."""
    reversed_options = tuple(reversed(options))
    logits = score_menu(scorer, collator, context, reversed_options, device)
    probabilities = (logits / temperature).softmax(-1)
    table = {
        option: float(probability)
        for option, probability in zip(reversed_options, probabilities.tolist())
    }
    choice = max(reversed_options, key=table.__getitem__)
    baseline_probabilities = baseline["probabilities"]  # type: ignore[assignment]
    largest_drift = max(
        abs(float(table[option]) - float(baseline_probabilities[option]))  # type: ignore[index]
        for option in options
    )
    ranked = sorted(table.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    if choice != baseline["choice"] and margin > 1e-9:
        raise ValueError(
            "option order changed the chosen option: "
            f"{baseline['choice']!r} -> {choice!r} (margin {margin:.3g})"
        )
    return {
        "choice": choice,
        "stable": choice == baseline["choice"],
        "top_gap": margin,
        "max_probability_drift": largest_drift,
    }


def dump(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
