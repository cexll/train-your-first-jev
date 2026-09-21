"""Executable checks for the properties this course claims.

Each check fails loudly with the observed values. They cover the vendored fixes
(snapshot copy, safe loader, deterministic option order), split hygiene, the
temperature contract and bin ownership; ``jev-course check`` runs all of them and
exits non-zero if any fail. No training run is needed: the checks use a fresh tiny
scorer and the real generator at miniature sizes.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from jevlike.data import write_synthetic
from jevlike.model import TinyScorer, load_checkpoint, select_device, trainable_state

from .calibration import (
    Scored,
    confidence_bins,
    metrics,
    read_temperature,
    validate_options,
    validate_temperature,
)
from .dataset import digest, load_examples, load_splits, make_synthetic_splits, overlap_report

CONFIG = {
    "encoder": "tiny",
    "hf_model": "unused",
    "width": 8,
    "rank": 8,
    "context_tokens": 16,
    "option_tokens": 8,
}


def check_snapshot_is_independent() -> str:
    """``trainable_state`` must copy, so later epochs cannot rewrite a saved best."""
    torch.manual_seed(3)
    model = TinyScorer(CONFIG["width"], CONFIG["rank"], CONFIG["context_tokens"])
    snapshot = trainable_state(model)
    before = {name: tensor.clone() for name, tensor in snapshot.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    changed = [name for name, tensor in snapshot.items() if not torch.equal(tensor, before[name])]
    if changed:
        raise AssertionError(
            f"snapshot shares storage with live parameters: {sorted(changed)[:3]}"
        )
    if not snapshot:
        raise AssertionError("no trainable state was captured")
    return f"{len(snapshot)} tensors survive in-place parameter updates"


def _marker_side_effect(path: str) -> str:
    Path(path).write_text("executed", encoding="utf-8")
    return path


class _HostilePayload:
    """A checkpoint-like object whose unpickling would run code."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (_marker_side_effect, (self.marker,))


def check_safe_loader() -> str:
    """Own checkpoints load; a pickle payload is refused before it can run code."""
    device = select_device("cpu")
    with tempfile.TemporaryDirectory() as workspace:
        workspace_path = Path(workspace)
        good = workspace_path / "good.pt"
        torch.manual_seed(4)
        model = TinyScorer(CONFIG["width"], CONFIG["rank"], CONFIG["context_tokens"])
        torch.save({"config": CONFIG, "state_dict": trainable_state(model)}, good)
        reloaded, _collator, config = load_checkpoint(good, device)
        if config != CONFIG or not torch.equal(next(iter(trainable_state(reloaded).values())),
                                              next(iter(trainable_state(model).values()))):
            raise AssertionError("a checkpoint written by this course did not round-trip")

        marker = workspace_path / "marker.txt"
        hostile = workspace_path / "hostile.pt"
        torch.save({"config": CONFIG, "state_dict": {}, "extra": _HostilePayload(str(marker))}, hostile)
        try:
            load_checkpoint(hostile, device)
        except ValueError:
            pass
        else:
            raise AssertionError("load_checkpoint accepted a non-tensor pickle payload")
        if marker.exists():
            raise AssertionError("the hostile payload executed code before being refused")

        plain = workspace_path / "plain.pt"
        torch.save({"state_dict": {}}, plain)
        try:
            load_checkpoint(plain, device)
        except ValueError:
            pass
        else:
            raise AssertionError("load_checkpoint accepted a checkpoint without a config")
    return "own checkpoint round-trips; pickle payload refused without executing"


def _generate(workspace: Path, name: str, hash_seed: str) -> dict[str, str]:
    output = workspace / name
    command = [
        sys.executable, "-m", "jev_course.cli", "data",
        "--out", str(output),
        "--train", "8", "--validation", "4", "--calibration", "4", "--test", "4",
        "--seed", "5",
    ]
    environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise AssertionError(
            f"data generation failed under PYTHONHASHSEED={hash_seed}: {completed.stderr.strip()}"
        )
    return {path.name: digest(path) for path in sorted(output.glob("*.jsonl"))}


def check_generation_is_hash_seed_independent() -> str:
    """The same seed must produce byte-identical splits in any interpreter."""
    with tempfile.TemporaryDirectory() as workspace:
        workspace_path = Path(workspace)
        first = _generate(workspace_path, "one", "1")
        second = _generate(workspace_path, "two", "981273")
    if first != second:
        raise AssertionError(f"split digests differ across hash seeds: {first} != {second}")
    return f"{len(first)} split files identical across hash seeds"


def check_splits_are_disjoint() -> str:
    """Four non-empty splits with no shared context, at the course's default sizes."""
    sizes = {"train": 2000, "validation": 400, "calibration": 400, "test": 400}
    with tempfile.TemporaryDirectory() as workspace:
        workspace_path = Path(workspace)
        report = make_synthetic_splits(workspace_path, sizes, 700001)
        if not report["disjoint"] or any(report["shared_contexts"].values()):
            raise AssertionError(f"splits are not disjoint: {report['shared_contexts']}")
        for name, size in sizes.items():
            if report["splits"][name]["rows"] != size:
                raise AssertionError(
                    f"split {name} has {report['splits'][name]['rows']} rows, expected {size}"
                )
        raw = workspace_path / "raw"
        write_synthetic(raw, sizes, 700001)
        shared = sum(overlap_report(load_splits(raw)).values())
    return (
        "deduplicated writer shares 0 contexts at default sizes; "
        f"upstream write_synthetic for the same seed shares {shared} context(s)"
    )


def check_row_validation() -> str:
    """Duplicate options, bad labels and empty files are refused, not repaired."""
    with tempfile.TemporaryDirectory() as workspace:
        workspace_path = Path(workspace)
        cases = {
            "duplicate_options.jsonl": (
                '{"context":"pick one","options":["a","a"],"label":0}',
                "repeats option",
            ),
            "bad_label.jsonl": (
                '{"context":"pick one","options":["a","b"],"label":2}',
                "label must be an option index",
            ),
            "bool_label.jsonl": (
                '{"context":"pick one","options":["a","b"],"label":true}',
                "integer index",
            ),
            "blank_context.jsonl": (
                '{"context":"   ","options":["a","b"],"label":0}',
                "blank context",
            ),
            "blank_option.jsonl": (
                '{"context":"pick one","options":["a"," \\t"],"label":1}',
                "blank option",
            ),
            "single_option.jsonl": (
                '{"context":"pick one","options":["a"],"label":0}',
                "at least two",
            ),
            "broken_json.jsonl": (
                '{"context": "unterminated,',
                "not valid JSON",
            ),
            "non_object.jsonl": (
                '["not","an","object"]',
                "JSON object",
            ),
            "empty.jsonl": ("", "no examples"),
        }
        for name, (content, expected) in cases.items():
            path = workspace_path / name
            path.write_text(content + "\n" if content else "", encoding="utf-8")
            try:
                load_examples(path)
            except ValueError as error:
                if expected not in str(error):
                    raise AssertionError(f"{name} raised {error!r}, expected {expected!r}") from error
            else:
                raise AssertionError(f"{name} was accepted")
    for bad in (["a", "a"], ["a"], ["", "b"]):
        try:
            validate_options(bad)
        except ValueError:
            continue
        raise AssertionError(f"options {bad!r} were accepted")
    return f"{len(cases) + 3} invalid inputs refused with a stated reason"


def check_temperature_contract() -> str:
    """Only finite positive numbers pass, in code and in the saved JSON."""
    accepted = [1.0, 0.25, 7, 3.5e-2]
    for value in accepted:
        if validate_temperature(value, "check") != float(value):
            raise AssertionError(f"temperature {value!r} was not returned unchanged")
    refused = [0, -1, float("nan"), float("inf"), float("-inf"), "1.0", None, True, [1.0]]
    for value in refused:
        try:
            validate_temperature(value, "check")
        except ValueError:
            continue
        raise AssertionError(f"temperature {value!r} was accepted")
    with tempfile.TemporaryDirectory() as workspace:
        path = Path(workspace) / "temperature.json"
        for payload, reason in (
            ({"temperature": 0.0}, "zero"),
            ({"temperature": float("nan")}, "nan"),
            ({"temperature": "1.0"}, "string"),
            ({"value": 1.0}, "missing key"),
        ):
            path.write_text(json.dumps(payload), encoding="utf-8")
            try:
                read_temperature(path)
            except ValueError:
                continue
            raise AssertionError(f"temperature file with {reason} was accepted")
        path.write_text(json.dumps({"temperature": 1.25}), encoding="utf-8")
        if read_temperature(path) != 1.25:
            raise AssertionError("a valid temperature file was not read back")
    return f"{len(accepted)} values accepted, {len(refused) + 4} refused"


def check_bin_ownership() -> str:
    """Every row lands in exactly one bin; a confidence of exactly 1.0 is counted."""
    bins = confidence_bins([1.0, 0.999, 0.5, 0.0])
    if [bins[9], bins[5], bins[0]] != [[0, 1], [2], [3]]:
        raise AssertionError(f"unexpected bin ownership: {bins}")
    certain = Scored(torch.tensor([0.0, -1000.0], dtype=torch.float64), 0)
    report = metrics([certain], 1.0)
    if report["binned_examples"] != 1 or report["ece_10bins"] != 0.0:
        raise AssertionError(f"a certain row was dropped from ECE: {report}")
    flat = Scored(torch.tensor([0.0, 0.0], dtype=torch.float64), 1)
    tie = metrics([flat], 1.0)
    if abs(tie["nll"] - math.log(2.0)) > 1e-12:
        raise AssertionError(f"equal logits are not an even split: {tie}")
    if not math.isfinite(metrics([flat, certain], 0.5)["nll"]):
        raise AssertionError("calibrated NLL is not finite")
    return "bin index = floor(p*10) clamped to 9, all rows counted"


CHECKS = (
    ("snapshot_is_independent", check_snapshot_is_independent),
    ("safe_checkpoint_loader", check_safe_loader),
    ("generation_is_hash_seed_independent", check_generation_is_hash_seed_independent),
    ("splits_are_disjoint_and_nonempty", check_splits_are_disjoint),
    ("row_and_option_validation", check_row_validation),
    ("temperature_contract", check_temperature_contract),
    ("bin_ownership", check_bin_ownership),
)


def run_checks() -> dict[str, object]:
    results = []
    for name, function in CHECKS:
        try:
            detail = function()
        except Exception as error:  # report every failure, not just the first
            results.append({"name": name, "ok": False, "detail": f"{type(error).__name__}: {error}"})
        else:
            results.append({"name": name, "ok": True, "detail": detail})
    passed = sum(1 for item in results if item["ok"])
    return {
        "checks": results,
        "passed": passed,
        "failed": len(results) - passed,
        "ok": passed == len(results),
    }
