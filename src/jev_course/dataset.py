"""Read JSONL splits, refuse ambiguous rows and prove the splits stay disjoint.

Row shape is decided by upstream ``jevlike.data.validate`` (string context, at
least two non-empty string options, in-range integer label); this module adds the
checks that decide whether a split may be used for training, calibration or the
final held-out exam at all.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jevlike.data import ChoiceExample, synthetic_example, validate

SPLITS = ("train", "validation", "calibration", "test")

# The step upstream's write_synthetic uses between consecutive example seeds.
STRIDE = 104729


def load_examples(path: str | Path) -> list[ChoiceExample]:
    """Return the validated rows of one split, or raise with the offending row.

    Upstream's ``validate`` accepts things a course must not silently tolerate, so
    this adds, with the row number in every error: malformed JSON, a non-object
    row, a boolean ``label`` (``True`` is an ``int`` in Python and upstream would
    treat it as option index 1), a blank context, whitespace-only options, and
    duplicate options inside a row (two identical candidates would silently share
    one probability and the row would no longer have a single correct answer).
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [(number, line) for number, line in enumerate(text.splitlines(), start=1)
             if line.strip()]
    if not lines:
        raise ValueError(f"no examples in {path}")
    rows: list[ChoiceExample] = []
    for number, line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{number} must be a JSON object per line")
        if isinstance(payload.get("label"), bool):
            raise ValueError(
                f"{path}:{number} label must be an integer index, not {payload['label']!r}"
            )
        row = validate(payload)
        if not row.context.strip():
            raise ValueError(f"{path}:{number} has a blank context")
        for option in row.options:
            if not option.strip():
                raise ValueError(f"{path}:{number} has a blank option {option!r}")
        if len(set(row.options)) != len(row.options):
            repeated = sorted({name for name in row.options if row.options.count(name) > 1})
            raise ValueError(
                f"{path}:{number} repeats option(s) {repeated}; "
                "duplicate options would silently share one probability"
            )
        rows.append(row)
    return rows


def context_set(examples: list[ChoiceExample]) -> set[str]:
    return {row.context for row in examples}


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digests(examples: list[ChoiceExample]) -> str:
    """Order-independent digest of a split's contexts, for manifest comparison."""
    joined = "\n".join(sorted(context_set(examples)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def describe(path: str | Path, examples: list[ChoiceExample]) -> dict[str, object]:
    return {
        "path": str(path),
        "rows": len(examples),
        "file_sha256": digest(path),
        "context_sha256": digests(examples),
    }


def overlap_report(splits: dict[str, list[ChoiceExample]]) -> dict[str, int]:
    """Pairwise count of shared context strings; every value must be zero."""
    names = sorted(splits)
    report: dict[str, int] = {}
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            shared = context_set(splits[names[left]]) & context_set(splits[names[right]])
            report[f"{names[left]}&{names[right]}"] = len(shared)
    return report


def require_disjoint(splits: dict[str, list[ChoiceExample]]) -> dict[str, int]:
    """Raise unless every split is non-empty and no context is shared by two splits."""
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"split {name!r} has no rows")
    report = overlap_report(splits)
    leaked = {pair: count for pair, count in report.items() if count}
    if leaked:
        raise ValueError(
            "splits share contexts (group leakage): "
            + ", ".join(f"{pair}={count}" for pair, count in sorted(leaked.items()))
        )
    return report


def load_splits(directory: str | Path, names: tuple[str, ...] = SPLITS) -> dict[str, list[ChoiceExample]]:
    directory = Path(directory)
    loaded: dict[str, list[ChoiceExample]] = {}
    for name in names:
        path = directory / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing split file {path}")
        loaded[name] = load_examples(path)
    return loaded


def _write_split(path: Path, rows: list[ChoiceExample]) -> None:
    """Write one JSONL split with upstream's field layout and dump settings."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "context": row.context, "options": row.options, "label": row.label,
            }) + "\n")


def _unique_examples(sizes: dict[str, int], seed: int) -> dict[str, list[ChoiceExample]]:
    """Upstream generator, one distinct context per row across all splits.

    Upstream's ``write_synthetic`` draws row ``i`` of a split from
    ``synthetic_example(seed + offset + i * 104729)`` and never compares contexts, so
    two splits can receive the same context: at this course's default sizes
    (2000/400/400/400, seed 700001) the upstream writer emits one context in both
    train and test. This function uses the same generator and the same stride but
    skips a row whose context is already used, which keeps the requested row counts
    and makes group leakage impossible by construction. With no collision the emitted
    rows are exactly upstream's.
    """
    rows: dict[str, list[ChoiceExample]] = {name: [] for name in sizes}
    seen: set[str] = set()
    cursor = seed
    for name, size in sizes.items():
        while len(rows[name]) < size:
            example = synthetic_example(cursor)
            cursor += STRIDE
            if example.context in seen:
                continue
            seen.add(example.context)
            rows[name].append(example)
    return rows


def make_synthetic_splits(
    directory: str | Path,
    sizes: dict[str, int],
    seed: int,
) -> dict[str, object]:
    """Write fresh, pairwise-disjoint splits and report their digests.

    The rows come from the upstream generator; the split writer is this course's, so
    that the four files are disjoint by construction. ``require_disjoint`` proves that
    afterwards instead of assuming it.
    """
    for name, size in sizes.items():
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"split {name!r} needs a positive integer size, got {size!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = _unique_examples(dict(sizes), seed)
    for name, examples in rows.items():
        _write_split(directory / f"{name}.jsonl", examples)
    loaded = load_splits(directory, tuple(sizes))
    report = require_disjoint(loaded)
    return {
        "seed": seed,
        "sizes": dict(sizes),
        "splits": {name: describe(directory / f"{name}.jsonl", examples)
                   for name, examples in loaded.items()},
        "shared_contexts": report,
        "disjoint": all(count == 0 for count in report.values()),
        "generator": ("jevlike.data.synthetic_example, stride 104729, with cross-split "
                      "context deduplication (upstream write_synthetic does not deduplicate)"),
    }


def dump(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def refuse_nonempty_directory(directory: str | Path, force: bool, what: str) -> None:
    """Refuse to write into a directory that already holds entries, unless forced.

    The course's commands write several files per directory; silently overwriting
    an existing directory (a previous run, or a user's evidence) destroys work the
    command was never asked to touch. ``data`` and ``run-all`` call this before
    creating anything, and both expose ``--force`` for deliberate re-runs.
    """
    directory = Path(directory)
    if force or not directory.is_dir():
        return
    entries = sorted(entry.name for entry in directory.iterdir())
    if not entries:
        return
    preview = ", ".join(entries[:5]) + ("…" if len(entries) > 5 else "")
    raise ValueError(
        f"{directory} already contains {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} "
        f"({preview}); pass --force to overwrite {what} there, or choose a new --out"
    )
