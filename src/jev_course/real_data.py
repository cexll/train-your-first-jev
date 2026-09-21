"""Build the course's four splits from the real CommonsenseQA release.

The synthetic generator in :mod:`jev_course.dataset` makes the lesson reproducible
but teaches it on invented questions. This module keeps the same four split files
(``train``/``validation``/``calibration``/``test``, same ``context``/``options``/
``label`` layout) and fills them from real multiple-choice questions instead:

* Source — CommonsenseQA (Talmor et al., 2019), the official ``random`` split as
  published by the authors at ``https://s3.amazonaws.com/commensenseqa/`` (the
  bucket name is misspelled in the original release, and that spelling is the
  canonical one; the historical Hugging Face loader
  ``datasets/commonsense_qa/commonsense_qa.py`` pinned the same URLs). Only the two
  files that carry answer keys are downloaded — ``train_rand_split.jsonl`` and
  ``dev_rand_split.jsonl``. ``test_rand_split_no_answers.jsonl`` is deliberately
  *not* fetched: it ships without answer keys, so using it would mean inventing
  labels. Every download is checked against a pinned SHA-256 and byte count.
* Licence — MIT. The dataset repository has no ``LICENSE`` file, so the licence is
  taken from the maintainer's own answer on the official issue tracker
  (https://github.com/jonathanherzig/commonsenseqa/issues/5: "Our dataset is
  licensed under an MIT license."), which matches the ``license: mit`` field of the
  authors' Hugging Face dataset card (https://huggingface.co/datasets/tau/commonsense_qa).
  The questions are English and crowdsourced; see the module-level ``LIMITATIONS``
  below for what that means for a beginner-sized lesson.

Why the official ``train``/``dev`` split boundary is not reused
---------------------------------------------------------------
CommonsenseQA groups its questions by a ConceptNet concept (``question_concept``),
and the official ``train`` and ``dev`` files share 736 concepts. Carving rows out of
them row-by-row would therefore put two questions about the same concept in two
different splits, which is exactly the leakage the course's ``require_disjoint``
check exists to prevent. So this module pools the labelled rows of both files and
re-splits them by concept group: one concept contributes to exactly one split.

The selection is deterministic and needs no seed: groups are ordered by
``sha256(concept)``, and each group is placed whole into the first split (in
``SPLITS`` order) that still has room for it. A group larger than every remaining
quota is left out rather than split. The result is then *proved*, not assumed: the
written files are reloaded through :func:`jev_course.dataset.load_splits`, checked
with :func:`jev_course.dataset.require_disjoint`, and the concept sets of the
reloaded splits are compared pairwise.

The full-course recipe
----------------------
:func:`prepare_course` keeps the same download, verification and group-reserving
machinery but serves the course's real training run instead of the beginner-sized
lesson: whole concept groups are reserved for ``validation`` (512 rows),
``calibration`` (256) and ``test`` (256) with the larger ``COURSE_RESERVE`` quotas,
and ``train`` then absorbs every remaining usable pool row whose concept group was
not reserved — on the pinned release exactly 9792 train rows. This is the recipe
the course model in ``runs/lora-full`` was trained on, so the helper reproduces
that data byte-for-byte.

Nothing here trains or scores anything; ``prepare`` and ``prepare_course`` only
write the four split files and return the report described in their docstrings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .dataset import SPLITS, describe, dump, load_splits, refuse_nonempty_directory, require_disjoint

# Contract: the lesson's beginner-sized splits. Changing these changes the lesson,
# not just the download, so they are constants rather than flags.
SIZES = {"train": 256, "validation": 64, "calibration": 64, "test": 64}

#: The full-course recipe, as trained in ``runs/lora-full``. The held-out splits are
#: reserved first with these quotas (the ``train`` entry only bounds the reservation
#: step, because ``train`` is expanded afterwards), then ``train`` takes every
#: remaining usable row. Also constants: changing them changes the course's data.
COURSE_RESERVE = {"train": 2048, "validation": 512, "calibration": 256, "test": 256}

#: The splits whose concept groups are reserved and never reach ``train``.
COURSE_HELD_OUT = ("validation", "calibration", "test")

#: The recipe, described once and recorded in every :func:`prepare_course` report so a
#: reader can tell what data a course run was trained on without reading this file.
COURSE_RECIPE = (
    "reserve whole concept groups for validation/calibration/test via select_splits "
    "with the COURSE_RESERVE quotas, then give train every remaining usable pool row "
    "whose concept group was not reserved"
)
COURSE_TRAIN_EXPANSION = (
    "all usable pool rows minus the concept groups placed in the held-out splits, in "
    "pool order (train_rand_split.jsonl then dev_rand_split.jsonl, line by line)"
)

#: The field that defines a leakage group; one concept never spans two splits.
GROUP_FIELD = "question.question_concept"

#: Canonical release location. ``commensenseqa`` is the authors' original (misspelt)
#: bucket name, as referenced by the historical Hugging Face dataset loader.
SOURCE_ROOT = "https://s3.amazonaws.com/commensenseqa/"

#: Pinned source files, with the digests this module verifies before parsing.
#: ``bytes`` and ``sha256`` were observed on 2026-09-20; ``etag_md5`` is the S3 ETag,
#: which equals the MD5 for a single-part upload and is therefore an independent
#: integrity check on the same object.
SOURCES: tuple[dict[str, object], ...] = (
    {
        "name": "train_rand_split.jsonl",
        "url": SOURCE_ROOT + "train_rand_split.jsonl",
        "bytes": 3785890,
        "sha256": "58ffa3c8472410e24b8c43f423d89c8a003d8284698a6ed7874355dedd09a2fb",
        "etag_md5": "cd6c6a82b627d2c5602b0e60931d2785",
        "rows": 9741,
    },
    {
        "name": "dev_rand_split.jsonl",
        "url": SOURCE_ROOT + "dev_rand_split.jsonl",
        "bytes": 471653,
        "sha256": "3210497fdaae614ac085d9eb873dd7f4d49b6f965a93adadc803e1229fd8a02a",
        "etag_md5": "43b6b0b8767502a2c3b64ecc6e7a4a6b",
        "rows": 1221,
    },
)

#: Recorded in the report so that "no test labels were invented" is auditable.
WITHHELD_SOURCE = {
    "name": "test_rand_split_no_answers.jsonl",
    "url": SOURCE_ROOT + "test_rand_split_no_answers.jsonl",
    "reason": "ships without answer keys; the course's test split is a held-out slice "
              "of the labelled pool instead of guessed labels",
}

USER_AGENT = "train-your-first-jev/0.1 (teaching repository; CommonsenseQA course data)"
CHUNK = 1 << 16
TIMEOUT_SECONDS = 120.0

#: Everything a reader should know before quoting a number produced on these splits.
LIMITATIONS: tuple[str, ...] = (
    "The splits are not the official CommonsenseQA splits: labelled rows from the "
    "official train and dev files are pooled and re-split by concept, so accuracy on "
    "them is not comparable with published random-split results.",
    "The official test split is never downloaded, because it carries no answer keys; "
    "the course's test.jsonl is a held-out slice of the labelled pool.",
    "Splits are concept-disjoint, which is stricter than the official random split; a "
    "concept with more rows than a split's entire quota cannot be placed whole and is "
    "excluded by construction (`selection.largest_unplaced_concept` names it).",
    "Only a beginner-sized slice of the labelled pool is used; the report's "
    "`selection` and `excluded_rows` sections carry the measured counts.",
    "Questions are English, crowdsourced, and contain no explicit licence header in "
    "the files themselves; the MIT licence comes from the dataset maintainer and the "
    "authors' dataset card, both cited in the report.",
)


class SourceUnavailable(RuntimeError):
    """The pinned source could not be fetched (network, DNS, HTTP status)."""


class SourceVerificationError(RuntimeError):
    """A fetched source does not match its pinned digest or byte count."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(url: str, expected_bytes: int, timeout: float = TIMEOUT_SECONDS) -> bytes:
    """Download one pinned file, refusing anything larger than its pinned size.

    Reading stops with an error as soon as more than ``expected_bytes`` have arrived,
    so a wrong or hostile response cannot fill memory before the digest check runs.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    chunks: list[bytes] = []
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_bytes:
                    raise SourceVerificationError(
                        f"{url} sent more than the pinned {expected_bytes} bytes; stopped reading"
                    )
                chunks.append(chunk)
    except urllib.error.HTTPError as error:
        raise SourceUnavailable(f"{url} answered HTTP {error.code} {error.reason}") from error
    except urllib.error.URLError as error:
        raise SourceUnavailable(f"{url} is unreachable: {error.reason}") from error
    return b"".join(chunks)


def verify(payload: bytes, source: dict[str, object]) -> dict[str, object]:
    """Check one payload against its pinned byte count, SHA-256 and MD5 ETag."""
    url = str(source["url"])
    if len(payload) != source["bytes"]:
        raise SourceVerificationError(
            f"{url} returned {len(payload)} bytes, expected {source['bytes']}"
        )
    observed = _sha256(payload)
    if observed != source["sha256"]:
        raise SourceVerificationError(
            f"{url} has sha256 {observed}, expected {source['sha256']}"
        )
    observed_md5 = hashlib.md5(payload).hexdigest()
    if observed_md5 != source["etag_md5"]:
        raise SourceVerificationError(
            f"{url} has md5 {observed_md5}, expected {source['etag_md5']}"
        )
    return {
        "url": url,
        "bytes": len(payload),
        "sha256": observed,
        "etag_md5": observed_md5,
        "sha256_verified": True,
    }


def download_sources(cache_dir: str | Path) -> tuple[dict[str, Path], list[dict[str, object]]]:
    """Fetch every pinned source into ``cache_dir``, reusing verified copies.

    A file already in ``cache_dir`` whose SHA-256 matches its pin is reused without a
    network round trip; anything else is downloaded again and verified. Returns the
    local paths plus the evidence records that go into the report.
    """
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    evidence: list[dict[str, object]] = []
    for source in SOURCES:
        name = str(source["name"])
        path = directory / name
        if path.exists() and _sha256(path.read_bytes()) == source["sha256"]:
            payload = path.read_bytes()
            record = verify(payload, source)
            record["reused_from_cache"] = True
        else:
            payload = fetch(str(source["url"]), int(source["bytes"]))
            record = verify(payload, source)
            record["reused_from_cache"] = False
            path.write_bytes(payload)
        paths[name] = path
        record["name"] = name
        record["rows"] = source["rows"]
        evidence.append(record)
    return paths, evidence


def _usable_row(payload: Any) -> tuple[dict[str, object] | None, str | None]:
    """Convert one official row, or name the reason it cannot be a course row.

    The reasons are the ones :func:`jev_course.dataset.load_examples` would raise on,
    plus the conversion's own requirements (a ConceptNet concept to group by, a usable
    ``answerKey``). Rows that fail are excluded and counted, never repaired: a row
    whose five options contain a repeat has no single correct answer, and guessing
    which one was meant would invent data.
    """
    if not isinstance(payload, dict):
        return None, "row is not a JSON object"
    question = payload.get("question")
    if not isinstance(question, dict):
        return None, "row has no question object"
    stem = question.get("stem")
    choices = question.get("choices")
    if not isinstance(stem, str) or not isinstance(choices, list):
        return None, "row has no string stem or choice list"
    if not stem.strip():
        return None, "blank context"
    if len(choices) < 2:
        return None, "fewer than two options"
    labels: list[Any] = []
    texts: list[Any] = []
    for choice in choices:
        if not isinstance(choice, dict):
            return None, "choice is not an object"
        labels.append(choice.get("label"))
        texts.append(choice.get("text"))
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        return None, "blank option"
    if len(set(texts)) != len(texts):
        return None, "duplicate options"
    if len(set(labels)) != len(labels):
        return None, "duplicate choice labels"
    answer = payload.get("answerKey")
    if not isinstance(answer, str) or answer not in labels:
        return None, "answerKey is not one of the row's choice labels"
    concept = question.get("question_concept")
    if not isinstance(concept, str) or not concept.strip():
        return None, "row has no question_concept to group by"
    row_id = payload.get("id")
    if not isinstance(row_id, str) or not row_id.strip():
        return None, "row has no id"
    return {
        "context": stem,
        "options": tuple(texts),
        "label": labels.index(answer),
        "concept": concept,
        "id": row_id,
    }, None


def parse_pool(paths: Iterable[str | Path]) -> tuple[list[dict[str, object]], Counter]:
    """Read the pinned files into usable rows, counting every rejected row's reason."""
    rows: list[dict[str, object]] = []
    rejected: Counter = Counter()
    seen: dict[str, str] = {}
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                rejected["malformed JSON"] += 1
                continue
            row, reason = _usable_row(payload)
            if row is None:
                rejected[str(reason)] += 1
                continue
            context = str(row["context"])
            concept = str(row["concept"])
            if context in seen:
                # The same question twice would break context-disjointness: one copy
                # could land in a different group (hence a different split) than the
                # other. Refuse the repeat rather than pick a winner.
                if seen[context] != concept:
                    raise ValueError(
                        f"{path}:{number} repeats a context under another concept: "
                        f"{seen[context]!r} then {concept!r}"
                    )
                rejected["duplicate context"] += 1
                continue
            seen[context] = concept
            rows.append(row)
    if not rows:
        raise ValueError(f"no usable rows in {list(paths)}")
    return rows, rejected


def select_splits(
    rows: list[dict[str, object]],
    sizes: dict[str, int],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Place whole concept groups into splits until every quota is met exactly.

    Groups are ordered by ``sha256(concept)`` — a fixed order that needs no seed and
    does not depend on ``PYTHONHASHSEED``, on the order the files were read, or on
    group size. Each group goes whole into the first split of ``SPLITS`` with room for
    it; a group that fits nowhere (larger than every remaining quota) is left out
    rather than split, because half a concept's questions in one split and half in
    another is the leakage this design exists to avoid.
    """
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["concept"]), []).append(row)
    order = sorted(groups, key=lambda concept: hashlib.sha256(concept.encode("utf-8")).hexdigest())
    room = dict(sizes)
    chosen: dict[str, list[dict[str, object]]] = {name: [] for name in sizes}
    placed_groups = {name: 0 for name in sizes}
    skipped: list[dict[str, object]] = []
    for concept in order:
        members = groups[concept]
        fits = [name for name in SPLITS if room[name] >= len(members)]
        if not fits:
            skipped.append({"concept": concept, "rows": len(members)})
            continue
        name = fits[0]
        room[name] -= len(members)
        placed_groups[name] += 1
        chosen[name].extend(members)
    short = {name: left for name, left in room.items() if left}
    if short:
        raise ValueError(
            "the labelled pool cannot fill the requested splits without splitting a "
            f"concept group; still missing {short}"
        )
    report = {
        "group_field": GROUP_FIELD,
        "group_order": "ascending sha256(question_concept)",
        "placement": "whole group into the first split in SPLITS order with enough room",
        "pool_rows": len(rows),
        "pool_concepts": len(groups),
        "placed_concepts": placed_groups,
        "unplaced_concepts": len(skipped),
        "unplaced_rows": sum(int(item["rows"]) for item in skipped),
        "largest_unplaced_concept": max(
            (item["concept"] for item in skipped),
            key=lambda concept: len(groups[str(concept)]),
            default=None,
        ),
    }
    return chosen, report


def _write_split(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one split in the same layout as :mod:`jev_course.dataset`'s writer.

    Same key order and same ``json.dumps`` defaults, so a real-data split file is
    read back by ``load_splits`` and digested exactly like a synthetic one.
    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "context": row["context"], "options": list(row["options"]),
                "label": row["label"],
            }) + "\n")


def _ids_digest(rows: list[dict[str, object]]) -> str:
    joined = "\n".join(sorted(str(row["id"]) for row in rows))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _concepts_by_split(
    splits: dict[str, list], concepts: dict[str, str],
) -> dict[str, set[str]]:
    """Map the reloaded rows back to their concept groups via their context text."""
    per_split: dict[str, set[str]] = {}
    for name, rows in splits.items():
        unknown = [row.context for row in rows if row.context not in concepts]
        if unknown:
            raise ValueError(f"split {name!r} holds contexts the source never produced: {unknown[:3]}")
        per_split[name] = {concepts[row.context] for row in rows}
    return per_split


def _pairwise(sets: dict[str, set[str]]) -> dict[str, int]:
    names = sorted(sets)
    return {
        f"{left}&{right}": len(sets[left] & sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1:]
    }


def _requested_sizes(sizes: dict[str, int] | None) -> dict[str, int]:
    """Validate a split-size mapping (defaulting to the lesson's ``SIZES``)."""
    requested = dict(SIZES if sizes is None else sizes)
    if set(requested) != set(SPLITS):
        raise ValueError(f"sizes must name exactly {list(SPLITS)}, got {sorted(requested)}")
    for name, size in requested.items():
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"split {name!r} needs a positive integer size, got {size!r}")
    return requested


def _open_cache(cache_dir: str | Path | None) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Resolve the download cache, creating a temporary one when none was given."""
    if cache_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="jev-course-sources-")
        return Path(temporary.name), temporary
    return Path(cache_dir), None


def _source_block(evidence: list[dict[str, object]]) -> dict[str, object]:
    """The provenance block every real-data report carries: source, licence, digests."""
    return {
        "name": "CommonsenseQA",
        "revision": "official random split, JSONL release as published by the authors",
        "paper": "https://arxiv.org/abs/1811.00937",
        "homepage": "https://www.tau-nlp.org/commonsenseqa",
        "repository": "https://github.com/jonathanherzig/commonsenseqa",
        "root": SOURCE_ROOT,
        "license": "MIT",
        "license_evidence": [
            "https://github.com/jonathanherzig/commonsenseqa/issues/5 — "
            "maintainer Jonathan Herzig: \"Our dataset is licensed under an MIT license.\"",
            "https://huggingface.co/datasets/tau/commonsense_qa — dataset card "
            "field `license: mit`",
        ],
        "license_note": "the released JSONL files carry no licence header of their own",
        "files": evidence,
        "withheld": WITHHELD_SOURCE,
    }


def _prove_written(
    out: Path,
    chosen: dict[str, list[dict[str, object]]],
    requested: dict[str, int],
    concepts: dict[str, str],
) -> tuple[dict[str, list], dict[str, set[str]], dict[str, int], dict[str, int]]:
    """Reload the written splits and prove row counts, context- and concept-disjointness.

    The artefacts are checked rather than assumed: the files are read back through
    :func:`jev_course.dataset.load_splits` (which re-applies every row rule),
    :func:`jev_course.dataset.require_disjoint` proves no context is shared, and the
    reloaded rows are mapped back to their concept groups so group leakage is caught
    too. Returns the reloaded splits, their concept sets, and both pairwise reports.
    """
    loaded = load_splits(out, SPLITS)
    shared_contexts = require_disjoint(loaded)
    for name in SPLITS:
        if len(loaded[name]) != requested[name]:
            raise ValueError(
                f"split {name!r} reloads with {len(loaded[name])} rows, "
                f"expected {requested[name]}"
            )
    per_split_concepts = _concepts_by_split(loaded, concepts)
    shared_concepts = _pairwise(per_split_concepts)
    leaked = {pair: count for pair, count in shared_concepts.items() if count}
    if leaked:
        raise ValueError(f"splits share concept groups (group leakage): {leaked}")
    return loaded, per_split_concepts, shared_contexts, shared_concepts


def _splits_block(
    out: Path,
    loaded: dict[str, list],
    per_split_concepts: dict[str, set[str]],
    chosen: dict[str, list[dict[str, object]]],
) -> dict[str, dict[str, object]]:
    return {
        name: {
            **describe(out / f"{name}.jsonl", loaded[name]),
            "concepts": len(per_split_concepts[name]),
            "source_ids_sha256": _ids_digest(chosen[name]),
        }
        for name in SPLITS
    }


def _write_splits(out: Path, chosen: dict[str, list[dict[str, object]]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        _write_split(out / f"{name}.jsonl", chosen[name])


def prepare(
    directory: str | Path,
    *,
    sizes: dict[str, int] | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Download CommonsenseQA, write the four course splits, and report what happened.

    ``directory`` receives ``train.jsonl``, ``validation.jsonl``, ``calibration.jsonl``
    and ``test.jsonl`` in the course's row layout; it is refused when it already holds
    entries unless ``force`` is set. ``cache_dir`` keeps the verified raw downloads
    (default: a temporary directory that is removed afterwards). The returned report
    records the source URLs, the verified digests, the licence and its evidence, the
    selection rule, per-split file/context digests, and the disjointness proofs — it is
    deterministic for a given source revision.
    """
    requested = _requested_sizes(sizes)

    out = Path(directory)
    refuse_nonempty_directory(out, force, "the four real-data split files")

    cache, temporary = _open_cache(cache_dir)
    try:
        paths, evidence = download_sources(cache)
        rows, rejected = parse_pool(paths.values())
        chosen, selection = select_splits(rows, requested)
        _write_splits(out, chosen)

        # Prove the artefacts, do not assume them: reload through the course loader
        # (which re-applies every row rule) and require context-disjointness.
        concepts = {str(row["context"]): str(row["concept"]) for row in rows}
        loaded, per_split_concepts, shared_contexts, shared_concepts = _prove_written(
            out, chosen, requested, concepts
        )

        report: dict[str, object] = {
            "schema": 1,
            "source": _source_block(evidence),
            "selection": selection,
            "sizes": requested,
            "splits": _splits_block(out, loaded, per_split_concepts, chosen),
            "shared_contexts": shared_contexts,
            "shared_concepts": shared_concepts,
            "disjoint": all(count == 0 for count in shared_contexts.values()),
            "group_disjoint": all(count == 0 for count in shared_concepts.values()),
            "excluded_rows": dict(sorted(rejected.items())),
            "excluded_rows_total": sum(rejected.values()),
            "limitations": list(LIMITATIONS),
        }
    finally:
        if temporary is not None:
            temporary.cleanup()
    return report


def prepare_course(
    directory: str | Path,
    cache_dir: str | Path | None = None,
) -> dict[str, object]:
    """Download CommonsenseQA, write the full-course splits, and report what happened.

    This is the recipe the course's real model (``runs/lora-full``) was trained on.
    The pinned release is downloaded and verified exactly as in :func:`prepare`, the
    whole concept groups for ``validation``/``calibration``/``test`` are reserved via
    :func:`select_splits` with the ``COURSE_RESERVE`` quotas, and ``train`` then
    absorbs *every* remaining usable pool row whose concept group was not reserved —
    on the pinned files that is 9792 train rows (the whole labelled pool minus the
    512/256/256 reserved rows). No held-out group is split or re-used for training:
    the reserved concepts appear only in their own split, and the returned report
    proves both row counts and context/concept disjointness on the reloaded files.

    ``directory`` receives ``train.jsonl``, ``validation.jsonl``, ``calibration.jsonl``
    and ``test.jsonl`` in the course's row layout and is refused when it already holds
    entries — unlike :func:`prepare` there is deliberately no ``force``: overwriting
    the course's training data is never automatic. ``cache_dir`` keeps the verified
    raw downloads (default: a temporary directory that is removed afterwards).
    """
    out = Path(directory)
    if out.is_dir():
        existing = sorted(entry.name for entry in out.iterdir())
        if existing:
            preview = ", ".join(existing[:5]) + ("…" if len(existing) > 5 else "")
            raise ValueError(
                f"{out} already contains {len(existing)} "
                f"entr{'y' if len(existing) == 1 else 'ies'} ({preview}); the full-course "
                "recipe never overwrites existing data — choose a new --out"
            )

    cache, temporary = _open_cache(cache_dir)
    try:
        paths, evidence = download_sources(cache)
        rows, rejected = parse_pool(paths.values())

        # Reserve whole concept groups for the held-out splits first; the `train`
        # entry of COURSE_RESERVE only bounds that reservation step, because train is
        # expanded to every remaining row afterwards.
        reserved, selection = select_splits(rows, dict(COURSE_RESERVE))
        held_out = {name: reserved[name] for name in COURSE_HELD_OUT}
        reserved_concepts = {
            str(row["concept"]) for split in held_out.values() for row in split
        }
        chosen: dict[str, list[dict[str, object]]] = {
            "train": [
                row for row in rows if str(row["concept"]) not in reserved_concepts
            ],
            **{name: list(split) for name, split in held_out.items()},
        }
        if not chosen["train"]:
            raise ValueError("the full-course recipe produced no train rows")
        sizes = {name: len(chosen[name]) for name in SPLITS}
        _write_splits(out, chosen)

        # Prove the artefacts, do not assume them: reload through the course loader
        # (which re-applies every row rule) and require context/concept-disjointness.
        concepts = {str(row["context"]): str(row["concept"]) for row in rows}
        loaded, per_split_concepts, shared_contexts, shared_concepts = _prove_written(
            out, chosen, sizes, concepts
        )

        # Coverage, not just disjointness: every usable pool row must land in exactly
        # one split (train takes the unreserved remainder, the held-out splits take
        # their reserved groups), so no row is silently dropped by the expansion.
        accounted = sizes["train"] + sum(len(split) for split in held_out.values())
        if accounted != len(rows):
            raise ValueError(
                f"the full-course recipe accounted for {accounted} of {len(rows)} "
                "usable pool rows"
            )
        pool_concepts = {str(row["concept"]) for row in rows}
        covered = {concept for name in SPLITS for concept in per_split_concepts[name]}
        if covered != pool_concepts:
            raise ValueError(
                f"the full-course splits cover {len(covered)} of {len(pool_concepts)} "
                f"pool concepts; missing {sorted(pool_concepts - covered)[:3]}"
            )

        report: dict[str, object] = {
            "schema": 1,
            "directory": str(out),
            "source": _source_block(evidence),
            "selection": selection,
            "sizes": sizes,
            "course": {
                "prepared_by": "jev_course.real_data.prepare_course",
                "recipe": COURSE_RECIPE,
                "reserve_sizes": dict(COURSE_RESERVE),
                "held_out_splits": list(COURSE_HELD_OUT),
                "train_expansion": COURSE_TRAIN_EXPANSION,
                "pool_rows": len(rows),
                "pool_concepts": len(pool_concepts),
                "reserved_concepts": len(reserved_concepts),
                "reserved_rows": sum(len(split) for split in held_out.values()),
            },
            "splits": _splits_block(out, loaded, per_split_concepts, chosen),
            "shared_contexts": shared_contexts,
            "shared_concepts": shared_concepts,
            "disjoint": all(count == 0 for count in shared_contexts.values()),
            "group_disjoint": all(count == 0 for count in shared_concepts.values()),
            "excluded_rows": dict(sorted(rejected.items())),
            "excluded_rows_total": sum(rejected.values()),
            "limitations": list(LIMITATIONS),
        }
    finally:
        if temporary is not None:
            temporary.cleanup()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jev_course.real_data",
        description="Download CommonsenseQA and write the course's four real-data splits.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/real"),
                        help="directory for the four split files")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="keep the verified raw downloads here (default: temporary)")
    parser.add_argument("--force", action="store_true",
                        help="write into a non-empty --out directory")
    parser.add_argument("--course", action="store_true",
                        help="use the full-course recipe (9792 train / 512 validation / "
                             "256 calibration / 256 test on the pinned release); the course "
                             "recipe never overwrites a non-empty --out (--force is ignored)")
    parser.add_argument("--json-out", default=None, help="also write the report file")
    args = parser.parse_args(argv)
    report = (prepare_course(args.out, cache_dir=args.cache_dir) if args.course
              else prepare(args.out, cache_dir=args.cache_dir, force=args.force))
    text = dump(report)
    print(text)
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the module's CLI
    raise SystemExit(main())
