# Provenance

This repository extends the vendored `jevlike` decision scorer with split hygiene,
independent calibration, checked execution and an optional PEFT LoRA training route.
The adapter route is a local extension, not upstream Jev or kev model equivalence.

## Upstream source

| | |
|---|---|
| Project | `jevlike` |
| Repository | https://github.com/vinnylarouge/jevlike |
| Revision | `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452` |
| Author | Minimal Labs |
| Licence | MIT (`vendor/jevlike/LICENSE`) |
| Vendored at | `vendor/jevlike/jevlike/` |

The baseline that was copied is the pinned revision held by the course author's
local mirror (`jev-local/.vendor/jevlike`). Machine-readable evidence, including
the SHA-256 of every file *before* patching and *after* patching, the licence
digest and the packaging deltas, is in
[`vendor/jevlike/PROVENANCE.json`](vendor/jevlike/PROVENANCE.json). To confirm the
baseline yourself, check out the revision above and compare those
`pristine_sha256` values.

Upstream files not vendored, because the lessons do not use them: `examples/`,
`docs/`, `data/`, `runs/`, `scripts/`, `tests/`, `README.md`, `AGENTS.md`.

## Changes applied to the vendored copy

The cumulative [`patches/0001-course-fixes.diff`](vendor/jevlike/patches/0001-course-fixes.diff)
records changes against the pristine baseline. Its current identity and source
digests are recorded in `vendor/jevlike/PROVENANCE.json`.

1. **`jevlike/model.py`, `trainable_state`** — `parameter.detach().cpu()` became
   `parameter.detach().cpu().clone()`. On CPU the old expression returned the live
   parameter tensor, so the "best validation" snapshot shared storage with the
   model: later epochs kept training and overwrote it, and the checkpoint written at
   the end held the final epoch instead of the best one. Nothing else changes —
   device, dtype and the parameter set are identical. `jev-course check` exercises
   this directly (`snapshot_is_independent`).

2. **`jevlike/model.py`, `load_checkpoint`** — `torch.load(..., weights_only=False)`
   became `weights_only=True`, plus validation that the payload is a dict holding a
   `config` dict with the expected keys and a `state_dict` mapping names to tensors.
   The on-disk checkpoint schema is unchanged; only the loader's tolerance for
   arbitrary pickled objects is removed. A checkpoint that needs more than tensors
   and plain config values is rejected with a `ValueError` instead of being
   unpickled. `jev-course check` (`safe_checkpoint_loader`) proves that a hostile
   payload is refused *before* its side effect runs, and that checkpoints written by
   `jevlike.train` still round-trip.

3. **`jevlike/data.py`, `synthetic_example`** — `options = list({...})` became
   `options = sorted({...})`. Set iteration order for strings depends on
   `PYTHONHASHSEED`, so the same `--seed` produced a different option order — and
   therefore different labels — in different processes. The option *set* is
   unchanged. `jev-course check` (`generation_is_hash_seed_independent`) generates
   splits under two different hash seeds and compares file digests.

4. **`jevlike/eval.py`, `metrics`** — ECE bins were selected with
   `lower <= p < lower + 0.1` over `linspace(0, 0.9, 10)`. A confidence of exactly
   `1.0` was never counted, and membership of the boundary values depended on
   float32 equality with the `linspace` edges. Bins are now
   `floor(p * 10)` clamped to `[0, 9]`, so every row lands in exactly one bin.
   `top1` and `top3` are unaffected; only the reported `ece` value changes, and it
   now covers all examples. `jev-course check` (`bin_ownership`) covers this.

5. **`jevlike/model.py`, `FrozenTransformerScorer.__init__` / `make_system` and
   `jevlike/train.py`, `main`** — an optional encoder revision. `--hf-revision`
   (default `None`) is recorded in the checkpoint config as `hf_revision`, and
   `make_system` reads it with `config.get("hf_revision")` and passes it to both
   `AutoTokenizer.from_pretrained` and `AutoModel.from_pretrained`, so a tokenizer
   and encoder always come from the same commit instead of whatever the hub default
   branch points at when a checkpoint is reloaded. `FrozenTransformerScorer` takes
   the revision as an optional third constructor argument. Nothing else changes: no
   architecture, no loss, no optimiser, no training loop. `revision=None` is the hub
   default branch, and because the flag is only written when supplied and read with
   `.get()`, checkpoints written before this key existed load unchanged and the
   tiny-encoder config in `src/jev_course/checks.py` needs no new field.

6. **Adapter training (`model.py`, `train.py`)** — `encoder=lora` uses PEFT
   rank-16, alpha-32 adapters on Qwen query/value projections and trains the decision
   head. The base stays frozen. Adapter and head learning rates use separate AdamW
   groups with OneCycle; tiny and frozen-HF training retain their original schedule.
   Saved weights contain adapters plus head, not the pretrained base. Reload requires
   the pinned base snapshot and rejects missing trainable adapter tensors.
   LoRA reload requires a 40-character commit revision in the saved config; this
   rejects omitted or symbolic revisions, but does not authenticate a modified file.
   This adapts the general LoRA-plus-head approach investigated in
   [kev](https://github.com/jaredpalmer/kev), but retains jevlike's pooled-option
   architecture rather than kev's joint pointer-token architecture.

Packaging deltas in `vendor/jevlike/pyproject.toml` (documented in the file itself
and in `PROVENANCE.json`): the `readme` key, the `dev`/`games` extras, the pytest
configuration and the four upstream console scripts were dropped; name, version,
description, `requires-python`, licence, authors, dependencies, the `transformers`
extra and the module entry points are upstream text. The lessons call
`python -m jevlike.data|train|eval|predict` instead of the console scripts.

## This repository's own code

`src/jev_course/` is original MIT code for this course: `dataset.py` (validated
loading, disjointness proofs and the deduplicating split writer), `calibration.py`
(temperature fit, held-out metrics, reload-and-predict), `checks.py` (the
self-checks above), `pipeline.py` (the full-flow runner) and `cli.py`. It imports
`jevlike` and never reimplements training. The split writer keeps upstream's
`synthetic_example` and seed stride but skips any context already used by an earlier
split, because upstream's own `write_synthetic` can place the same context in two
splits: at the course's default sizes (2000/400/400/400, seed 700001) it shares one
context between train and test. `jev-course check` reproduces and reports that
collision. The license for this repository is [`LICENSE`](LICENSE).
