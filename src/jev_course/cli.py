"""Command-line interface: ``jev-course <data|validate-data|check|fit|test|predict|run-all>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibration import (
    dump,
    fit_report,
    held_out_report,
    load_scorer,
    predict_report,
    read_temperature,
    validate_options,
)
from .checks import run_checks
from .dataset import describe, load_splits, make_synthetic_splits, refuse_nonempty_directory, require_disjoint
from .pipeline import REPO_ROOT, run_all

DEFAULT_SIZES = {
    "train": 2000,
    "validation": 400,
    "calibration": 400,
    "test": 400,
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def _emit(payload: object, json_out: str | None) -> None:
    text = dump(payload)
    print(text)
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(text + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-course",
        description="Train, evaluate, calibrate and reload a small Jev-like option scorer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    guide = sub.add_parser("guide", help="逐课训练开源基座的 LoRA 决策模型")
    guide.add_argument("--out", type=Path, default=Path("runs/my-jev"))
    guide.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    guide.add_argument("--data-directory", type=Path, default=None)

    data = sub.add_parser("data", help="generate four disjoint splits")
    data.add_argument("--out", type=Path, default=Path("data/course"),
                      help="relative paths resolve against the repository root")
    data.add_argument("--train", type=_positive_int, default=DEFAULT_SIZES["train"])
    data.add_argument("--validation", type=_positive_int, default=DEFAULT_SIZES["validation"])
    data.add_argument("--calibration", type=_positive_int, default=DEFAULT_SIZES["calibration"])
    data.add_argument("--test", type=_positive_int, default=DEFAULT_SIZES["test"])
    data.add_argument("--seed", type=int, default=700001)
    data.add_argument("--json-out", default=None, help="also write the report file")
    data.add_argument("--force", action="store_true",
                      help="overwrite a non-empty output directory")

    validate = sub.add_parser(
        "validate-data",
        help="preflight a directory of train/validation/calibration/test.jsonl",
        description=(
            "Check four split files before training on them. Rows are refused for "
            "malformed JSON, a non-object row, blanks, duplicate options, a boolean "
            "or out-of-range label; the splits must be non-empty and share no "
            "context. Upstream jevlike.train does none of this by itself."
        ),
    )
    validate.add_argument("directory", help="directory holding the four .jsonl files")
    validate.add_argument("--json-out", default=None)

    check = sub.add_parser("check", help="run the course self-checks")
    check.add_argument("--json-out", default=None)

    fit = sub.add_parser("fit", help="fit a temperature on the calibration split only")
    fit.add_argument("checkpoint")
    fit.add_argument("calibration")
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    fit.add_argument("--batch-size", type=_positive_int, default=64)

    test = sub.add_parser("test", help="report raw and calibrated numbers on the held-out split")
    test.add_argument("checkpoint")
    test.add_argument("test")
    test.add_argument("--temperature", required=True)
    test.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    test.add_argument("--json-out", default=None)
    test.add_argument("--batch-size", type=_positive_int, default=64)

    predict = sub.add_parser("predict", help="reload and score one menu of options")
    predict.add_argument("checkpoint")
    predict.add_argument("--temperature", required=True)
    predict.add_argument("--context", required=True)
    predict.add_argument("--option", action="append", required=True)
    predict.add_argument("--label", type=int, default=None,
                         help="optional correct index, recorded with the prediction")
    predict.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    predict.add_argument("--permute-check", action="store_true",
                         help="re-score the same menu in reverse order and compare")
    predict.add_argument("--json-out", default=None)

    run = sub.add_parser("run-all", help="run the full flow as checked subprocesses")
    run.add_argument("--out", type=Path, default=Path("runs/course"),
                     help="output directory; relative paths resolve against the repository root")
    run.add_argument("--force", action="store_true",
                     help="overwrite a non-empty --out directory (refused by default)")
    run.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda", "auto"))
    run.add_argument("--encoder", choices=("tiny", "hf", "lora"), default="tiny")
    run.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B")
    run.add_argument("--hf-revision", default=None)
    run.add_argument("--data-directory", type=Path, default=None)
    run.add_argument("--eval-batch-size", type=_positive_int, default=64)
    run.add_argument("--train", type=_positive_int, default=DEFAULT_SIZES["train"])
    run.add_argument("--validation", type=_positive_int, default=DEFAULT_SIZES["validation"])
    run.add_argument("--calibration", type=_positive_int, default=DEFAULT_SIZES["calibration"])
    run.add_argument("--test", type=_positive_int, default=DEFAULT_SIZES["test"])
    run.add_argument("--seed", type=int, default=700001, help="split generation seed")
    run.add_argument("--train-seed", type=int, default=7)
    run.add_argument("--width", type=int, default=64)
    run.add_argument("--rank", type=int, default=64)
    run.add_argument("--context-tokens", type=int, default=192)
    run.add_argument("--option-tokens", type=int, default=32)
    run.add_argument("--epochs", type=int, default=8)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--learning-rate", type=float, default=2e-3)
    run.add_argument("--predict-context", default=None,
                     help="context for the final reload-predict; defaults to the first test row")
    run.add_argument("--predict-option", action="append", dest="predict_options", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        exit_code = _dispatch(parser, args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        exit_code = 130
    except Exception as error:
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        exit_code = 1
    sys.exit(exit_code)


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.command == "guide":
        from .guide import run_guide
        return run_guide(args.out, args.device, args.data_directory)
    if args.command == "data":
        out = args.out if args.out.is_absolute() else Path(REPO_ROOT) / args.out
        refuse_nonempty_directory(out, args.force, "the four split files")
        report = make_synthetic_splits(
            out,
            {"train": args.train, "validation": args.validation,
             "calibration": args.calibration, "test": args.test},
            args.seed,
        )
        _emit(report, args.json_out)
        return 0

    if args.command == "validate-data":
        directory = Path(args.directory)
        splits = load_splits(directory)
        report = require_disjoint(splits)
        _emit({
            "directory": str(directory),
            "splits": {name: describe(directory / f"{name}.jsonl", rows)
                       for name, rows in splits.items()},
            "shared_contexts": report,
            "disjoint": all(count == 0 for count in report.values()),
            "ok": True,
        }, args.json_out)
        return 0

    if args.command == "check":
        report = run_checks()
        _emit(report, args.json_out)
        return 0 if report["ok"] else 1

    if args.command == "fit":
        model, collator, _config, device = load_scorer(args.checkpoint, args.device)
        report = fit_report(model, collator, args.calibration, device, batch_size=args.batch_size)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(dump(report) + "\n", encoding="utf-8")
        _emit(report, None)
        return 0

    if args.command == "test":
        model, collator, _config, device = load_scorer(args.checkpoint, args.device)
        temperature = read_temperature(args.temperature)
        report = held_out_report(model, collator, args.test, temperature, device, batch_size=args.batch_size)
        _emit(report, args.json_out)
        return 0

    if args.command == "predict":
        model, collator, _config, device = load_scorer(args.checkpoint, args.device)
        temperature = read_temperature(args.temperature)
        options = validate_options(args.option)
        report = predict_report(
            model, collator, args.context, options, temperature, device,
            permute_check=args.permute_check,
        )
        if args.label is not None:
            if not 0 <= args.label < len(options):
                raise ValueError(f"--label {args.label} is not an option index")
            report["matches_label"] = report["choice"] == options[args.label]
            report["label"] = args.label
        _emit(report, args.json_out)
        return 0

    if args.command == "run-all":
        # run_all owns path resolution: it decides the single absolute root the step
        # subprocesses (cwd=REPO_ROOT) and this process both use.
        return run_all(
            args.out,
            force=args.force,
            encoder=args.encoder,
            hf_model=args.hf_model,
            hf_revision=args.hf_revision,
            data_directory=args.data_directory,
            eval_batch_size=args.eval_batch_size,
            device="cpu" if args.device == "auto" else args.device,
            sizes={"train": args.train, "validation": args.validation,
                   "calibration": args.calibration, "test": args.test},
            data_seed=args.seed,
            train_seed=args.train_seed,
            width=args.width,
            rank=args.rank,
            context_tokens=args.context_tokens,
            option_tokens=args.option_tokens,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            predict_context=args.predict_context,
            predict_options=args.predict_options,
        )

    parser.error(f"unknown command {args.command!r}")  # unreachable
    return 2


if __name__ == "__main__":
    main()