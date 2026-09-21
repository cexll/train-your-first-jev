"""Score one context against command-line options."""

from __future__ import annotations

import argparse
import json

import torch

from .data import ChoiceExample
from .model import load_checkpoint, select_device
from .train import move


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--context", required=True)
    parser.add_argument("--option", action="append", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if len(args.option) < 2:
        parser.error("pass --option at least twice")
    device = select_device(args.device)
    model, collator, _ = load_checkpoint(args.checkpoint, device)
    batch = move(collator([
        ChoiceExample(args.context, tuple(args.option), 0)
    ]), device)
    model.eval()
    with torch.no_grad():
        probabilities = model(batch).softmax(-1)[0, :len(args.option)].cpu().tolist()
    print(json.dumps([
        {"option": option, "probability": probability}
        for option, probability in zip(args.option, probabilities)
    ], indent=2))


if __name__ == "__main__":
    main()
