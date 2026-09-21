"""Evaluate accuracy, calibration and a shuffled-context control."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .data import JsonlDataset
from .model import load_checkpoint, select_device
from .train import move


@torch.no_grad()
def metrics(model, loader, device, shuffle_context=False):
    model.eval()
    top1 = top3 = total = 0
    confidences, predictions, labels = [], [], []
    for host_batch in loader:
        batch = move(host_batch, device)
        logits = model(batch, shuffle_context=shuffle_context).cpu()
        probabilities = logits.softmax(-1)
        ranking = logits.topk(min(3, logits.shape[1]), dim=-1).indices
        batch_labels = batch["labels"].cpu()
        top1 += int(ranking[:, 0].eq(batch_labels).sum())
        top3 += int(ranking.eq(batch_labels[:, None]).any(1).sum())
        total += batch_labels.numel()
        confidence, prediction = probabilities.max(-1)
        confidences.append(confidence)
        predictions.append(prediction)
        labels.append(batch["labels"].cpu())
    confidence, prediction, labels = map(
        torch.cat, (confidences, predictions, labels)
    )
    # Fixed bin ownership: bin = floor(p * 10) clamped to [0, 9]. The upstream loop
    # tested "lower <= p < lower + 0.1" against linspace boundaries, which dropped
    # confidence exactly 1.0 and relied on float32 equality with those boundaries.
    bins = (confidence * 10).floor().clamp(0, 9).long()
    ece = 0.0
    for index in range(10):
        selected = bins.eq(index)
        if selected.any():
            gap = prediction[selected].eq(labels[selected]).float().mean()
            gap -= confidence[selected].mean()
            ece += float(selected.float().mean() * gap.abs())
    return {
        "top1": top1 / total,
        "top3": top3 / total,
        "ece": ece, "examples": labels.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model, collator, _ = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(
        JsonlDataset(args.data), batch_size=args.batch_size, collate_fn=collator,
    )
    print(json.dumps({
        "model": metrics(model, loader, device),
        "shuffled_context": metrics(model, loader, device, True),
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
