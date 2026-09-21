"""Train a scorer from JSONL examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import JsonlDataset
from .model import make_system, select_device, trainable_state


def move(batch, device):
    return {name: tensor.to(device) for name, tensor in batch.items()}


@torch.no_grad()
def mean_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = move(batch, device)
        loss = F.cross_entropy(model(batch), batch["labels"], reduction="sum")
        total += float(loss)
        count += batch["labels"].numel()
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="runs/model.pt")
    parser.add_argument("--encoder", choices=("tiny", "hf", "lora"), default="tiny")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=192)
    parser.add_argument("--option-tokens", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    config = {
        "encoder": args.encoder, "hf_model": args.hf_model,
        "width": args.width, "rank": args.rank,
        "context_tokens": args.context_tokens, "option_tokens": args.option_tokens,
    }
    if args.hf_revision is not None:
        config["hf_revision"] = args.hf_revision
    if args.encoder == "lora":
        config["lora_rank"] = args.lora_rank
        config["lora_alpha"] = args.lora_alpha
    model, collator = make_system(config, device)
    train_loader = DataLoader(
        JsonlDataset(args.train), batch_size=args.batch_size, shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        JsonlDataset(args.validation), batch_size=args.batch_size,
        collate_fn=collator,
    )
    scheduler = None
    if args.encoder == "lora":
        encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
        head_parameters = [p for p in model.head.parameters() if p.requires_grad]
        parameters = encoder_parameters + head_parameters
        optimiser = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": args.learning_rate},
                {"params": head_parameters, "lr": args.head_learning_rate},
            ],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr=[group["lr"] for group in optimiser.param_groups],
            total_steps=args.epochs * len(train_loader),
        )
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    best_loss, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        epoch_lrs = (
            [group["lr"] for group in optimiser.param_groups]
            if scheduler is not None else None
        )
        total, count = 0.0, 0
        for host_batch in train_loader:
            batch = move(host_batch, device)
            loss = F.cross_entropy(model(batch), batch["labels"])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach()) * batch["labels"].numel()
            count += batch["labels"].numel()
        validation_loss = mean_loss(model, validation_loader, device)
        if validation_loss < best_loss:
            best_loss, best_state = validation_loss, trainable_state(model)
        record = {
            "epoch": epoch + 1, "train_nll": total / count,
            "validation_nll": validation_loss, "device": str(device),
        }
        if epoch_lrs is not None:
            record["adapter_lr"] = epoch_lrs[0]
            record["head_lr"] = epoch_lrs[1]
        print(json.dumps(record))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": best_state}, output)
    print(json.dumps({"checkpoint": str(output), "best_validation_nll": best_loss}))


if __name__ == "__main__":
    main()
