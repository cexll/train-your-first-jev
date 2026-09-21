"""Screen-dependent Jev-like scorer for ViZDoom."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from jevlike.model import AttentionHead
from torch import nn


DOOM_OPTION_IDS = tuple(range(7))
CHESS_OPTION_IDS = tuple(range(7, 12))
TOTAL_OPTIONS = 12


def position_2d(rows: int, columns: int, width: int) -> torch.Tensor:
    """Fixed separable 2D sinusoidal positions with shape (rows * columns, width)."""
    if width % 4:
        raise ValueError("width must be divisible by four")
    quarter = width // 4
    frequency = torch.exp(-math.log(10_000) * torch.arange(quarter) / max(1, quarter - 1))
    y = torch.arange(rows)[:, None] * frequency[None, :]
    x = torch.arange(columns)[:, None] * frequency[None, :]
    row = torch.cat((y.sin(), y.cos()), -1)[:, None, :].expand(-1, columns, -1)
    column = torch.cat((x.sin(), x.cos()), -1)[None, :, :].expand(rows, -1, -1)
    return torch.cat((row, column), -1).reshape(rows * columns, width)


class DoomScorerV2(nn.Module):
    """Score buttons from RGB and motion patches with explicit positions in K."""

    def __init__(self, actions: int = 7, width: int = 32, rank: int = 32,
                 reads: int = 1) -> None:
        super().__init__()
        if width != rank:
            raise ValueError("v2 uses width == rank so fixed positions can enter K directly")
        if reads < 1:
            raise ValueError("reads must be positive")
        self.width = width
        self.rank = rank
        self.reads = reads
        self.grid_rows = 8
        self.grid_columns = 10
        self.stem = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, width, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
        )
        self.patch = nn.Conv2d(width, width, kernel_size=4, stride=4)
        self.options = nn.Embedding(actions, width)
        self.head = AttentionHead(width, rank)
        self.extra_heads = nn.ModuleList(
            AttentionHead(width, rank) for _ in range(reads - 1)
        )
        self.value_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.register_buffer("positions", position_2d(self.grid_rows, self.grid_columns, width))

    def encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = observations[:, :3]
        motion = observations[:, 3:4]
        mean = rgb.mean(dim=(2, 3), keepdim=True)
        std = rgb.std(dim=(2, 3), keepdim=True).clamp_min(0.08)
        rgb = (rgb - mean) / std
        motion = (motion - 0.5) * 2.0
        image = torch.cat((rgb, motion), 1)
        image = F.pad(image, (0, 0, 4, 4))
        features = self.patch(self.stem(image)).permute(0, 2, 3, 1).flatten(1, 2)
        positions = self.positions.to(features.dtype).unsqueeze(0).expand(len(features), -1, -1)
        return features, positions

    def forward(self, observations: torch.Tensor,
                option_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value, _ = self._forward(observations, option_ids, capture=False)
        return logits, value

    def forward_trace(self, observations: torch.Tensor,
                      option_ids: torch.Tensor | None = None):
        return self._forward(observations, option_ids, capture=True)

    def _forward(self, observations: torch.Tensor,
                 option_ids: torch.Tensor | None, capture: bool):
        raw_context, positions = self.encode(observations)
        if option_ids is None:
            option_vectors = self.options.weight.unsqueeze(0).expand(len(raw_context), -1, -1)
        else:
            option_ids = option_ids.to(device=raw_context.device, dtype=torch.long)
            if option_ids.ndim == 1:
                option_vectors = self.options(option_ids).unsqueeze(0).expand(len(raw_context), -1, -1)
            elif option_ids.ndim == 2 and option_ids.shape[0] == len(raw_context):
                option_vectors = self.options(option_ids)
            else:
                raise ValueError("option_ids must have shape (N,) or (batch, N)")
        traces = []
        for head in (self.head, *self.extra_heads):
            context = head.context_norm(raw_context.float())
            options = head.option_norm(option_vectors.float())
            query = head.query(options)
            # The fixed term is deliberately added after the learned projection. It
            # cannot be washed out by context normalisation or routed only into V.
            key = head.key(context) + positions
            values = head.value(context)
            scores = torch.einsum("bnr,blr->bnl", query, key) / math.sqrt(self.rank)
            attention = scores.softmax(-1)
            attended = torch.einsum("bnl,blr->bnr", attention, values)
            logits = (query * attended).sum(-1) / math.sqrt(self.rank)
            traces.append((logits, query, key, values, attention))
        logits = torch.stack([row[0] for row in traces]).mean(0)
        value_context = self.head.context_norm(raw_context.float())
        value = self.value_head(value_context.mean(1)).squeeze(-1)
        if not capture:
            return logits, value, {}
        trace = {
            "query_matrix": torch.stack([row[1] for row in traces]).mean(0).detach(),
            "key_matrix": torch.stack([row[2] for row in traces]).mean(0).detach(),
            "value_matrix": torch.stack([row[3] for row in traces]).mean(0).detach(),
            "attention_map": torch.stack([row[4] for row in traces]).mean(0).detach(),
            "logits_matrix": logits.detach(),
            "probabilities": logits.softmax(-1).detach(),
        }
        return logits, value, trace


class PlainConvPolicy(nn.Module):
    """Architecture control: the same visual stem followed by a flat MLP."""

    def __init__(self, actions: int = 7, width: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.width = width
        self.rank = width
        self.actions = actions
        self.stem = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, width, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
        )
        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 32 * 40, hidden),
            nn.SiLU(),
            nn.Linear(hidden, actions),
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 1),
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        rgb = observations[:, :3]
        motion = observations[:, 3:4]
        mean = rgb.mean(dim=(2, 3), keepdim=True)
        std = rgb.std(dim=(2, 3), keepdim=True).clamp_min(0.08)
        image = torch.cat(((rgb - mean) / std, (motion - 0.5) * 2.0), 1)
        return self.stem(F.pad(image, (0, 0, 4, 4)))

    def forward(self, observations: torch.Tensor,
                option_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(observations)
        logits = self.policy(features)
        if option_ids is not None:
            option_ids = option_ids.to(device=logits.device, dtype=torch.long)
            logits = logits[:, option_ids] if option_ids.ndim == 1 else logits.gather(1, option_ids)
        return logits, self.value_head(features).squeeze(-1)

    def forward_trace(self, observations: torch.Tensor,
                      option_ids: torch.Tensor | None = None):
        logits, value = self(observations, option_ids)
        return logits, value, {"probabilities": logits.softmax(-1).detach()}


def observation(frame: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Pack RGB plus a signed greyscale frame-difference channel into uint8."""
    current = np.ascontiguousarray(frame)
    if previous is None:
        difference = np.zeros(current.shape[:2], dtype=np.float32)
    else:
        difference = (current.astype(np.float32) - previous.astype(np.float32)).mean(-1)
    encoded = np.clip(np.rint(difference) + 128, 0, 255).astype(np.uint8)
    return np.concatenate((current, encoded[..., None]), -1)


def observation_tensor(items: list[np.ndarray] | np.ndarray, device: torch.device) -> torch.Tensor:
    array = np.stack(items) if isinstance(items, list) else items
    return torch.from_numpy(np.ascontiguousarray(array)).to(device).permute(0, 3, 1, 2).float() / 255.0


def benchmark(model: DoomScorerV2, item: np.ndarray, device: torch.device,
              runs: int = 200, option_ids: torch.Tensor | None = None) -> float:
    model = model.to(device).eval()
    tensor = observation_tensor(item[None], device)
    if option_ids is not None:
        option_ids = option_ids.to(device)
    with torch.inference_mode():
        for _ in range(20):
            model(tensor, option_ids)
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(tensor, option_ids)
        if device.type == "mps":
            torch.mps.synchronize()
    return (time.perf_counter() - start) * 1000 / runs


def attention_entropy_by_action(model: DoomScorerV2, items: np.ndarray,
                                device: torch.device,
                                option_ids: torch.Tensor | None = None,
                                limit: int = 256) -> list[float]:
    """Normalised live attention entropy, one value per active option."""
    sample = items[:limit]
    with torch.inference_mode():
        _, _, trace = model.forward_trace(observation_tensor(sample, device), option_ids)
    attention = trace["attention_map"].float()
    entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1)
    entropy = entropy / math.log(attention.shape[-1])
    return entropy.mean(0).cpu().tolist()
