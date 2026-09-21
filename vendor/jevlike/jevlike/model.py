"""One-pass option scorers with tiny, frozen, or adapter-tuned pretrained encoders."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import torch
from torch import nn


class AttentionHead(nn.Module):
    """Turn context tokens and option vectors into one score per option."""

    def __init__(self, input_width: int, rank: int) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(input_width)
        self.option_norm = nn.LayerNorm(input_width)
        self.query = nn.Linear(input_width, rank, bias=False)
        self.key = nn.Linear(input_width, rank, bias=False)
        self.value = nn.Linear(input_width, rank, bias=False)
        self.rank = rank

    def forward(
        self, context: torch.Tensor, context_mask: torch.Tensor,
        options: torch.Tensor, option_mask: torch.Tensor,
        shuffle_context: bool = False,
    ) -> torch.Tensor:
        context = self.context_norm(context.float())
        options = self.option_norm(options.float())
        if shuffle_context and context.shape[0] > 1:
            context = context.roll(1, dims=0)
            context_mask = context_mask.roll(1, dims=0)
        query = self.query(options)
        key = self.key(context)
        value = self.value(context)
        scores = torch.einsum("bnr,blr->bnl", query, key) / math.sqrt(self.rank)
        scores = scores.masked_fill(
            ~context_mask[:, None, :], torch.finfo(scores.dtype).min
        )
        attended = torch.einsum("bnl,blr->bnr", scores.softmax(-1), value)
        logits = (query * attended).sum(-1) / math.sqrt(self.rank)
        return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)


class TinyScorer(nn.Module):
    def __init__(self, width: int, rank: int, context_tokens: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.position = nn.Embedding(context_tokens, width)
        self.head = AttentionHead(width, rank)

    def forward(self, batch: dict[str, torch.Tensor], shuffle_context: bool = False):
        context_ids = batch["context_ids"]
        positions = torch.arange(context_ids.shape[1], device=context_ids.device)
        context = self.embedding(context_ids) + self.position(positions)
        option_tokens = self.embedding(batch["option_ids"])
        weights = batch["option_token_mask"].unsqueeze(-1)
        options = (option_tokens * weights).sum(2) / weights.sum(2).clamp_min(1)
        return self.head(
            context, batch["context_mask"], options, batch["option_mask"],
            shuffle_context,
        )


class FrozenTransformerScorer(nn.Module):
    """Score options with a pretrained encoder that is frozen or adapter-tuned.

    ``lora_rank=None`` (the upstream default) is the frozen route: the encoder is a
    fixed feature extractor, forced into eval mode and ``torch.no_grad()`` on every
    forward so a training loop can never reach it. Passing ``lora_rank`` is the
    adapter route: PEFT injects LoRA adapters into ``q_proj``/``v_proj``, the
    pretrained weights stay frozen, the adapters and the head are what trains, and
    the encoder keeps whatever mode the caller set with ``train()``/``eval()``.
    """

    def __init__(
        self, model_name: str, rank: int, revision: str | None = None,
        lora_rank: int | None = None, lora_alpha: int = 32,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        # revision=None keeps the hub default branch, so unpinned callers are unchanged.
        self.encoder = AutoModel.from_pretrained(model_name, revision=revision)
        # Frozen weights first: the adapters injected below are created afterwards and
        # keep requires_grad=True, so they end up the encoder's only trainable ones.
        self.encoder.requires_grad_(False)
        self.encoder_frozen = lora_rank is None
        if self.encoder_frozen:
            self.encoder.eval()
        else:
            self._add_lora(lora_rank, lora_alpha)
        self.head = AttentionHead(self.encoder.config.hidden_size, rank)

    def _add_lora(self, lora_rank: int, lora_alpha: int) -> None:
        from peft import LoraConfig, TaskType, inject_adapter_in_model

        # Injected in place rather than wrapped in a PeftModel: the encoder stays the
        # plain transformers model, so both routes run the one forward below.
        inject_adapter_in_model(
            LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"], bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            ),
            self.encoder,
        )

    def _encoder_scope(self) -> AbstractContextManager[None]:
        """Frozen route: eval plus no_grad. Adapter route: leave the caller's mode."""
        if self.encoder_frozen:
            self.encoder.eval()
            return torch.no_grad()
        return nullcontext()

    def forward(self, batch: dict[str, torch.Tensor], shuffle_context: bool = False):
        with self._encoder_scope():
            context = self.encoder(
                input_ids=batch["context_ids"],
                attention_mask=batch["context_mask"],
            ).last_hidden_state
            shape = batch["option_ids"].shape
            flat_ids = batch["option_ids"].reshape(-1, shape[-1])
            flat_mask = batch["option_token_mask"].reshape(-1, shape[-1])
            hidden = self.encoder(
                input_ids=flat_ids, attention_mask=flat_mask,
            ).last_hidden_state
            pooled = (hidden * flat_mask.unsqueeze(-1)).sum(1)
            pooled = pooled / flat_mask.sum(1, keepdim=True).clamp_min(1)
            options = pooled.reshape(shape[0], shape[1], -1)
        return self.head(
            context, batch["context_mask"], options, batch["option_mask"],
            shuffle_context,
        )


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_system(config: dict, device: torch.device):
    from .data import ByteCollator, HuggingFaceCollator

    if config["encoder"] == "tiny":
        model = TinyScorer(
            config["width"], config["rank"], config["context_tokens"]
        )
        collator = ByteCollator(config["context_tokens"], config["option_tokens"])
    elif config["encoder"] in ("hf", "lora"):
        from transformers import AutoTokenizer

        # "hf" leaves the pretrained weights frozen; "lora" adds trainable adapters to
        # the same encoder, built from the same pinned revision and tokenizer. The
        # adapter settings are read only for "lora", so a stray key can never turn the
        # frozen route into a trainable one.
        lora = (
            {"lora_rank": config.get("lora_rank", 16),
             "lora_alpha": config.get("lora_alpha", 32)}
            if config["encoder"] == "lora" else {}
        )
        name = config["hf_model"]
        revision = config.get("hf_revision")
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = FrozenTransformerScorer(name, config["rank"], revision, **lora)
        collator = HuggingFaceCollator(
            tokenizer, config["context_tokens"], config["option_tokens"]
        )
    else:
        raise ValueError(f"unknown encoder {config['encoder']!r}")
    return model.to(device), collator


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def load_checkpoint(path: str | Path, device: torch.device):
    """Load a checkpoint written by :func:`jevlike.train.main`.

    The file must hold a plain ``{"config": {...}, "state_dict": {name: tensor}}``
    payload. ``weights_only=True`` makes the loader refuse arbitrary pickled
    objects, so an untrusted checkpoint fails here instead of executing code.
    Checkpoints that carry extra Python objects must be re-saved in that plain
    form; this loader never falls back to ``weights_only=False``.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # unpicklable payload, corrupt or missing file
        raise ValueError(
            f"refusing to load {path}: not a plain-tensor checkpoint ({error})"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} must be a dict")
    for key in ("config", "state_dict"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"checkpoint {path} needs a '{key}' dict")
    if not all(
        isinstance(name, str) and isinstance(weight, torch.Tensor)
        for name, weight in payload["state_dict"].items()
    ):
        raise ValueError(f"checkpoint {path} state_dict must map names to tensors")
    for key in ("encoder", "hf_model", "width", "rank", "context_tokens", "option_tokens"):
        if key not in payload["config"]:
            raise ValueError(f"checkpoint {path} config is missing '{key}'")
    if payload["config"]["encoder"] == "lora":
        revision = payload["config"].get("hf_revision")
        if not isinstance(revision, str) or len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise ValueError("LoRA checkpoint requires a pinned 40-character hf_revision")
    model, collator = make_system(payload["config"], device)
    try:
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    except RuntimeError as error:
        # Shape disagreement (e.g. adapter rank differs from the configured one) is a
        # mismatch like any other, so it raises ValueError rather than a bare RuntimeError.
        raise ValueError(f"checkpoint {path} does not fit the model: {error}") from error
    # Checkpoints carry the trainable weights only, so the frozen encoder is expected
    # to be absent and gets rebuilt from the hub. LoRA adapters are stored under
    # "encoder." too, hence the test is "does this name train", not "is it the
    # encoder": a stripped or corrupt adapter must not pass as a missing frozen block.
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    required_missing = [name for name in missing if name in trainable]
    if unexpected or required_missing:
        raise ValueError(
            f"checkpoint mismatch: missing={required_missing}, unexpected={unexpected}"
        )
    return model, collator, payload["config"]
