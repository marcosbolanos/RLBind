from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "torch is required for embedding. Install it with your CUDA/CPU setup."
        ) from exc
    return torch


def _require_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "transformers is required for embedding. Install it with `uv add transformers`."
        ) from exc
    return AutoModel, AutoTokenizer


def _pool_embeddings(hidden_state, attention_mask, pooling: str):
    torch = _require_torch()
    if pooling == "cls":
        return hidden_state[:, 0, :]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        summed = (hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        return summed / denom
    raise ValueError(f"Unsupported pooling: {pooling}")


@dataclass
class HFEmbedder:
    model_name: str
    device: str
    pooling: str
    max_length: int | None
    trust_remote_code: bool

    def __post_init__(self) -> None:
        AutoModel, AutoTokenizer = _require_transformers()
        torch = _require_torch()

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._model.to(self._device)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            outputs = self._model(**encoded)
        hidden_state = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        pooled = _pool_embeddings(hidden_state, encoded["attention_mask"], self.pooling)
        return pooled.float().cpu().numpy()
