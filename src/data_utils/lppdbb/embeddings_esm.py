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


def _require_esm(model_name: str):
    try:
        import esm
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "esm is required for ESM embeddings. Install it with `uv add esm`."
        ) from exc

    esm_version = getattr(esm, "__version__", "unknown")
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "ESMC is not available in this esm install "
            f"(version {esm_version}). "
            "Update esm to an ESMC-enabled release."
        ) from exc
    return ESMC, ESMProtein, LogitsConfig


def _pool_sequence_embeddings(array: np.ndarray, pooling: str) -> np.ndarray:
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Unexpected embeddings shape: {array.shape}")
    if pooling == "cls":
        return array[0]
    if pooling == "mean":
        return array.mean(axis=0)
    raise ValueError(f"Unsupported pooling: {pooling}")


@dataclass
class ESMEmbedder:
    model_name: str
    device: str
    pooling: str

    def __post_init__(self) -> None:
        torch = _require_torch()
        ESMC, ESMProtein, LogitsConfig = _require_esm(self.model_name)

        self._torch = torch
        self._ESMProtein = ESMProtein
        self._logits_config = LogitsConfig(sequence=True, return_embeddings=True)
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._client = ESMC.from_pretrained(self.model_name).to(self._device)

    def _to_numpy(self, tensor) -> np.ndarray:
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().float().numpy()
        return np.asarray(tensor, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embeddings: list[np.ndarray] = []
        for sequence in texts:
            protein = self._ESMProtein(sequence=sequence)
            protein_tensor = self._client.encode(protein)
            outputs = self._client.logits(protein_tensor, self._logits_config)
            pooled = _pool_sequence_embeddings(
                self._to_numpy(outputs.embeddings),
                self.pooling,
            )
            embeddings.append(pooled)
        return np.stack(embeddings, axis=0)
