from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.data_utils.lppdbb.rdkit.distances import compute_pairwise_distances


def _require_rdkit():
    try:
        from rdkit.ML.Cluster import Butina  # noqa: WPS433  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "RDKit is required for LP-PDBBind clustering. "
            "Install it with `uv sync --extra rdkit`."
        ) from exc
    return Butina


def butina_cluster_distances(
    distances: Sequence[float] | np.ndarray,
    num_items: int,
    *,
    threshold: float = 0.5,
) -> tuple[tuple[int, ...], ...]:
    Butina = _require_rdkit()
    if not num_items:
        return tuple()
    cutoff = 1.0 - threshold
    clusters = Butina.ClusterData(
        distances,
        num_items,
        cutoff,
        isDistData=True,
    )
    return tuple(tuple(cluster) for cluster in clusters)


def butina_cluster_ligands(
    mols: Sequence[Any],
    *,
    threshold: float = 0.5,
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[tuple[int, ...], ...]:
    if not mols:
        return tuple()
    distances = compute_pairwise_distances(mols, radius=radius, n_bits=n_bits)
    return butina_cluster_distances(
        distances,
        len(mols),
        threshold=threshold,
    )


def cluster_membership(
    clusters: Sequence[Sequence[int]],
    item_ids: Sequence[str],
) -> dict[str, int]:
    membership: dict[str, int] = {}
    for cluster_id, members in enumerate(clusters):
        for member_idx in members:
            membership[item_ids[member_idx]] = cluster_id
    return membership
