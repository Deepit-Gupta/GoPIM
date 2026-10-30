from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ISUPlan:
    degrees: tuple[int, ...]
    ordered_vertices: tuple[int, ...]
    selected_vertices: frozenset[int]
    assignments: dict[int, tuple[int, ...]]
    average_degree: float
    update_fraction: float
    low_degree_update_period: int

    def should_update(self, vertex_id: int, epoch: int) -> bool:
        if not 0 <= vertex_id < len(self.degrees):
            raise IndexError("vertex_id is outside the graph")
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        if vertex_id in self.selected_vertices:
            return True
        return (epoch + 1) % self.low_degree_update_period == 0


def loss_aware_update_mask(
    plan: ISUPlan,
    epoch: int,
    node_losses: np.ndarray | None,
    *,
    loss_weight: float = 0.75,
) -> np.ndarray:
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    if not 0 <= loss_weight <= 1:
        raise ValueError("loss_weight must be in [0, 1]")
    if node_losses is None:
        losses = np.full(len(plan.degrees), np.nan, dtype=np.float64)
    else:
        losses = np.asarray(node_losses, dtype=np.float64)
        if losses.shape != (len(plan.degrees),):
            raise ValueError("node_losses must contain one value per vertex")

    degrees = np.asarray(plan.degrees, dtype=np.float64)
    degree_order = np.lexsort((np.arange(len(degrees)), -degrees))
    degree_rank = np.empty(len(degrees), dtype=np.float64)
    degree_rank[degree_order] = np.linspace(1.0, 0.0, len(degrees))

    observed = np.isfinite(losses)
    score = degree_rank.copy()
    if observed.any():
        observed_indices = np.flatnonzero(observed)
        loss_order = np.lexsort((observed_indices, -losses[observed]))
        loss_rank = np.linspace(1.0, 0.0, len(observed_indices))
        score[observed_indices[loss_order]] = (
            loss_weight * loss_rank
            + (1.0 - loss_weight) * degree_rank[observed_indices[loss_order]]
        )

    selected_count = len(plan.selected_vertices)
    selected_order = np.lexsort((np.arange(len(score)), -score))
    selected = np.zeros(len(score), dtype=bool)
    selected[selected_order[:selected_count]] = True
    if (epoch + 1) % plan.low_degree_update_period == 0:
        return np.ones(len(score), dtype=bool)
    return selected


def build_isu_plan(
    edge_index,
    num_vertices: int,
    crossbar_count: int,
    *,
    sparse_threshold: float = 8.0,
    sparse_fraction: float = 0.8,
    dense_fraction: float = 0.5,
    low_degree_update_period: int = 20,
) -> ISUPlan:
    if num_vertices <= 0 or crossbar_count <= 0:
        raise ValueError("num_vertices and crossbar_count must be positive")
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((2, 0), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edge_count]")
    if edges.size and (edges.min() < 0 or edges.max() >= num_vertices):
        raise ValueError("edge_index contains an out-of-range vertex")
    if low_degree_update_period <= 0:
        raise ValueError("low_degree_update_period must be positive")
    for fraction in (sparse_fraction, dense_fraction):
        if not 0 < fraction <= 1:
            raise ValueError("update fractions must be in (0, 1]")

    degrees = np.bincount(edges[0], minlength=num_vertices)
    average_degree = float(degrees.mean())
    fraction = sparse_fraction if average_degree <= sparse_threshold else dense_fraction
    selected_count = min(num_vertices, math.ceil(fraction * num_vertices))
    ordered = np.lexsort((np.arange(num_vertices), -degrees))
    selected_order = ordered[:selected_count]
    selected = frozenset(int(vertex) for vertex in selected_order)

    effective_crossbars = min(crossbar_count, selected_count)
    interval_size = math.ceil(selected_count / effective_crossbars)
    assignments: dict[int, list[int]] = {
        crossbar_id: [] for crossbar_id in range(effective_crossbars)
    }
    for interval_start in range(0, selected_count, effective_crossbars):
        interval = selected_order[interval_start : interval_start + effective_crossbars]
        for crossbar_id, vertex in enumerate(interval):
            assignments[crossbar_id].append(int(vertex))

    return ISUPlan(
        tuple(int(value) for value in degrees),
        tuple(int(value) for value in ordered),
        selected,
        {key: tuple(value) for key, value in assignments.items()},
        average_degree,
        fraction,
        low_degree_update_period,
    )
