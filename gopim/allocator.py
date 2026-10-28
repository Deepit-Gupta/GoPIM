from dataclasses import dataclass
import heapq
import math

from .config import HardwareConfig
from .stages import StageSpec, crossbars_needed


@dataclass(frozen=True)
class Allocation:
    stage_id: str
    replicas: int
    predicted_seconds: float
    useful_replica_limit: int
    crossbars_per_replica: int


@dataclass(frozen=True)
class AllocationResult:
    allocations: dict[str, Allocation]
    crossbars_used: int
    crossbars_available: int
    initial_bottleneck_seconds: float
    final_bottleneck_seconds: float


def allocate_replicas(
    stages: list[StageSpec],
    predicted_seconds: dict[str, float],
    config: HardwareConfig,
    *,
    crossbar_budget: int | None = None,
) -> AllocationResult:
    """Greedily reduce the current HP bottleneck using replica marginal gains."""
    if not stages:
        raise ValueError("At least one stage is required")
    if len({stage.stage_id for stage in stages}) != len(stages):
        raise ValueError("Stage IDs must be unique")
    if set(predicted_seconds) != {stage.stage_id for stage in stages}:
        raise ValueError("A predicted duration is required for every stage")
    if any(not math.isfinite(value) or value <= 0 for value in predicted_seconds.values()):
        raise ValueError("Predicted durations must be finite and positive")
    config.validate_resource_capacity()

    hardware_budget = min(
        config.total_crossbars,
        config.crossbar_resource_limit_bytes // config.bytes_per_crossbar,
    )
    budget = hardware_budget if crossbar_budget is None else crossbar_budget
    if budget <= 0 or budget > hardware_budget:
        raise ValueError(
            f"crossbar_budget must be between 1 and {hardware_budget}"
        )
    crossbars_per_replica = {
        stage.stage_id: crossbars_needed(
            stage, config.crossbar_rows, config.crossbar_cols
        )
        for stage in stages
    }
    initial_resources = sum(crossbars_per_replica.values())
    if initial_resources > budget:
        raise ValueError(
            "Crossbar budget cannot hold one mapping per stage: "
            f"{initial_resources} > {budget}"
        )

    replica_limit = {
        stage.stage_id: max(
            1,
            min(
                budget // crossbars_per_replica[stage.stage_id],
                1 + (budget - initial_resources) // crossbars_per_replica[stage.stage_id],
            ),
        )
        for stage in stages
    }
    initial = {stage.stage_id: predicted_seconds[stage.stage_id] for stage in stages}
    replicas = {stage.stage_id: 1 for stage in stages}
    version = {stage.stage_id: 0 for stage in stages}
    hp: list[tuple[float, int, str]] = []
    hv: list[tuple[float, int, str]] = []

    def push(stage_id: str) -> None:
        current_time = initial[stage_id] / replicas[stage_id]
        heapq.heappush(hp, (-current_time, version[stage_id], stage_id))
        if replicas[stage_id] < replica_limit[stage_id]:
            gain = initial[stage_id] / replicas[stage_id] - initial[stage_id] / (
                replicas[stage_id] + 1
            )
            heapq.heappush(hv, (-gain, version[stage_id], stage_id))

    def resources_required(target_seconds: float) -> int:
        if target_seconds <= 0:
            return budget + 1
        total = 0
        for stage_id, duration in initial.items():
            ratio = duration / target_seconds
            required = max(
                1,
                math.ceil(ratio - 1e-10 * max(1.0, ratio)),
            )
            total += min(required, replica_limit[stage_id]) * crossbars_per_replica[
                stage_id
            ]
        return total

    # Batch the repeated heap pops that would lower the bottleneck to the same
    # time level; the heap loop below allocates any remaining indivisible budget.
    low, high = 0.0, max(initial.values())
    for _ in range(100):
        middle = (low + high) / 2
        if resources_required(middle) <= budget:
            high = middle
        else:
            low = middle
    for stage_id, duration in initial.items():
        ratio = duration / high
        replicas[stage_id] = min(
            replica_limit[stage_id],
            max(
                1,
                math.ceil(ratio - 1e-10 * max(1.0, ratio)),
            ),
        )
        version[stage_id] = replicas[stage_id] - 1

    for stage in stages:
        push(stage.stage_id)

    initial_bottleneck = max(initial.values())
    used = sum(
        replicas[stage_id] * crossbars_per_replica[stage_id]
        for stage_id in replicas
    )
    while True:
        while hp:
            negative_time, current_version, stage_id = heapq.heappop(hp)
            if (
                current_version == version[stage_id]
                and replicas[stage_id] < replica_limit[stage_id]
            ):
                break
        else:
            break

        deferred: list[tuple[float, int, str]] = []
        selected_gain = 0.0
        while hv:
            candidate = heapq.heappop(hv)
            negative_gain, candidate_version, candidate_id = candidate
            if candidate_version != version[candidate_id]:
                continue
            if candidate_id == stage_id:
                selected_gain = -negative_gain
                break
            deferred.append(candidate)
        for candidate in deferred:
            heapq.heappush(hv, candidate)
        if selected_gain <= 0:
            break
        resource_cost = crossbars_per_replica[stage_id]
        affordable_replicas = (budget - used) // resource_cost
        if affordable_replicas <= 0:
            replica_limit[stage_id] = replicas[stage_id]
            continue

        other_bottleneck = max(
            (
                initial[other_id] / replicas[other_id]
                for other_id in initial
                if other_id != stage_id
            ),
            default=math.inf,
        )
        if math.isinf(other_bottleneck):
            target_replicas = replica_limit[stage_id]
        else:
            target_replicas = max(
                replicas[stage_id] + 1,
                math.ceil(initial[stage_id] / other_bottleneck),
            )
        batch_size = min(
            affordable_replicas,
            replica_limit[stage_id] - replicas[stage_id],
            target_replicas - replicas[stage_id],
        )
        if batch_size <= 0:
            batch_size = 1
        replicas[stage_id] += batch_size
        used += batch_size * resource_cost
        version[stage_id] += 1
        push(stage_id)

    allocations = {
        stage.stage_id: Allocation(
            stage.stage_id,
            replicas[stage.stage_id],
            initial[stage.stage_id] / replicas[stage.stage_id],
            replica_limit[stage.stage_id],
            crossbars_per_replica[stage.stage_id],
        )
        for stage in stages
    }
    return AllocationResult(
        allocations,
        used,
        budget,
        initial_bottleneck,
        max(item.predicted_seconds for item in allocations.values()),
    )
