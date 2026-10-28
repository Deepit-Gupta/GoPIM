from dataclasses import dataclass

from .allocator import allocate_replicas
from .config import HardwareConfig
from .hardware import StageTimingModel
from .pipeline import PipelineScheduler
from .stages import StageSpec, crossbars_needed


@dataclass(frozen=True)
class SchemeResult:
    scheme: str
    replicas: dict[str, int]
    crossbars_used: int
    makespan_seconds: float
    read_array_energy_joules: float
    write_array_energy_joules: float
    speedup_vs_serial: float


def _static_equal_replicas(
    stages: list[StageSpec], config: HardwareConfig, budget: int
) -> tuple[dict[str, int], int]:
    costs = {
        stage.stage_id: crossbars_needed(
            stage, config.crossbar_rows, config.crossbar_cols
        )
        for stage in stages
    }
    minimum = sum(costs.values())
    if minimum > budget:
        raise ValueError(
            "Static pipelining needs one mapping per stage: "
            f"{minimum} crossbars required, {budget} available"
        )

    replicas_per_stage = max(1, budget // minimum)
    replicas = {stage.stage_id: replicas_per_stage for stage in stages}
    used = replicas_per_stage * minimum
    for stage in stages:
        cost = costs[stage.stage_id]
        if used + cost <= budget:
            replicas[stage.stage_id] += 1
            used += cost
    return replicas, used


def compare_allocation_schemes(
    stages: list[StageSpec],
    config: HardwareConfig,
    *,
    micro_batches: int = 4,
    crossbar_budget: int | None = None,
) -> list[SchemeResult]:
    if not stages:
        raise ValueError("At least one stage is required")
    if micro_batches <= 0:
        raise ValueError("micro_batches must be positive")

    hardware_budget = min(
        config.total_crossbars,
        config.crossbar_resource_limit_bytes // config.bytes_per_crossbar,
    )
    budget = hardware_budget if crossbar_budget is None else crossbar_budget
    if budget <= 0 or budget > hardware_budget:
        raise ValueError(
            f"crossbar_budget must be between 1 and {hardware_budget}"
        )

    model = StageTimingModel(config)
    baseline = {
        stage.stage_id: model.estimate(stage, replicas=1)
        for stage in stages
    }
    read_array_energy_joules = micro_batches * sum(
        item.read_array_energy_joules for item in baseline.values()
    )
    write_array_energy_joules = micro_batches * sum(
        item.write_array_energy_joules for item in baseline.values()
    )
    minimum_pipeline_budget = sum(
        crossbars_needed(stage, config.crossbar_rows, config.crossbar_cols)
        for stage in stages
    )
    if minimum_pipeline_budget > budget:
        raise ValueError(
            "Pipelined schemes need one mapping per stage: "
            f"{minimum_pipeline_budget} crossbars required, {budget} available"
        )

    serial_replicas = {
        stage.stage_id: max(
            1,
            budget
            // crossbars_needed(
                stage, config.crossbar_rows, config.crossbar_cols
            ),
        )
        for stage in stages
    }
    serial_stage_times = {
        stage.stage_id: model.estimate(
            stage, replicas=serial_replicas[stage.stage_id]
        ).seconds
        for stage in stages
    }
    serial_seconds = micro_batches * sum(serial_stage_times.values())
    serial_crossbars = max(
        serial_replicas[stage.stage_id]
        * crossbars_needed(stage, config.crossbar_rows, config.crossbar_cols)
        for stage in stages
    )
    serial = SchemeResult(
        "serial",
        serial_replicas,
        serial_crossbars,
        serial_seconds,
        read_array_energy_joules,
        write_array_energy_joules,
        1.0,
    )

    static_replicas, static_used = _static_equal_replicas(
        stages, config, budget
    )
    dynamic_allocation = allocate_replicas(
        stages,
        {stage_id: item.seconds for stage_id, item in baseline.items()},
        config,
        crossbar_budget=budget,
    )

    def pipeline_result(
        scheme: str, replicas: dict[str, int], crossbars_used: int
    ) -> SchemeResult:
        scheduler = PipelineScheduler()
        durations = [
            (
                stage.stage_id,
                model.estimate(stage, replicas=replicas[stage.stage_id]).seconds,
            )
            for stage in stages
        ]
        makespan = 0.0
        for micro_batch in range(micro_batches):
            makespan = scheduler.schedule(micro_batch, durations)
        return SchemeResult(
            scheme,
            replicas,
            crossbars_used,
            makespan,
            read_array_energy_joules,
            write_array_energy_joules,
            serial_seconds / makespan,
        )

    static = pipeline_result(
        "slimgnn_like_static_equal_replication", static_replicas, static_used
    )
    gopim = pipeline_result(
        "gopim_greedy_bottleneck",
        {
            stage_id: item.replicas
            for stage_id, item in dynamic_allocation.allocations.items()
        },
        dynamic_allocation.crossbars_used,
    )
    return [serial, static, gopim]
