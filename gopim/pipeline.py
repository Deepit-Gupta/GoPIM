from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineEvent:
    micro_batch: int
    stage_id: str
    start_seconds: float
    end_seconds: float


class PipelineScheduler:
    """Flow-shop schedule for inter-micro-batch and intra-micro-batch overlap."""

    def __init__(self) -> None:
        self._stage_finish: dict[str, float] = {}
        self.events: list[PipelineEvent] = []

    def schedule(self, micro_batch: int, stage_durations: list[tuple[str, float]]) -> float:
        if micro_batch < 0:
            raise ValueError("micro_batch cannot be negative")
        previous_stage_finish = 0.0
        for stage_id, duration in stage_durations:
            if duration <= 0:
                raise ValueError("Stage duration must be positive")
            start = max(
                previous_stage_finish,
                self._stage_finish.get(stage_id, 0.0),
            )
            finish = start + duration
            self.events.append(PipelineEvent(micro_batch, stage_id, start, finish))
            self._stage_finish[stage_id] = finish
            previous_stage_finish = finish
        return previous_stage_finish
