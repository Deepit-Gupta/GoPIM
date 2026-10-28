from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable


class Operation(str, Enum):
    COMBINATION = "CO"
    AGGREGATION = "AG"


class Direction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(frozen=True)
class StageFeatures:
    ifm_rows_co: int
    ifm_cols_co: int
    weight_rows_co: int
    weight_cols_co: int
    adjacency_rows_ag: int
    adjacency_cols_ag: int
    feature_rows_ag: int
    feature_cols_ag: int
    sparsity: float
    layer: int

    def __post_init__(self) -> None:
        dimensions = (
            self.ifm_rows_co,
            self.ifm_cols_co,
            self.weight_rows_co,
            self.weight_cols_co,
            self.adjacency_rows_ag,
            self.adjacency_cols_ag,
            self.feature_rows_ag,
            self.feature_cols_ag,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Stage matrix dimensions must be positive")
        if not 0 <= self.sparsity <= 1:
            raise ValueError("Graph sparsity must be in [0, 1]")
        if self.layer < 0:
            raise ValueError("Layer index must be non-negative")

    def as_vector(self) -> tuple[float, ...]:
        return (
            float(self.ifm_rows_co),
            float(self.ifm_cols_co),
            float(self.weight_rows_co),
            float(self.weight_cols_co),
            float(self.adjacency_rows_ag),
            float(self.adjacency_cols_ag),
            float(self.feature_rows_ag),
            float(self.feature_cols_ag),
            float(self.sparsity),
            float(self.layer),
        )


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    phase: str
    layer: int
    operation: Operation
    direction: Direction
    features: StageFeatures


def four_stages_per_layer(
    layer_dimensions: Iterable[tuple[int, int]], node_count: int, edge_count: int
) -> list[StageSpec]:
    """Create CO/AG forward and backward stages for each (input, output) width."""
    layer_dimensions = list(layer_dimensions)
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if edge_count < 0:
        raise ValueError("edge_count cannot be negative")

    sparsity = 1.0 - min(1.0, edge_count / max(1, node_count * node_count))
    stages: list[StageSpec] = []
    for layer, (input_width, output_width) in enumerate(layer_dimensions):
        if input_width <= 0 or output_width <= 0:
            raise ValueError("Layer dimensions must be positive")
        co = StageFeatures(
            node_count,
            input_width,
            input_width,
            output_width,
            node_count,
            node_count,
            node_count,
            output_width,
            sparsity,
            layer,
        )
        ag = StageFeatures(
            node_count,
            output_width,
            output_width,
            output_width,
            node_count,
            node_count,
            node_count,
            output_width,
            sparsity,
            layer,
        )
        for operation, features in (
            (Operation.COMBINATION, co),
            (Operation.AGGREGATION, ag),
        ):
            stages.append(
                StageSpec(
                    f"layer-{layer}-{operation.value}-{Direction.FORWARD.value}",
                    operation.value,
                    layer,
                    operation,
                    Direction.FORWARD,
                    features,
                )
            )
    for layer, (input_width, output_width) in reversed(
        list(enumerate(layer_dimensions))
    ):
        co = StageFeatures(
            node_count,
            input_width,
            input_width,
            output_width,
            node_count,
            node_count,
            node_count,
            output_width,
            sparsity,
            layer,
        )
        ag = StageFeatures(
            node_count,
            output_width,
            output_width,
            output_width,
            node_count,
            node_count,
            node_count,
            output_width,
            sparsity,
            layer,
        )
        for operation, features in (
            (Operation.COMBINATION, co),
            (Operation.AGGREGATION, ag),
        ):
            phase = "LC" if operation is Operation.COMBINATION else "GC"
            stages.append(
                StageSpec(
                    f"layer-{layer}-{phase}-backward",
                    phase,
                    layer,
                    operation,
                    Direction.BACKWARD,
                    features,
                )
            )
    return stages


def crossbars_needed(stage: StageSpec, rows: int = 64, cols: int = 64) -> int:
    features = stage.features
    if stage.operation is Operation.COMBINATION:
        row_tiles = math.ceil(features.weight_rows_co / rows)
        col_tiles = math.ceil(features.weight_cols_co / cols)
        return max(1, row_tiles * col_tiles * 2)
    nonzero_edges = math.ceil(
        features.adjacency_rows_ag
        * features.adjacency_cols_ag
        * (1.0 - features.sparsity)
    )
    adjacency_tiles = math.ceil(nonzero_edges / (rows * cols))
    feature_tiles = math.ceil(features.feature_cols_ag / cols)
    return max(1, adjacency_tiles * feature_tiles)
