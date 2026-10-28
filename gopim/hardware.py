from dataclasses import dataclass
import math
from collections import Counter

import torch

from .config import HardwareConfig
from .stages import Direction, Operation, StageSpec


@dataclass(frozen=True)
class TimingBreakdown:
    seconds: float
    read_operations: int
    write_operations: int
    bytes_moved: int
    read_array_energy_joules: float
    write_array_energy_joules: float

    @property
    def array_energy_joules(self) -> float:
        return self.read_array_energy_joules + self.write_array_energy_joules


class GlobalBuffer:
    def __init__(self, config: HardwareConfig):
        self.capacity_bytes = config.global_buffer_bytes
        self.read_bytes = 0
        self.write_bytes = 0

    def transfer(self, byte_count: int, *, write: bool = False) -> int:
        if byte_count < 0:
            raise ValueError("byte_count cannot be negative")
        self.read_bytes += 0 if write else byte_count
        self.write_bytes += byte_count if write else 0
        return math.ceil(byte_count / self.capacity_bytes)


class SRAMWeightManager:
    def __init__(self, config: HardwareConfig):
        self.write_endurance = config.weight_manager_write_endurance
        self.writes = 0

    def update(self, weight_count: int) -> None:
        if weight_count < 0:
            raise ValueError("weight_count cannot be negative")
        if self.writes + weight_count > self.write_endurance:
            raise OverflowError("SRAM weight-manager write endurance exceeded")
        self.writes += weight_count


class ReRAMWriteTracker:
    def __init__(self) -> None:
        self.weight_values = 0
        self.feature_values = 0

    def update_weights(self, value_count: int) -> None:
        if value_count < 0:
            raise ValueError("value_count cannot be negative")
        self.weight_values += value_count

    def update_features(self, value_count: int) -> None:
        if value_count < 0:
            raise ValueError("value_count cannot be negative")
        self.feature_values += value_count


class GOPIMDataflow:
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.global_buffer = GlobalBuffer(config)
        self.weight_manager = SRAMWeightManager(config)
        self.reram_writes = ReRAMWriteTracker()
        self.operations: Counter[str] = Counter()

    def run_combination(
        self, features: torch.Tensor, mapped_weights: torch.Tensor
    ) -> torch.Tensor:
        self.global_buffer.transfer(mapped_weights.numel() * mapped_weights.element_size())
        self.operations["fetch_weights_to_gb"] += 1
        self.operations["map_weights_to_crossbars"] += 1
        self.global_buffer.transfer(features.numel() * features.element_size())
        self.operations["load_vertex_features"] += 1
        output = torch.nn.functional.linear(features, mapped_weights)
        self.global_buffer.transfer(output.numel() * output.element_size(), write=True)
        self.operations["combination_mvm"] += 1
        return output

    def run_aggregation(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        aggregate,
    ) -> torch.Tensor:
        self.global_buffer.transfer(features.numel() * features.element_size())
        self.global_buffer.transfer(edge_index.numel() * edge_index.element_size())
        self.operations["load_adjacency"] += 1
        output = aggregate(features)
        self.global_buffer.transfer(output.numel() * output.element_size(), write=True)
        self.operations["aggregation_mvm"] += 1
        self.operations["adder"] += 1
        return output

    def activate(self, values: torch.Tensor) -> torch.Tensor:
        self.operations["relu"] += 1
        return torch.relu(values)

    def record_backward(
        self,
        layer_dimensions: list[tuple[int, int]],
        feature_update_count: int,
        feature_width: int,
    ) -> None:
        if feature_update_count < 0 or feature_width <= 0:
            raise ValueError("Invalid backward update size")
        for input_width, output_width in reversed(layer_dimensions):
            self.operations["backward_error_mvm"] += 1
            self.operations["weight_gradient_multiplication"] += 1
            self.weight_manager.update(input_width * output_width)
            self.operations["sram_weight_gradient_update"] += 1
            self.reram_writes.update_weights(input_width * output_width)
            self.operations["rewrite_weights_to_reram"] += 1
        if feature_update_count:
            self.reram_writes.update_features(feature_update_count * feature_width)
            self.operations["rewrite_features_to_reram"] += 1

    def summary(self) -> dict[str, int]:
        return {
            "global_buffer_capacity_bytes": self.global_buffer.capacity_bytes,
            "global_buffer_read_bytes": self.global_buffer.read_bytes,
            "global_buffer_write_bytes": self.global_buffer.write_bytes,
            "sram_weight_values_written": self.weight_manager.writes,
            "reram_weight_values_written": self.reram_writes.weight_values,
            "reram_feature_values_written": self.reram_writes.feature_values,
            "crossbar_operations": sum(self.operations[key] for key in (
                "combination_mvm",
                "aggregation_mvm",
                "backward_error_mvm",
            )),
            **{f"operation_{key}": value for key, value in self.operations.items()},
        }


class ActivationModule:
    @staticmethod
    def apply(values: torch.Tensor) -> torch.Tensor:
        return torch.relu(values)


class StageTimingModel:
    """Analytical stage timing and first-order NeuroSim-derived array energy."""

    def __init__(self, config: HardwareConfig):
        self.config = config

    def read_array_energy_per_operation(self) -> float:
        config = self.config
        row_length = (
            config.crossbar_cols
            * config.cell_width_in_feature_size
            * config.feature_size_meters
        )
        column_length = (
            config.crossbar_rows
            * config.cell_height_in_feature_size
            * config.feature_size_meters
        )
        bitline_capacitance = (
            column_length * config.wire_capacitance_farad_per_meter
        )
        wordline_capacitance = (
            row_length * config.wire_capacitance_farad_per_meter
        )
        read_cells_per_sense_amp = math.ceil(
            config.crossbar_cols / config.columns_per_sense_amp
        )
        array_energy = (
            bitline_capacitance
            * config.read_voltage**2
            * read_cells_per_sense_amp
            + wordline_capacitance
            * config.technology_supply_voltage**2
            * config.crossbar_rows
            * config.read_active_row_fraction
        ) * config.columns_per_sense_amp
        return array_energy

    def write_array_energy_per_row(self) -> float:
        config = self.config
        row_length = (
            config.crossbar_cols
            * config.cell_width_in_feature_size
            * config.feature_size_meters
        )
        wordline_capacitance = (
            row_length * config.wire_capacitance_farad_per_meter
        )
        bitline_capacitance = (
            row_length * config.wire_capacitance_farad_per_meter
        )
        average_cell_resistance = (
            config.reram_resistance_on_ohms
            + config.reram_resistance_off_ohms
        ) / 2
        cell_energy = (
            config.crossbar_cols
            * config.write_voltage**2
            / average_cell_resistance
            * config.write_latency_seconds
            * config.write_pulses_per_cell
        )
        line_energy_per_row = (
            wordline_capacitance * config.technology_supply_voltage**2
            + (bitline_capacitance + wordline_capacitance)
            / 2
            * config.write_voltage**2
            * config.crossbar_cols
        )
        return cell_energy + line_energy_per_row * config.write_pulses_per_cell

    def estimate(
        self, stage: StageSpec, replicas: int = 1, batch_size: int = 64
    ) -> TimingBreakdown:
        if replicas <= 0 or batch_size <= 0:
            raise ValueError("replicas and batch_size must be positive")
        f = stage.features
        if stage.operation is Operation.COMBINATION:
            operations = f.ifm_rows_co * f.ifm_cols_co * f.weight_cols_co
            read_ops = math.ceil(operations / (self.config.crossbar_rows**2))
        else:
            nonzero_edges = math.ceil(
                f.adjacency_rows_ag
                * f.adjacency_cols_ag
                * (1.0 - f.sparsity)
            )
            operations = nonzero_edges * f.feature_cols_ag
            read_ops = math.ceil(
                operations / (self.config.crossbar_rows * self.config.crossbar_cols)
            )
        read_ops = max(1, read_ops * math.ceil(batch_size / 64))
        write_ops = 0
        if stage.direction is Direction.BACKWARD:
            write_ops = max(1, math.ceil(operations / self.config.crossbar_cols))

        concurrent_reads = math.ceil(read_ops / replicas)
        concurrent_writes = math.ceil(write_ops / replicas) if write_ops else 0
        bytes_moved = (
            f.ifm_rows_co * f.ifm_cols_co * 4
            + f.weight_rows_co * f.weight_cols_co * 4
            + f.feature_rows_ag * f.feature_cols_ag * 4
        )
        seconds = (
            concurrent_reads * self.config.read_latency_seconds
            + concurrent_writes * self.config.write_latency_seconds
        )
        return TimingBreakdown(
            seconds,
            read_ops,
            write_ops,
            bytes_moved,
            read_ops * self.read_array_energy_per_operation(),
            write_ops * self.write_array_energy_per_row(),
        )


def make_stage_executor(config: HardwareConfig):
    """Yield the GOPIM forward-dataflow operations in their specified order."""
    buffer = GlobalBuffer(config)
    activation = ActivationModule()

    def execute(
        features: torch.Tensor,
        weights: torch.Tensor,
        adjacency_aggregate,
    ) -> torch.Tensor:
        buffer.transfer(weights.numel() * weights.element_size())
        combination = features @ weights
        buffer.transfer(combination.numel() * combination.element_size(), write=True)
        buffer.transfer(combination.numel() * combination.element_size())
        aggregation = adjacency_aggregate(combination)
        buffer.transfer(aggregation.numel() * aggregation.element_size(), write=True)
        output = activation.apply(aggregation)
        buffer.transfer(output.numel() * output.element_size(), write=True)
        return output

    return execute
