from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    tile_count: int = 65_536
    pes_per_tile: int = 8
    crossbars_per_pe: int = 32
    crossbar_rows: int = 64
    crossbar_cols: int = 64
    crossbar_cell_bits: int = 2
    crossbar_resource_limit_bytes: int = 16 * 1024**3
    global_buffer_bytes: int = 128 * 1024
    weight_manager: str = "SRAM"
    weight_manager_write_endurance: int = 10**16
    read_latency_seconds: float = 29.31e-9
    write_latency_seconds: float = 50.88e-9
    reram_resistance_on_ohms: float = 100_000.0
    reram_resistance_off_ohms: float = 1_000_000.0
    read_voltage: float = 0.5
    write_voltage: float = 4.0
    technology_supply_voltage: float = 0.9
    feature_size_meters: float = 40e-9
    cell_width_in_feature_size: float = 4.0
    cell_height_in_feature_size: float = 4.0
    wire_capacitance_farad_per_meter: float = 0.2e-9
    columns_per_sense_amp: int = 8
    read_active_row_fraction: float = 0.5
    write_pulses_per_cell: int = 1
    activation: str = "relu"
    micro_batch_size: int = 64
    sparse_average_degree_threshold: float = 8.0
    sparse_update_fraction: float = 0.8
    dense_update_fraction: float = 0.5
    low_degree_update_period: int = 20

    def __post_init__(self) -> None:
        positive = {
            "tile_count": self.tile_count,
            "pes_per_tile": self.pes_per_tile,
            "crossbars_per_pe": self.crossbars_per_pe,
            "crossbar_rows": self.crossbar_rows,
            "crossbar_cols": self.crossbar_cols,
            "crossbar_cell_bits": self.crossbar_cell_bits,
            "crossbar_resource_limit_bytes": self.crossbar_resource_limit_bytes,
            "global_buffer_bytes": self.global_buffer_bytes,
            "weight_manager_write_endurance": self.weight_manager_write_endurance,
            "micro_batch_size": self.micro_batch_size,
            "low_degree_update_period": self.low_degree_update_period,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Hardware values must be positive: {', '.join(invalid)}")
        if self.read_latency_seconds <= 0 or self.write_latency_seconds <= 0:
            raise ValueError("Read and write latencies must be positive")
        positive_physical = {
            "reram_resistance_on_ohms": self.reram_resistance_on_ohms,
            "reram_resistance_off_ohms": self.reram_resistance_off_ohms,
            "read_voltage": self.read_voltage,
            "write_voltage": self.write_voltage,
            "technology_supply_voltage": self.technology_supply_voltage,
            "feature_size_meters": self.feature_size_meters,
            "cell_width_in_feature_size": self.cell_width_in_feature_size,
            "cell_height_in_feature_size": self.cell_height_in_feature_size,
            "wire_capacitance_farad_per_meter": self.wire_capacitance_farad_per_meter,
            "read_active_row_fraction": self.read_active_row_fraction,
            "columns_per_sense_amp": self.columns_per_sense_amp,
            "write_pulses_per_cell": self.write_pulses_per_cell,
        }
        invalid = [name for name, value in positive_physical.items() if value <= 0]
        if invalid:
            raise ValueError(f"Physical values must be positive: {', '.join(invalid)}")
        if self.read_active_row_fraction > 1:
            raise ValueError("read_active_row_fraction must be at most 1")
        if self.reram_resistance_off_ohms <= self.reram_resistance_on_ohms:
            raise ValueError("ReRAM off resistance must exceed on resistance")
        if self.weight_manager.upper() != "SRAM":
            raise ValueError("GOPIM weight manager must use SRAM")
        if self.activation.lower() != "relu":
            raise ValueError("GOPIM activation module must use ReLU")
        for name, fraction in (
            ("sparse_update_fraction", self.sparse_update_fraction),
            ("dense_update_fraction", self.dense_update_fraction),
        ):
            if not 0 < fraction <= 1:
                raise ValueError(f"{name} must be in (0, 1]")

    @property
    def crossbars_per_tile(self) -> int:
        return self.pes_per_tile * self.crossbars_per_pe

    @property
    def total_crossbars(self) -> int:
        return self.tile_count * self.crossbars_per_tile

    @property
    def bytes_per_crossbar(self) -> int:
        return (
            self.crossbar_rows
            * self.crossbar_cols
            * self.crossbar_cell_bits
            // 8
        )

    @property
    def crossbar_capacity_bytes(self) -> int:
        return self.total_crossbars * self.bytes_per_crossbar

    @property
    def array_layout(self) -> tuple[int, int]:
        return (4, 8)

    def validate_resource_capacity(self) -> None:
        if self.crossbar_capacity_bytes > self.crossbar_resource_limit_bytes:
            raise ValueError(
                "Configured crossbars exceed the resource limit: "
                f"{self.crossbar_capacity_bytes} > "
                f"{self.crossbar_resource_limit_bytes} bytes"
            )
