from dataclasses import dataclass
import math
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import HardwareConfig
from .hardware import StageTimingModel
from .stages import StageSpec


@dataclass(frozen=True)
class ProfileRecord:
    features: tuple[float, ...]
    duration_seconds: float
    dataset: str
    stage_id: str


class ExecutionTimeMLP(nn.Module):
    """Three-layer MLP: 10 inputs, 256 hidden neurons, and one output."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(10, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class StageTimePredictor:
    def __init__(self, model: ExecutionTimeMLP | None = None):
        self.model = model or ExecutionTimeMLP()
        self.feature_mean: torch.Tensor | None = None
        self.feature_scale: torch.Tensor | None = None
        self.target_mean: torch.Tensor | None = None
        self.target_scale: torch.Tensor | None = None

    @staticmethod
    def _transform_features(features: torch.Tensor) -> torch.Tensor:
        transformed = features.clone()
        transformed[:, :8] = torch.log1p(transformed[:, :8].clamp_min(0))
        return transformed

    def fit(
        self,
        records: list[ProfileRecord],
        *,
        epochs: int = 150,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if not records:
            raise ValueError("At least one profile record is required")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        features = torch.tensor(
            np.asarray([record.features for record in records], dtype=np.float32)
        )
        durations = np.asarray(
            [record.duration_seconds for record in records], dtype=np.float32
        )
        if not np.isfinite(durations).all() or np.any(durations <= 0):
            raise ValueError("Profile durations must be finite and positive")
        targets = torch.tensor(np.log(durations).reshape(-1, 1), dtype=torch.float32)

        features = self._transform_features(features)
        self.feature_mean = features.mean(dim=0)
        self.feature_scale = features.std(dim=0, unbiased=False).clamp_min(0.25)
        self.target_mean = targets.mean(dim=0)
        self.target_scale = targets.std(dim=0, unbiased=False).clamp_min(1e-6)
        normalized_features = (features - self.feature_mean) / self.feature_scale
        normalized_targets = (targets - self.target_mean) / self.target_scale

        torch.manual_seed(seed)
        self.model = ExecutionTimeMLP()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(normalized_features)
            loss = loss_function(predictions, normalized_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

    def predict(self, features: tuple[float, ...] | list[float]) -> float:
        if len(features) != 10:
            raise ValueError("The MLP accepts exactly 10 stage features")
        if self.feature_mean is None or self.target_mean is None:
            raise RuntimeError("The execution-time predictor has not been fitted")
        inputs = torch.tensor(features, dtype=torch.float32).reshape(1, 10)
        self.model.eval()
        with torch.no_grad():
            transformed = self._transform_features(inputs)
            normalized = (transformed - self.feature_mean) / self.feature_scale
            log_duration = self.model(normalized) * self.target_scale + self.target_mean
        log_value = float(log_duration.item())
        if not math.isfinite(log_value) or not -80 <= log_value <= 80:
            raise FloatingPointError(
                f"Predicted log-duration is outside the safe numeric range: {log_value}"
            )
        return max(math.exp(log_value), 1e-12)

    def predict_stages(self, stages: list[StageSpec]) -> dict[str, float]:
        return {
            stage.stage_id: self.predict(stage.features.as_vector())
            for stage in stages
        }

    def save(self, path: str | Path) -> None:
        if self.feature_mean is None:
            raise RuntimeError("Fit the predictor before saving it")
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "target_mean": self.target_mean,
                "target_scale": self.target_scale,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StageTimePredictor":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = ExecutionTimeMLP()
        model.load_state_dict(checkpoint["model_state"])
        predictor = cls(model)
        predictor.feature_mean = checkpoint["feature_mean"]
        predictor.feature_scale = checkpoint["feature_scale"]
        predictor.target_mean = checkpoint["target_mean"]
        predictor.target_scale = checkpoint["target_scale"]
        return predictor


def profile_with_timing_model(
    stages: list[StageSpec],
    dataset_name: str,
    config: HardwareConfig,
    batch_size: int = 64,
) -> list[ProfileRecord]:
    simulator = StageTimingModel(config)
    return [
        ProfileRecord(
            stage.features.as_vector(),
            simulator.estimate(stage, batch_size=batch_size).seconds,
            dataset_name,
            stage.stage_id,
        )
        for stage in stages
    ]


def load_profile_csv(path: str | Path) -> list[ProfileRecord]:
    records: list[ProfileRecord] = []
    feature_columns = (
        "R_IFM_CO",
        "C_IFM_CO",
        "R_W_CO",
        "C_W_CO",
        "R_A_AG",
        "C_A_AG",
        "R_F_AG",
        "C_F_AG",
        "sparsity",
        "layer",
    )
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"dataset", "stage_id", "duration_seconds", *feature_columns}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Profile CSV is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                features = tuple(float(row[column]) for column in feature_columns)
                duration = float(row["duration_seconds"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid profile values on CSV line {line_number}"
                ) from error
            records.append(
                ProfileRecord(
                    features,
                    duration,
                    row["dataset"],
                    row["stage_id"],
                )
            )
    if not records:
        raise ValueError("Profile CSV contains no timing records")
    return records
