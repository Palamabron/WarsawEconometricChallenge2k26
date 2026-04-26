import importlib.util
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: str = "data"
    checkpoint_file: str
    pass_file: str
    run_file: str
    shot_file: str
    pressure_file: str


class OutputSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models_dir: str = "outputs/models"
    predictions_dir: str = "outputs/predictions"
    reports_dir: str = "outputs/reports"


class CrossValidationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_folds: int = Field(default=5, ge=2)
    random_state: int = 42
    group_by: str = "fixture_id"
    stratify: bool = True


class ProjectConfig(BaseModel):
    """Typed view of the YAML config.

    Extra top-level sections stay allowed so experimental model settings can be
    added without changing this loader first.
    """

    model_config = ConfigDict(extra="allow")

    data: DataSettings
    outputs: OutputSettings = Field(default_factory=OutputSettings)
    cross_validation: CrossValidationSettings = Field(default_factory=CrossValidationSettings)
    features: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    hyperparameter_optimization: dict[str, Any] = Field(default_factory=dict)
    ensemble: dict[str, Any] = Field(default_factory=dict)
    performance_targets: dict[str, Any] = Field(default_factory=dict)
    random_seed: int = 42


class Config:
    """Load validated YAML config and expose legacy dotted-key access."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.settings = self._load_config()
        self.config = self.settings.model_dump(mode="python")
        self.use_gpu = self._detect_gpu()
        self.project_root = Path(__file__).parent.parent

    def _load_config(self) -> ProjectConfig:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f) or {}

        return ProjectConfig.model_validate(raw_config)

    def _detect_gpu(self) -> bool:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible == "" or cuda_visible == "-1":
            return False

        if importlib.util.find_spec("cudf") is not None:
            return True

        try:
            import torch

            if torch.cuda.is_available():
                return True
        except ImportError:
            pass

        return False

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def get_data_path(self, filename: str) -> Path:
        data_dir = self.get("data.data_dir", "data")
        return self.project_root / data_dir / filename

    def get_output_path(self, output_type: str, filename: str) -> Path:
        output_dir = self.get(f"outputs.{output_type}_dir", f"outputs/{output_type}")
        path = self.project_root / output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    @property
    def n_folds(self) -> int:
        return self.get("cross_validation.n_folds", 5)

    @property
    def random_state(self) -> int:
        return self.get("cross_validation.random_state", self.get("random_seed", 42))

    @property
    def group_by_column(self) -> str:
        return self.get("cross_validation.group_by", "fixture_id")

    def __repr__(self) -> str:
        return f"Config(gpu={self.use_gpu}, config_path='{self.config_path}')"


_config: Config | None = None
_config_path: Path | None = None


def get_config(config_path: str = "config.yaml") -> Config:
    global _config, _config_path
    requested_path = Path(config_path).expanduser().resolve()
    if _config is None or _config_path != requested_path:
        _config = Config(config_path)
        _config_path = requested_path
    return _config
