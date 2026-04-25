"""Configuration loader with environment detection"""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Configuration manager with GPU/CPU environment detection"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.use_gpu = self._detect_gpu()
        self.project_root = Path(__file__).parent.parent

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _detect_gpu(self) -> bool:
        """
        Detect CUDA-capable GPU availability

        Returns:
            bool: True if GPU available, False otherwise
        """
        # Check environment variable
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible == "" or cuda_visible == "-1":
            return False

        # Try importing CUDA libraries
        try:
            import cudf

            return True
        except ImportError:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                return True
        except ImportError:
            pass

        return False

    def get(self, key: str, default=None):
        """Get configuration value by key (supports nested keys with dots)"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def get_data_path(self, filename: str) -> Path:
        """Get full path to data file"""
        data_dir = self.get("data.data_dir", "data")
        return self.project_root / data_dir / filename

    def get_output_path(self, output_type: str, filename: str) -> Path:
        """Get full path to output file (models/predictions/reports)"""
        output_dir = self.get(f"outputs.{output_type}_dir", f"outputs/{output_type}")
        path = self.project_root / output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    @property
    def n_folds(self) -> int:
        """Number of cross-validation folds"""
        return self.get("cross_validation.n_folds", 5)

    @property
    def random_state(self) -> int:
        """Random seed for reproducibility (prefers cross_validation.random_state)"""
        return self.get("cross_validation.random_state", self.get("random_seed", 42))

    @property
    def group_by_column(self) -> str:
        """Column name for GroupKFold (fixture_id)"""
        return self.get("cross_validation.group_by", "fixture_id")

    def __repr__(self) -> str:
        return f"Config(gpu={self.use_gpu}, config_path='{self.config_path}')"


# Global config instance
_config: Config | None = None
_config_path: Path | None = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create global config instance"""
    global _config, _config_path
    requested_path = Path(config_path).expanduser().resolve()
    if _config is None or _config_path != requested_path:
        _config = Config(config_path)
        _config_path = requested_path
    return _config
