"""Data ingestion module with GPU/CPU backend support"""

import logging
import warnings
from typing import Union

import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)

# Try importing cuDF for GPU acceleration
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None


class DataIngestion:
    """
    Data loader with automatic GPU/CPU backend detection

    Handles loading and validation of all WEC2026 dataset files with optimized dtypes
    and proper handling of temporal boundaries.
    """

    def __init__(self, use_gpu: bool | None = None, config_path: str = "config.yaml"):
        """
        Initialize data ingestion

        Args:
            use_gpu: Force GPU (True) or CPU (False). None for auto-detect.
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)

        # Determine backend
        if use_gpu is None:
            use_gpu = self.config.use_gpu and CUDF_AVAILABLE
        elif use_gpu and not CUDF_AVAILABLE:
            warnings.warn("GPU requested but cuDF not available. Falling back to pandas.")
            use_gpu = False

        self.use_gpu = use_gpu
        self.backend = "cudf" if use_gpu else "pandas"

        logger.info("DataIngestion initialized: backend=%s", self.backend)

    def _get_dataframe_backend(self):
        """Get appropriate DataFrame library (cudf or pandas)"""
        return cudf if self.use_gpu else pd

    def load_checkpoint_data(self) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """
        Load main checkpoint dataset (players_quarters_final.csv)

        Returns:
            DataFrame with player statistics at 15-minute checkpoints
        """
        df_lib = self._get_dataframe_backend()
        filepath = self.config.get_data_path(self.config.get("data.checkpoint_file"))

        logger.info("Loading checkpoint data from %s", filepath)

        # Optimized dtypes
        dtypes = {
            "player_appearance_id": "int32",
            "player_id": "int32",
            "fixture_id": "int32",
            "checkpoint_period": "category",
            "checkpoint_min": "int16",
            "position": "category",
            "formation": "category",
            "minute_in": "int16",
            "minute_out": "int16",
            "jersey_number": "int16",
            "scored_after": "int8",
        }

        # Float columns
        float_cols = [
            "last15_sprints",
            "last15_hsr",
            "last15_distance",
            "last15_mean_max_speed",
            "last15_peak_speed",
            "last15_shots",
            "last15_shots_on_target",
            "last15_shots_under_press",
            "last15_shots_top_third",
            "cumul_sprints",
            "cumul_hsr",
            "cumul_distance",
            "cumul_mean_max_speed",
            "cumul_peak_speed",
            "cumul_shots",
            "cumul_shots_on_target",
            "cumul_shots_under_press",
            "cumul_shots_top_third",
        ]

        for col in float_cols:
            dtypes[col] = "float32"

        df = df_lib.read_csv(
            filepath,
            dtype=dtypes,
            parse_dates=["date"],
            true_values=["TRUE"],
            false_values=["FALSE"],
            na_values=["NULL"],
        )

        # Validate data
        self._validate_checkpoint_data(df)

        logger.info("Loaded %s checkpoint observations", len(df))
        return df

    def load_pass_data(self) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """Load pass event data"""
        df_lib = self._get_dataframe_backend()
        filepath = self.config.get_data_path(self.config.get("data.pass_file"))

        logger.info("Loading pass data from %s", filepath)

        dtypes = {
            "id": "int32",
            "period": "category",
            "player_appearance_id": "int32",
            "addressee_player_appearance_id": "float32",  # Can be NULL
            "minute": "int16",
            "stage": "category",
        }

        df = df_lib.read_csv(
            filepath,
            dtype=dtypes,
            true_values=["TRUE"],
            false_values=["FALSE"],
            na_values=["NULL"],
        )

        logger.info("Loaded %s pass events", len(df))
        return df

    def load_run_data(self) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """Load high-speed run event data"""
        df_lib = self._get_dataframe_backend()
        filepath = self.config.get_data_path(self.config.get("data.run_file"))

        logger.info("Loading run data from %s", filepath)

        dtypes = {
            "id": "int32",
            "period": "category",
            "stage": "category",
            "possession": "int32",
            "run_type": "category",
            "minute": "int16",
            "min_speed": "float32",
            "max_speed": "float32",
            "distance": "float32",
            "player_appearance_id": "int32",
        }

        df = df_lib.read_csv(
            filepath,
            dtype=dtypes,
            true_values=["TRUE"],
            false_values=["FALSE"],
            na_values=["NULL"],
        )

        logger.info("Loaded %s run events", len(df))
        return df

    def load_shot_data(self) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """
        Load shot event data

        WARNING: shot outcome columns must be dropped to prevent data leakage
        """
        df_lib = self._get_dataframe_backend()
        filepath = self.config.get_data_path(self.config.get("data.shot_file"))

        logger.info("Loading shot data from %s", filepath)

        dtypes = {
            "id": "int32",
            "period": "category",
            "player_appearance_id": "int32",
            "body_part": "category",
            "technique": "category",
            "play_pattern": "category",
            "minute": "int16",
            "possession": "int32",
            "stage": "category",
        }

        df = df_lib.read_csv(
            filepath,
            dtype=dtypes,
            true_values=["TRUE"],
            false_values=["FALSE"],
            na_values=["NULL"],
        )

        # CRITICAL: Drop outcome columns to prevent leakage
        leakage_cols = ["outcome", "result", "goal", "scored"]
        existing_leakage = [col for col in leakage_cols if col in df.columns]
        if existing_leakage:
            warnings.warn(f"Dropping potential leakage columns: {existing_leakage}")
            df = df.drop(columns=existing_leakage)

        logger.info("Loaded %s shot events", len(df))
        return df

    def load_pressure_data(self) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """Load behaviour under pressure data"""
        df_lib = self._get_dataframe_backend()
        filepath = self.config.get_data_path(self.config.get("data.pressure_file"))

        logger.info("Loading pressure data from %s", filepath)

        dtypes = {
            "id": "int32",
            "period": "category",
            "player_appearance_id": "int32",
            "addressee_player_appearance_id": "float32",  # Can be NULL
            "pressing_player_appearance_id": "int32",
            "press_induced_outcome": "category",
            "pass_angle": "float32",  # Can be NULL
            "minute": "int16",
            "stage": "category",
        }

        df = df_lib.read_csv(
            filepath,
            dtype=dtypes,
            true_values=["TRUE"],
            false_values=["FALSE"],
            na_values=["NULL"],
        )

        logger.info("Loaded %s pressure events", len(df))
        return df

    def load_all(
        self,
    ) -> tuple[
        Union[pd.DataFrame, "cudf.DataFrame"], dict[str, Union[pd.DataFrame, "cudf.DataFrame"]]
    ]:
        """
        Load all datasets

        Returns:
            Tuple of (checkpoint_df, event_dfs_dict)
        """
        logger.info("Loading all WEC2026 datasets")

        checkpoint_df = self.load_checkpoint_data()

        event_dfs = {
            "pass": self.load_pass_data(),
            "run": self.load_run_data(),
            "shot": self.load_shot_data(),
            "pressure": self.load_pressure_data(),
        }

        logger.info(
            "All datasets loaded successfully: total_events=%s",
            sum(len(df) for df in event_dfs.values()),
        )

        return checkpoint_df, event_dfs

    def _validate_checkpoint_data(self, df: Union[pd.DataFrame, "cudf.DataFrame"]) -> None:
        """
        Validate checkpoint data integrity

        Args:
            df: Checkpoint DataFrame to validate
        """
        # Convert to pandas for validation if using cuDF
        if self.use_gpu:
            df_pd = df.to_pandas()
        else:
            df_pd = df

        # Check for duplicates
        dup_cols = ["player_appearance_id", "checkpoint"]
        if df_pd.duplicated(subset=dup_cols).any():
            warnings.warn("Found duplicate player_appearance_id + checkpoint combinations")

        # Validate substitution boundaries
        invalid_subs = df_pd[
            ~(
                (df_pd["minute_in"] <= df_pd["checkpoint_min"])
                & (df_pd["checkpoint_min"] <= df_pd["minute_out"])
            )
        ]

        if len(invalid_subs) > 0:
            warnings.warn(
                f"Found {len(invalid_subs)} observations where checkpoint is outside "
                f"player's time on pitch (minute_in to minute_out)"
            )

        # Check target variable distribution
        target_dist = df_pd["scored_after"].value_counts()
        pos_pct = (target_dist.get(1, 0) / len(df_pd)) * 100

        logger.info(
            "Target distribution: negative=%s (%.2f%%) positive=%s (%.2f%%)",
            target_dist.get(0, 0),
            100 - pos_pct,
            target_dist.get(1, 0),
            pos_pct,
        )

        if pos_pct < 5 or pos_pct > 7:
            warnings.warn(f"Target class imbalance ({pos_pct:.2f}%) differs from expected ~5.8%")


def safe_temporal_filter(
    events_df: Union[pd.DataFrame, "cudf.DataFrame"], checkpoint_min: int, checkpoint_period: str
) -> Union[pd.DataFrame, "cudf.DataFrame"]:
    """
    Apply strict temporal filtering to prevent data leakage

    Only includes events that occurred at or before the checkpoint minute.

    Args:
        events_df: Event DataFrame to filter
        checkpoint_min: Checkpoint minute boundary
        checkpoint_period: Checkpoint period (half_1 or half_2)

    Returns:
        Filtered DataFrame
    """
    mask = (events_df["minute"] <= checkpoint_min) & (events_df["period"] == checkpoint_period)
    return events_df[mask]
