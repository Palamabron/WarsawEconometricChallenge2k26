import logging
import warnings
from typing import Union

import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None


class DataIngestion:
    """Load WEC2026 CSV files with either pandas or cuDF.

    Args:
        use_gpu: Force cuDF when True, force pandas when False, or auto-detect
            when None.
        config_path: Path to the project YAML config.
    """

    def __init__(self, use_gpu: bool | None = None, config_path: str = "config.yaml"):
        self.config = get_config(config_path)

        if use_gpu is None:
            use_gpu = self.config.use_gpu and CUDF_AVAILABLE
        elif use_gpu and not CUDF_AVAILABLE:
            warnings.warn("GPU requested but cuDF not available. Falling back to pandas.")
            use_gpu = False

        self.use_gpu = use_gpu
        self.backend = "cudf" if use_gpu else "pandas"

        logger.info("DataIngestion initialized: backend=%s", self.backend)

    def _get_dataframe_backend(self):
        return cudf if self.use_gpu else pd

    def load_checkpoint_data(self) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        """Load checkpoint-level player observations with narrower dtypes."""
        df_lib = self._get_dataframe_backend()
        filepath = self.config.get_data_path(self.config.get("data.checkpoint_file"))

        logger.info("Loading checkpoint data from %s", filepath)

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

        self._validate_checkpoint_data(df)

        logger.info("Loaded %s checkpoint observations", len(df))
        return df

    def load_pass_data(self) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        df_lib = self._get_dataframe_backend()
        filepath = self.config.get_data_path(self.config.get("data.pass_file"))

        logger.info("Loading pass data from %s", filepath)

        dtypes = {
            "id": "int32",
            "period": "category",
            "player_appearance_id": "int32",
            "addressee_player_appearance_id": "float32",
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
        """Load shot event data and drop outcome-like leakage columns."""
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

        leakage_cols = ["outcome", "result", "goal", "scored"]
        existing_leakage = [col for col in leakage_cols if col in df.columns]
        if existing_leakage:
            warnings.warn(f"Dropping potential leakage columns: {existing_leakage}")
            df = df.drop(columns=existing_leakage)

        logger.info("Loaded %s shot events", len(df))
        return df

    def load_pressure_data(self) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        df_lib = self._get_dataframe_backend()
        filepath = self.config.get_data_path(self.config.get("data.pressure_file"))

        logger.info("Loading pressure data from %s", filepath)

        dtypes = {
            "id": "int32",
            "period": "category",
            "player_appearance_id": "int32",
            "addressee_player_appearance_id": "float32",
            "pressing_player_appearance_id": "int32",
            "press_induced_outcome": "category",
            "pass_angle": "float32",
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
        if self.use_gpu:
            df_pd = df.to_pandas()
        else:
            df_pd = df

        dup_cols = ["player_appearance_id", "checkpoint"]
        if df_pd.duplicated(subset=dup_cols).any():
            warnings.warn("Found duplicate player_appearance_id + checkpoint combinations")

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
    """Keep events from the same period at or before a checkpoint.

    Args:
        events_df: Event frame with `minute` and `period`.
        checkpoint_min: Period-relative checkpoint minute.
        checkpoint_period: Period label to match.

    Returns:
        Filtered event frame.
    """
    mask = (events_df["minute"] <= checkpoint_min) & (events_df["period"] == checkpoint_period)
    return events_df[mask]
