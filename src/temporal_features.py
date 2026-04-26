"""Fold-safe temporal feature engineering for WEC2026.

This module turns the checkpoint table plus event logs into a modeling table.
The transformer is intentionally fitted inside each CV fold: reference rates
such as xT zone values and pressure turnover baselines are learned only from
the training fold and then applied to validation checkpoints.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PERIOD_OFFSETS = {
    "half_1": 0,
    "half_2": 45,
    "extra_time_1": 90,
    "extra_time_2": 105,
}

WINDOWS = {
    "last5": 5,
    "last15": 15,
    "cumul": None,
}

TARGET_COL = "scored_after"
ID_COLS = {
    "player_appearance_id",
    "player_id",
    "fixture_id",
    "date",
    "checkpoint",
    "scored_after",
}
NON_MODEL_COLS = {
    "jersey_number",
    # These are only known after the appearance ends, so they leak future substitution information.
    "minute_out",
    "subbed",
}
REDUNDANT_MODEL_FEATURES = frozenset(
    {
        # Position is represented by explicit role flags, which avoids one-hot duplicates.
        "position",
        # Event-log run aggregates duplicate the supplied checkpoint physical summaries.
        "last15_sprint_event_count",
        "last15_hsr_event_count",
        "last15_run_distance",
        "last15_run_mean_speed",
        "last15_run_max_speed",
        "cumul_sprint_event_count",
        "cumul_hsr_event_count",
        "cumul_run_distance",
        "cumul_run_mean_speed",
        "cumul_run_max_speed",
        # Event-log shot counts/ratios duplicate supplied checkpoint shot summaries.
        "last15_shot_event_count",
        "cumul_shot_event_count",
        "cumul_shot_under_pressure_rate",
        "cumul_shot_top_share",
        # Expected turnovers were almost a scaled pressure count; quality keeps the adjustment.
        "last5_pressure_expected_turnovers",
        "last15_pressure_expected_turnovers",
        "cumul_pressure_expected_turnovers",
    }
)


def remove_redundant_model_features(columns: Iterable[str]) -> list[str]:
    """Drop deterministic or near-deterministic feature aliases before modeling."""
    return [c for c in columns if c not in REDUNDANT_MODEL_FEATURES]


def add_absolute_time(
    df: pd.DataFrame, period_col: str, minute_col: str, out_col: str
) -> pd.DataFrame:
    """Add absolute match minutes using WEC period labels."""
    out = df.copy()
    offsets = out[period_col].astype(str).map(PERIOD_OFFSETS).fillna(0)
    out[out_col] = offsets + pd.to_numeric(out[minute_col], errors="coerce").fillna(0)
    return out


def prepare_checkpoint_frame(checkpoint_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize checkpoint fields and derive context columns."""
    df = checkpoint_df.copy()
    df["checkpoint_period"] = df["checkpoint_period"].astype(str)
    df = add_absolute_time(df, "checkpoint_period", "checkpoint_min", "checkpoint_abs_min")
    df["minute_in"] = pd.to_numeric(df["minute_in"], errors="coerce").fillna(1)
    df["minute_out"] = pd.to_numeric(df["minute_out"], errors="coerce").fillna(120)
    df["minutes_played_at_checkpoint"] = (
        df["checkpoint_abs_min"].clip(lower=df["minute_in"]) - df["minute_in"] + 1
    ).clip(lower=1)
    df["is_on_pitch"] = (
        (df["minute_in"] <= df["checkpoint_abs_min"])
        & (df["checkpoint_abs_min"] <= df["minute_out"])
    ).astype(int)
    df["is_second_half"] = (df["checkpoint_period"] == "half_2").astype(int)
    df["is_extra_time"] = df["checkpoint_period"].str.startswith("extra_time").astype(int)
    df["is_late_game"] = (df["checkpoint_abs_min"] >= 60).astype(int)
    df["is_home"] = df["is_home"].astype(str).str.upper().eq("TRUE").astype(int)
    df["subbed"] = df["subbed"].astype(str).str.upper().eq("TRUE").astype(int)
    df["position"] = df["position"].astype(str)
    df["formation"] = df["formation"].astype(str)
    df["is_attacker"] = df["position"].isin(["A", "F"]).astype(int)
    df["is_midfielder"] = (df["position"] == "M").astype(int)
    df["is_defender"] = (df["position"] == "D").astype(int)
    df["is_goalkeeper"] = (df["position"] == "G").astype(int)
    return df


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1", "YES"])


def _safe_div(
    num: pd.Series | np.ndarray | float, den: pd.Series | np.ndarray | float
) -> pd.Series:
    return pd.Series(num).astype(float) / pd.Series(den).replace(0, np.nan).astype(float)


def _entropy(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def _window_mask(merged: pd.DataFrame, window: int | None) -> pd.Series:
    mask = (merged["event_abs_min"] >= merged["minute_in"]) & (
        merged["event_abs_min"] <= merged["checkpoint_abs_min"]
    )
    if window is not None:
        mask &= merged["event_abs_min"] > (merged["checkpoint_abs_min"] - window)
    return mask


def _merge_events(checkpoints: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events.merge(
        checkpoints[
            [
                "player_appearance_id",
                "checkpoint",
                "checkpoint_abs_min",
                "checkpoint_period",
                "minute_in",
            ]
        ],
        on="player_appearance_id",
        how="inner",
    )


def _empty_features(checkpoints: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = checkpoints[["player_appearance_id", "checkpoint"]].copy()
    for col in columns:
        out[col] = 0.0
    return out


def _window_duration(checkpoints: pd.DataFrame, window: int | None) -> pd.Series:
    """Return the observable minutes for a checkpoint/window pair."""
    played = pd.to_numeric(checkpoints["minutes_played_at_checkpoint"], errors="coerce").fillna(1)
    if window is None:
        return played.clip(lower=1)
    return played.clip(lower=1, upper=window)


def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return default
    return float(values.mean())


@dataclass
class TemporalFeatureBuilder:
    """Fold-safe feature builder for checkpoint-level goal prediction."""

    zones: list[str] = field(default_factory=lambda: ["bottom", "middle", "top"])
    zone_xt_: dict[str, float] = field(default_factory=dict)
    pressure_turnover_rate_: dict[str, float] = field(default_factory=dict)
    pressure_global_turnover_rate_: float = 0.0
    proportion_priors_: dict[str, float] = field(default_factory=dict)
    position_baselines_: pd.DataFrame | None = None
    fitted_: bool = False

    def fit(
        self, checkpoint_df: pd.DataFrame, event_dfs: dict[str, pd.DataFrame]
    ) -> TemporalFeatureBuilder:
        """Fit reference statistics on training-fold data only."""
        checkpoints = prepare_checkpoint_frame(checkpoint_df)
        train_appearances = set(checkpoints["player_appearance_id"])

        pass_df = self._prepare_passes(event_dfs["pass"])
        shot_df = self._prepare_shots(event_dfs["shot"])
        pressure_df = self._prepare_pressure(event_dfs["pressure"])

        pass_train = pass_df[pass_df["player_appearance_id"].isin(train_appearances)]
        shot_train = shot_df[shot_df["player_appearance_id"].isin(train_appearances)]
        pressure_train = pressure_df[pressure_df["player_appearance_id"].isin(train_appearances)]

        self.zone_xt_ = self._fit_zone_xt(pass_train, shot_train)
        self.pressure_turnover_rate_ = self._fit_pressure_turnover_rates(pressure_train)
        self.pressure_global_turnover_rate_ = _safe_mean(pressure_train["is_turnover"], 0.0)
        self.proportion_priors_ = self._fit_proportion_priors(
            pass_train, shot_train, pressure_train
        )
        self.position_baselines_ = self._fit_position_baselines(checkpoints)
        self.fitted_ = True
        return self

    def fit_transform(
        self, checkpoint_df: pd.DataFrame, event_dfs: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        self.fit(checkpoint_df, event_dfs)
        return self.transform(checkpoint_df, event_dfs)

    def transform(
        self, checkpoint_df: pd.DataFrame, event_dfs: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create model features for a checkpoint subset."""
        if not self.fitted_:
            raise ValueError("TemporalFeatureBuilder must be fitted before transform().")

        checkpoints = prepare_checkpoint_frame(checkpoint_df)
        features = checkpoints.copy()

        pass_features = self._aggregate_pass_features(checkpoints, event_dfs["pass"])
        run_features = self._aggregate_run_features(checkpoints, event_dfs["run"])
        shot_features = self._aggregate_shot_features(checkpoints, event_dfs["shot"])
        pressure_features = self._aggregate_pressure_features(checkpoints, event_dfs["pressure"])
        press_applied = self._aggregate_pressing_applied_features(
            checkpoints, event_dfs["pressure"]
        )

        for block in [pass_features, run_features, shot_features, pressure_features, press_applied]:
            features = features.merge(block, on=["player_appearance_id", "checkpoint"], how="left")

        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)
        features = self._add_derived_features(features)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return features

    def get_feature_sets(self, columns: Iterable[str]) -> dict[str, list[str]]:
        """Return research-question-oriented feature families."""
        cols = remove_redundant_model_features(columns)
        id_like = ID_COLS | NON_MODEL_COLS | {"checkpoint_period"}

        def keep(predicate) -> list[str]:
            return [c for c in cols if c not in id_like and predicate(c)]

        base = keep(
            lambda c: (
                c.startswith(("last15_", "cumul_"))
                or c
                in {
                    "checkpoint_min",
                    "checkpoint_abs_min",
                    "minutes_played_at_checkpoint",
                    "is_home",
                    "is_second_half",
                    "is_extra_time",
                    "is_late_game",
                    "is_attacker",
                    "is_midfielder",
                    "is_defender",
                    "is_goalkeeper",
                    "position",
                    "formation",
                }
            )
        )
        shots_sprints = keep(lambda c: any(k in c for k in ["shot", "sprint", "hsr", "speed"]))
        passing = keep(
            lambda c: (
                c.startswith(("pass_", "last5_pass_", "last15_pass_", "cumul_pass_")) or "xt_" in c
            )
        )
        pressure = keep(lambda c: "pressure" in c or "press_" in c)
        recent = keep(lambda c: c.startswith(("last5_", "last15_")) or "recent_" in c)
        cumulative = keep(lambda c: c.startswith("cumul_"))
        external = keep(
            lambda c: (
                c
                in {
                    "checkpoint_min",
                    "checkpoint_abs_min",
                    "is_home",
                    "is_second_half",
                    "is_extra_time",
                    "is_late_game",
                    "position",
                    "formation",
                    "minute_in",
                    "minutes_played_at_checkpoint",
                }
            )
        )
        all_features = [c for c in cols if c not in id_like and c != TARGET_COL]
        return {
            "base_checkpoint": sorted(set(base)),
            "sprints_shots": sorted(set(shots_sprints)),
            "base_plus_passing": sorted(set(base + passing)),
            "base_plus_pressure": sorted(set(base + pressure)),
            "recent_only": sorted(set(recent)),
            "cumulative_only": sorted(set(cumulative)),
            "external_context": sorted(set(external)),
            "all_features": sorted(set(all_features)),
        }

    def _prepare_passes(self, pass_df: pd.DataFrame) -> pd.DataFrame:
        df = pass_df.copy()
        df["period"] = df["period"].astype(str)
        df = add_absolute_time(df, "period", "minute", "event_abs_min")
        df["accurate_bool"] = _as_bool(df["accurate"])
        df["stage"] = df["stage"].fillna("unknown").replace("", "unknown").astype(str)
        df["addressee_missing"] = df["addressee_player_appearance_id"].isna().astype(int)
        df = df.sort_values(["player_appearance_id", "period", "event_abs_min", "id"])
        df["dest_stage"] = df.groupby(["player_appearance_id", "period"])["stage"].shift(-1)
        df["origin_xt"] = df["stage"].map(self.zone_xt_).fillna(0.0)
        df["dest_xt"] = df["dest_stage"].map(self.zone_xt_).fillna(df["origin_xt"])
        df["xt_added"] = df["dest_xt"] - df["origin_xt"]
        return df

    def _prepare_runs(self, run_df: pd.DataFrame) -> pd.DataFrame:
        df = run_df.copy()
        df["period"] = df["period"].astype(str)
        df = add_absolute_time(df, "period", "minute", "event_abs_min")
        df["stage"] = df["stage"].astype(str)
        df["run_type"] = df["run_type"].astype(str)
        df["is_sprint_event"] = (df["run_type"] == "sprint").astype(int)
        df["is_hsr_event"] = (df["run_type"] == "hsr").astype(int)
        return df

    def _prepare_shots(self, shot_df: pd.DataFrame) -> pd.DataFrame:
        df = shot_df.copy()
        df["period"] = df["period"].astype(str)
        df = add_absolute_time(df, "period", "minute", "event_abs_min")
        for col in ["stage", "body_part", "technique", "play_pattern"]:
            df[col] = df[col].fillna("unknown").replace("", "unknown").astype(str)
        df["under_pressure_bool"] = _as_bool(df["under_pressure"])
        df["is_set_piece"] = df["play_pattern"].isin(
            ["corner_kick", "direct_free_kick", "indirect_free_kick", "penalty"]
        )
        df["is_counter"] = df["play_pattern"].eq("counter_attack")
        df["is_head"] = df["body_part"].eq("head")
        df["is_foot"] = df["body_part"].isin(["left_foot", "right_foot"])
        df["is_blocked"] = df["block_player_appearance_id"].notna()
        return df

    def _prepare_pressure(self, pressure_df: pd.DataFrame) -> pd.DataFrame:
        df = pressure_df.copy()
        df["period"] = df["period"].astype(str)
        df = add_absolute_time(df, "period", "minute", "event_abs_min")
        df["stage"] = df["stage"].fillna("unknown").replace("", "unknown").astype(str)
        df["outcome"] = df["press_induced_outcome"].fillna("unknown").astype(str)
        df["accurate_bool"] = _as_bool(df["accurate"])
        df["is_turnover"] = df["outcome"].eq("turnover")
        df["is_positive_pressure_action"] = df["outcome"].isin(
            ["forward_pass", "backward_pass", "lateral_pass", "ball_carry"]
        )
        df["is_forward_escape"] = df["outcome"].isin(["forward_pass", "ball_carry"])
        df["is_backward_escape"] = df["outcome"].eq("backward_pass")
        df["is_lateral_escape"] = df["outcome"].eq("lateral_pass")
        df["pass_angle_abs"] = pd.to_numeric(df["pass_angle"], errors="coerce").abs()
        return df

    def _fit_zone_xt(self, pass_df: pd.DataFrame, shot_df: pd.DataFrame) -> dict[str, float]:
        if pass_df.empty and shot_df.empty:
            return dict.fromkeys(self.zones, 0.0)

        accurate = pass_df[pass_df["accurate_bool"]].copy()
        if accurate.empty:
            transition = np.eye(len(self.zones))
        else:
            transition_counts = pd.crosstab(accurate["stage"], accurate["dest_stage"]).reindex(
                index=self.zones, columns=self.zones, fill_value=0
            )
            transition = transition_counts.to_numpy(dtype=float)
            row_sums = transition.sum(axis=1, keepdims=True)
            transition = np.divide(
                transition,
                row_sums,
                out=np.eye(len(self.zones), dtype=float),
                where=row_sums != 0,
            )

        shot_counts = shot_df["stage"].value_counts()
        pass_counts = accurate["stage"].value_counts()
        shot_probs = []
        conversion = {"bottom": 0.02, "middle": 0.05, "top": 0.12}
        for zone in self.zones:
            n_shots = shot_counts.get(zone, 0)
            n_passes = pass_counts.get(zone, 0)
            shot_probs.append(n_shots / max(n_shots + n_passes, 1))

        values = np.zeros(len(self.zones), dtype=float)
        shot_probs_arr = np.asarray(shot_probs)
        goal_probs = np.asarray([conversion.get(zone, 0.05) for zone in self.zones])
        for _ in range(25):
            old = values.copy()
            values = shot_probs_arr * goal_probs + (1 - shot_probs_arr) * transition.dot(values)
            if np.max(np.abs(values - old)) < 1e-5:
                break
        return dict(zip(self.zones, values, strict=True))

    def _fit_pressure_turnover_rates(self, pressure_df: pd.DataFrame) -> dict[str, float]:
        if pressure_df.empty:
            return dict.fromkeys(self.zones, 0.0)
        rates = pressure_df.groupby("stage")["is_turnover"].mean().to_dict()
        return {
            zone: float(rates.get(zone, pressure_df["is_turnover"].mean())) for zone in self.zones
        }

    def _fit_proportion_priors(
        self,
        pass_df: pd.DataFrame,
        shot_df: pd.DataFrame,
        pressure_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Fit train-fold priors used to shrink noisy event-window proportions."""
        priors: dict[str, float] = {}
        if not pass_df.empty:
            priors.update(
                {
                    "pass_accuracy": _safe_mean(pass_df["accurate_bool"], 0.0),
                    "pass_missing_receiver_rate": _safe_mean(pass_df["addressee_missing"], 0.0),
                    "pass_top_share": float((pass_df["stage"] == "top").mean()),
                    "pass_middle_share": float((pass_df["stage"] == "middle").mean()),
                    "pass_bottom_share": float((pass_df["stage"] == "bottom").mean()),
                }
            )
        if not shot_df.empty:
            priors.update(
                {
                    "shot_under_pressure_rate": _safe_mean(shot_df["under_pressure_bool"], 0.0),
                    "shot_top_share": float((shot_df["stage"] == "top").mean()),
                    "shot_head_rate": _safe_mean(shot_df["is_head"], 0.0),
                    "shot_foot_rate": _safe_mean(shot_df["is_foot"], 0.0),
                    "shot_blocked_rate": _safe_mean(shot_df["is_blocked"], 0.0),
                }
            )
        if not pressure_df.empty:
            priors.update(
                {
                    "pressure_retention": _safe_mean(
                        pressure_df["is_positive_pressure_action"], 0.0
                    ),
                    "pressure_forward_escape_rate": _safe_mean(
                        pressure_df["is_forward_escape"], 0.0
                    ),
                    "pressure_backward_escape_rate": _safe_mean(
                        pressure_df["is_backward_escape"], 0.0
                    ),
                    "pressure_lateral_escape_rate": _safe_mean(
                        pressure_df["is_lateral_escape"], 0.0
                    ),
                    "pressure_turnover_rate": _safe_mean(pressure_df["is_turnover"], 0.0),
                    "pressure_top_share": float((pressure_df["stage"] == "top").mean()),
                }
            )
        return priors

    def _fit_position_baselines(self, checkpoints: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "last15_sprints",
            "last15_hsr",
            "last15_distance",
            "last15_shots",
            "cumul_shots",
            "cumul_distance",
        ]
        available = [c for c in cols if c in checkpoints.columns]
        return checkpoints.groupby("position", observed=True)[available].mean()

    def _aggregate_pass_features(
        self, checkpoints: pd.DataFrame, pass_df: pd.DataFrame
    ) -> pd.DataFrame:
        columns = []
        for prefix in WINDOWS:
            columns.extend(
                [
                    f"{prefix}_pass_count",
                    f"{prefix}_pass_has_events",
                    f"{prefix}_pass_accuracy",
                    f"{prefix}_pass_receiver_diversity",
                    f"{prefix}_pass_missing_receiver_rate",
                    f"{prefix}_pass_top_share",
                    f"{prefix}_pass_middle_share",
                    f"{prefix}_pass_bottom_share",
                    f"{prefix}_xt_added",
                    f"{prefix}_positive_xt_count",
                ]
            )
        if pass_df.empty:
            return _empty_features(checkpoints, columns)

        passes = self._prepare_passes(pass_df)
        merged = _merge_events(checkpoints, passes)
        outputs = checkpoints[["player_appearance_id", "checkpoint"]].copy()
        keys = ["player_appearance_id", "checkpoint"]

        for prefix, window in WINDOWS.items():
            subset = merged[_window_mask(merged, window)]
            if subset.empty:
                agg = _empty_features(checkpoints, [c for c in columns if c.startswith(prefix)])
            else:
                grouped = subset.groupby(keys, sort=False)
                agg = grouped.agg(
                    **{
                        f"{prefix}_pass_count": ("id", "count"),
                        f"{prefix}_pass_has_events": ("id", lambda x: int(len(x) > 0)),
                        f"{prefix}_pass_accuracy": ("accurate_bool", "mean"),
                        f"{prefix}_pass_receiver_diversity": (
                            "addressee_player_appearance_id",
                            lambda x: x.dropna().nunique(),
                        ),
                        f"{prefix}_pass_missing_receiver_rate": ("addressee_missing", "mean"),
                        f"{prefix}_pass_top_share": ("stage", lambda x: (x == "top").mean()),
                        f"{prefix}_pass_middle_share": ("stage", lambda x: (x == "middle").mean()),
                        f"{prefix}_pass_bottom_share": ("stage", lambda x: (x == "bottom").mean()),
                        f"{prefix}_xt_added": ("xt_added", "sum"),
                        f"{prefix}_positive_xt_count": ("xt_added", lambda x: (x > 0).sum()),
                    }
                ).reset_index()
            outputs = outputs.merge(agg, on=keys, how="left")
        return outputs.fillna(0)

    def _aggregate_run_features(
        self, checkpoints: pd.DataFrame, run_df: pd.DataFrame
    ) -> pd.DataFrame:
        columns = []
        for prefix in WINDOWS:
            columns.extend(
                [
                    f"{prefix}_run_event_count",
                    f"{prefix}_run_has_events",
                    f"{prefix}_run_distance",
                    f"{prefix}_run_max_speed",
                    f"{prefix}_run_mean_speed",
                    f"{prefix}_sprint_event_count",
                    f"{prefix}_hsr_event_count",
                    f"{prefix}_run_top_share",
                    f"{prefix}_run_entropy",
                ]
            )
        if run_df.empty:
            return _empty_features(checkpoints, columns)

        runs = self._prepare_runs(run_df)
        merged = _merge_events(checkpoints, runs)
        outputs = checkpoints[["player_appearance_id", "checkpoint"]].copy()
        keys = ["player_appearance_id", "checkpoint"]

        for prefix, window in WINDOWS.items():
            subset = merged[_window_mask(merged, window)]
            if subset.empty:
                agg = _empty_features(checkpoints, [c for c in columns if c.startswith(prefix)])
            else:
                grouped = subset.groupby(keys, sort=False)
                agg = grouped.agg(
                    **{
                        f"{prefix}_run_event_count": ("id", "count"),
                        f"{prefix}_run_has_events": ("id", lambda x: int(len(x) > 0)),
                        f"{prefix}_run_distance": ("distance", "sum"),
                        f"{prefix}_run_max_speed": ("max_speed", "max"),
                        f"{prefix}_run_mean_speed": ("max_speed", "mean"),
                        f"{prefix}_sprint_event_count": ("is_sprint_event", "sum"),
                        f"{prefix}_hsr_event_count": ("is_hsr_event", "sum"),
                        f"{prefix}_run_top_share": ("stage", lambda x: (x == "top").mean()),
                        f"{prefix}_run_entropy": ("run_type", _entropy),
                    }
                ).reset_index()
            outputs = outputs.merge(agg, on=keys, how="left")
        return outputs.fillna(0)

    def _aggregate_shot_features(
        self, checkpoints: pd.DataFrame, shot_df: pd.DataFrame
    ) -> pd.DataFrame:
        columns = []
        for prefix in WINDOWS:
            columns.extend(
                [
                    f"{prefix}_shot_event_count",
                    f"{prefix}_shot_has_events",
                    f"{prefix}_shot_under_pressure_rate",
                    f"{prefix}_shot_top_share",
                    f"{prefix}_shot_counter_count",
                    f"{prefix}_shot_set_piece_count",
                    f"{prefix}_shot_head_rate",
                    f"{prefix}_shot_foot_rate",
                    f"{prefix}_shot_blocked_rate",
                ]
            )
        if shot_df.empty:
            return _empty_features(checkpoints, columns)

        shots = self._prepare_shots(shot_df)
        merged = _merge_events(checkpoints, shots)
        outputs = checkpoints[["player_appearance_id", "checkpoint"]].copy()
        keys = ["player_appearance_id", "checkpoint"]

        for prefix, window in WINDOWS.items():
            subset = merged[_window_mask(merged, window)]
            if subset.empty:
                agg = _empty_features(checkpoints, [c for c in columns if c.startswith(prefix)])
            else:
                grouped = subset.groupby(keys, sort=False)
                agg = grouped.agg(
                    **{
                        f"{prefix}_shot_event_count": ("id", "count"),
                        f"{prefix}_shot_has_events": ("id", lambda x: int(len(x) > 0)),
                        f"{prefix}_shot_under_pressure_rate": ("under_pressure_bool", "mean"),
                        f"{prefix}_shot_top_share": ("stage", lambda x: (x == "top").mean()),
                        f"{prefix}_shot_counter_count": ("is_counter", "sum"),
                        f"{prefix}_shot_set_piece_count": ("is_set_piece", "sum"),
                        f"{prefix}_shot_head_rate": ("is_head", "mean"),
                        f"{prefix}_shot_foot_rate": ("is_foot", "mean"),
                        f"{prefix}_shot_blocked_rate": ("is_blocked", "mean"),
                    }
                ).reset_index()
            outputs = outputs.merge(agg, on=keys, how="left")
        return outputs.fillna(0)

    def _aggregate_pressure_features(
        self, checkpoints: pd.DataFrame, pressure_df: pd.DataFrame
    ) -> pd.DataFrame:
        columns = []
        for prefix in WINDOWS:
            columns.extend(
                [
                    f"{prefix}_pressure_count",
                    f"{prefix}_pressure_has_events",
                    f"{prefix}_pressure_retention",
                    f"{prefix}_pressure_forward_escape_rate",
                    f"{prefix}_pressure_backward_escape_rate",
                    f"{prefix}_pressure_lateral_escape_rate",
                    f"{prefix}_pressure_turnover_rate",
                    f"{prefix}_pressure_expected_turnovers",
                    f"{prefix}_pressure_quality",
                    f"{prefix}_pressure_angle_mean_abs",
                    f"{prefix}_pressure_angle_std_abs",
                    f"{prefix}_pressure_top_share",
                ]
            )
        if pressure_df.empty:
            return _empty_features(checkpoints, columns)

        pressure = self._prepare_pressure(pressure_df)
        pressure["expected_turnover"] = (
            pressure["stage"]
            .map(self.pressure_turnover_rate_)
            .fillna(self.pressure_global_turnover_rate_)
        )
        merged = _merge_events(checkpoints, pressure)
        outputs = checkpoints[["player_appearance_id", "checkpoint"]].copy()
        keys = ["player_appearance_id", "checkpoint"]

        for prefix, window in WINDOWS.items():
            subset = merged[_window_mask(merged, window)]
            if subset.empty:
                agg = _empty_features(checkpoints, [c for c in columns if c.startswith(prefix)])
            else:
                grouped = subset.groupby(keys, sort=False)
                agg = grouped.agg(
                    **{
                        f"{prefix}_pressure_count": ("id", "count"),
                        f"{prefix}_pressure_has_events": ("id", lambda x: int(len(x) > 0)),
                        f"{prefix}_pressure_retention": ("is_positive_pressure_action", "mean"),
                        f"{prefix}_pressure_forward_escape_rate": ("is_forward_escape", "mean"),
                        f"{prefix}_pressure_backward_escape_rate": (
                            "is_backward_escape",
                            "mean",
                        ),
                        f"{prefix}_pressure_lateral_escape_rate": (
                            "is_lateral_escape",
                            "mean",
                        ),
                        f"{prefix}_pressure_turnover_rate": ("is_turnover", "mean"),
                        f"{prefix}_pressure_expected_turnovers": ("expected_turnover", "sum"),
                        f"{prefix}_pressure_angle_mean_abs": ("pass_angle_abs", "mean"),
                        f"{prefix}_pressure_angle_std_abs": ("pass_angle_abs", "std"),
                        f"{prefix}_pressure_top_share": ("stage", lambda x: (x == "top").mean()),
                    }
                ).reset_index()
                actual = grouped["is_turnover"].sum().reset_index(name="actual_turnovers")
                agg = agg.merge(actual, on=keys, how="left")
                agg[f"{prefix}_pressure_quality"] = (
                    agg[f"{prefix}_pressure_expected_turnovers"] - agg["actual_turnovers"]
                )
                agg = agg.drop(columns=["actual_turnovers"])
            outputs = outputs.merge(agg, on=keys, how="left")
        return outputs.fillna(0)

    def _aggregate_pressing_applied_features(
        self, checkpoints: pd.DataFrame, pressure_df: pd.DataFrame
    ) -> pd.DataFrame:
        columns = [
            "last15_press_applied_count",
            "last15_press_forced_turnover_count",
            "cumul_press_applied_count",
            "cumul_press_forced_turnover_count",
        ]
        if pressure_df.empty:
            return _empty_features(checkpoints, columns)

        pressure = self._prepare_pressure(pressure_df).rename(
            columns={"player_appearance_id": "pressed_player_appearance_id"}
        )
        pressure["player_appearance_id"] = pd.to_numeric(
            pressure["pressing_player_appearance_id"], errors="coerce"
        )
        pressure = pressure.dropna(subset=["player_appearance_id"])
        pressure["player_appearance_id"] = pressure["player_appearance_id"].astype(
            checkpoints["player_appearance_id"].dtype
        )
        merged = _merge_events(checkpoints, pressure)
        outputs = checkpoints[["player_appearance_id", "checkpoint"]].copy()
        keys = ["player_appearance_id", "checkpoint"]

        specs = {
            "last15": 15,
            "cumul": None,
        }
        for prefix, window in specs.items():
            subset = merged[_window_mask(merged, window)]
            if subset.empty:
                agg = _empty_features(
                    checkpoints,
                    [f"{prefix}_press_applied_count", f"{prefix}_press_forced_turnover_count"],
                )
            else:
                agg = (
                    subset.groupby(keys, sort=False)
                    .agg(
                        **{
                            f"{prefix}_press_applied_count": ("id", "count"),
                            f"{prefix}_press_forced_turnover_count": ("is_turnover", "sum"),
                        }
                    )
                    .reset_index()
                )
            outputs = outputs.merge(agg, on=keys, how="left")
        return outputs.fillna(0)

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        def smooth_rate(col: str, count_col: str, prior_key: str, strength: float = 8.0) -> None:
            prior = self.proportion_priors_.get(prior_key, 0.0)
            counts = pd.to_numeric(out[count_col], errors="coerce").fillna(0).clip(lower=0)
            values = pd.to_numeric(out[col], errors="coerce").fillna(prior)
            out[f"{col}_smoothed"] = ((values * counts) + (prior * strength)) / (counts + strength)

        for prefix, window in WINDOWS.items():
            duration = _window_duration(out, window)
            for family, count_col in {
                "pass": f"{prefix}_pass_count",
                "run": f"{prefix}_run_event_count",
                "shot": f"{prefix}_shot_event_count",
                "pressure": f"{prefix}_pressure_count",
            }.items():
                out[f"{prefix}_{family}_events_per_minute"] = (
                    _safe_div(out[count_col], duration).fillna(0).values
                )

            smooth_specs = {
                f"{prefix}_pass_accuracy": (f"{prefix}_pass_count", "pass_accuracy"),
                f"{prefix}_pass_missing_receiver_rate": (
                    f"{prefix}_pass_count",
                    "pass_missing_receiver_rate",
                ),
                f"{prefix}_pass_top_share": (f"{prefix}_pass_count", "pass_top_share"),
                f"{prefix}_pass_middle_share": (f"{prefix}_pass_count", "pass_middle_share"),
                f"{prefix}_pass_bottom_share": (f"{prefix}_pass_count", "pass_bottom_share"),
                f"{prefix}_shot_under_pressure_rate": (
                    f"{prefix}_shot_event_count",
                    "shot_under_pressure_rate",
                ),
                f"{prefix}_shot_top_share": (f"{prefix}_shot_event_count", "shot_top_share"),
                f"{prefix}_shot_head_rate": (f"{prefix}_shot_event_count", "shot_head_rate"),
                f"{prefix}_shot_foot_rate": (f"{prefix}_shot_event_count", "shot_foot_rate"),
                f"{prefix}_shot_blocked_rate": (
                    f"{prefix}_shot_event_count",
                    "shot_blocked_rate",
                ),
                f"{prefix}_pressure_retention": (
                    f"{prefix}_pressure_count",
                    "pressure_retention",
                ),
                f"{prefix}_pressure_forward_escape_rate": (
                    f"{prefix}_pressure_count",
                    "pressure_forward_escape_rate",
                ),
                f"{prefix}_pressure_backward_escape_rate": (
                    f"{prefix}_pressure_count",
                    "pressure_backward_escape_rate",
                ),
                f"{prefix}_pressure_lateral_escape_rate": (
                    f"{prefix}_pressure_count",
                    "pressure_lateral_escape_rate",
                ),
                f"{prefix}_pressure_turnover_rate": (
                    f"{prefix}_pressure_count",
                    "pressure_turnover_rate",
                ),
                f"{prefix}_pressure_top_share": (
                    f"{prefix}_pressure_count",
                    "pressure_top_share",
                ),
            }
            for col, (count_col, prior_key) in smooth_specs.items():
                if col in out.columns and count_col in out.columns:
                    smooth_rate(col, count_col, prior_key)

        out["shot_accuracy_cumul"] = (
            _safe_div(out["cumul_shots_on_target"], out["cumul_shots"]).fillna(0).values
        )
        out["shot_pressure_ratio_cumul"] = (
            _safe_div(out["cumul_shots_under_press"], out["cumul_shots"]).fillna(0).values
        )
        out["top_third_shot_ratio_cumul"] = (
            _safe_div(out["cumul_shots_top_third"], out["cumul_shots"]).fillna(0).values
        )

        out["recent_shot_ratio"] = (
            _safe_div(out["last15_shots"], out["cumul_shots"] + 1).fillna(0).values
        )
        out["recent_sprint_ratio"] = (
            _safe_div(out["last15_sprints"], out["cumul_sprints"] + 1).fillna(0).values
        )
        out["recent_hsr_ratio"] = (
            _safe_div(out["last15_hsr"], out["cumul_hsr"] + 1).fillna(0).values
        )
        out["recent_distance_ratio"] = (
            _safe_div(out["last15_distance"], out["cumul_distance"] + 1).fillna(0).values
        )
        out["recent_xt_ratio"] = (
            _safe_div(out["last15_xt_added"], out["cumul_xt_added"].abs() + 1).fillna(0).values
        )
        out["last15_shot_to_pass_ratio"] = (
            _safe_div(out["last15_shots"], out["last15_pass_count"]).fillna(0).clip(0, 10).values
        )
        out["cumul_shot_to_pass_ratio"] = (
            _safe_div(out["cumul_shots"], out["cumul_pass_count"]).fillna(0).clip(0, 10).values
        )
        out["recent_pressure_quality_delta"] = out["last15_pressure_quality"] - (
            out["cumul_pressure_quality"] - out["last15_pressure_quality"]
        )
        out["workload_ratio_hsr"] = (
            _safe_div(
                out["last15_hsr"], out["cumul_hsr"] / (out["checkpoint_abs_min"] / 15).clip(lower=1)
            )
            .fillna(1)
            .clip(0, 5)
            .values
        )
        out["workload_ratio_sprints"] = (
            _safe_div(
                out["last15_sprints"],
                out["cumul_sprints"] / (out["checkpoint_abs_min"] / 15).clip(lower=1),
            )
            .fillna(1)
            .clip(0, 5)
            .values
        )
        out["speed_preservation"] = (
            _safe_div(out["last15_peak_speed"], out["cumul_peak_speed"]).fillna(1).clip(0, 2).values
        )
        out["last15_run_distance_per_event"] = (
            _safe_div(out["last15_run_distance"], out["last15_run_event_count"]).fillna(0).values
        )
        out["cumul_run_distance_per_event"] = (
            _safe_div(out["cumul_run_distance"], out["cumul_run_event_count"]).fillna(0).values
        )
        for prefix in WINDOWS:
            passive_pressure = (
                out[f"{prefix}_pressure_backward_escape_rate"]
                + out[f"{prefix}_pressure_lateral_escape_rate"]
                + out[f"{prefix}_pressure_turnover_rate"]
            )
            out[f"{prefix}_pressure_verticality_ratio"] = (
                _safe_div(out[f"{prefix}_pressure_forward_escape_rate"], passive_pressure)
                .fillna(0)
                .clip(0, 10)
                .values
            )
        out["has_intensity_surge"] = (out["workload_ratio_hsr"] > 1.2).astype(int)
        out["is_fatigued"] = (out["workload_ratio_hsr"] < 0.8).astype(int)

        if self.position_baselines_ is not None:
            baselines = self.position_baselines_.add_prefix("pos_avg_").reset_index()
            out = out.merge(baselines, on="position", how="left")
            for col in [
                "last15_sprints",
                "last15_hsr",
                "last15_distance",
                "last15_shots",
                "cumul_shots",
                "cumul_distance",
            ]:
                avg_col = f"pos_avg_{col}"
                if avg_col in out.columns:
                    out[f"position_delta_{col}"] = out[col] - out[avg_col].fillna(out[col].mean())
                    out = out.drop(columns=[avg_col])

        out["attacker_x_recent_shots"] = out["is_attacker"] * out["last15_shots"]
        out["attacker_x_cumul_shots"] = out["is_attacker"] * out["cumul_shots"]
        out["attacker_x_xt"] = out["is_attacker"] * out["cumul_xt_added"]
        out["home_x_attacker"] = out["is_home"] * out["is_attacker"]
        out["late_x_attacker"] = out["is_late_game"] * out["is_attacker"]
        out["surge_x_shots"] = out["has_intensity_surge"] * out["cumul_shots"]
        out["fatigue_x_shots"] = out["is_fatigued"] * out["cumul_shots"]
        out["pressure_x_xt"] = out["cumul_pressure_quality"] * out["cumul_xt_added"]
        out["attacker_x_last15_pass_top_share"] = (
            out["is_attacker"] * out["last15_pass_top_share_smoothed"]
        )
        out["attacker_x_last15_xt"] = out["is_attacker"] * out["last15_xt_added"]
        out["late_x_shot_pressure"] = (
            out["is_late_game"] * out["last15_shot_under_pressure_rate_smoothed"]
        )
        out["late_x_recent_distance"] = out["is_late_game"] * out["recent_distance_ratio"]
        out["minutes_x_recent_xt_ratio"] = (
            out["minutes_played_at_checkpoint"] * out["recent_xt_ratio"]
        )
        out["pass_xt_per_pass"] = (
            _safe_div(out["cumul_xt_added"], out["cumul_pass_count"]).fillna(0).values
        )
        out["shots_per_90_so_far"] = (
            _safe_div(out["cumul_shots"] * 90, out["minutes_played_at_checkpoint"]).fillna(0).values
        )
        out["sprints_per_90_so_far"] = (
            _safe_div(out["cumul_sprints"] * 90, out["minutes_played_at_checkpoint"])
            .fillna(0)
            .values
        )
        return out


def modeling_columns(df: pd.DataFrame, feature_cols: Iterable[str] | None = None) -> list[str]:
    """Return usable feature columns, excluding IDs and target."""
    if feature_cols is not None:
        return [
            c
            for c in feature_cols
            if c in df.columns and c not in ID_COLS and c not in NON_MODEL_COLS and c != TARGET_COL
        ]
    excluded = ID_COLS | NON_MODEL_COLS | {"is_on_pitch"}
    return [c for c in df.columns if c not in excluded and c != TARGET_COL]
