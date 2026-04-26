import pandas as pd

from src.temporal_features import (
    TemporalFeatureBuilder,
    modeling_columns,
    prepare_checkpoint_frame,
    remove_redundant_model_features,
)


def test_prepare_checkpoint_uses_absolute_match_time_for_second_half():
    df = pd.DataFrame(
        {
            "player_appearance_id": [1],
            "player_id": [10],
            "fixture_id": [100],
            "date": ["2025-01-01"],
            "checkpoint": ["H2_15"],
            "checkpoint_period": ["half_2"],
            "checkpoint_min": [15],
            "position": ["A"],
            "is_home": ["TRUE"],
            "formation": ["4-3-3"],
            "minute_in": [46],
            "minute_out": [90],
            "subbed": ["FALSE"],
            "jersey_number": [9],
            "scored_after": [0],
        }
    )

    out = prepare_checkpoint_frame(df)

    assert out.loc[0, "checkpoint_abs_min"] == 60
    assert out.loc[0, "is_on_pitch"] == 1


def test_temporal_builder_excludes_future_events_and_uses_all_event_blocks():
    checkpoints = pd.DataFrame(
        {
            "player_appearance_id": [1],
            "player_id": [10],
            "fixture_id": [100],
            "date": ["2025-01-01"],
            "checkpoint": ["H1_15"],
            "checkpoint_period": ["half_1"],
            "checkpoint_min": [15],
            "position": ["A"],
            "is_home": ["TRUE"],
            "formation": ["4-3-3"],
            "minute_in": [1],
            "minute_out": [90],
            "subbed": ["FALSE"],
            "jersey_number": [9],
            "last15_sprints": [1],
            "last15_hsr": [2],
            "last15_distance": [30.0],
            "last15_mean_max_speed": [7.0],
            "last15_peak_speed": [8.0],
            "last15_shots": [1],
            "last15_shots_on_target": [0],
            "last15_shots_under_press": [0],
            "last15_shots_top_third": [1],
            "cumul_sprints": [1],
            "cumul_hsr": [2],
            "cumul_distance": [30.0],
            "cumul_mean_max_speed": [7.0],
            "cumul_peak_speed": [8.0],
            "cumul_shots": [1],
            "cumul_shots_on_target": [0],
            "cumul_shots_under_press": [0],
            "cumul_shots_top_third": [1],
            "scored_after": [0],
        }
    )
    event_dfs = {
        "pass": pd.DataFrame(
            {
                "id": [1, 2],
                "period": ["half_1", "half_1"],
                "player_appearance_id": [1, 1],
                "addressee_player_appearance_id": [2, 2],
                "accurate": ["TRUE", "TRUE"],
                "minute": [10, 20],
                "stage": ["middle", "top"],
            }
        ),
        "run": pd.DataFrame(
            {
                "id": [1, 2],
                "period": ["half_1", "half_1"],
                "stage": ["top", "top"],
                "possession": [1, 1],
                "run_type": ["sprint", "hsr"],
                "minute": [12, 30],
                "min_speed": [6.0, 6.0],
                "max_speed": [8.5, 8.0],
                "distance": [20.0, 25.0],
                "player_appearance_id": [1, 1],
            }
        ),
        "shot": pd.DataFrame(
            {
                "id": [1, 2],
                "period": ["half_1", "half_1"],
                "player_appearance_id": [1, 1],
                "body_part": ["right_foot", "head"],
                "technique": ["normal", "normal"],
                "play_pattern": ["regular_play", "corner_kick"],
                "own_goal_player_appearance_id": [None, None],
                "block_player_appearance_id": [None, None],
                "minute": [14, 18],
                "possession": [1, 2],
                "stage": ["top", "top"],
                "under_pressure": ["TRUE", "FALSE"],
            }
        ),
        "pressure": pd.DataFrame(
            {
                "id": [1, 2],
                "period": ["half_1", "half_1"],
                "player_appearance_id": [1, 1],
                "addressee_player_appearance_id": [2, 2],
                "accurate": ["TRUE", "FALSE"],
                "pressing_player_appearance_id": [3, 3],
                "press_induced_outcome": ["forward_pass", "lateral_pass"],
                "pass_angle": [30.0, 10.0],
                "minute": [11, 12],
                "stage": ["middle", "top"],
            }
        ),
    }

    features = TemporalFeatureBuilder().fit_transform(checkpoints, event_dfs)

    assert features.loc[0, "cumul_pass_count"] == 1
    assert features.loc[0, "cumul_pass_has_events"] == 1
    assert features.loc[0, "cumul_pass_events_per_minute"] == 1 / 15
    assert features.loc[0, "cumul_run_event_count"] == 1
    assert features.loc[0, "cumul_run_has_events"] == 1
    assert features.loc[0, "cumul_shot_event_count"] == 1
    assert features.loc[0, "cumul_shot_has_events"] == 1
    assert features.loc[0, "cumul_pressure_count"] == 2
    assert features.loc[0, "cumul_pressure_has_events"] == 1
    assert features.loc[0, "cumul_pressure_retention"] == 1
    assert "cumul_pressure_retention_smoothed" in features.columns
    assert features.loc[0, "cumul_pressure_forward_escape_rate"] == 0.5
    assert features.loc[0, "cumul_pressure_lateral_escape_rate"] == 0.5
    assert features.loc[0, "cumul_pressure_verticality_ratio"] == 1
    assert features.loc[0, "cumul_shot_to_pass_ratio"] == 1
    assert features.loc[0, "last15_shot_to_pass_ratio"] == 1
    assert features.loc[0, "cumul_run_distance_per_event"] == 20


def test_temporal_builder_excludes_second_half_events_after_checkpoint():
    checkpoints = pd.DataFrame(
        {
            "player_appearance_id": [1],
            "player_id": [10],
            "fixture_id": [100],
            "date": ["2025-01-01"],
            "checkpoint": ["H2_15"],
            "checkpoint_period": ["half_2"],
            "checkpoint_min": [15],
            "position": ["A"],
            "is_home": ["TRUE"],
            "formation": ["4-3-3"],
            "minute_in": [1],
            "minute_out": [90],
            "subbed": ["FALSE"],
            "jersey_number": [9],
            "last15_sprints": [0],
            "last15_hsr": [0],
            "last15_distance": [0.0],
            "last15_mean_max_speed": [0.0],
            "last15_peak_speed": [0.0],
            "last15_shots": [0],
            "last15_shots_on_target": [0],
            "last15_shots_under_press": [0],
            "last15_shots_top_third": [0],
            "cumul_sprints": [0],
            "cumul_hsr": [0],
            "cumul_distance": [0.0],
            "cumul_mean_max_speed": [0.0],
            "cumul_peak_speed": [0.0],
            "cumul_shots": [0],
            "cumul_shots_on_target": [0],
            "cumul_shots_under_press": [0],
            "cumul_shots_top_third": [0],
            "scored_after": [0],
        }
    )
    event_dfs = {
        "pass": pd.DataFrame(
            {
                "id": [1, 2, 3],
                "period": ["half_1", "half_2", "half_2"],
                "player_appearance_id": [1, 1, 1],
                "addressee_player_appearance_id": [2, 2, 2],
                "accurate": ["TRUE", "TRUE", "TRUE"],
                "minute": [40, 10, 20],
                "stage": ["middle", "top", "top"],
            }
        ),
        "run": pd.DataFrame(
            columns=[
                "id",
                "period",
                "stage",
                "possession",
                "run_type",
                "minute",
                "min_speed",
                "max_speed",
                "distance",
                "player_appearance_id",
            ]
        ),
        "shot": pd.DataFrame(
            columns=[
                "id",
                "period",
                "player_appearance_id",
                "body_part",
                "technique",
                "play_pattern",
                "own_goal_player_appearance_id",
                "block_player_appearance_id",
                "minute",
                "possession",
                "stage",
                "under_pressure",
            ]
        ),
        "pressure": pd.DataFrame(
            columns=[
                "id",
                "period",
                "player_appearance_id",
                "addressee_player_appearance_id",
                "accurate",
                "pressing_player_appearance_id",
                "press_induced_outcome",
                "pass_angle",
                "minute",
                "stage",
            ]
        ),
    }

    features = TemporalFeatureBuilder().fit_transform(checkpoints, event_dfs)

    assert features.loc[0, "checkpoint_abs_min"] == 60
    assert features.loc[0, "cumul_pass_count"] == 2
    assert features.loc[0, "last15_pass_count"] == 1
    assert features.loc[0, "cumul_run_has_events"] == 0
    assert features.loc[0, "cumul_shot_has_events"] == 0


def test_modeling_columns_exclude_future_substitution_fields():
    df = pd.DataFrame(
        {
            "player_appearance_id": [1],
            "fixture_id": [100],
            "checkpoint": ["H1_15"],
            "checkpoint_min": [15],
            "minute_in": [1],
            "minute_out": [90],
            "subbed": [1],
            "last15_shots": [0],
            "scored_after": [0],
        }
    )

    cols = modeling_columns(df)

    assert "last15_shots" in cols
    assert "minute_out" not in cols
    assert "subbed" not in cols


def test_feature_sets_prune_known_redundant_aliases():
    df = pd.DataFrame(
        {
            "last15_sprints": [1],
            "last15_sprint_event_count": [1],
            "last15_run_distance": [20.0],
            "last15_distance": [20.0],
            "position": ["A"],
            "is_attacker": [1],
            "formation": ["4-3-3"],
            "scored_after": [0],
        }
    )

    cols = remove_redundant_model_features(modeling_columns(df))
    feature_sets = TemporalFeatureBuilder().get_feature_sets(modeling_columns(df))

    assert "last15_sprints" in cols
    assert "is_attacker" in cols
    assert "last15_sprint_event_count" not in cols
    assert "last15_run_distance" not in cols
    assert "position" not in cols
    assert "last15_sprint_event_count" not in feature_sets["all_features"]
    assert "position" not in feature_sets["external_context"]
    assert "formation" in feature_sets["external_context"]


def test_temporal_builder_uses_period_and_minute_for_stoppage_time():
    checkpoints = pd.DataFrame(
        {
            "player_appearance_id": [1],
            "player_id": [10],
            "fixture_id": [100],
            "date": ["2025-01-01"],
            "checkpoint": ["H1_45"],
            "checkpoint_period": ["half_1"],
            "checkpoint_min": [45],
            "position": ["A"],
            "is_home": ["TRUE"],
            "formation": ["4-3-3"],
            "minute_in": [1],
            "minute_out": [90],
            "subbed": ["FALSE"],
            "jersey_number": [9],
            "last15_sprints": [0],
            "last15_hsr": [0],
            "last15_distance": [0.0],
            "last15_mean_max_speed": [0.0],
            "last15_peak_speed": [0.0],
            "last15_shots": [0],
            "last15_shots_on_target": [0],
            "last15_shots_under_press": [0],
            "last15_shots_top_third": [0],
            "cumul_sprints": [0],
            "cumul_hsr": [0],
            "cumul_distance": [0.0],
            "cumul_mean_max_speed": [0.0],
            "cumul_peak_speed": [0.0],
            "cumul_shots": [0],
            "cumul_shots_on_target": [0],
            "cumul_shots_under_press": [0],
            "cumul_shots_top_third": [0],
            "scored_after": [0],
        }
    )
    event_dfs = {
        "pass": pd.DataFrame(
            {
                "id": [1, 2],
                "period": ["half_1", "half_1"],
                "player_appearance_id": [1, 1],
                "addressee_player_appearance_id": [2, 2],
                "accurate": ["TRUE", "TRUE"],
                "minute": [44, 47],
                "stage": ["middle", "top"],
            }
        ),
        "run": pd.DataFrame(
            {
                "id": [1],
                "period": ["half_1"],
                "stage": ["top"],
                "possession": [1],
                "run_type": ["sprint"],
                "minute": [47],
                "min_speed": [6.0],
                "max_speed": [8.5],
                "distance": [20.0],
                "player_appearance_id": [1],
            }
        ),
        "shot": pd.DataFrame(
            {
                "id": [1],
                "period": ["half_1"],
                "player_appearance_id": [1],
                "body_part": ["right_foot"],
                "technique": ["normal"],
                "play_pattern": ["regular_play"],
                "own_goal_player_appearance_id": [None],
                "block_player_appearance_id": [None],
                "minute": [47],
                "possession": [1],
                "stage": ["top"],
                "under_pressure": ["TRUE"],
            }
        ),
        "pressure": pd.DataFrame(
            {
                "id": [1],
                "period": ["half_1"],
                "player_appearance_id": [1],
                "addressee_player_appearance_id": [2],
                "accurate": ["TRUE"],
                "pressing_player_appearance_id": [3],
                "press_induced_outcome": ["forward_pass"],
                "pass_angle": [30.0],
                "minute": [47],
                "stage": ["middle"],
            }
        ),
    }

    features = TemporalFeatureBuilder().fit_transform(checkpoints, event_dfs)

    assert features.loc[0, "cumul_pass_count"] == 1
    assert features.loc[0, "cumul_run_event_count"] == 0
    assert features.loc[0, "cumul_shot_event_count"] == 0
    assert features.loc[0, "cumul_pressure_count"] == 0
