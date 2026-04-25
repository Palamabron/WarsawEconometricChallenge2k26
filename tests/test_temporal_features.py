import pandas as pd

from src.temporal_features import TemporalFeatureBuilder, prepare_checkpoint_frame


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
                "press_induced_outcome": ["forward_pass", "turnover"],
                "pass_angle": [30.0, 10.0],
                "minute": [11, 19],
                "stage": ["middle", "top"],
            }
        ),
    }

    features = TemporalFeatureBuilder().fit_transform(checkpoints, event_dfs)

    assert features.loc[0, "cumul_pass_count"] == 1
    assert features.loc[0, "cumul_run_event_count"] == 1
    assert features.loc[0, "cumul_shot_event_count"] == 1
    assert features.loc[0, "cumul_pressure_count"] == 1
    assert features.loc[0, "cumul_pressure_retention"] == 1
