import pytest
from pydantic import ValidationError

from src.config import Config


def _write_config(tmp_path, n_folds: int = 5):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  data_dir: data
  checkpoint_file: players_quarters_final.csv
  pass_file: player_appearance_pass.csv
  run_file: player_appearance_run.csv
  shot_file: player_appearance_shot_limited.csv
  pressure_file: player_appearance_behaviour_under_pressure.csv
cross_validation:
  n_folds: {n_folds}
  random_state: 123
  group_by: fixture_id
  stratify: true
""".lstrip()
    )
    return config_path


def test_config_preserves_dotted_key_access(tmp_path):
    config = Config(str(_write_config(tmp_path, n_folds=3)))

    assert config.n_folds == 3
    assert config.random_state == 123
    assert config.group_by_column == "fixture_id"
    assert config.get("missing.path", "fallback") == "fallback"


def test_config_validates_cross_validation_fold_count(tmp_path):
    with pytest.raises(ValidationError):
        Config(str(_write_config(tmp_path, n_folds=1)))
