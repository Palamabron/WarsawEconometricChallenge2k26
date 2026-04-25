"""
Feature Factory - Orchestrates all feature engineering

Coordinates Expected Threat, Press Resistance, and Physical Metrics feature
generation with strict temporal integrity enforcement.
"""

import warnings
from typing import Union, Dict, Optional

import numpy as np
import pandas as pd

from src.config import get_config
from src.features.expected_threat import ExpectedThreatCalculator, aggregate_xt_features
from src.features.press_resistance import calculate_press_resistance_features
from src.features.physical_metrics import calculate_physical_features, calculate_fatigue_indicators

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None


class FeatureFactory:
    """
    Stateless feature engineering orchestrator

    Generates all features while maintaining strict temporal boundaries
    to prevent data leakage.
    """

    def __init__(self, config_path: str = "config.yaml", use_gpu: bool = False):
        """
        Initialize feature factory

        Args:
            config_path: Path to configuration file
            use_gpu: Use GPU acceleration (cuDF)
        """
        self.config = get_config(config_path)
        self.use_gpu = use_gpu and CUDF_AVAILABLE

        if self.use_gpu and not CUDF_AVAILABLE:
            warnings.warn("GPU requested but cuDF not available. Using pandas.")
            self.use_gpu = False

        print(f"FeatureFactory initialized (GPU={self.use_gpu})")

    def engineer_features(
        self,
        checkpoint_df: Union[pd.DataFrame, 'cudf.DataFrame'],
        event_dfs: Dict[str, Union[pd.DataFrame, 'cudf.DataFrame']],
        fold_indices: Optional[np.ndarray] = None
    ) -> Union[pd.DataFrame, 'cudf.DataFrame']:
        """
        Engineer all features for the dataset

        This is a STATELESS operation - no information leaks between calls

        Args:
            checkpoint_df: Base checkpoint data
            event_dfs: Dictionary of event DataFrames (pass, run, shot, pressure)
            fold_indices: Optional indices for cross-validation isolation

        Returns:
            Augmented feature matrix
        """
        print("=" * 60)
        print("Starting Feature Engineering Pipeline")
        print("=" * 60)

        # Make copies to avoid modifying originals
        if self.use_gpu:
            df = checkpoint_df.to_pandas().copy()
            events = {k: v.to_pandas() for k, v in event_dfs.items()}
        else:
            df = checkpoint_df.copy()
            events = {k: v.copy() for k, v in event_dfs.items()}

        # If fold indices provided, filter to training data only
        if fold_indices is not None:
            df = df.iloc[fold_indices].copy()
            print(f"Filtered to {len(df)} training observations")

        # 1. Expected Threat Features
        print("\n1. Computing Expected Threat features...")
        df = self._add_expected_threat_features(df, events)

        # 2. Press Resistance Features
        print("\n2. Computing Press Resistance features...")
        df = self._add_press_resistance_features(df, events)

        # 3. Physical Metrics Features
        print("\n3. Computing Physical Metrics features...")
        df = self._add_physical_metrics_features(df, events)

        # 4. Derived Features
        print("\n4. Computing Derived features...")
        df = self._add_derived_features(df)

        # 5. Feature Validation
        print("\n5. Validating features...")
        df = self._validate_features(df, checkpoint_df)

        print("\n" + "=" * 60)
        print(f"Feature engineering complete: {df.shape[1]} total features")
        print("=" * 60)

        # Convert back to cuDF if needed
        if self.use_gpu:
            return cudf.from_pandas(df)

        return df

    def _add_expected_threat_features(
        self,
        df: pd.DataFrame,
        events: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Add Expected Threat features"""
        try:
            # Initialize and fit xT calculator
            xt_calc = ExpectedThreatCalculator(
                zones=self.config.get('features.expected_threat.zones', ['bottom', 'middle', 'top']),
                convergence_threshold=self.config.get('features.expected_threat.convergence_threshold', 0.001),
                max_iterations=self.config.get('features.expected_threat.max_iterations', 10),
                use_gpu=False  # Use pandas for now for compatibility
            )

            # Fit on pass and shot data
            xt_calc.fit(events['pass'], events['shot'])

            # Aggregate features per checkpoint
            df = aggregate_xt_features(
                events['pass'],
                df,
                xt_calc,
                use_gpu=False
            )

            print(f"  ✓ Added {4} Expected Threat features")

        except Exception as e:
            warnings.warn(f"Failed to compute Expected Threat features: {e}")
            # Add zero columns as fallback
            df['last15_xt_added'] = 0.0
            df['cumul_xt_added'] = 0.0
            df['last15_xt_count'] = 0
            df['cumul_xt_count'] = 0

        return df

    def _add_press_resistance_features(
        self,
        df: pd.DataFrame,
        events: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Add Press Resistance features"""
        try:
            positive_outcomes = self.config.get(
                'features.press_resistance.positive_outcomes',
                ['forward_pass', 'backward_pass', 'ball_carry']
            )
            negative_outcomes = self.config.get(
                'features.press_resistance.negative_outcomes',
                ['turnover']
            )

            df = calculate_press_resistance_features(
                events['pressure'],
                events['pass'],
                df,
                positive_outcomes=positive_outcomes,
                negative_outcomes=negative_outcomes,
                use_gpu=False
            )

            print(f"  ✓ Added {10} Press Resistance features")

        except Exception as e:
            warnings.warn(f"Failed to compute Press Resistance features: {e}")
            # Add zero columns as fallback
            for prefix in ['last15', 'cumul']:
                df[f'{prefix}_press_retention'] = 0.0
                df[f'{prefix}_progressive_press'] = 0
                df[f'{prefix}_press_angle_std'] = 0.0
                df[f'{prefix}_press_quality'] = 0.0

        return df

    def _add_physical_metrics_features(
        self,
        df: pd.DataFrame,
        events: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Add Physical Metrics features"""
        try:
            df = calculate_physical_features(
                events['run'],
                df,
                use_gpu=False
            )

            # Add fatigue indicators
            df = calculate_fatigue_indicators(
                df,
                fatigue_threshold=self.config.get('features.physical.fatigue_threshold_low', 0.8),
                momentum_threshold=self.config.get('features.physical.momentum_threshold_high', 1.2)
            )

            print(f"  ✓ Added {9} Physical Metrics features")

        except Exception as e:
            warnings.warn(f"Failed to compute Physical Metrics features: {e}")
            # Add zero columns as fallback
            df['workload_ratio_hsr'] = 1.0
            df['workload_ratio_sprints'] = 1.0
            df['speed_decay'] = 1.0
            df['positional_sprint_deviation'] = 0.0
            df['positional_hsr_deviation'] = 0.0
            df['run_type_diversity'] = 0.0
            df['is_fatigued'] = 0
            df['has_momentum'] = 0
            df['intensity_state'] = 'normal'

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived/interaction features"""
        # Interaction features
        df['xt_per_pass'] = df['cumul_xt_added'] / (df['cumul_xt_count'] + 1)
        # pressure_efficiency: progressive evasions per pressure event.
        # cumul_press_count is the total number of pressure events (a count, not a rate).
        df['pressure_efficiency'] = df['cumul_progressive_press'] / (df['cumul_press_count'] + 1)

        # Physical efficiency
        df['sprint_efficiency'] = df['cumul_sprints'] / (df['cumul_distance'] + 1)
        df['hsr_efficiency'] = df['cumul_hsr'] / (df['cumul_distance'] + 1)

        # Position encoding
        df['is_attacker'] = df['position'].isin(['A', 'F']).astype(int)
        df['is_midfielder'] = (df['position'] == 'M').astype(int)
        df['is_defender'] = (df['position'] == 'D').astype(int)
        df['is_goalkeeper'] = (df['position'] == 'G').astype(int)

        # Temporal features
        df['is_late_game'] = (df['checkpoint_min'] >= 60).astype(int)
        df['is_second_half'] = (df['checkpoint_period'] == 'half_2').astype(int)

        # Shot efficiency
        df['shot_accuracy'] = df['cumul_shots_on_target'] / (df['cumul_shots'] + 1)
        df['shots_under_pressure_ratio'] = df['cumul_shots_under_press'] / (df['cumul_shots'] + 1)

        print(f"  ✓ Added {13} Derived features")

        return df

    def _validate_features(
        self,
        engineered_df: pd.DataFrame,
        original_df: Union[pd.DataFrame, 'cudf.DataFrame']
    ) -> pd.DataFrame:
        """
        Validate feature engineering integrity.

        Fills NaN and replaces Inf values, then returns the cleaned DataFrame.

        Args:
            engineered_df: DataFrame with engineered features
            original_df: Original checkpoint DataFrame

        Returns:
            Cleaned DataFrame with NaN/Inf replaced.
        """
        df = engineered_df.copy()

        # Check for NaN/Inf values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_cols = [col for col in numeric_cols if df[col].isna().any()]
        inf_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]

        if nan_cols:
            warnings.warn(f"Found NaN values in columns: {nan_cols[:5]}")
            for col in nan_cols:
                df[col] = df[col].fillna(0)

        if inf_cols:
            warnings.warn(f"Found Inf values in columns: {inf_cols[:5]}")
            for col in inf_cols:
                df[col] = df[col].replace([np.inf, -np.inf], [1e6, -1e6])

        # Check feature count
        orig_cols = set(original_df.columns if isinstance(original_df, pd.DataFrame) else original_df.to_pandas().columns)
        new_features = set(df.columns) - orig_cols
        print(f"  ✓ Created {len(new_features)} new features")

        # Validate no leakage indicators
        forbidden_keywords = ['outcome', 'result', 'goal', 'scored']
        suspicious_cols = [
            col for col in df.columns
            if any(keyword in col.lower() for keyword in forbidden_keywords)
            and col != 'scored_after'  # Target is allowed
        ]

        if suspicious_cols:
            warnings.warn(f"Suspicious column names detected (potential leakage): {suspicious_cols}")

        return df

    def get_feature_names(self, include_base: bool = False) -> list:
        """
        Get list of engineered feature names

        Args:
            include_base: Include base features from checkpoint data

        Returns:
            List of feature column names
        """
        # Expected Threat
        xt_features = [
            'last15_xt_added', 'cumul_xt_added',
            'last15_xt_count', 'cumul_xt_count'
        ]

        # Press Resistance
        press_features = [
            'last15_press_retention', 'cumul_press_retention',
            'last15_progressive_press', 'cumul_progressive_press',
            'last15_press_angle_std', 'cumul_press_angle_std',
            'last15_press_quality', 'cumul_press_quality',
            'last15_press_count', 'cumul_press_count',
        ]

        # Physical Metrics
        physical_features = [
            'workload_ratio_hsr', 'workload_ratio_sprints',
            'speed_decay', 'positional_sprint_deviation',
            'positional_hsr_deviation', 'run_type_diversity',
            'is_fatigued', 'has_momentum', 'intensity_state'
        ]

        # Derived
        derived_features = [
            'xt_per_pass', 'pressure_efficiency',
            'sprint_efficiency', 'hsr_efficiency',
            'is_attacker', 'is_midfielder', 'is_defender', 'is_goalkeeper',
            'is_late_game', 'is_second_half',
            'shot_accuracy', 'shots_under_pressure_ratio'
        ]

        engineered = xt_features + press_features + physical_features + derived_features

        if include_base:
            base_features = [
                'last15_sprints', 'last15_hsr', 'last15_distance',
                'last15_mean_max_speed', 'last15_peak_speed',
                'last15_shots', 'last15_shots_on_target',
                'cumul_sprints', 'cumul_hsr', 'cumul_distance',
                'cumul_mean_max_speed', 'cumul_peak_speed',
                'cumul_shots', 'cumul_shots_on_target',
                'position', 'is_home', 'checkpoint_min', 'is_second_half'
            ]
            return base_features + engineered

        return engineered
