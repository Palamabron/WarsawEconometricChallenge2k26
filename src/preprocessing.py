from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CategoricalEncoder:
    """Fit category mappings on training data and reuse them on later splits."""

    def __init__(self) -> None:
        self.encoders: dict[str, LabelEncoder] = {}
        self.mappings: dict[str, dict[str, int]] = {}
        self.fitted: bool = False

    def fit(
        self, df: pd.DataFrame, categorical_cols: list[str] | None = None
    ) -> "CategoricalEncoder":
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in categorical_cols:
            if col in df.columns:
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                self.encoders[col] = encoder
                self.mappings[col] = {
                    category: code for code, category in enumerate(encoder.classes_)
                }

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("CategoricalEncoder must be fitted before transform")

        df = df.copy()

        for col in self.encoders:
            if col in df.columns:
                # Handle unseen categories by mapping to -1 without per-row encoder calls.
                df[col] = df[col].astype(str).map(self.mappings[col]).fillna(-1).astype(int)

        return df

    def fit_transform(
        self, df: pd.DataFrame, categorical_cols: list[str] | None = None
    ) -> pd.DataFrame:
        self.fit(df, categorical_cols)
        return self.transform(df)

    def get_feature_names(self) -> list[str]:
        return list(self.encoders.keys())


def safe_fillna(df: pd.DataFrame, value: Any = 0) -> pd.DataFrame:
    return df.fillna(value)


def safe_clip_infinite(df: pd.DataFrame, lower: float = -1e6, upper: float = 1e6) -> pd.DataFrame:
    import numpy as np

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.copy()

    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], [upper, lower])

    return df
