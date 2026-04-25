"""
Type aliases for cleaner type annotations across the codebase.
"""

from typing import TypeAlias, Union

import pandas as pd

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None

# DataFrame that could be pandas or cuDF
DataFrame: TypeAlias = Union[pd.DataFrame, "cudf.DataFrame"]

# Series that could be pandas or cuDF
Series: TypeAlias = Union[pd.Series, "cudf.Series"]
