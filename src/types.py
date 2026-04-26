"""
Type aliases for cleaner type annotations across the codebase.
"""

import pandas as pd

try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    cudf = None

# DataFrame that could be pandas or cuDF
type DataFrame = pd.DataFrame | cudf.DataFrame

# Series that could be pandas or cuDF
type Series = pd.Series | cudf.Series
