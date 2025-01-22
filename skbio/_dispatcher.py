import numpy as np
import pandas as pd

from ._config import get_option


def create_table(data, columns=None, index=None, backend=None):
    """Create a table object using the specified backend."""
    if backend is None:
        backend = get_option("table_backend")

    if backend == "pandas":
        return pd.DataFrame(data, index=index, columns=columns)
    elif backend == "numpy":
        return np.array(data)
    else:
        raise ValueError(f"Unsupported backend '{backend}'")


def create_table_1d(data, index=None, backend=None):
    """Create a 1d array using the specified backend."""
    if backend is None:
        backend = get_option("table_backend")

    if backend == "pandas":
        return pd.Series(data, index=index)
    elif backend == "numpy":
        return np.array(data)
    else:
        raise ValueError(f"Unsupported backend '{backend}'")
