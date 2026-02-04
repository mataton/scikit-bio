"""Defines Handling of different Missing classes for Metadata module."""
# ----------------------------------------------------------------------------
# Copyright (c) 2016-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
import numpy as np

from ._enan import make_nan_with_payload as _make_nan_with_payload
from ._enan import get_payload_from_nan as _get_payload_from_nan


def _encode_terms(namespace):
    enum = _MISSING_ENUMS[namespace]
    namespace = _NAMESPACE_LOOKUP.index(namespace)

    def encode(x):
        if not isinstance(x, str):
            return x
        try:
            code = enum.index(x)
        except ValueError:
            return x
        return _make_nan_with_payload(code, namespace=namespace)

    return encode


def _handle_insdc_missing(series):
    # # Check if there are any INSDC sentinel strings to encode
    # sentinel_values = set(_MISSING_ENUMS["INSDC:missing"])
    # has_sentinels = series.apply(lambda x: isinstance(x, str) and x in
    # sentinel_values).any()

    # if not has_sentinels:
    #     # No sentinels to encode, return as-is
    #     return series

    # # Need to encode sentinels - must use object dtype to preserve
    # NaN payloads
    # series = series.astype(object).copy()
    # encode = _encode_terms("INSDC:missing")
    # for idx in series.index:
    #     series[idx] = encode(series[idx])
    # return series

    # print('\n')
    # series = series.astype(object)
    # print(f"series type: {series.dtype}")
    # print(series)
    # res = series.apply(_encode_terms("INSDC:missing"))
    # print(f"res type: {res.dtype}")
    # print(res)
    # # vvv this is original line
    # # return series.apply(_encode_terms("INSDC:missing"))
    # return res

    # series = series.astype(object).copy()
    # encode = _encode_terms("INSDC:missing")
    # print(f"series before encode")
    # print(series)
    # for idx in series.index:
    #     series.at[idx] = encode(series.at[idx])
    # print(f"series after code")
    # print(series)

    # return series
    # from ._enan import get_payload_from_nan, _float_to_int
    # import numpy as np

    # encode = _encode_terms("INSDC:missing")

    # # Method 1: using .apply()
    # res1 = series.apply(encode)

    # # Method 2: using list comprehension
    # encoded_values = [encode(x) for x in series]
    # res2 = pd.Series(encoded_values, index=series.index, name=series.name,
    # dtype=object)

    # # Check each value
    # for idx in series.index:
    #     val1 = res1.at[idx]
    #     val2 = res2.at[idx]

    #     print(f"\n--- {idx} ---")
    #     if isinstance(val1, float) and np.isnan(val1):
    #         print(f"apply():  int={_float_to_int(val1)},
    # payload={get_payload_from_nan(val1)}")
    #     else:
    #         print(f"apply():  {val1!r}")

    #     if isinstance(val2, float) and np.isnan(val2):
    #         print(f"list:     int={_float_to_int(val2)},
    # payload={get_payload_from_nan(val2)}")
    #     else:
    #         print(f"list:     {val2!r}")

    # return res2
    from ._enan import _float_to_int

    encode = _encode_terms("INSDC:missing")
    encoded_values = [encode(x) for x in series]
    result = pd.Series(
        encoded_values, index=series.index, name=series.name, dtype=object
    )

    # Confirm payload is intact
    print(f"\n=== _handle_insdc_missing END ===")
    for idx in result.index:
        val = result.at[idx]
        if isinstance(val, float) and np.isnan(val):
            print(f"encoded result[{idx}]: int={_float_to_int(val)}")

    return result


def _handle_blank(series):
    return series


def _handle_no_missing(series):
    if series.isna().any():
        raise ValueError(
            "Missing values are not allowed in series/column"
            " (name=%r) when using scheme 'no-missing'." % series.name
        )
    return series


BUILTIN_MISSING = {
    "INSDC:missing": _handle_insdc_missing,
    "blank": _handle_blank,
    "no-missing": _handle_no_missing,
}
_MISSING_ENUMS = {
    "INSDC:missing": (
        "not applicable",
        "missing",
        "not collected",
        "not provided",
        "restricted access",
    )
}

# list index reflects the nan namespace, the "blank"/"no-missing" enums don't
# apply here, since they aren't actually encoded in the NaNs
_NAMESPACE_LOOKUP = ["INSDC:missing"]
DEFAULT_MISSING = "blank"


def series_encode_missing(series: pd.Series, enumeration: str) -> pd.Series:
    """Return encoded Missing values."""
    if not isinstance(enumeration, str):
        TypeError("Wrong type for `enumeration`, expected string")
    try:
        encoder = BUILTIN_MISSING[enumeration]
    except KeyError:
        raise ValueError(
            "Unknown enumeration: %r, (available: %r)"
            % (enumeration, list(BUILTIN_MISSING.keys()))
        )

    new = encoder(series)
    return new


def series_extract_missing(series: pd.Series) -> pd.Series:
    """Return extracted Missing types from passed Series."""
    from ._enan import _float_to_int

    def _decode(x):
        if np.issubdtype(type(x), np.floating) and np.isnan(x):
            code, namespace = _get_payload_from_nan(x)
            if namespace is None:
                return x
            elif namespace == 255:
                raise ValueError("Custom enumerations are not yet supported")
            else:
                try:
                    enum = _MISSING_ENUMS[_NAMESPACE_LOOKUP[namespace]]
                except (IndexError, KeyError):
                    return x

            try:
                return enum[code]
            except IndexError:
                return x

        return x

    print(f"\n=== series_extract_missing DEBUG ===")
    print(f"input series dtype: {series.dtype}")

    # Check payload in original series
    for idx in series.index:
        val = series.at[idx]
        if isinstance(val, float) and np.isnan(val):
            print(f"original series[{idx}]: int={_float_to_int(val)}")

    na_mask = series.isna()
    missing_indices = series.index[na_mask]

    # Check payload after accessing via .at[]
    for idx in missing_indices:
        val = series.at[idx]
        print(f"series.at[{idx}]: int={_float_to_int(val)}")

    decoded_values = [_decode(series.at[idx]) for idx in missing_indices]
    print(f"decoded_values: {decoded_values}")

    return pd.Series(
        decoded_values, index=missing_indices, name=series.name, dtype=object
    )
