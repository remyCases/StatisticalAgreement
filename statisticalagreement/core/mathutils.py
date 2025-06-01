# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from typing import Optional, Self, TypeVar
from attrs import define
import numpy as np

from statisticalagreement.core._types import NDArrayFloat, NDArrayUInt


@define
class AssertFloatResult:
    _flag: bool
    diff_ulp: Optional[NDArrayUInt] = None

    @classmethod
    def _as(cls, flag: bool) -> Self:
        return cls(flag)
    

    def __bool__(self) -> bool:
        return self._flag


T = TypeVar('T', float, NDArrayFloat)
def almost_equal_float(first: T, second: T, max_ulps: int=4) -> AssertFloatResult:
    """
    Asserts if two floats are almost equal.

    Parameters
    ----------
    first : float
    second : float
    max_ulps : int, default: 4
        Maximum difference of ulp to assert equality.

    Returns
    -------
    bool

    Examples
    --------
    >>> almost_equal_float(1.0, 1.01)
    AssertFloatResult(_flag=False)
    """
    # using difference of ulp as a more reliable way to test equality in float/double
    # same method as used in GoogleTest C++ library
    # see https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/

    first_arr = np.asarray(first, dtype=np.float64)
    second_arr = np.asarray(second, dtype=np.float64)

    # handling shape differences
    if first_arr.ndim != first_arr.ndim:
        return AssertFloatResult(False)
    if second_arr.shape != second_arr.shape:
        return AssertFloatResult(False)
    
    # handling nan cases
    nan_mask = np.isnan(first_arr) | np.isnan(second_arr)
    if np.any(nan_mask):
        return AssertFloatResult(False)
    
    valid_mask = np.full_like(first_arr, True, dtype=bool)

    # handling infinity cases
    inf_mask = np.isinf(first_arr) | np.isinf(second_arr)
    finite_mask = ~inf_mask
    zero_mask = (first_arr == 0.0) & (second_arr == 0.0)

    if np.any(inf_mask):
        # Both must be infinite with same sign
        sign_match = np.sign(first_arr) == np.sign(second_arr)
        valid_mask[inf_mask] = sign_match[inf_mask]
        if not np.all(valid_mask):
            return AssertFloatResult(False)

    if np.any(finite_mask):
        # Sign comparison for finite elements
        sign_diff = (np.signbit(first_arr) != np.signbit(second_arr))[finite_mask]
        if np.any(sign_diff):
            # Allow 0.0/-0.0 combinations
            valid_mask[finite_mask] = ~sign_diff | zero_mask[finite_mask]

    # Early exit if any invalid found
    if not np.all(valid_mask):
        return AssertFloatResult(False)

    first_int: NDArrayUInt = first_arr.view(np.uint64)
    second_int: NDArrayUInt = second_arr.view(np.uint64)
    all_int = np.stack([first_int, second_int])

    ulps_diff = np.max(all_int, axis=0) - np.min(all_int, axis=0)
    valid_mask[finite_mask] = (ulps_diff <= max_ulps)[finite_mask] | zero_mask[finite_mask]

    if np.all(valid_mask):
        return AssertFloatResult(True)

    return AssertFloatResult(False, ulps_diff)


def assert_float(first: T, second: T, max_ulps: int=4) -> None:

    if not (cmp := almost_equal_float(first, second, max_ulps)):
        raise AssertionError(
            f"ULP failing: {first} vs {second} (diff={cmp.diff_ulp} ULP > {max_ulps})"
        )


def vectorize_cov(x: np.typing.ArrayLike, y: np.typing.ArrayLike, bias: bool=False, ddof: Optional[int]=None) -> NDArrayFloat:

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # Reshape 1D arrays to 2D (row vectors)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)

    if x_arr.ndim > 2:
        raise ValueError("x has more than 2 dimensions")
    if y_arr.ndim > 2:
        raise ValueError("y has more than 2 dimensions")

    if ddof is None:
        ddof = 0 if bias else 1

    N = x_arr.shape[-1]
    fact = N - ddof
    concat = np.concatenate((x_arr[:, np.newaxis, :], y_arr[:, np.newaxis, :]), axis=1)
    centered = concat - np.mean(concat, axis=-1, keepdims=True)  # Shape: (M, 2, N)

    cov = np.matmul(
        centered,
        centered.transpose(0, 2, 1)
    )  # Shape: (M, 2, 2)
    
    return cov.transpose(1, 2, 0) / fact  # Shape: (2, 2, M)
