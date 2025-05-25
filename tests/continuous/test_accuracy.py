# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import numpy as np
import pytest

from statisticalagreement.core._continuous_agreement import accuracy, precision
from statisticalagreement.core._types import NDArrayFloat
from statisticalagreement.core.classutils import TransformedEstimator
from statisticalagreement.core.mathutils import assert_float


@pytest.mark.parametrize("x_name", [
    ("basic_array"),
    ("random_array_float64"),
    ("zeros_array"),
    ("ones_array"),
])
def test_accuracy_perfect_agreement(
    x_name: str, 
    request: pytest.FixtureRequest
) -> None:
    x: NDArrayFloat = request.getfixturevalue(x_name)
    rho = TransformedEstimator(estimate=np.float64(1.0))
    acc = accuracy(x, x, rho, alpha=0.05)

    assert_float(acc.estimate, 1.0, max_ulps=4)
    assert acc.variance is not None
    assert_float(acc.variance, 0.0, max_ulps=4)
    assert_float(acc.limit, 1.0, max_ulps=4)


@pytest.mark.parametrize("x_name", [
    ("basic_array"),
    ("random_array_float64"),
    ("ones_array"),
])
def test_accuracy_added_denormalized_number(
    x_name: str, 
    request: pytest.FixtureRequest
) -> None:
    x: NDArrayFloat = request.getfixturevalue(x_name)
    y: NDArrayFloat = x + np.random.normal(0, 1e-9)
    rho = precision(x, y, alpha=0.05)
    acc = accuracy(x, y, rho, alpha=0.05)

    assert_float(acc.estimate, 1.0, max_ulps=4)
    assert acc.variance is not None
    assert_float(acc.variance, 0.0, max_ulps=4)
    assert_float(acc.limit, 1.0, max_ulps=4)
