# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import pytest

from statisticalagreement.core._continuous_agreement import cp_exact
from statisticalagreement.core._types import NDArrayFloat
from statisticalagreement.core.mathutils import assert_float


@pytest.mark.parametrize("x_name", [
    ("basic_array"),
    ("random_array_float64")
])
def test_cp_exact_perfect_agreement(
    x_name: str, 
    request: pytest.FixtureRequest
) -> None:
    x: NDArrayFloat = request.getfixturevalue(x_name)
    cp = cp_exact(x, x, alpha=0.05, delta_criterion=1.0, cp_allowance=0.0)

    assert_float(cp.estimate, 1.0, max_ulps=4)
    assert cp.variance is not None
    assert_float(cp.variance, 0.0, max_ulps=4)
    assert_float(cp.limit, 1.0, max_ulps=4)
