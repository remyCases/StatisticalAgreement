# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import pytest

from statisticalagreement.core._continuous_agreement import msd_exact
from statisticalagreement.core._types import NDArrayFloat
from statisticalagreement.core.mathutils import assert_float


@pytest.mark.parametrize("x_name", [
    ("basic_array"),
    ("random_array_float64")
])
def test_msd_perfect_agreement(
    x_name: str, 
    request: pytest.FixtureRequest
) -> None:
    x: NDArrayFloat = request.getfixturevalue(x_name)
    msd = msd_exact(x, x, alpha=0.05)

    assert_float(msd.estimate, 0.0, max_ulps=4)
    assert msd.variance is not None
    assert_float(msd.variance, 0.0, max_ulps=4)
    assert_float(msd.limit, 0.0, max_ulps=4)
