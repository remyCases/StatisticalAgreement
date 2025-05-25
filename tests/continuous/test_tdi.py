# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import numpy as np
import pytest

from statisticalagreement.core._continuous_agreement import msd_exact, tdi_approx
from statisticalagreement.core._types import NDArrayFloat
from statisticalagreement.core.classutils import TransformedEstimator
from statisticalagreement.core.mathutils import assert_float


@pytest.mark.parametrize("x_name", [
    ("basic_array"),
    ("random_array_float64"),
    ("zeros_array"),
    ("ones_array"),
])
def test_tdi_approx_perfect_agreement(
    x_name: str, 
    request: pytest.FixtureRequest
) -> None:
    _x: NDArrayFloat = request.getfixturevalue(x_name)
    msd = TransformedEstimator(estimate=np.float64(0.0), limit=np.float64(0.0))
    tdi = tdi_approx(msd, pi_criterion=0.95, tdi_allowance=0.0)

    assert_float(tdi.estimate, 0.0, max_ulps=4)
    assert_float(tdi.limit, 0.0, max_ulps=4)


@pytest.mark.parametrize("x_name", [
    ("basic_array"),
    ("random_array_float64"),
    ("ones_array"),
])
def test_rbs_added_denormalized_number(
    x_name: str, 
    request: pytest.FixtureRequest
) -> None:
    x: NDArrayFloat = request.getfixturevalue(x_name)
    y: NDArrayFloat = x + np.random.normal(0, 1e-9)
    msd = msd_exact(x, y, alpha=0.05)
    tdi = tdi_approx(msd, pi_criterion=0.95, tdi_allowance=0.0)

    assert_float(tdi.estimate, 0.0, max_ulps=4)
    assert_float(tdi.limit, 0.0, max_ulps=4)
