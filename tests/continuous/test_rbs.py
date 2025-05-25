# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import numpy as np
import pytest

from statisticalagreement.core._continuous_agreement import rbs
from statisticalagreement.core._types import NDArrayFloat
from statisticalagreement.core.mathutils import assert_float


@pytest.mark.parametrize("x_name", [
    ("basic_array"),
    ("random_array_float64")
])
def test_rbs_perfect_agreement(
    x_name: str, 
    request: pytest.FixtureRequest
) -> None:
    x: NDArrayFloat = request.getfixturevalue(x_name)
    _rbs = rbs(x, x, cp_allowance=0.0)

    assert_float(_rbs.estimate, np.nan, max_ulps=4)
