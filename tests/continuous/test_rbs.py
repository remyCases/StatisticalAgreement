# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from typing import Tuple
import numpy as np
import pytest

from statisticalagreement.core._continuous_agreement import rbs
from statisticalagreement.core._types import NDArrayFloat
from statisticalagreement.core.mathutils import assert_float
from statisticalagreement.simulation.monte_carlo import MonteCarlo
from tests.continuous.conftest import DENORMALIZED_FLOAT, N_SIMULATIONS


@pytest.mark.parametrize("x_name", [
    ("basic_array"),
    ("random_array_float64"),
    ("zeros_array"),
    ("ones_array"),
])
def test_rbs_perfect_agreement(
    x_name: str, 
    request: pytest.FixtureRequest
) -> None:
    x: NDArrayFloat = request.getfixturevalue(x_name)
    _rbs = rbs(x, x, cp_allowance=0.0)

    assert np.isnan(_rbs.estimate)


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
    y: NDArrayFloat = x + np.random.normal(0, DENORMALIZED_FLOAT)
    _rbs = rbs(x, y, cp_allowance=0.0)

    assert np.isnan(_rbs.estimate)


@pytest.mark.stochastic
def test_monte_carlo_rbs_variance(
    monte_carlo_arrays: Tuple[NDArrayFloat, NDArrayFloat]
) -> None:

    np.random.seed(0)
    mc = MonteCarlo()
    vars = np.empty(2*N_SIMULATIONS)

    x = monte_carlo_arrays[0]
    eps = monte_carlo_arrays[1]
    y1 = x + eps
    y2 = x - eps

    for i in range(N_SIMULATIONS):

        rbs1 = rbs(x[i,:], y1[i,:], cp_allowance=0.0)
        mc.append(rbs1.transformed_estimate)
        vars[2*i] = rbs1.transformed_variance

        rbs2 = rbs(x[i,:], y2[i,:], cp_allowance=0.0)
        mc.append(rbs2.transformed_estimate)
        vars[2*i+1] = rbs2.transformed_variance

    res = mc.compute()
    var_empiric, se_empiric = res.var, res.standard_error
    mean_var = sum(vars) / (2*N_SIMULATIONS)
    ratio = (mean_var - var_empiric) / var_empiric

    assert mean_var < var_empiric + 1.96*se_empiric
    assert mean_var > var_empiric - 1.96*se_empiric
    assert np.abs(ratio) < .02
    assert_float(mean_var, var_empiric, max_ulps=4)
