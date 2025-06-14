# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from typing import Tuple
import numpy as np
import pytest

from statisticalagreement.core._categorical_agreement import squared_kappa
from statisticalagreement.core._types import NDArrayInt
from statisticalagreement.core.mathutils import assert_float
from statisticalagreement.simulation.monte_carlo import MonteCarlo
from tests.continuous.conftest import N_SIMULATIONS


@pytest.mark.parametrize("x_name", [
    ("basic_array"),
    ("basic_array_not_starting_at_zeros"),
    ("random_array_int64"),
    ("zeros_array"),
    ("ones_array"),
])
def test_kappa_perfect_agreement(
    x_name: str, 
    request: pytest.FixtureRequest
) -> None:
    x: NDArrayInt = request.getfixturevalue(x_name)
    kappa = squared_kappa(x, x, alpha=0.05)

    assert_float(kappa.estimate, 1.0, max_ulps=4)
    assert_float(kappa.variance, 0.0, max_ulps=4)
    assert_float(kappa.limit, 1.0, max_ulps=4)


@pytest.mark.stochastic
def test_monte_carlo_kappa_variance(
    monte_carlo_arrays: Tuple[NDArrayInt, NDArrayInt]
) -> None:

    np.random.seed(0)
    mc = MonteCarlo()
    vars = np.empty(2*N_SIMULATIONS)

    x = monte_carlo_arrays[0]
    eps = monte_carlo_arrays[1]
    y1 = x + eps
    y2 = x - eps

    for i in range(N_SIMULATIONS):

        kappa1 = squared_kappa(x[i,:], y1[i,:], alpha=0.05)
        mc.append(kappa1.transformed_estimate)
        vars[2*i] = kappa1.transformed_variance

        kappa2 = squared_kappa(x[i,:], y2[i,:], alpha=0.05)
        mc.append(kappa2.transformed_estimate)
        vars[2*i+1] = kappa2.transformed_variance

    res = mc.compute()
    var_empiric, se_empiric = res.var, res.standard_error
    mean_var = sum(vars) / (2*N_SIMULATIONS)
    ratio = (mean_var - var_empiric) / var_empiric

    assert mean_var < var_empiric + 1.96*se_empiric
    assert mean_var > var_empiric - 1.96*se_empiric
    assert np.abs(ratio) < .02
    assert_float(mean_var, var_empiric, max_ulps=4)
