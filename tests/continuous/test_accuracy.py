# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from typing import Tuple
import numpy as np
import pytest

from statisticalagreement.core._continuous_agreement import accuracy, precision
from statisticalagreement.core._types import NDArrayFloat
from statisticalagreement.core.classutils import TransformedEstimator
from statisticalagreement.core.mathutils import assert_float
from statisticalagreement.simulation.monte_carlo import MonteCarlo
from tests.continuous.conftest import DENORMALIZED_FLOAT, N_SIMULATIONS


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
    y: NDArrayFloat = x + np.random.normal(0, DENORMALIZED_FLOAT)
    rho = precision(x, y, alpha=0.05)
    acc = accuracy(x, y, rho, alpha=0.05)

    assert_float(acc.estimate, 1.0, max_ulps=4)
    assert_float(acc.variance, 0.0, max_ulps=4)
    assert_float(acc.limit, 1.0, max_ulps=4)


def test_accuracy_gaussian_data(
    gaussian_arrays: Tuple[NDArrayFloat, NDArrayFloat]
) -> None:
    x = gaussian_arrays[0]
    y = gaussian_arrays[1]

    var_x, _, _, var_y = np.cov(x, y, bias=True, dtype=np.float64).flatten()
    acc_theoric = 2 * np.sqrt(var_x*var_y) / (var_x + var_y + (np.mean(x) - np.mean(y))**2)

    rho = precision(x, y, alpha=0.05)
    acc = accuracy(x, y, rho, alpha=0.05)

    assert_float(acc.estimate, acc_theoric, max_ulps=4)


@pytest.mark.stochastic
def test_monte_carlo_accuracy_variance(
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

        rho1 = precision(x[i,:], y1[i,:], alpha=0.05)
        acc1 = accuracy(x[i,:], y1[i,:], rho1, 0.05)
        mc.append(acc1.transformed_estimate)
        vars[2*i] = acc1.transformed_variance

        rho2 = precision(x[i,:], y2[i,:], alpha=0.05)
        acc2 = accuracy(x[i,:], y2[i,:], rho2, 0.05)
        mc.append(acc2.transformed_estimate)
        vars[2*i+1] = acc2.transformed_variance

    res = mc.compute()
    var_empiric, se_empiric = res.var, res.standard_error
    mean_var = sum(vars) / (2*N_SIMULATIONS)
    ratio = (mean_var - var_empiric) / var_empiric

    assert mean_var < var_empiric + 1.96*se_empiric
    assert mean_var > var_empiric - 1.96*se_empiric
    assert np.abs(ratio) < .02
    assert_float(mean_var, var_empiric, max_ulps=4)
