# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import numpy as np
from scipy.stats import norm, sem

from statisticalagreement.core.mathutils import assert_float
from statisticalagreement.simulation.monte_carlo import MonteCarlo

N_REPETITION = 5000


def assert_almost_mc_ci(array: np.typing.NDArray[np.float64]) -> None:
    mc = MonteCarlo().compute(array)
    assert_float(mc.mean, np.mean(array, dtype=np.float64))
    assert_float(mc.var, np.var(array, dtype=np.float64))
    assert_float(mc.standard_error, float(sem(array, ddof=0)))


def test_constant_monte_carlo() -> None:
    array = np.repeat(1.0, N_REPETITION).astype(np.float64)
    assert_almost_mc_ci(array)


def test_normal_monte_carlo() -> None:
    array = norm.rvs(loc=0.0, scale=1.0, size=N_REPETITION)
    array = np.asarray(array, dtype=np.float64)
    assert_almost_mc_ci(array)
