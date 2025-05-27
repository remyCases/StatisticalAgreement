# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.


import numpy as np
import pytest

from statisticalagreement.core.mathutils import assert_float, vectorize_cov


@pytest.mark.parametrize(("M", "N", "ddof"), [
    (3, 50, 1),
    (100, 1000, 1),
    (100, 1000, 5),
    (1, 1, 0),
    (100, 1, 0),
])
def test_vectorized_cov(M: int, N: int, ddof: int) -> None:
    np.random.seed(0)

    x = np.random.normal(loc=1.0, scale=100.0, size=(M, N))
    y = np.random.normal(loc=0.0, scale=200.0, size=(M, N))
    cov = np.zeros((2, 2, M), dtype=np.float64)
    var_x = np.zeros((M), dtype=np.float64)
    var_y = np.zeros((M), dtype=np.float64)

    for i in range(M):
        cov[:,:,i] = np.cov(x[i,:], y[i,:], ddof=ddof)
        var_x[i] = np.var(x[i,:], ddof=ddof)
        var_y[i] = np.var(y[i,:], ddof=ddof)

    vec_cov = vectorize_cov(x, y, ddof=ddof)
    vec_cov_arr = vec_cov.reshape(4, -1)
    assert_float(vec_cov_arr[0, :], var_x, max_ulps=4)
    assert_float(vec_cov_arr[3, :], var_y, max_ulps=4)

    assert_float(vec_cov, cov, max_ulps=4)
