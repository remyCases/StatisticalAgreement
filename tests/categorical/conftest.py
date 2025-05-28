# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from typing import Tuple
import numpy as np
import pytest

from statisticalagreement.core._types import NDArrayInt


DENORMALIZED_FLOAT = 1e-100
N_SAMPLES = 1000
N_SIMULATIONS = 5000

@pytest.fixture
def basic_array() -> NDArrayInt:
    return np.array([1, 1, 2, 2, 1], dtype=np.int64)


@pytest.fixture
def random_array_int64() -> NDArrayInt:
    return np.array([5, 8, 6, 5, 9, 5, 4, 6, 6, 5, 3, 2, 1, 3, 4, 6, 2, 5, 5, 3, 2, 2,
       2, 9, 1, 8, 1, 5, 1, 7, 1, 9, 1, 4, 7, 6, 4, 4, 5, 5, 9, 4, 8, 4,
       6, 4, 3, 7, 1, 6, 3, 4, 9, 9, 2, 5, 1, 5, 8, 4, 6, 3, 5, 1, 5, 8,
       4, 9, 7, 5, 5, 6, 7, 3, 6, 7, 7, 9, 5, 2, 9, 5, 2, 4, 2, 6, 3, 7,
       4, 2, 8, 5, 9, 6, 5, 5, 2, 4, 5, 7], dtype=np.int64)


@pytest.fixture
def ones_array() -> NDArrayInt:
    return np.ones(100, dtype=np.int64)


@pytest.fixture
def random_arrays() -> Tuple[NDArrayInt, NDArrayInt]:
    np.random.seed(0)
    x = np.random.randint(1, 10, 1000)
    y = x + np.random.randint(0, 2, 1000)
    return x, y


@pytest.fixture
def monte_carlo_arrays() -> Tuple[NDArrayInt, NDArrayInt]:
    np.random.seed(0)
    x = np.random.randint(1, 10, size=(N_SIMULATIONS, N_SAMPLES))
    eps = np.random.randint(0, 2, size=(N_SIMULATIONS, N_SAMPLES))
    return x, eps
