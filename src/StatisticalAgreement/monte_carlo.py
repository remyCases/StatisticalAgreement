# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatiscalAgreement.

import numpy as np
from typing import NamedTuple

class McCi(NamedTuple):
    mean: float
    var: float
    lower: float
    upper: float

class MonteCarlo():
    def __init__(self):
        self._n = 0
        self._sum = 0
        self._sum_sq = 0

    def append(self, x:float):
        self._n += 1
        self._sum += x
        self._sum_sq += x*x

    def compute(self, data=None):
        if data is not None:
            self._n = len(data)
            self._sum = np.sum(data)
            self._sum_sq = np.sum(data*data)

        self._mean = self._sum / self._n
        self._var = self._sum_sq / self._n - self._mean * self._mean
        return McCi(
            mean=self._mean,
            var=self._var,
            lower=self._mean - 1.96*np.sqrt(self._var/self._n),
            upper=self._mean + 1.96*np.sqrt(self._var/self._n)
            )