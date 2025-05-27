# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from typing import Optional, Self
from attrs import define, field
import numpy as np

from statisticalagreement.core._types import NDArrayFloat


@define
class McCi:
    mean: float
    var: float
    standard_error: float


@define
class MonteCarlo:
    _n: int = 0
    _sum: float = 0.0
    _sum_sq: float = 0.0
    _mean: float = field(init=False)
    _var: float = field(init=False)
    _standard_error: float = field(init=False)


    def append(self, x: float) -> Self:
        self._n += 1
        self._sum += x
        self._sum_sq += x*x

        return self


    def compute(self, data: Optional[NDArrayFloat]=None) -> McCi:
        if data is not None:
            self._n = len(data)
            self._sum = np.sum(data)
            self._sum_sq = np.sum(data*data)

        if self._n == 0:
            raise ValueError("No data available to compute statistics.")

        self._mean = self._sum / self._n

        if self._n < 2:
            raise ValueError("At least two samples are required to compute variance and standard error.")

        self._var = (self._sum_sq - self._sum * self._sum / self._n) / (self._n - 1)
        self._standard_error = np.sqrt(self._var / self._n)

        return McCi(
            mean=self._mean,
            var=self._var,
            standard_error=self._standard_error
        )
