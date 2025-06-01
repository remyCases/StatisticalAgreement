# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from statisticalagreement.core._types import NDArrayInt
from statisticalagreement.core.classutils import TransformedEstimator


def contingency(
    x: NDArrayInt, 
    y: NDArrayInt, 
    c: int
) -> NDArrayInt:
    ...


def cohen_kappa(
        x: NDArrayInt,
        y: NDArrayInt,
        alpha: float
    ) -> TransformedEstimator:
    ...


def abs_kappa(
        x: NDArrayInt,
        y: NDArrayInt,
        alpha: float
    ) -> TransformedEstimator:
    ...


def squared_kappa(
        x: NDArrayInt,
        y: NDArrayInt,
        alpha: float
    ) -> TransformedEstimator:
    ...
