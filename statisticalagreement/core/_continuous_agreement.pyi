# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from statisticalagreement.core._types import NDArrayFloat
from statisticalagreement.core.classutils import TransformedEstimator


def precision(
        x: NDArrayFloat,
        y: NDArrayFloat,
        alpha: float
    ) -> TransformedEstimator:
    ...


def accuracy(
        x: NDArrayFloat,
        y: NDArrayFloat,
        t_precision: TransformedEstimator,
        alpha: float
    ) -> TransformedEstimator:
    ...


def ccc_lin(
        x: NDArrayFloat,
        y: NDArrayFloat,
        t_precision: TransformedEstimator,
        t_accuracy: TransformedEstimator,
        alpha: float,
        allowance_whitin_sample_deviation: float
    ) -> TransformedEstimator:
    ...


def ccc_ustat(
        x: NDArrayFloat,
        y: NDArrayFloat,
        alpha: float,
        allowance_whitin_sample_deviation: float
    ) -> TransformedEstimator:
    ...


def msd_exact(
        x: NDArrayFloat,
        y: NDArrayFloat,
        alpha: float
    ) -> TransformedEstimator:
    ...


def _rbs_allowance(cp_allowance: float) -> float:
    ...


def rbs(
        x: NDArrayFloat, 
        y: NDArrayFloat, 
        cp_allowance: float
    ) -> TransformedEstimator:
    ...


def cp_approx(
        x: NDArrayFloat,
        y: NDArrayFloat,
        msd: TransformedEstimator, 
        alpha: float, 
        delta_criterion: float, 
        cp_allowance: float
    ) -> TransformedEstimator:
    ...


def cp_exact(
        x: NDArrayFloat,
        y: NDArrayFloat, 
        alpha: float,
        delta_criterion: float,
        cp_allowance: float
    ) -> TransformedEstimator:
    ...


def tdi_approx(
        msd: TransformedEstimator, 
        pi_criterion: float, 
        tdi_allowance: float
    ) -> TransformedEstimator:
    ...
