# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import warnings
import numpy as np

from StatisticalAgreement.core import _categorical_agreement, _continuous_agreement
from StatisticalAgreement.core.classutils import TransformedEstimator
from StatisticalAgreement.core._types import NDArrayFloat


def ccc_methods(
        x: NDArrayFloat,
        y: NDArrayFloat,
        method: str,
        alpha: float,
        allowance: float
    ) -> TransformedEstimator:

    if method == "approx":
        # Lin LI. A concordance correlation coefficient to evaluate reproducibility.
        # Biometrics. 1989 Mar;45(1):255-68. PMID: 2720055.
        rho = _continuous_agreement.precision(x, y, alpha)
        acc = _continuous_agreement.accuracy(x, y, rho, alpha)
        return _continuous_agreement.ccc_lin(x, y, rho, acc, alpha, allowance)

    if method == "ustat":
        # King TS, Chinchilli VM.
        # Robust estimators of the concordance correlation coefficient.
        # J Biopharm Stat. 2001;11(3):83-105. doi: 10.1081/BIP-100107651. PMID: 11725932.
        warnings.warn("The current implementation of the variance of the estimator is invalid.")
        return _continuous_agreement.ccc_ustat(x, y, alpha, allowance)

    raise ValueError("Wrong method called for ccc computation, \
                     current possible methods are approx or ustat.")


def cp_methods(
        x: NDArrayFloat,
        y: NDArrayFloat,
        method: str,
        alpha: float,
        criterion: float,
        allowance: float
    ) -> TransformedEstimator:

    if method == "approx":
        _msd = _continuous_agreement.msd_exact(x, y, alpha)
        return _continuous_agreement.cp_approx(x, y, _msd, alpha, criterion, allowance)

    if method == "exact":
        return _continuous_agreement.cp_exact(x, y, alpha, criterion, allowance)

    raise ValueError("Wrong method called for cp computation, \
                     current possible methods are approx or exact.")


def tdi_methods(
        x: NDArrayFloat,
        y: NDArrayFloat,
        method: str,
        alpha: float,
        criterion: float,
        allowance: float
    ) -> TransformedEstimator:

    if method == "approx":
        _msd = _continuous_agreement.msd_exact(x, y, alpha)
        return _continuous_agreement.tdi_approx(_msd, criterion, allowance)

    raise ValueError("Wrong method called for tdi computation, \
                     current possible methods are approx.")


def msd_methods(
        x: NDArrayFloat,
        y: NDArrayFloat,
        method: str,
        alpha: float
    ) -> TransformedEstimator:

    if method == "approx":
        return _continuous_agreement.msd_exact(x, y, alpha)

    raise ValueError("Wrong method called for msd computation, \
                     current possible methods are approx.")


def kappa_methods(
        x: NDArrayFloat,
        y: NDArrayFloat,
        method: str,
        alpha: float
    ) -> TransformedEstimator:

    c = max(len(np.unique(x)), len(np.unique(y)))
    if method == "cohen":
        return _categorical_agreement.cohen_kappa(x, y, c, alpha)

    if method in {"ciccetti", "abs"}:
        return _categorical_agreement.abs_kappa(x, y, c, alpha)

    if method in {"fleiss", "squared"}:
        return _categorical_agreement.squared_kappa(x, y, c, alpha)

    raise ValueError("Wrong method called for kappa computation, \
                     current possible methods are cohen, abs or squared.")

