# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

# Based on Lin, L. I.-K. (2000).
## Total deviation index for measuring individual agreement with applications in
## laboratory performance and bioequivalence. Statistics in Medicine, 19(2), 255–270

import warnings
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from .classutils import Indices, FlagData, TransformedEstimator, Estimator
from . import continuous_agreement
from . import categorical_agreement

DEFAULT_ALPHA = 0.05
CP_DELTA = 0.5
TDI_PI = 0.9
CP_ALLOWANCE = 0.9
TDI_ALLOWANCE = 10
WITHIN_SAMPLE_DEVIATION = 0.15

def _ccc_methods(x, y, method: str, alpha: float, allowance: float) -> TransformedEstimator:

    if method == "approx":
        # Lin LI. A concordance correlation coefficient to evaluate reproducibility.
        # Biometrics. 1989 Mar;45(1):255-68. PMID: 2720055.
        rho = continuous_agreement.precision(x, y, alpha)
        acc = continuous_agreement.accuracy(x, y, rho, alpha)
        return continuous_agreement.ccc_lin(x, y, rho, acc, alpha, allowance)

    if method == "ustat":
        # King TS, Chinchilli VM.
        # Robust estimators of the concordance correlation coefficient.
        # J Biopharm Stat. 2001;11(3):83-105. doi: 10.1081/BIP-100107651. PMID: 11725932.
        return continuous_agreement.ccc_ustat(x, y, alpha, allowance)

    raise ValueError("Wrong method called for ccc computation, \
                     current possible methods are approx or ustat.")

def _cp_methods(x, y, method: str, alpha: float, criterion: float, allowance: float) -> TransformedEstimator:

    if method == "approx":
        _msd = continuous_agreement.msd_exact(x, y, alpha)
        return continuous_agreement.cp_approx(x, y, _msd, alpha, criterion, allowance)

    if method == "exact":
        return continuous_agreement.cp_exact(x, y, alpha, criterion, allowance)

    raise ValueError("Wrong method called for cp computation, \
                     current possible methods are approx or exact.")

def _tdi_methods(x, y, method: str, alpha: float, criterion: float, allowance: float) -> TransformedEstimator:

    if method == "approx":
        _msd = continuous_agreement.msd_exact(x, y, alpha)
        return continuous_agreement.tdi_approx(_msd, criterion, allowance)

    raise ValueError("Wrong method called for tdi computation, \
                     current possible methods are approx.")

def _msd_methods(x, y, method: str, alpha: float) -> TransformedEstimator:

    if method == "approx":
        return continuous_agreement.msd_exact(x, y, alpha)

    raise ValueError("Wrong method called for msd computation, \
                     current possible methods are approx.")

def _kappa_methods(x, y, method: str, alpha: float) -> TransformedEstimator:

    c = max(len(np.unique(x)), len(np.unique(y)))
    if method == "cohen":
        return categorical_agreement.cohen_kappa(x, y, c, alpha)

    if method == "ciccetti" or method == "abs":
        return categorical_agreement.abs_kappa(x, y, c, alpha)

    if method == "fleiss" or method == "squared":
        return categorical_agreement.squared_kappa(x, y, c, alpha)

    raise ValueError("Wrong method called for kappa computation, \
                     current possible methods are cohen, abs or squared.")

class AgreementIndex:
    def __init__(self, name: Indices):
        self._name = name

    def __call__(self, x, y,
                 method="approx",
                 alpha=DEFAULT_ALPHA,
                 criterion=0.0,
                 allowance=0.0,
                 transformed=False) -> Estimator | TransformedEstimator:
        '''
        Compute index estimate and its confident interval

        Parameters
        ----------
        x : array_like of float
            Target values.
        y : array_like of float
            Observation values, should have the same length as x.
        method : str, default: approx
            Method used to compute the index.
        alpha : float, default: 0.05
            Confident level used in confident interval computation.
        criterion : float, optional
            Criterion used in some index computation (CP and TDI).
        allowance : float, optional
            Allowance level to assert agreement.
        transformed : bool, default: False
            If true return the transformedEstimator with all data used to computed estimate and confident limit,
            else return only estimate and confident limit.

        Returns
        -------
        Estimator
            dataclass storing estimate of index, its confident limit and allowance if given.

        Raises
        ------
        ValueError
            If wrong method is given or if less than 4 data are given.

        Examples
        --------
        >>> X = np.array([12, 10, 13, 10])
        >>> Y = np.array([11, 12, 16, 9])
        >>> sa.ccc(X, Y, method='approx', alpha=0.05, allowance=0.10)
        Estimator(estimate=0.5714285714285715, limit=-0.4247655971444191, allowance=0.99)
        '''

        if len(x) <= 3 or len(y) <= 3:
            raise ValueError("Not enough data to compute indices, \
                             need at least four elements on each array_like input.")

        if self._name == Indices.CCC:
            index = _ccc_methods(x, y, method, alpha, allowance)

        elif self._name == Indices.CP:
            index = _cp_methods(x, y, method, alpha, criterion, allowance)

        elif self._name == Indices.TDI:
            index = _tdi_methods(x, y, method, alpha, criterion, allowance)

        elif self._name == Indices.MSD:
            index = _msd_methods(x, y, method, alpha)

        elif self._name == Indices.KAPPA:
            index = _kappa_methods(x, y, method, alpha)

        if transformed:
            return index

        return index.as_estimator()

ccc = AgreementIndex(name=Indices.CCC)
cp = AgreementIndex(name=Indices.CP)
tdi = AgreementIndex(name=Indices.TDI)
msd = AgreementIndex(name=Indices.MSD)
kappa = AgreementIndex(name=Indices.KAPPA)
get_contingency_table = categorical_agreement.contingency

def agreement(x, y, delta_criterion, pi_criterion, alpha=DEFAULT_ALPHA,
              allowance_whitin_sample_deviation=WITHIN_SAMPLE_DEVIATION,
              cp_allowance=CP_ALLOWANCE,
              tdi_allowance=TDI_ALLOWANCE,
              log=False,
              display=False):

    flag = FlagData.OK

    if log:
        if np.sum(x<=0) + np.sum(y<=0) > 0:
            flag = FlagData.NEGATIVE
            raise ValueError("Input data can't be negative for a log transformation")
        else:
            x=np.log(x)
            y=np.log(y)
            delta_criterion=np.log(1.0 + delta_criterion / 100.0)

    res = pd.DataFrame(columns=["estimate", "limit", "variance", "transformed_function",
                                "transformed_estimate", "transformed_variance", "allowance",
                                "robust"])

    if np.var(x)==0 or np.var(y)==0:
        flag = FlagData.CONSTANT
        warnings.warn("Input values are constant, can't compute ccc-related indexes")

    if flag != FlagData.CONSTANT:
        _rho = continuous_agreement.precision(x, y, alpha)
        _acc = continuous_agreement.accuracy(x, y, _rho, alpha)
        _ccc_lin = continuous_agreement.ccc_lin(x, y, _rho, _acc, alpha, allowance_whitin_sample_deviation)
        _ccc_ustat = continuous_agreement.ccc_ustat(x, y, alpha, allowance_whitin_sample_deviation)

    _msd = continuous_agreement.msd_exact(x, y, alpha)
    _rbs = continuous_agreement.rbs(x, y, cp_allowance)
    _cp_approx = continuous_agreement.cp_approx(x, y, _msd, alpha, delta_criterion, cp_allowance)
    _cp = continuous_agreement.cp_exact(x, y, alpha, delta_criterion, cp_allowance)
    _tdi = continuous_agreement.tdi_approx(_msd, pi_criterion, tdi_allowance)

    res.loc["acc", :] = _acc.to_series()
    res.loc["rho", :] = _rho.to_series()
    res.loc["ccc", :] = _ccc_lin.to_series()
    res.loc["ccc_robust", :] = _ccc_ustat.to_series()
    res.loc["msd", :] = _msd.to_series()
    res.loc["rbs", :] = _rbs.to_series()

    if flag != FlagData.NEGATIVE:
        res.loc["cp_approx", :] = _cp_approx.to_series()
        res.loc["cp", :] = _cp.to_series()
        res.loc["tdi", :] = _tdi.to_series()
        res.loc["cp_approx", "criterion"] = delta_criterion
        res.loc["cp", "criterion"] = delta_criterion
        res.loc["tdi", "criterion"] = pi_criterion

    if display:
        print(res)
        print(shapiro(x - y))

    return flag, res
