# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

# Based on Lin, L. I.-K. (2000).
## Total deviation index for measuring individual agreement with applications in
## laboratory performance and bioequivalence. Statistics in Medicine, 19(2), 255–270

from typing import Protocol, Union, runtime_checkable
import warnings
import numpy as np
import pandas as pd
from scipy.stats import shapiro

from StatisticalAgreement.core import _continuous_agreement
from StatisticalAgreement.core.classutils import ConfidentLimit, Estimator, FlagData, Indices, TransformFunc, TransformedEstimator
from StatisticalAgreement.core._methods import ccc_methods, cp_methods, kappa_methods, msd_methods, tdi_methods


DEFAULT_ALPHA = 0.05
CP_DELTA = 0.5
TDI_PI = 0.9
CP_ALLOWANCE = 0.9
TDI_ALLOWANCE = 10
WITHIN_SAMPLE_DEVIATION = 0.15


@runtime_checkable
class AgreementFunctor(Protocol):
    def __call__(
            self, 
            x: np.typing.ArrayLike,
            y: np.typing.ArrayLike,
            method: str="approx",
            alpha: float=DEFAULT_ALPHA,
            criterion: float=0.0,
            allowance: float=0.0,
            transformed: bool=False
        ) -> Union[Estimator, TransformedEstimator]:
        ...


class AgreementIndex:
    def __init__(self, name: Indices):
        self._name = name

    def __call__(
            self, 
            x: np.typing.ArrayLike,
            y: np.typing.ArrayLike,
            method: str="approx",
            alpha: float=DEFAULT_ALPHA,
            criterion: float=0.0,
            allowance: float=0.0,
            transformed: bool=False
        ) -> Union[Estimator, TransformedEstimator]:
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
        >>> x = np.array([12, 10, 13, 10])
        >>> y = np.array([11, 12, 16, 9])
        >>> sa.ccc(x, y, method='approx', alpha=0.05, allowance=0.10)
        Estimator(estimate=0.5714285714285715, limit=-0.4247655971444191, allowance=0.99)
        '''

        try:
            x_float = np.asarray(x, dtype=np.float64)
            y_float = np.asarray(y, dtype=np.float64)
        except ValueError as e:
            raise TypeError("Input must be convertible to float") from e

        if x_float.ndim == 0 or y_float.ndim == 0:
            raise ValueError("Input must be at least 1-dimensional")
    

        if len(x_float) <= 3 or len(y_float) <= 3:
            raise ValueError("Not enough data to compute indices, \
                             need at least four elements on each array_like input.")

        if self._name == Indices.CCC:
            index = ccc_methods(x_float, y_float, method, alpha, allowance)

        elif self._name == Indices.CP:
            index = cp_methods(x_float, y_float, method, alpha, criterion, allowance)

        elif self._name == Indices.TDI:
            index = tdi_methods(x_float, y_float, method, alpha, criterion, allowance)

        elif self._name == Indices.MSD:
            index = msd_methods(x_float, y_float, method, alpha)

        elif self._name == Indices.KAPPA:
            index = kappa_methods(x_float, y_float, method, alpha)
        else:
            raise ValueError("Unknown index")

        if transformed:
            return index

        return index.as_estimator()


def agreement(
        x: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        delta_criterion: float,
        pi_criterion: float, 
        alpha: float=DEFAULT_ALPHA,
        allowance_whitin_sample_deviation: float=WITHIN_SAMPLE_DEVIATION,
        cp_allowance: float=CP_ALLOWANCE,
        tdi_allowance: float=TDI_ALLOWANCE,
        log: bool=False,
        display: bool=False
    ):

    try:
        x_float = np.asarray(x, dtype=np.float64)
        y_float = np.asarray(y, dtype=np.float64)
    except ValueError as e:
        raise TypeError("Input must be convertible to float") from e

    if x_float.ndim == 0 or y_float.ndim == 0:
        raise ValueError("Input must be at least 1-dimensional")

    flag = FlagData.OK

    if log:
        if np.any(x_float<=0) or np.any(y_float<=0):
            flag = FlagData.NEGATIVE
            raise ValueError("Input data can't be negative for a log transformation")

        x=np.log(x_float)
        y=np.log(y_float)
        delta_criterion=np.log(1.0 + delta_criterion / 100.0)

    res = pd.DataFrame(columns=["estimate", "limit", "variance", "transformed_function",
                                "transformed_estimate", "transformed_variance", "allowance",
                                "robust"])

    if np.var(x_float)==0 or np.var(y_float)==0:
        flag = FlagData.CONSTANT
        warnings.warn("Input values are constant, can't compute ccc-related indexes")

    if flag != FlagData.CONSTANT:
        _rho = _continuous_agreement.precision(x_float, y_float, alpha)
        _acc = _continuous_agreement.accuracy(x_float, y_float, _rho, alpha)
        _ccc_lin = _continuous_agreement.ccc_lin(x_float, y_float, _rho, _acc, alpha,
                                                allowance_whitin_sample_deviation)
        _ccc_ustat = _continuous_agreement.ccc_ustat(x_float, y_float, alpha, allowance_whitin_sample_deviation)
    else:
        _rho = _acc = _ccc_lin = _ccc_ustat = TransformedEstimator(
            estimate=np.nan,
            variance=np.nan,
            transformed_variance=np.nan,
            transformed_function=TransformFunc.NONE,
            allowance=1-allowance_whitin_sample_deviation**2,
            alpha=alpha,
            confident_limit=ConfidentLimit.NONE,
            n=0
        )

    _msd = _continuous_agreement.msd_exact(x_float, y_float, alpha)
    _rbs = _continuous_agreement.rbs(x_float, y_float, cp_allowance)
    _cp_approx = _continuous_agreement.cp_approx(x_float, y_float, _msd, alpha, delta_criterion, cp_allowance)
    _cp = _continuous_agreement.cp_exact(x_float, y_float, alpha, delta_criterion, cp_allowance)
    _tdi = _continuous_agreement.tdi_approx(_msd, pi_criterion, tdi_allowance)

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
        print(shapiro(x_float - y_float))

    return flag, res
