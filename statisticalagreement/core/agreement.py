# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

# Based on Lin, L. I.-K. (2000).
## Total deviation index for measuring individual agreement with applications in
## laboratory performance and bioequivalence. Statistics in Medicine, 19(2), 255–270

from typing import Dict, Optional, Protocol, Tuple, Union, runtime_checkable
import warnings
import numpy as np
from scipy.stats import shapiro

from statisticalagreement.core import _continuous_agreement
from statisticalagreement.core.classutils import ConfidentLimit, Estimator, FlagData, Indices, TransformFunc, TransformedEstimator
from statisticalagreement.core._methods import categorical_methods, continuous_methods


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
            /,
            *,
            axis: Optional[int]=None,
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
            axis: Optional[int]=None,
            method: str="approx",
            alpha: float=DEFAULT_ALPHA,
            criterion: float=0.0,
            allowance: float=0.0,
            transformed: bool=False
        ) -> Union[Estimator, TransformedEstimator]:
    
        if self._name == Indices.KAPPA:
            index = categorical_methods(self._name, x, y, axis, method, alpha, criterion, allowance)
        else:
            index = continuous_methods(self._name, x, y, axis, method, alpha, criterion, allowance)
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
    ) -> Tuple[FlagData, Dict[str, Dict[str, Union[float, str, bool]]]]:

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
    
    res: Dict[str, Dict[str, Union[float, str, bool]]] = {}

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

    res["acc"] = _acc.to_dict()
    res["rho"] = _rho.to_dict()
    res["ccc"] = _ccc_lin.to_dict()
    res["ccc_robust"] = _ccc_ustat.to_dict()
    res["msd"] = _msd.to_dict()
    res["rbs"] = _rbs.to_dict()

    if flag != FlagData.NEGATIVE:
        res["cp_approx"] = _cp_approx.to_dict()
        res["cp"] = _cp.to_dict()
        res["tdi"] = _tdi.to_dict()
        res["cp_approx"]["criterion"] = delta_criterion
        res["cp"]["criterion"] = delta_criterion
        res["tdi"]["criterion"] = pi_criterion

    if display:
        print(shapiro(x_float - y_float))

    return flag, res
