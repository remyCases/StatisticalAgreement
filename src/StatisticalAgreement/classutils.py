# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of adventOfCode project from https://github.com/remyCases/StatiscalAgreement.

from enum import Enum
import numpy as np
from scipy.stats import norm, t

class TransformFunc(Enum):
    Id = 0
    Log = 1
    Z = 2
    Logit = 3

    def apply(self, x: float) -> float:
        if self == TransformFunc.Id:
            return x
        if self == TransformFunc.Log:
            return np.log(x)
        if self == TransformFunc.Z:
            return np.log((1+x)/(1-x)) * 0.5
        if self == TransformFunc.Logit:
            return np.log(x/(1-x))
        
    def apply_inv(self, x: float) -> float:
        if self == TransformFunc.Id:
            return x
        if self == TransformFunc.Log:
            return np.exp(x)
        if self == TransformFunc.Z:
            return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        if self == TransformFunc.Logit:
            return np.exp(x)/(np.exp(x) + 1)
        
class ConfidentLimit(Enum):
    Lower = 0
    Upper = 1

class TransformEstimator:
    _esp: float
    _var: float
    _t: TransformFunc

    def __init__(self, point_estimator: float, variance_estimator: float, transform_func: TransformFunc) -> None:
        self._esp = point_estimator
        self._var = variance_estimator
        self._t = transform_func

    def ci(self, alpha: float, limit: ConfidentLimit, n = 30) -> float:
        if n >= 30:
            coeff = norm.ppf(1 - alpha)
        else:
            coeff = t.ppf(1 - alpha, n - 1)

        if limit == ConfidentLimit.Upper:
            transformed_limit = self._t.apply(self._esp) + coeff * np.sqrt(self._var)
        if limit == ConfidentLimit.Lower:
            transformed_limit = self._t.apply(self._esp) - coeff * np.sqrt(self._var)

        return self._t.apply_inv(transformed_limit)
