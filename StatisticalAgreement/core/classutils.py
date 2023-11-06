# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from dataclasses import dataclass, field, InitVar

class Indices(Enum):
    ccc = 0
    cp = 1
    tdi = 2
    msd = 3
    kappa = 4

class FlagData(Enum):
    Data_Ok = 0
    Constant = 1
    Negative = 2

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

@dataclass
class Estimator:
    estimate: float
    limit: float
    allowance: float

    def to_series(self):
        return pd.Series({
            "estimate": self.estimate,
            "limit": self.limit,
            "allowance": self.allowance,
            })
    
@dataclass
class TransformedEstimator:
    estimate: float
    variance: float | None = None
    transformed_function: TransformFunc | None = None
    transformed_estimate: float = field(init=False)
    transformed_variance: float | None = None
    limit: float | None = None
    allowance: float | None = None
    robust: bool = False
    alpha: InitVar[float | None] = None
    confident_limit: InitVar[ConfidentLimit | None] = None
    n: InitVar[int] = 30

    def __post_init__(self, alpha, confident_limit, n) -> None:
        if self.variance is not None:
            if n >= 30:
                coeff = norm.ppf(1 - alpha)
            else:
                coeff = t.ppf(1 - alpha, n - 1)

            self.transformed_estimate = self.transformed_function.apply(self.estimate)
            if self.transformed_variance is None:
                self.transformed_variance = self.variance

            if confident_limit == ConfidentLimit.Upper:
                transformed_limit = self.transformed_estimate + coeff * np.sqrt(self.transformed_variance)
            if confident_limit == ConfidentLimit.Lower:
                transformed_limit = self.transformed_estimate - coeff * np.sqrt(self.transformed_variance)

            self.limit = self.transformed_function.apply_inv(transformed_limit)
        else:
            self.transformed_estimate = None

    def to_series(self):
        return pd.Series({
            "estimate": self.estimate,
            "limit": self.limit,
            "variance": self.variance,
            "transformed_function": self.transformed_function,
            "transformed_estimate": self.transformed_estimate,
            "transformed_variance": self.transformed_variance,
            "allowance": self.allowance,
            "robust": self.robust,
            })
    
    def as_estimator(self) -> Estimator:
        return Estimator(estimate=self.estimate, limit=self.limit, allowance=self.allowance)