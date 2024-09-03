# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from enum import Enum
from dataclasses import dataclass, field, InitVar
from scipy.stats import norm, t
import numpy as np
import pandas as pd

class Indices(Enum):
    CCC = 0
    CP = 1
    TDI = 2
    MSD = 3
    KAPPA = 4

class FlagData(Enum):
    OK = 0
    CONSTANT = 1
    NEGATIVE = 2

class TransformFunc(Enum):
    ID = 0
    LOG = 1
    Z = 2
    LOGIT = 3

    def apply(self, x: float) -> float:
        match self:
            case TransformFunc.ID:
                return x
            case TransformFunc.LOG:
                return np.log(x)
            case TransformFunc.Z:
                return np.log((1.0 + x)/(1.0 - x)) * 0.5
            case TransformFunc.LOGIT:
                return np.log(x / (1.0 - x))

    def apply_inv(self, x: float) -> float:
        match self:
            case TransformFunc.ID:
                return x
            case TransformFunc.LOG:
                return np.exp(x)
            case TransformFunc.Z:
                return (np.exp(2.0 * x) - 1.0) / (np.exp(2.0 * x) + 1.0)
            case TransformFunc.LOGIT:
                return np.exp(x)/(np.exp(x) + 1.0)

class ConfidentLimit(Enum):
    LOWER = 0
    UPPER = 1

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

            if confident_limit == ConfidentLimit.UPPER:
                transformed_limit = self.transformed_estimate + coeff * np.sqrt(self.transformed_variance)
            if confident_limit == ConfidentLimit.LOWER:
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
