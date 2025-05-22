# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from enum import Enum
from typing import Optional

from attr import define, field
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
    NONE = "none"
    ID = "identity"
    LOG = "logarithmic"
    Z = "fisher_z"
    LOGIT = "logit"


    def apply(self, x: np.typing.ArrayLike) -> np.typing.ArrayLike:
        """
        Apply transformation to input data.

        Parameters
        ----------
        x : array_like
            Input data. Must satisfy:
            - LOG: x > 0
            - Z: |x| < 1
            - LOGIT: 0 < x < 1

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        x_array = np.asarray(x, dtype=np.float64)
        if x_array.size == 0:
            return x_array
        
        match self:
            case TransformFunc.ID:
                return x_array
            case TransformFunc.LOG:
                if np.any(x_array <= 0):
                    raise ValueError("LOG transformation requires all x > 0")
                return np.log(x)
            case TransformFunc.Z:
                if np.any(np.abs(x_array) >= 1):
                    raise ValueError("Z transformation requires |x| < 1")
                return np.log((1.0 + x_array)/(1.0 - x_array)) * 0.5
            case TransformFunc.LOGIT:
                if np.any((x_array <= 0) | (x_array >= 1)):
                    raise ValueError("LOGIT transformation requires 0 < x < 1")
                return np.log(x_array / (1.0 - x_array))
            case TransformFunc.NONE:
                raise ValueError("Cannot apply a NONE transformer.")


    def apply_inv(self, x: np.typing.ArrayLike) -> np.typing.ArrayLike:
        x_array = np.asarray(x, dtype=np.float64)
        if x_array.size == 0:
            return x_array

        match self:
            case TransformFunc.ID:
                return x_array
            case TransformFunc.LOG:
                return np.exp(x_array)
            case TransformFunc.Z:
                exp_2x = np.exp(2.0 * x_array)
                return (exp_2x - 1.0) / (exp_2x + 1.0)
            case TransformFunc.LOGIT:
                exp_x = np.exp(x_array)
                return exp_x/(exp_x + 1.0)
            case TransformFunc.NONE:
                raise ValueError("Cannot apply a NONE transformer.")


class ConfidentLimit(Enum):
    NONE = -1
    LOWER = 0
    UPPER = 1


@define
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


@define
class TransformedEstimator:
    estimate: float
    variance: Optional[float] = None
    transformed_function: Optional[TransformFunc] = None
    transformed_estimate: Optional[float] = field(init=False)
    transformed_variance: Optional[float] = None
    limit: Optional[float] = None
    allowance: Optional[float] = None
    robust: bool = False
    alpha: Optional[float] = None
    confident_limit: ConfidentLimit = ConfidentLimit.NONE
    n: int = 30

    def __post_init__(self, alpha: float, confident_limit: ConfidentLimit, n: int) -> None:
        if self.variance is not None:
            if n >= 30:
                coeff = norm.ppf(1 - alpha)
            else:
                coeff = t.ppf(1 - alpha, n - 1)

            self.transformed_estimate = self.transformed_function.apply(self.estimate)
            if self.transformed_variance is None:
                self.transformed_variance = self.variance

            transformed_limit = self.transformed_estimate
            if confident_limit == ConfidentLimit.UPPER:
                transformed_limit += coeff * np.sqrt(self.transformed_variance)
            if confident_limit == ConfidentLimit.LOWER:
                transformed_limit -= coeff * np.sqrt(self.transformed_variance)

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
