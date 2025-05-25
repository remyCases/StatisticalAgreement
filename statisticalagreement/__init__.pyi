# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from typing import Dict, Tuple, Union
import numpy as np
from statisticalagreement.core.classutils import Estimator, TransformedEstimator, FlagData
from statisticalagreement.simulation.mc_simulation import mc_simulation as mc_simulation


def ccc(
        x: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        /,
        *,
        method: str = ...,
        alpha: float = ...,
        criterion: float = ...,
        allowance: float = ...,
        transformed: bool = ...
    ) -> Union[Estimator, TransformedEstimator]: 
    """
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
        >>> sa.ccc(x, y, method="approx", alpha=0.05, allowance=0.10)
        Estimator(estimate=0.5714285714285715, limit=-0.4247655971444191, allowance=0.99)
        """
    ...

def cp(
        x: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        /,
        *,
        method: str = ...,
        alpha: float = ...,
        criterion: float = ...,
        allowance: float = ...,
        transformed: bool = ...
    ) -> Union[Estimator, TransformedEstimator]:
    """
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
        >>> sa.ccc(x, y, method="approx", alpha=0.05, allowance=0.10)
        Estimator(estimate=0.5714285714285715, limit=-0.4247655971444191, allowance=0.99)
        """
    ...
def tdi(
        x: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        /,
        *,
        method: str = ...,
        alpha: float = ...,
        criterion: float = ...,
        allowance: float = ...,
        transformed: bool = ...
    ) -> Union[Estimator, TransformedEstimator]:
    """
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
        >>> sa.ccc(x, y, method="approx", alpha=0.05, allowance=0.10)
        Estimator(estimate=0.5714285714285715, limit=-0.4247655971444191, allowance=0.99)
        """
    ...
def msd(
        x: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        /,
        *,
        method: str = ...,
        alpha: float = ...,
        criterion: float = ...,
        allowance: float = ...,
        transformed: bool = ...
    ) -> Union[Estimator, TransformedEstimator]:
    """
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
        >>> sa.ccc(x, y, method="approx", alpha=0.05, allowance=0.10)
        Estimator(estimate=0.5714285714285715, limit=-0.4247655971444191, allowance=0.99)
        """
    ...
def kappa(
        x: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        /,
        *,
        method: str = ...,
        alpha: float = ...,
        criterion: float = ...,
        allowance: float = ...,
        transformed: bool = ...
    ) -> Union[Estimator, TransformedEstimator]:
    """
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
        >>> sa.ccc(x, y, method="approx", alpha=0.05, allowance=0.10)
        Estimator(estimate=0.5714285714285715, limit=-0.4247655971444191, allowance=0.99)
        """
    ...

def agreement(
        x: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        /,
        *,
        delta_criterion: float,
        pi_criterion: float,
        alpha: float = ...,
        allowance_whitin_sample_deviation: float = ...,
        cp_allowance: float = ...,
        tdi_allowance: float = ...,
        log: bool = ...,
        display: bool = ...
    ) -> Tuple[FlagData, Dict[str, Dict[str, Union[float, str, bool]]]]:
    ...

def get_contingency_table(
        x: np.typing.NDArray[np.int64],
        y: np.typing.NDArray[np.int64],
        c: int
    ) -> np.typing.NDArray[np.int64]:
    ...
