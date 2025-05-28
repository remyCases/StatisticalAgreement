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
        Compute the concordance correlation coefficient between x and y.

        Parameters
        ----------
        x : array_like of float
            Target values.
        y : array_like of float
            Observation values, should have the same length as x.
        method : str, default: approx
            Method used to compute the index. Expected values are approx or ustat.
        alpha : float, default: 0.05
            Confident level used in confident interval computation.
        criterion : None
            Not used.
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
        Compute the proportion of target values that lie in a given interval around the observation values.

        Parameters
        ----------
        x : array_like of float
            Target values.
        y : array_like of float
            Observation values, should have the same length as x.
        method : str, default: approx
            Method used to compute the index. Expected values are approx or exact.
        alpha : float, default: 0.05
            Confident level used in confident interval computation.
        criterion : float, optional
            Distance from the observation values.
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
        >>> sa.cp(x, y, method="approx", alpha=0.05, criterion=5.0, allowance=0.5)
        Estimator(estimate=0.9746526813225317, limit=3.301923138582779e-18, allowance=0.5)
    
        This result implies that it is expected that 97% of data from x lie in the following interval [y - 5, y + 5].
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
        Compute the boundary for which a given proportion of target values lie around the observation values.

        Parameters
        ----------
        x : array_like of float
            Target values.
        y : array_like of float
            Observation values, should have the same length as x.
        method : str, default: approx
            Method used to compute the index. Expected values are approx.
        alpha : float, default: 0.05
            Confident level used in confident interval computation.
        criterion : float, optional
            Proportion of target values. Has to be between 0.0 and 1.0.
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
        >>> sa.tdi(x, y, alpha=0.05, criterion=0.9, allowance=10.0)
        Estimator(estimate=3.6780045229005722, limit=11.84116682273355, allowance=10.0)

        This result implies that it is expected that 90% of data from x lie in the following interval [y - 3.7, y + 3,7].
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
        Evaluate the aggregated deviation from the identity line Y = X.

        Parameters
        ----------
        x : array_like of float
            Target values.
        y : array_like of float
            Observation values, should have the same length as x.
        method : str, default: approx
            Method used to compute the index. Expected values are approx.
        alpha : float, default: 0.05
            Confident level used in confident interval computation.
        criterion : None
            Not used.
        allowance : None
            Not used.
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
        >>>  sa.msd(x, y, alpha=0.05)
        Estimator(estimate=5.0, limit=51.824424224849665, allowance=None)
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
        Measure the accordance between target and observation values.

        Parameters
        ----------
        x : array_like of int
            Target values.
        y : array_like of int
            Observation values, should have the same length as x.
        method : str, default: approx
            Method used to compute the index. Expected values are cohen, ciccetti, abs, fleiss or squared.
        alpha : float, default: 0.05
            Confident level used in confident interval computation.
        criterion : None
            Not used.
        allowance : None
            Not used.
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
        >>> x = np.array([1, 1, 3, 3, 1])
        >>> y = np.array([1, 1, 3, 2, 2])
        >>> sa.kappa(x, y, method="abs", alpha=0.05)
        Estimator(estimate=0.08695652173913045, limit=-0.00710419, allowance=None)
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

def contingency_table(
        x: np.typing.NDArray[np.int64],
        y: np.typing.NDArray[np.int64],
    ) -> np.typing.NDArray[np.int64]:
    """
        Compute the contingency table between target and observation values.

        Parameters
        ----------
        x : array_like of int
            Target values.
        y : array_like of int
            Observation values, should have the same length as x.

        Returns
        -------
        Array_like of int
            Matrix of size (c+1, c+1) in which c is the number of categorical classes from x and y.

        Examples
        --------
        >>> x = np.array([1, 1, 3, 3, 1])
        >>> y = np.array([1, 1, 3, 2, 2])
        >>> sa.contingency_table(x, y)
        array([[2, 1, 0, 3],
               [0, 0, 0, 0],
               [0, 1, 1, 2],
               [2, 2, 1, 5]], dtype=int64)
    """
    ...
