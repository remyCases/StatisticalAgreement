# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from typing import Optional
import warnings

import numpy as np

from statisticalagreement.core import _categorical_agreement, _continuous_agreement
from statisticalagreement.core.classutils import Indices, TransformedEstimator
from statisticalagreement.core._types import NDArrayFloat, NDArrayInt


def _ccc_methods(
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


def _cp_methods(
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


def _tdi_methods(
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


def _msd_methods(
        x: NDArrayFloat,
        y: NDArrayFloat,
        method: str,
        alpha: float
    ) -> TransformedEstimator:

    if method == "approx":
        return _continuous_agreement.msd_exact(x, y, alpha)

    raise ValueError("Wrong method called for msd computation, \
                     current possible methods are approx.")


def _kappa_methods(
        x: NDArrayInt,
        y: NDArrayInt,
        method: str,
        alpha: float
    ) -> TransformedEstimator:

    if method == "cohen":
        return _categorical_agreement.cohen_kappa(x, y, alpha)

    if method in {"ciccetti", "abs"}:
        return _categorical_agreement.abs_kappa(x, y, alpha)

    if method in {"fleiss", "squared"}:
        return _categorical_agreement.squared_kappa(x, y, alpha)

    raise ValueError("Wrong method called for kappa computation, \
                     current possible methods are cohen, abs or squared.")


def continuous_methods(
        index_name: Indices,
        x: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        axis: Optional[int],
        method: str,
        alpha: float,
        criterion: float,
        allowance: float,
    ) -> TransformedEstimator:

    x = np.asarray(x)
    y = np.asarray(y)

    if axis is None:
        x = np.reshape(x, (-1,))
        y = np.reshape(y, (-1,))
        axis = -1

    axis_int = int(axis)
    if axis_int != axis:
        raise ValueError('`axis` must be an integer.')
    axis = axis_int

    n = x.shape[axis]
    if n != y.shape[axis]:
        raise ValueError('`x` and `y` must have the same length along `axis`.')

    if n < 4:
        raise ValueError("Not enough data to compute indices, \
                         need at least four elements on each array_like input.")

    try:
        x, y = np.broadcast_arrays(x, y)
    except (ValueError, RuntimeError) as e:
        message = "`x` and `y` must be broadcastable."
        raise ValueError(message) from e
    
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    axis = -1

    dtype = np.result_type(x.dtype, y.dtype)
    if np.isdtype(dtype, "integral"):
        dtype = np.asarray(1.).dtype

    x = np.astype(x, dtype, copy=False)
    y = np.astype(y, dtype, copy=False)

    if index_name == Indices.CCC:
        return _ccc_methods(x, y, method, alpha, allowance)

    elif index_name == Indices.CP:
        return _cp_methods(x, y, method, alpha, criterion, allowance)

    elif index_name == Indices.TDI:
        return _tdi_methods(x, y, method, alpha, criterion, allowance)

    elif index_name == Indices.MSD:
        return _msd_methods(x, y, method, alpha)
    
    raise ValueError("Unknown index")


def categorical_methods(
        index_name: Indices,
        x: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        axis: Optional[int],
        method: str,
        alpha: float,
        _criterion: float,
        _allowance: float
    ) -> TransformedEstimator:

    x = np.asarray(x)
    y = np.asarray(y)

    if axis is None:
        x = np.reshape(x, (-1,))
        y = np.reshape(y, (-1,))
        axis = -1

    axis_int = int(axis)
    if axis_int != axis:
        raise ValueError('`axis` must be an integer.')
    axis = axis_int

    n = x.shape[axis]
    if n != y.shape[axis]:
        raise ValueError('`x` and `y` must have the same length along `axis`.')

    if n < 4:
        raise ValueError("Not enough data to compute indices, \
                         need at least four elements on each array_like input.")

    try:
        x, y = np.broadcast_arrays(x, y)
    except (ValueError, RuntimeError) as e:
        message = "`x` and `y` must be broadcastable."
        raise ValueError(message) from e
    
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    axis = -1

    dtype = np.result_type(x.dtype, y.dtype)
    if not np.isdtype(dtype, "integral"):
        raise TypeError("Input must be convertible to int")
    dtype = np.asarray(1.).dtype

    x = np.astype(x, dtype, copy=False)
    y = np.astype(y, dtype, copy=False)

    if method == "approx":
        method = "cohen"

    if index_name == Indices.KAPPA:
        return _kappa_methods(x, y, method, alpha)
    
    raise ValueError("Unknown index")
