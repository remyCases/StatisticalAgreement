# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import numpy as np

from statisticalagreement.core._types import NDArrayFloat, NDArrayInt
from statisticalagreement.core.classutils import TransformFunc, ConfidentLimit, TransformedEstimator

def contingency(
        x: NDArrayInt, 
        y: NDArrayInt, 
        c: int
    ) -> NDArrayInt:

    matrix_contingency = np.zeros((c+1, c+1), dtype=np.int64)
    for _x, _y in zip(x, y):
        matrix_contingency[_x][_y] += 1
        matrix_contingency[_x][c] += 1
        matrix_contingency[c][_y] += 1
        matrix_contingency[c][c] += 1

    return matrix_contingency


def _generic_kappa(
        x: NDArrayInt,
        y: NDArrayInt,
        w: NDArrayFloat,
        c: int,
        alpha: float
    ) -> TransformedEstimator:

    if x.shape != y.shape:
        raise ValueError("x and y should have the same number of dimensions.")
    if x.ndim != 1 and w.ndim != 2 and w.shape != (x.shape, x.shape):
        raise ValueError("Incorrect dimensions of w.")

    mat = contingency(x, y, c)
    n: int = mat[c][c]

    p0 = np.einsum("ii,ii", w, mat[:c, :c])
    pc = np.einsum("ij,i,j", w, mat[:c, c], mat[c, :c])

    if p0 == n:
        k_hat = 1.0
    else:
        k_hat = (p0 - pc / float(n)) / (n - pc / float(n))

    m_wi = np.einsum("i,ji->j", mat[c, :c], w) / float(n)
    m_wj = np.einsum("i,ji->j", mat[:c, c], w) / float(n)
    coeff = (w - (m_wi[:, np.newaxis] + m_wj)*(1-k_hat))**2
    factor = np.einsum("ij,ij", mat[:c, :c], coeff) / float(n)

    if pc == n*n:
        var_k_hat = 0.0
    else:
        var_k_hat = (factor - (k_hat - pc / float(n*n) *(1-k_hat))**2) / (n * (1 - pc / float(n*n))**2)

    kappa = TransformedEstimator(
        estimate=k_hat,
        variance=var_k_hat,
        transformed_function=TransformFunc.ID,
        alpha=alpha,
        confident_limit=ConfidentLimit.LOWER,
        n=n
    )
    return kappa


def cohen_kappa(
        x: NDArrayInt,
        y: NDArrayInt,
        c: int,
        alpha: float
    ) -> TransformedEstimator:

    weights = np.diag(np.ones(c))

    return _generic_kappa(x, y, weights, c, alpha)


def abs_kappa(
        x: NDArrayInt,
        y: NDArrayInt,
        c: int,
        alpha: float
    ) -> TransformedEstimator:

    weights = 1.0 - np.abs(np.arange(c)[:, np.newaxis] - np.arange(c)) / float(c)
    
    return _generic_kappa(x, y, weights, c, alpha)

def squared_kappa(
        x: NDArrayInt,
        y: NDArrayInt,
        c: int,
        alpha: float
    ) -> TransformedEstimator:

    weights = 1.0 - (np.arange(c)[:, np.newaxis] - np.arange(c))**2 / float(c*c)
    
    return _generic_kappa(x, y, weights, c, alpha)
