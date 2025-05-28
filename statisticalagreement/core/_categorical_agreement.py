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

    matrix = np.zeros((c, c), dtype=np.int64)
    for i in range(len(x)):
        matrix[x[i], y[i]] += 1
        
    full_matrix = np.zeros((c+1, c+1), dtype=np.int64)
    full_matrix[:c, :c] = matrix
    full_matrix[:c, c] = matrix.sum(axis=1)
    full_matrix[c, :c] = matrix.sum(axis=0)
    full_matrix[c, c] = matrix.sum()
    return full_matrix


def _generic_kappa(
        x: NDArrayInt,
        y: NDArrayInt,
        w: NDArrayFloat,
        c: int,
        alpha: float
    ) -> TransformedEstimator:

    if x.shape != y.shape:
        raise ValueError("x and y should have the same number of dimensions.")
    if w.shape != (c, c):
        raise ValueError(f"w must have a shape ({c},{c}), given: {w.shape}")

    mat = contingency(x, y, c)
    n: int = mat[c][c]

    p0 = np.einsum("ii,ii", w, mat[:c, :c], optimize=True)
    pc = np.einsum("ij,i,j", w, mat[:c, c], mat[c, :c], optimize=True)

    if p0 == n:
        k_hat = 1.0
    else:
        k_hat = float(n*p0 - pc) / (n*n - pc)

    p_ij = mat[:c, :c] / n
    w_mean = w * p_ij
    w_mean_i = w_mean.sum(axis=1)
    w_mean_j = w_mean.sum(axis=0)

    coeff = (w - (w_mean_i[:, np.newaxis] + w_mean_j)*(1-k_hat))**2
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
        alpha: float
    ) -> TransformedEstimator:

    classes = np.unique(np.concatenate([x, y]))
    c = len(classes)
    label_map = {label: idx for idx, label in enumerate(classes)}
    x_mapped = np.vectorize(label_map.get)(x)
    y_mapped = np.vectorize(label_map.get)(y)
    weights = np.diag(np.ones(c))

    return _generic_kappa(x_mapped, y_mapped, weights, c, alpha)


def abs_kappa(
        x: NDArrayInt,
        y: NDArrayInt,
        alpha: float
    ) -> TransformedEstimator:

    classes = np.unique(np.concatenate([x, y]))
    c = len(classes)
    label_map = {label: idx for idx, label in enumerate(classes)}
    x_mapped = np.vectorize(label_map.get)(x)
    y_mapped = np.vectorize(label_map.get)(y)
    weights = 1.0 - np.abs(np.arange(c)[:, np.newaxis] - np.arange(c)) / float(c)
    
    return _generic_kappa(x_mapped, y_mapped, weights, c, alpha)

def squared_kappa(
        x: NDArrayInt,
        y: NDArrayInt,
        alpha: float
    ) -> TransformedEstimator:

    classes = np.unique(np.concatenate([x, y]))
    c = len(classes)
    label_map = {label: idx for idx, label in enumerate(classes)}
    x_mapped = np.vectorize(label_map.get)(x)
    y_mapped = np.vectorize(label_map.get)(y)
    weights = 1.0 - (np.arange(c)[:, np.newaxis] - np.arange(c))**2 / float(c*c)
    
    return _generic_kappa(x_mapped, y_mapped, weights, c, alpha)
