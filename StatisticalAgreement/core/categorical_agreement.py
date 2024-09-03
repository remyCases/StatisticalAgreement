# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from itertools import product
import numpy as np
from .classutils import TransformFunc, ConfidentLimit, TransformedEstimator

def contingency(x, y, c: int):
    matrix_contingency = np.zeros((c+1, c+1))
    for _x, _y in zip(x, y):
        matrix_contingency[_x][_y] += 1
        matrix_contingency[_x][c] += 1
        matrix_contingency[c][_y] += 1
        matrix_contingency[c][c] += 1

    return matrix_contingency

def cohen_kappa(x, y, c: int, alpha: float) -> TransformedEstimator:
    mat = contingency(x, y, c)
    p0 = 0
    pc = 0
    factor = 0
    n = mat[c][c]

    m_wi = np.zeros(c)
    m_wj = np.zeros(c)

    for i in range(c):
        p0 += mat[i][i]/n
        pc += mat[i][c]/n * mat[c][i]/n

        m_wi[i] = mat[c][i]/n
        m_wj[i] = mat[i][c]/n

    k_hat = (p0 - pc) / (1 - pc)

    for i, j in product(range(c), range(c)):
        if i != j:
            w = 0
        else:
            w = 1

        factor += mat[i][j]/n*(w - (m_wi[i] + m_wj[j])*(1-k_hat))**2

    var_k_hat = (factor - (k_hat - pc*(1-k_hat))**2) / (n * (1-pc)**2)

    kappa = TransformedEstimator(
        estimate=k_hat,
        variance=var_k_hat,
        transformed_variance=var_k_hat,
        transformed_function=TransformFunc.ID,
        alpha=alpha,
        confident_limit=ConfidentLimit.LOWER,
        n=n
    )
    return kappa

def abs_kappa(x, y, c: int, alpha: float) -> TransformedEstimator:
    mat = contingency(x, y, c)
    p0 = 0
    pc = 0
    factor = 0
    n = mat[c][c]

    m_wi = np.zeros(c)
    m_wj = np.zeros(c)
    w = 0

    for i, j in product(range(c), range(c)):
        w = 1 - np.abs(i - j) / c
        p0 += w * mat[i][j]/n
        pc += w * mat[i][c]/n * mat[c][j]/n

        m_wi[i] += mat[c][j] * w/n
        m_wj[j] += mat[i][c] * w/n

    k_hat = (p0 - pc) / (1 - pc)

    for i, j in product(range(c), range(c)):
        w = 1 - np.abs(i - j) / c
        factor += mat[i][j]/n*(w - (m_wi[i] + m_wj[j])*(1-k_hat))**2

    var_k_hat = (factor - (k_hat - pc*(1-k_hat))**2) / (n * (1-pc)**2)

    kappa = TransformedEstimator(
        estimate=k_hat,
        variance=var_k_hat,
        transformed_variance=var_k_hat,
        transformed_function=TransformFunc.ID,
        alpha=alpha,
        confident_limit=ConfidentLimit.LOWER,
        n=n
    )
    return kappa

def squared_kappa(x, y, c: int, alpha: float) -> TransformedEstimator:
    mat = contingency(x, y, c)
    p0 = 0
    pc = 0
    factor = 0
    n = mat[c][c]

    m_wi = np.zeros(c)
    m_wj = np.zeros(c)
    w = 0

    for i, j in product(range(c), range(c)):
        w = 1 - (i - j)**2 / c**2
        p0 += w * mat[i][j]/n
        pc += w * mat[i][c]/n * mat[c][j]/n

        m_wi[i] += mat[c][j] * w/n
        m_wj[j] += mat[i][c] * w/n

    k_hat = (p0 - pc) / (1 - pc)

    for i, j in product(range(c), range(c)):
        w = 1 - (i - j)**2 / c**2
        factor += mat[i][j]/n*(w - (m_wi[i] + m_wj[j])*(1-k_hat))**2

    var_k_hat = (factor - (k_hat - pc*(1-k_hat))**2) / (n * (1-pc)**2)

    kappa = TransformedEstimator(
        estimate=k_hat,
        variance=var_k_hat,
        transformed_variance=var_k_hat,
        transformed_function=TransformFunc.ID,
        alpha=alpha,
        confident_limit=ConfidentLimit.LOWER,
        n=n
    )
    return kappa
