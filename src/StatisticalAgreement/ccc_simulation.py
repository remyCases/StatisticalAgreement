# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of adventOfCode project from https://github.com/remyCases/StatiscalAgreement.

import numpy as np
from scipy.stats import multivariate_normal
import src.StatisticalAgreement as sa
from .monte_carlo import MonteCarlo
import matplotlib.pyplot as plt

def ccc_simulation(n_mc: int, n_data: int):
    mean = np.array([0, 0])
    cov = np.array([[1, 0.95], [0.95, 1]])
    array_ccc_ustat = np.empty(n_mc)
    array_z_ustat = np.empty(n_mc)

    mc_ustat = MonteCarlo()

    for i in range(n_mc):
        multidim = multivariate_normal.rvs(mean=mean, cov=cov, size=n_data)
        agrtest = sa.Agreement(multidim[:, 0], multidim[:, 1])
        agrtest.ccc_ustat()
        array_ccc_ustat[i] = agrtest._ccc_ustat.estimator
        array_z_ustat[i] = agrtest._z_ustat.estimator

    plt.hist(array_ccc_ustat)
    plt.show()
    plt.hist(array_z_ustat)
    plt.show()
    print(f"{n_data} ccc : {mc_ustat.compute(array_ccc_ustat)}")
    print(f"{n_data} z : {mc_ustat.compute(array_z_ustat)}")