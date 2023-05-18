# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of adventOfCode project from https://github.com/remyCases/StatiscalAgreement.

import numpy as np
import numpy.typing as npt
from typing import NamedTuple
import pandas as pd
from itertools import product
from scipy.stats import multivariate_normal, shapiro
import src.StatisticalAgreement as sa
from .monte_carlo import MonteCarlo

class Models(NamedTuple):
    mean: npt.ArrayLike
    cov: npt.ArrayLike

# Based on 
# King TS, Chinchilli VM. 
# Robust estimators of the concordance correlation coefficient. 
# J Biopharm Stat. 2001;11(3):83-105. doi: 10.1081/BIP-100107651. PMID: 11725932.
# pages 90-94 
MODELS = {
    '1': Models(mean=np.array([0, 0]), 
                cov=np.array([[1, 0.95], [0.95, 1]])),
    '2': Models(mean=np.array([-np.sqrt(0.1)/2, np.sqrt(0.1)/2]), 
                cov=np.array([[1.1**2, 0.95*1.1*0.9], [0.95*1.1*0.9, 0.9**2]])),
    '3': Models(mean=np.array([-np.sqrt(0.25)/2, np.sqrt(0.25)/2]), 
                cov=np.array([[(4/3)**2, 0.5*4/3*2/3], [0.5*4/3*2/3, (2/3)**2]]))}

def ccc_simulation():
    data_possibilities = [10, 20, 40, 80]
    models_possibilities = ['1', '2', '3']
    tupleIndex = tuple(product(data_possibilities, models_possibilities, ['ccc', 'z']))
    multiIndex = pd.MultiIndex.from_tuples(tupleIndex, names=('n', 'case', 'estimator'))
    
    tupleCol = tuple(product(models_possibilities, data_possibilities))
    multiCol = pd.MultiIndex.from_tuples(tupleCol, names=('case', 'n'))

    normalityDf = pd.DataFrame(index=multiIndex, columns=['pvalue'])
    estimateDf = pd.DataFrame(index=['ccc (mc estimate of ccc)', 
                                     'sigma_sq_z (mc estimate of z-variance)', 
                                     'var_z (variance of mc estimation of z)'], columns=multiCol)

    for m in models_possibilities:
        for d in data_possibilities:
            ccc_simulation_from_model_and_ndata(nIteration=1000, nData=d, model=m, 
                                                normalityDf=normalityDf, estimateDf=estimateDf)
    print(normalityDf, '\n')
    print(estimateDf, '\n')

def ccc_simulation_from_model_and_ndata(nIteration: int, nData: int, model: str, normalityDf, estimateDf):
    mean, cov = MODELS[model]
    array_ccc_ustat = np.empty(nIteration)
    array_z_ustat = np.empty(nIteration)
    array_z_ustat_var = np.empty(nIteration)

    mc_ustat = MonteCarlo()

    for i in range(nIteration):
        multidim = multivariate_normal.rvs(mean=mean, cov=cov, size=nData)
        agrtest = sa.Agreement(multidim[:, 0], multidim[:, 1])
        agrtest.ccc_ustat()
        array_ccc_ustat[i] = agrtest._ccc_ustat.estimator
        array_z_ustat[i] = agrtest._z_ustat.estimator
        array_z_ustat_var[i] = agrtest._z_ustat.variance

    mc_ccc_res = mc_ustat.compute(array_ccc_ustat)
    mc_z_res = mc_ustat.compute(array_z_ustat)
    mc_z_var_res = mc_ustat.compute(array_z_ustat_var)
    
    normalityDf.loc[(nData, model, 'ccc'), 'pvalue'] = shapiro(array_ccc_ustat).pvalue
    normalityDf.loc[(nData, model, 'z'), 'pvalue'] = shapiro(array_z_ustat).pvalue

    estimateDf.loc['ccc (mc estimate of ccc)', (model, nData)] = mc_ccc_res.mean
    estimateDf.loc['sigma_sq_z (mc estimate of z-variance)', (model, nData)] = mc_z_var_res.mean
    estimateDf.loc['var_z (variance of mc estimation of z)', (model, nData)] = mc_z_res.var