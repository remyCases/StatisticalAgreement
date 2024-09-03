# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

from typing import NamedTuple
from itertools import product
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import multivariate_normal, shapiro
import StatisticalAgreement as sa
from .monte_carlo import MonteCarlo

class Models(NamedTuple):
    mean: npt.ArrayLike
    cov: npt.ArrayLike

# Based on
# Lin LI.
# A concordance correlation coefficient to evaluate reproducibility.
# Biometrics. 1989 Mar;45(1):255-68. PMID: 2720055.

# and

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

EXPECTED_VALUES = pd.DataFrame(data={"ccc": [0.950, 0.887, 0.360],
                                     "transformed_ccc": [1.832, 1.408, 0.377],
                                     "msd": [0.100, 0.239, 0.1583],
                                     "transformed_msd": [-2.303, -1.431, 0.640],},
                               index=['1', '2', '3'])

def mc_simulation(name_of_index: str, str_criterion=""):
    try:
        criterion = float(str_criterion)
    except ValueError:
        criterion = 0

    data_possibilities = [10, 20, 50]
    models_possibilities = ['1', '2', '3']

    tuples_row = tuple(product(models_possibilities, [f'{name_of_index}',
                                                      f's_{name_of_index}',
                                                      f'transformed_{name_of_index}', 
                                                      f's_transformed_{name_of_index}']))
    multi_index = pd.MultiIndex.from_tuples(tuples_row, names=('case', 'estimator'))

    tuples_col = tuple(product(data_possibilities, ['mean', 'std', 'pvalue']))
    multi_col = pd.MultiIndex.from_tuples(tuples_col, names=('n', ''))

    result_df = pd.DataFrame(index=multi_index, columns=multi_col)

    for m in models_possibilities:
        for d in data_possibilities:
            _simulation_from_model_and_ndata(n_iteration=5000, n_data=d, model=m, result_df=result_df,
                                             name_of_index=name_of_index, criterion=criterion)
            result_df.loc[(m, f'{name_of_index}'), "true_value"] = EXPECTED_VALUES.loc[m, f'{name_of_index}']
            result_df.loc[(m, f'transformed_{name_of_index}'), "true_value"] = EXPECTED_VALUES.loc[m, f'transformed_{name_of_index}']

    print(f"{name_of_index} with normal data:")
    print(result_df, '\n')

def _simulation_from_model_and_ndata(n_iteration: int, n_data: int, model: str, result_df, name_of_index: str, criterion: float):
    mean, cov = MODELS[model]
    array_index = np.empty(n_iteration)
    array_index_std = np.empty(n_iteration)
    array_transformed_index = np.empty(n_iteration)
    array_transformed_index_std = np.empty(n_iteration)
    index_func = getattr(sa, name_of_index)

    for i in range(n_iteration):
        multidim = multivariate_normal.rvs(mean=mean, cov=cov, size=n_data)
        index = index_func(multidim[:, 0], multidim[:, 1], method='approx', alpha=0.05, transformed=True, criterion=criterion)

        array_index[i] = index.estimate
        array_transformed_index[i] = index.transformed_estimate

        try:
            array_index_std[i] = np.sqrt(index.variance)
            array_transformed_index_std[i] = np.sqrt(index.transformed_variance)
        except TypeError:
            array_index_std[i] = 0
            array_transformed_index_std[i] = 0

    mc = MonteCarlo()
    mc_index = mc.compute(array_index)
    mc_index_std = mc.compute(array_index_std)
    mc_transformed_index = mc.compute(array_transformed_index)
    mc_transformed_index_std = mc.compute(array_transformed_index_std)

    result_df.loc[(model, name_of_index), (n_data, 'mean')] = mc_index.mean
    result_df.loc[(model, f'transformed_{name_of_index}'), (n_data, 'mean')] = mc_transformed_index.mean

    result_df.loc[(model, f's_{name_of_index}'), (n_data, 'mean')] = mc_index_std.mean
    result_df.loc[(model, f's_transformed_{name_of_index}'), (n_data, 'mean')] = mc_transformed_index_std.mean

    result_df.loc[(model, f'{name_of_index}'), (n_data, 'std')] = np.sqrt(mc_index.var)
    result_df.loc[(model, f'transformed_{name_of_index}'), (n_data, 'std')] = np.sqrt(mc_transformed_index.var)

    result_df.loc[(model, f'{name_of_index}'), (n_data, 'pvalue')] = shapiro(array_index).pvalue
    result_df.loc[(model, f'transformed_{name_of_index}'), (n_data, 'pvalue')] = shapiro(array_transformed_index).pvalue
