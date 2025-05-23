# Statistical Agreement

How to assert agreement using statistical indices ?

## Overview

This repo implements some indices used in statistical agreement such as total deviation index (TDI) and coverage probability (CP).

Statistical Agreement is an ensemble of processes to declare (or not) if two (or more) measurement methods lead to the same results.

Currently, only implementations for basic continuous or categorical models are planned.

## Installation

```sh
pip install StatisticalAgreement==0.4.0
```

## Usage

You can find examples in the example folder.

Here is an example of CCC usage with Gaussian simulated data:

```python
from scipy. stats import multivariate_normal
import numpy as np
import StatisticalAgreement as sa

import seaborn as sns    
import matplotlib.pyplot as plt

mean=np.array([-np.sqrt(0.1)/2, np.sqrt(0.1)/2])
cov=np.array([[1.1**2, 0.95*1.1*0.9], [0.95*1.1*0.9, 0.9**2]])
xy = multivariate_normal.rvs(mean=mean, cov=cov, size=100)

x = xy[:, 0]
y = xy[:, 1]

ax = sns.histplot(x - y)
ax.set(xlabel="Difference of methods")
plt.show()

# Return approximate estimate of CCC 
# with a alpha risk of 5% 
# and an allowance of whithin sample deviation of 10%.
ccc = sa.ccc(x, y, method="approx", alpha=0.05, allowance=0.10)
print(f"Approximate estimate of CCC: {ccc.estimate:.4f}\n\
Lower confident interval of the estimate with confident level of 95%: {ccc.limit:.4f}\n")
```

```text
Approximate estimate of CCC: 0.8943
Lower confident interval of the estimate with confident level of 95%: 0.8625
```

Since `allowance > limit`, then there is no allowance by criterion defined by the user.

The distribution of the difference of methods can be displayed for visual analysis.
![Distribution of difference of methods](plots/histplot_difference_methods_simalution_example.png?raw=true "Distribution of difference of methods")

Running the `main.py` with the argument `-e` will display the examples.

## Current Implementations

For each index listed in the following table:

- **naive** designes an implemetation using a parametric hypothesis (like a **normal** hypothesis), and thus only accurate if the hypothesis is true.
- **robust** designes an implemetation not depending of any kind of hypothesis.
- **tested** indicates if the implementation of the said index is tested with a monte-carlo test and results are correct in regards of the scientific literature.
- **bootstrap** indicates if an alternative way to compute confident interval using a resample method is implemented.
- **unified model** indicates if there is an implementation for models using continuous and categorical data (for instance with multiple raters and/or readings) - *not planned currently*

|Index | Naive | Tested | Robust |  Tested | Bootstrap | Unified model |
|--|:--:|:--:|:--:|:--:|:--:|:--:|
| MSD |:heavy_check_mark:|:heavy_check_mark:[^1]|:x:|:x:|:x:|:x:|
| TDI |:heavy_check_mark:|WIP|:x:|:x:|:x:|:x:|
| CP |:heavy_check_mark:|WIP|:x:|:x:|:x:|:x:|
| Accuracy |:heavy_check_mark:|:x:|:x:|:x:|:x:|:x:|
| Precision |:heavy_check_mark:|:x:|:x:|:x:|:x:|:x:|
| CCC |:heavy_check_mark:|:heavy_check_mark:[^1]|WIP|:x:|:x:|:x:|
| Kappa |:heavy_check_mark:|:heavy_check_mark:[^3]|:x:|:x:|:x:|:x:|
| Weighted Kappa |:heavy_check_mark:[^2]|:heavy_check_mark:[^3]|:x:|:x:|:x:|:x:|

## Test result

Implementation of the indices are tested with a monte-carlo simulation. The goal is to match results from the scientific literature. Currently tests of mc simulations can be display running `main.py` with the `-s i` argument where `i` is the index simulated.

Currently only `msd` and `ccc` tests are implemented. One can compare `msd` simulation results with \cite{LIN2000} and `ccc` one with \cite{LIN1989}.

## References

Bibtex is available [here](bibliography.bib).

[^1]: With normal data only
[^2]: Absolute and Squared Weighted Kappa
[^3]: Minimal testing based on examples found in \cite{LIN2013}

## Troubleshooting

For VS Code users on Windows, using a venv to run the script can be prohibited due to ExecutionPolicy.

```powershell
Get-ExecutionPolicy -List
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```
