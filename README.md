# Statistical Agreement
## How to assert agreement using statistical indices ?

This repo implements some indices used in statistical agreement such as total deviation index (TDI) and coverage probability (CP). 
Currently, only implementations for basic continuous or categorical models are planned.

### Usage

This project is not a proper python package yet. It will be distributed in the future via Pypi. Thus, to use it, you need to clone the current repo and include in your project.


You can find examples in the example folder *(WIP, look at the main.py file instead)*. Current functions are:
```python
import src.StatisticalAgreement as sa
X = np.array([10, 11, 12, 10, 13])
Y = np.array([11, 11, 11, 12, 16])

# Return approximate estimate of CCC 
# with a alpha risk of 5% 
# and an allowance of whithin sample deviation of 10%.
sa.ccc(X, Y, method="approx", alpha=0.05, allowance_whitin_sample_deviation=0.10)

# Return approximate estimate of CP with criterion of 2
# with a alpha risk of 5% 
# and an allowance of 80%.
sa.cp(X, Y, delta_criterion=2, method="approx", alpha=0.05, cp_allowance=0.8)

# Return approximate estimate of TDI with criterion of 90%
# with a alpha risk of 5% 
# and an allowance of 100.
sa.tdi(X, Y, pi_criterion=0.9, alpha=0.05, tdi_allowance=100)

# Return estimates of CCC, CP, TDI, MSD and others
# with a alpha risk of 5% 
# and display a summary of all estimates.
delta_criterion_for_cp=2
pi_criterion_for_tdi=0.9

sa.agreement(X, Y, 
    delta_criterion_for_cp, 
    pi_criterion_for_tdi, 
    display=True)
```

### Current Implementations

For each index listed in the following table:
- **naive** designed an implemetation using a parametric hypothesis (like a **normal** hypothesis), and thus only accurate if the hypothesis is true.
- **robust** designed an implemetation not depending of any kind of hypothesis 
- **tested** if the implementation of the said index is tested with a monte-carlo test and results are correct in regards of scientific litterature. 
- **bootstrap** if an alternative way to compute confident interval using a resample method is implemented
- **unified model** if there is an implementation for model using continuous and categorical data (for instance with multiple raters and/or readings) - *not planned currently*

|Index | Naive | Robust | Tested | Bootstrap | Unified model | 
|-----|:-:|:-----:|:---:|:----:|:----:|
| MSD |:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:x:
| TDI |:heavy_check_mark:|:x:|:x:|:x:|:x:
| CP |:heavy_check_mark:|:heavy_check_mark:|:x:|:x:|:x:
| Accuracy |:heavy_check_mark:|:x:|:x:|:x:|:x:
| Precision |:heavy_check_mark:|:x:|:x:|:x:|:x:
| CCC |:heavy_check_mark:|WIP|WIP|:x:|:x:
| Kappa |:x:|:x:|:x:|:x:|:x:
| Weighted Kappa |:x:|:x:|:x:|:x:|:x:

### Bibliography

Bibtex is available [here](bibliography.bib).