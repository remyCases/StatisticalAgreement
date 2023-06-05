# Statistical Agreement
## How to assert agreement using statistical indices ?

This repo implements some indices used in statistical agreement such as total deviation index (TDI) and coverage probability (CP). 
Currently, only implementations for basic continuous or categorical models are planned.

### Usage

This project is not a proper python package yet. It will be distributed in the future via Pypi. Thus, to use it, you need to clone the current repo and include in your project.


You can find examples in the example folder. Current functions are:
```python
import src.StatisticalAgreement as sa
sa.ccc(...)
sa.cp(...)
sa.tdi(...)
sa.agreement(...)
```

Running the `main.py` with the argument `-e` will display the examples.

### Current Implementations

For each index listed in the following table:
- **naive** designed an implemetation using a parametric hypothesis (like a **normal** hypothesis), and thus only accurate if the hypothesis is true.
- **robust** designed an implemetation not depending of any kind of hypothesis 
- **tested** if the implementation of the said index is tested with a monte-carlo test and results are correct in regards of scientific literature. 
- **bootstrap** if an alternative way to compute confident interval using a resample method is implemented
- **unified model** if there is an implementation for model using continuous and categorical data (for instance with multiple raters and/or readings) - *not planned currently*

|Index | Naive | Tested | Robust |  Tested | Bootstrap | Unified model | 
|--|:--:|:--:|:--:|:--:|:--:|:--:|
| MSD |:heavy_check_mark:|WIP|:heavy_check_mark:|:x:|:x:|:x:
| TDI |:heavy_check_mark:|WIP|:x:|:x:|:x:|:x:
| CP |:heavy_check_mark:|WIP|:heavy_check_mark:|:x:|:x:|:x:
| Accuracy |:heavy_check_mark:|:x:|:x:|:x:|:x:|:x:
| Precision |:heavy_check_mark:|:x:|:x:|:x:|:x:|:x:
| CCC |:heavy_check_mark:|:heavy_check_mark:[^1]|WIP|:x:|:x:|:x:
| Kappa |:x:|:x:|:x:|:x:|:x:|:x:
| Weighted Kappa |:x:|:x:|:x:|:x:|:x:|:x:

[^1]: With normal data only
### Test result

Implementation of the indices are tested with a monte-carlo simulation. The goal is to match results from the scientific literature. Currently tests of mc simulation can be display running `main.py` with the `-s` argument.

### Bibliography

Bibtex is available [here](bibliography.bib).