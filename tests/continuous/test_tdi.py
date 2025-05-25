# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import numpy as np
import pytest

from statisticalagreement.core._continuous_agreement import tdi_approx
from statisticalagreement.core._types import NDArrayFloat
from statisticalagreement.core.classutils import TransformedEstimator
from statisticalagreement.core.mathutils import assert_float


@pytest.mark.parametrize("x", [
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
    (np.array([5.22112622e-20, -2.07078331e+35, -5.17244799e-16, -8.09744291e-04,
        1.03226041e+31,  2.22421609e+18,  5.57698383e+14,  1.76780654e+33,
        5.73614039e+10, -1.18674943e-05, -2.43257918e-32, -4.74306548e-27,
        2.09445540e+34,  4.63172944e-10, -4.54774188e+05,  3.89914062e-31,
       -8.47040508e-20,  2.27500063e-08,  8.24331512e+27,  1.37715916e+20,
       -2.91699153e-35, -3.90364286e+17,  4.41938720e+07, -9.92234873e+36,
        1.36338386e+20, -1.89339725e+06, -3.88101107e-21,  7.33347901e-34,
        1.87416585e+24,  4.42420220e-26, -7.07988113e+25,  1.02108557e+29,
        8.72597011e-26, -1.41442334e-03, -1.92646784e+08,  1.33653162e-20,
        1.16235731e+16,  3.22684339e+37,  1.23108256e+00, -1.08786450e+06,
       -4.78332712e-24,  9.10944223e+21, -2.12798598e-17, -2.73197582e-30,
       -2.51416317e+20, -3.11404925e-21, -1.31308519e-15, -8.77899858e-13,
        2.11439524e+30, -5.65110035e-02,  1.26750137e-37,  4.31651880e+07,
       -2.36640808e-05, -3.95399094e+05,  8.30932969e-17,  2.20349236e+27,
       -1.22623047e+05, -2.74032797e-23, -1.03791590e-30, -9.61696320e+07,
       -2.36307140e-02,  1.63520831e+26, -3.13801575e-03, -6.06626462e+27,
        5.09073688e+05,  1.20274084e+10, -2.98037040e+07, -1.43269261e-07,
       -2.10278979e+03,  9.54432384e+08,  3.49677889e+17, -4.85443948e-25,
        3.11706296e+10, -1.79536108e-37, -4.15512003e+10, -4.59278899e+18,
        1.14652194e-32, -1.63124247e-32,  6.59990566e-31, -3.05130016e-29,
       -9.02408650e-27, -5.44453241e-12,  3.07500826e+13, -5.60638321e+10,
       -1.34229529e+33, -8.12934706e+12,  5.82139764e+28,  7.54790710e+02,
        7.14034715e-39, -6.45279237e+16, -1.53052767e-27, -6.88750552e+14,
       -4.84970752e+03, -6.68250942e+14, -1.15351419e+10,  8.68598415e+14,
       -6.47698375e+19,  2.70349675e-22, -2.81302900e+07, -3.33942274e-19]))
])
def test_tdi_approx_perfect_agreement(
    x: NDArrayFloat
) -> None:

    msd = TransformedEstimator(estimate=np.float64(0.0), limit=np.float64(0.0))
    tdi = tdi_approx(msd, pi_criterion=0.95, tdi_allowance=0.0)

    assert_float(tdi.estimate, 0.0, max_ulps=4)
    assert_float(tdi.limit, 0.0, max_ulps=4)
