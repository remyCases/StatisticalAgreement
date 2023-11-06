# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import numpy as np

def almost_equal_float(first: float, second: float, max_ulps=4) -> bool:
    # using difference of ulp as a more reliable way to test equality in float/double
    # same method as used in GoogleTest C++ library
    # see https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/

    # catching number with difference of sign and inf/nan
    if (first < 0 != second < 0):

        # 0 == -0 can't be catched by integer representation
        if (first == second):
            return True
        return False
    
    else:

        first_int = np.array(first).view('int64')
        second_int = np.array(second).view('int64')

        ulps_diff = abs(first_int - second_int)

        if ulps_diff < max_ulps:
            return True
   
    return False