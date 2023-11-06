# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatisticalAgreement.

import unittest
from unittest.util import safe_repr
import numpy as np
from scipy.stats import norm, sem
from StatisticalAgreement.simulation.monte_carlo import MonteCarlo

N_REPETITION = 5000
    
class TestMonteCarlo(unittest.TestCase):
    # using difference of ulp as a more reliable way to test equality in float/double
    # same method as used in GoogleTest C++ library
    # see https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    def assertAlmostEqual(self, first, second, max_ulps=4):

        # catching obvious case
        if (first == second):
            return
        
        # catching number with difference of sign and inf/nan
        if (first < 0 != second < 0):
            standard_msg = '%s != %s within %s ulp' % (
                safe_repr(first),
                safe_repr(second),
                safe_repr(max_ulps))
        else:

            first_int = np.array(first).view('int64')
            second_int = np.array(second).view('int64')

            ulps_diff = abs(first_int - second_int)

            if ulps_diff < max_ulps:
                return
            
            standard_msg = '%s != %s within %s ulp' % (
                safe_repr(first),
                safe_repr(second),
                safe_repr(max_ulps))
            
        msg = self._formatMessage(None, standard_msg)
        raise self.failureException(msg)
    
    def assertAlmostMcCi(self, array : np.array):
        mc = MonteCarlo().compute(array)
        self.assertAlmostEqual(mc.mean, np.mean(array))
        self.assertAlmostEqual(mc.var, np.var(array))
        self.assertAlmostEqual(mc.standard_error, sem(array, ddof=0))
    
    def test_constant_monte_carlo(self):
        array = np.repeat(1.0, N_REPETITION)
        self.assertAlmostMcCi(array)
        
    def test_normal_monte_carlo(self):
        array = norm.rvs(loc=0.0, scale=1.0, size=N_REPETITION)
        self.assertAlmostMcCi(array)
        
if __name__ == '__main__':
    unittest.main()