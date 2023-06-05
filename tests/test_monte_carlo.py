import unittest
import numpy as np
from scipy.stats import norm
from StatisticalAgreement.simulation.monte_carlo import MonteCarlo, McCi

N_REPETITION = 5000

class TestMonteCarlo(unittest.TestCase):

    def test_constant_monte_carlo(self):
        mc = MonteCarlo()
        
        expected_result = McCi(
            mean=1.0,
            var=0.0,
            standard_error=0.0)
        
        self.assertEqual(mc.compute(np.repeat(1.0, N_REPETITION)), 
                         expected_result)
        
    def test_normal_monte_carlo(self):
        mc = MonteCarlo()
        expected_result = McCi(
            mean=0.0,
            var=1.0,
            standard_error=1/np.sqrt(N_REPETITION))
        
        array = norm.rvs(loc=0.0, scale=1.0, size=N_REPETITION)
        self.assertEqual(mc.compute(array), 
                         expected_result)
        
if __name__ == '__main__':
    unittest.main()