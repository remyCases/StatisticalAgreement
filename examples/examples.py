import numpy as np
import StatisticalAgreement as sa

def main():
    X = np.array([12, 10, 13, 10])
    Y = np.array([11, 12, 16, 9])

    # Return approximate estimate of CCC 
    # with a alpha risk of 5% 
    # and an allowance of whithin sample deviation of 10%.
    print(f"Approximate estimate of CCC: {sa.ccc(X, Y, method='approx', alpha=0.05, allowance=0.10)}\n")

    # Return approximate estimate of CP with criterion of 2
    # with a alpha risk of 5% 
    # and an allowance of 80%.
    print(f"Approximate estimate of CP(2): {sa.cp(X, Y, method='approx', alpha=0.05, criterion=2, allowance=0.8)}\n")

    # Return approximate estimate of TDI with criterion of 90%
    # with a alpha risk of 5% 
    # and an allowance of 100.
    print(f"Approximate estimate of TDI(0.90): {sa.tdi(X, Y, method='approx', alpha=0.05, criterion=0.9, allowance=100)}\n")

    # Return estimates of CCC, CP, TDI, MSD and others
    # with a alpha risk of 5% 
    # and display a summary of all estimates.
    delta_criterion_for_cp=2
    pi_criterion_for_tdi=0.9
    
    sa.agreement(X, Y, 
        delta_criterion_for_cp, 
        pi_criterion_for_tdi, 
        display=True)