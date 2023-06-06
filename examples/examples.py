import numpy as np
import StatisticalAgreement as sa

def main():
    x_cat = np.repeat([0, 0, 0, 1, 1, 1, 2, 2, 2], [11, 2, 19, 1, 3, 3, 0, 8, 82])
    y_cat = np.repeat([0, 1, 2, 0, 1, 2, 0, 1, 2], [11, 2, 19, 1, 3, 3, 0, 8, 82])
    print(f"Estimate of Cohen's Kappa: {sa.cohen_kappa(x_cat, y_cat, method='exact', alpha=0.05)}")
    print(f"Of following contingency Matrix: \n{sa.core.agreement._contingency(x_cat, y_cat, 3)}\n")

    X = np.array([12, 10, 13, 10])
    Y = np.array([11, 12, 16, 9])

    # Return approximate estimate of MSD 
    # with a alpha risk of 5% 
    print(f"Approximate estimate of MSD: {sa.msd(X, Y, method='approx', alpha=0.05)}\n")

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