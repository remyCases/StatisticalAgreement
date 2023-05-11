import numpy as np
import pandas as pd
import src.StatisticalAgreement as sa

if __name__ == "__main__":
    X = np.array([10, 11, 12, 10, 13, 15, 100, 102, 101, 100, 20, 25, 22])
    Y = np.array([11, 11, 11, 12, 16, 16, 100, 103, 99, 99, 22, 21, 21])
    agr = sa.Agreement(X, Y)
    agr.CCC_approximation().CP_TDI_approximation().show()