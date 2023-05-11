import numpy as np
import pandas as pd
from scipy.stats import norm, t, ncx2
import src.StatisticalAgreement as sa

ALPHA = 0.05
CP_DELTA = 0.5
CP_ALLOWANCE = 0.9

def main(X, Y):
    n = len(X)
    res = pd.DataFrame(columns=["estimator", "variance", "limit", "allowance"], 
                       index=["msd", "accuracy", "precision", "ccc", "cp", "tdi", "rbs"])

    mu_hat_x = np.mean(X)
    mu_hat_y = np.mean(Y)
    mu_d = mu_hat_x - mu_hat_y
    s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(X, Y, bias=True).flatten()
    s_sq_hat_d = n / (n - 3) * (s_sq_hat_biased_x + s_sq_hat_biased_y - 2 * s_hat_biased_xy)

    eps_hat = np.sum((X-Y)**2) / (n - 1)
    var_esp_hat = 2 / (n - 2) * ( 1 - mu_d**4 / eps_hat**4)
    esp_range = sa.TransformEstimator(eps_hat, var_esp_hat, sa.TransformFunc.Log)
    res.loc["msd", "estimator"] = eps_hat
    res.loc["msd", "variance"] = var_esp_hat
    res.loc["msd", "limit"] = esp_range.ci(ALPHA, sa.ConfidentLimit.Upper, n)

    rbs_hat = mu_d**2 / s_sq_hat_d
    delta_plus = (CP_DELTA + mu_d) / np.sqrt(s_sq_hat_d)
    n_delta_plus = norm.pdf(-delta_plus)
    delta_minus = (CP_DELTA - mu_d) / np.sqrt(s_sq_hat_d)
    n_delta_minus = norm.pdf(delta_minus)

    cp_hat = ncx2.cdf(CP_DELTA, df=1, nc=rbs_hat)
    var_cp_hat = 1/2 * ((delta_plus * n_delta_plus + delta_minus * n_delta_minus)**2 + 
                        (n_delta_plus - n_delta_minus)**2) / ((n-3)*(1-cp_hat)**2*cp_hat**2)
    cp_range = sa.TransformEstimator(cp_hat, var_cp_hat, sa.TransformFunc.Logit)
    res.loc["cp", "estimator"] = cp_hat
    res.loc["cp", "variance"] = var_cp_hat
    res.loc["cp", "limit"] = cp_range.ci(ALPHA, sa.ConfidentLimit.Lower, n)
    res.loc["cp", "allowance"] = CP_ALLOWANCE

    res.loc["rbs", "estimator"] = rbs_hat
    print(res)
    print(sa.cp_tdi_approximation(rbs_hat, CP_ALLOWANCE))


if __name__ == "__main__":
    X = np.array([10, 11, 12, 10, 13, 15, 100, 102, 101, 100, 20, 25, 22])
    Y = np.array([11, 11, 11, 12, 16, 16, 100, 103, 99, 99, 22, 21, 21])
    agr = sa.Agreement(X, Y)
    agr.CCC_approximation().show()
    main(X, Y)