import numpy as np
import pandas as pd
from typing import TypeVar
from scipy.stats import norm, ncx2
from .classutils import TransformFunc, ConfidentLimit, TransformEstimator

ALPHA = 0.05
CP_DELTA = 0.5
CP_ALLOWANCE = 0.9
WITHIN_SAMPLE_DEVIATION = 0.15

def cp_tdi_approximation(rbs: float, cp_allowance: float) -> bool:
    if cp_allowance == 0.75 and rbs <= 1/2:
        return True
    if cp_allowance == 0.8 and rbs <= 8:
        return True
    if cp_allowance == 0.85 and rbs <= 2:
        return True
    if cp_allowance == 0.9 and rbs <= 1:
        return True
    if cp_allowance == 0.95 and rbs <= 1/2:
        return True
    return False

SAgreement = TypeVar("SAgreement", bound="Agreement")

class Agreement:
    def __init__(self, X, Y) -> None:
        self._X = X
        self._Y = Y
        self._n = len(X)
        self.res = pd.DataFrame(columns=["estimator", "variance", "limit", "allowance"], 
                                index=["msd", "accuracy", "precision", "ccc", "cp", "tdi", "rbs"])

    def CCC_approximation(self) -> SAgreement:
        mu_d = np.mean(self._X) - np.mean(self._Y)
        s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(self._X, self._Y, bias=True).flatten()

        sqr_var = np.sqrt(s_sq_hat_biased_x * s_sq_hat_biased_y)
        rho_hat = s_hat_biased_xy / sqr_var
        nu_sq_hat = mu_d**2 / sqr_var
        omega_hat = np.sqrt(s_sq_hat_biased_x / s_sq_hat_biased_y)

        acc_hat = 2 * sqr_var / (s_sq_hat_biased_x + s_sq_hat_biased_y + mu_d**2)
        var_acc_hat = (acc_hat**2*nu_sq_hat*(omega_hat + 1/omega_hat - 2*rho_hat) + 
                    0.5*acc_hat**2*(omega_hat**2+1/omega_hat**2+2*rho_hat**2) + 
                    (1+rho_hat**2)*(acc_hat*nu_sq_hat-1)) / ((self._n-2)*(1-acc_hat)**2)
        
        acc_range = TransformEstimator(acc_hat, var_acc_hat, TransformFunc.Logit)
        self.res.loc["accuracy", "estimator"] = acc_hat
        self.res.loc["accuracy", "variance"] = var_acc_hat
        self.res.loc["accuracy", "limit"] = acc_range.ci(ALPHA, ConfidentLimit.Lower, self._n)

        var_rho_hat = (1 - rho_hat**2/2)/(self._n-3)
        rho_range = TransformEstimator(rho_hat, var_rho_hat, TransformFunc.Z)
        self.res.loc["precision", "estimator"] = rho_hat
        self.res.loc["precision", "variance"] = var_rho_hat
        self.res.loc["precision", "limit"] = rho_range.ci(ALPHA, ConfidentLimit.Lower, self._n)

        ccc_hat = acc_hat * rho_hat
        var_ccc_hat = 1 / (self._n - 2) * ( (1-rho_hat**2)*ccc_hat**2/((1-ccc_hat**2)*rho_hat**2)
                                           + 2*ccc_hat**3*(1-ccc_hat)*nu_sq_hat / (rho_hat*(1-ccc_hat**2)**2)
                                           - ccc_hat**4 * nu_sq_hat**2 / (2*rho_hat**2*(1-ccc_hat**2)**2))
        ccc_range = TransformEstimator(ccc_hat, var_ccc_hat, TransformFunc.Z)
        self.res.loc["ccc", "estimator"] = ccc_hat
        self.res.loc["ccc", "variance"] = var_ccc_hat
        self.res.loc["ccc", "limit"] = ccc_range.ci(ALPHA, ConfidentLimit.Lower, self._n)
        self.res.loc["ccc", "allowance"] = 1 - WITHIN_SAMPLE_DEVIATION**2

        return self
    
    def CP_TDI_approximation(self) -> SAgreement:

        D = self._X - self._Y
        s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(self._X, self._Y, bias=True).flatten()
        s_sq_hat_d = self._n / (self._n - 3) * (s_sq_hat_biased_x + s_sq_hat_biased_y - 2 * s_hat_biased_xy)
        mu_d = np.mean(D)

        eps_hat = np.sum((D)**2) / (self._n - 1)
        var_esp_hat = 2 / (self._n - 2) * ( 1 - mu_d**4 / eps_hat**4)
        esp_range = TransformEstimator(eps_hat, var_esp_hat, TransformFunc.Log)
        self.res.loc["msd", "estimator"] = eps_hat
        self.res.loc["msd", "variance"] = var_esp_hat
        self.res.loc["msd", "limit"] = esp_range.ci(ALPHA, ConfidentLimit.Upper, self._n)

        rbs_hat = mu_d**2 / s_sq_hat_d
        self.res.loc["rbs", "estimator"] = rbs_hat
        self.res.loc["rbs", "allowance"] = cp_tdi_approximation(rbs_hat, CP_ALLOWANCE)

        delta_plus = (CP_DELTA + mu_d) / np.sqrt(s_sq_hat_d)
        n_delta_plus = norm.pdf(-delta_plus)
        delta_minus = (CP_DELTA - mu_d) / np.sqrt(s_sq_hat_d)
        n_delta_minus = norm.pdf(delta_minus)

        cp_hat = ncx2.cdf(CP_DELTA, df=1, nc=rbs_hat)
        var_cp_hat = 1/2 * ((delta_plus * n_delta_plus + delta_minus * n_delta_minus)**2 + 
                            (n_delta_plus - n_delta_minus)**2) / ((self._n-3)*(1-cp_hat)**2*cp_hat**2)
        cp_range = TransformEstimator(cp_hat, var_cp_hat, TransformFunc.Logit)
        self.res.loc["cp", "estimator"] = cp_hat
        self.res.loc["cp", "variance"] = var_cp_hat
        self.res.loc["cp", "limit"] = cp_range.ci(ALPHA, ConfidentLimit.Lower, self._n)
        self.res.loc["cp", "allowance"] = CP_ALLOWANCE

        return self
    
    def show(self) -> None:
        print(self.res)