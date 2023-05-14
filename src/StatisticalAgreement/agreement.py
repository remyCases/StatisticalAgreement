# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of adventOfCode project from https://github.com/remyCases/StatiscalAgreement.

# Based on Lin, L. I.-K. (2000). 
## Total deviation index for measuring individual agreement with applications in 
## laboratory performance and bioequivalence. Statistics in Medicine, 19(2), 255–270

import numpy as np
import pandas as pd
from typing import TypeVar
from scipy.stats import norm, ncx2
from .classutils import TransformFunc, ConfidentLimit, TransformEstimator, Estimator

ALPHA = 0.05
CP_DELTA = 0.5
TDI_PI = 0.9
CP_ALLOWANCE = 0.9
TDI_ALLOWANCE = 10
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
    def __init__(self, x, y, delta_criterion_for_cp=CP_DELTA, pi_criterion_for_tdi=TDI_PI) -> None:
        self._x = x
        self._y = y
        self._delta_criterion = delta_criterion_for_cp
        self._pi_criterion = pi_criterion_for_tdi
        self._n = len(x)

    def ccc_approximation(self) -> SAgreement:
        # Lin LI. A concordance correlation coefficient to evaluate reproducibility. 
        # Biometrics. 1989 Mar;45(1):255-68. PMID: 2720055.
        mu_d = np.mean(self._x) - np.mean(self._y)
        s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(self._x, self._y, bias=True).flatten()

        sqr_var = np.sqrt(s_sq_hat_biased_x * s_sq_hat_biased_y)
        rho_hat = s_hat_biased_xy / sqr_var
        nu_sq_hat = mu_d**2 / sqr_var
        omega_hat = np.sqrt(s_sq_hat_biased_x / s_sq_hat_biased_y)

        acc_hat = 2 * sqr_var / (s_sq_hat_biased_x + s_sq_hat_biased_y + mu_d**2)
        var_acc_hat = (acc_hat**2*nu_sq_hat*(omega_hat + 1/omega_hat - 2*rho_hat) + 
                    0.5*acc_hat**2*(omega_hat**2+1/omega_hat**2+2*rho_hat**2) + 
                    (1+rho_hat**2)*(acc_hat*nu_sq_hat-1)) / ((self._n-2)*(1-acc_hat)**2)
        
        acc_range = TransformEstimator(acc_hat, var_acc_hat, TransformFunc.Logit)
        self._acc = Estimator(
            name="acc",
            estimator=acc_hat,
            variance=var_acc_hat,
            limit=acc_range.ci(ALPHA, ConfidentLimit.Lower, self._n)
        )

        var_rho_hat = (1 - rho_hat**2/2)/(self._n-3)
        rho_range = TransformEstimator(rho_hat, var_rho_hat, TransformFunc.Z)
        self._rho = Estimator(
            name="rho",
            estimator=rho_hat,
            variance=var_rho_hat,
            limit=rho_range.ci(ALPHA, ConfidentLimit.Lower, self._n)
        )

        ccc_hat = acc_hat * rho_hat
        var_ccc_hat = 1 / (self._n - 2) * ( (1-rho_hat**2)*ccc_hat**2/((1-ccc_hat**2)*rho_hat**2)
                                           + 2*ccc_hat**3*(1-ccc_hat)*nu_sq_hat / (rho_hat*(1-ccc_hat**2)**2)
                                           - ccc_hat**4 * nu_sq_hat**2 / (2*rho_hat**2*(1-ccc_hat**2)**2))
        ccc_range = TransformEstimator(ccc_hat, var_ccc_hat, TransformFunc.Z)
        self._ccc = Estimator(
            name="ccc",
            estimator=ccc_hat,
            variance=var_ccc_hat,
            limit=ccc_range.ci(ALPHA, ConfidentLimit.Lower, self._n)
        )

        return self

    def ccc_ustat(self) -> SAgreement:
        # King TS, Chinchilli VM. 
        # Robust estimators of the concordance correlation coefficient. 
        # J Biopharm Stat. 2001;11(3):83-105. doi: 10.1081/BIP-100107651. PMID: 11725932.

        sx = np.sum(self._x)
        sy = np.sum(self._y)
        ssx = np.sum(self._x**2)
        ssy = np.sum(self._y**2)

        xy = self._x * self._y
        dss = self._x**2 - ssx / self._n + self._y**2 - ssy / self._n
        ds_sq = self._x * sy + self._y * sx - sx * sy / self._n
        sxy = np.sum(xy)

        u1 = -4 / (self._n - 1) * sxy
        u2 = 2 / (self._n - 1) * (ssx + ssy)
        u3 = -4 / (self._n * (self._n - 1)) * (sx * sy)

        h = (self._n - 1) * (u3 - u1)
        g = u1 + self._n * u2+ (self._n - 1)*u3

        v_u1 = 64 / (self._n-1)**2 * np.sum(xy**2)
        v_u2 = 4 / ((self._n-1)**2 * self._n**2) * np.sum(dss**2)
        v_u3 = 64 / ((self._n-1)**2 * self._n**2) * np.sum(ds_sq**2)

        cov_u1_u2 = -16 / (self._n-1)**2 * np.sum(xy * dss)
        cov_u1_u3 = 64 / ((self._n-1)**2 * self._n) * np.sum(xy * ds_sq)
        cov_u2_u3 = -16 / ((self._n-1)**2 * self._n) * np.sum(dss * ds_sq)

        s_sq_h = (self._n-1)**2 * (v_u3 + v_u1 - 2 * cov_u1_u3)
        s_sq_g = (self._n-1)**2 * v_u3 + v_u1 + self._n**2 * v_u2 + 2*(self._n-1) * cov_u1_u3 \
            + 2 * self._n**2 * cov_u2_u3
        s_hg = -(self._n-1)*(self._n-2)*cov_u1_u3 + self._n*(self._n-1)*cov_u2_u3 + (self._n-1)**2*v_u3 \
            - (self._n-1)*v_u1 - self._n*(self._n-1)*cov_u1_u2
        
        ccc_hat = h/g
        var_ccc_hat = ccc_hat**2 * (s_sq_h / h**2 - 2 * s_hg / (h * g) + s_sq_g / g**2)
        var_z_hat = var_ccc_hat / (1 - ccc_hat**2)**2

        ccc_range = TransformEstimator(ccc_hat, var_ccc_hat, TransformFunc.Z)
        self._ccc_ustat = Estimator(
            name="ccc_ustat",
            estimator=ccc_hat,
            variance=var_ccc_hat,
            limit=ccc_range.ci(ALPHA, ConfidentLimit.Lower, self._n)
        )
        self._z_ustat = Estimator(
            name="z_ustat",
            estimator=TransformFunc.Z.apply(ccc_hat),
            variance=var_z_hat,
            limit=np.nan
        )
        return self
    
    def cp_tdi_approximation(self) -> SAgreement:

        D = self._x - self._y
        s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(self._x, self._y, bias=True).flatten()
        s_sq_hat_d = self._n / (self._n - 3) * (s_sq_hat_biased_x + s_sq_hat_biased_y - 2 * s_hat_biased_xy)
        mu_d = np.mean(D)

        eps_sq_hat = np.sum(D**2) / (self._n - 1)
        var_esp_hat = 2 / (self._n - 2) * ( 1 - mu_d**4 / eps_sq_hat**2)
        esp_range = TransformEstimator(eps_sq_hat, var_esp_hat, TransformFunc.Log)
        self._msd = Estimator(
            name="msd",
            estimator=eps_sq_hat,
            variance=var_esp_hat,
            limit=esp_range.ci(ALPHA, ConfidentLimit.Upper, self._n)
        )

        rbs_hat = mu_d**2 / s_sq_hat_d
        self._rbs = Estimator(
            name="rbs",
            estimator=rbs_hat,
            variance=np.nan,
            limit=np.nan
        )

        delta_plus = (self._delta_criterion + mu_d) / np.sqrt(s_sq_hat_d)
        n_delta_plus = norm.pdf(-delta_plus)
        delta_minus = (self._delta_criterion - mu_d) / np.sqrt(s_sq_hat_d)
        n_delta_minus = norm.pdf(delta_minus)

        cp_hat = ncx2.cdf(self._delta_criterion, df=1, nc=rbs_hat)
        var_cp_hat = 1/2 * ((delta_plus * n_delta_plus + delta_minus * n_delta_minus)**2 + 
                            (n_delta_plus - n_delta_minus)**2) / ((self._n-3)*(1-cp_hat)**2*cp_hat**2)
        cp_range = TransformEstimator(cp_hat, var_cp_hat, TransformFunc.Logit)
        self._cp = Estimator(
            name="cp",
            estimator=cp_hat,
            variance=var_cp_hat,
            limit=cp_range.ci(ALPHA, ConfidentLimit.Lower, self._n)
        )

        coeff_tdi = norm.ppf(1 - (1 - self._pi_criterion) / 2)
        tdi_hat = coeff_tdi * np.sqrt(eps_sq_hat)
        self._tdi = Estimator(
            name="tdi",
            estimator=tdi_hat,
            variance=np.nan,
            limit=np.nan
        )

        return self
    
    def show(self) -> None:
        self.res = pd.DataFrame(columns=["estimator", "variance", "limit", "criterion", "allowance"], 
                                index=["msd", "accuracy", "precision", "ccc", "cp", "tdi", "rbs"])
        
        self.res.loc["accuracy", "estimator"] = self._acc.estimator
        self.res.loc["accuracy", "variance"] = self._acc.variance
        self.res.loc["accuracy", "limit"] = self._acc.limit
                
        self.res.loc["precision", "estimator"] = self._rho.estimator
        self.res.loc["precision", "variance"] = self._rho.variance
        self.res.loc["precision", "limit"] = self._rho.limit

        self.res.loc["ccc", "estimator"] = self._ccc.estimator
        self.res.loc["ccc", "variance"] = self._ccc.variance
        self.res.loc["ccc", "limit"] = self._ccc.limit
        self.res.loc["ccc", "allowance"] = 1 - WITHIN_SAMPLE_DEVIATION**2

        self.res.loc["msd", "estimator"] = self._msd.estimator
        self.res.loc["msd", "variance"] = self._msd.variance
        self.res.loc["msd", "limit"] = self._msd.limit

        self.res.loc["rbs", "estimator"] = self._rbs.estimator
        self.res.loc["rbs", "allowance"] = cp_tdi_approximation(self._rbs.estimator, CP_ALLOWANCE)

        self.res.loc["cp", "estimator"] = self._cp.estimator
        self.res.loc["cp", "variance"] = self._cp.variance
        self.res.loc["cp", "limit"] = self._cp.limit
        self.res.loc["cp", "criterion"] = self._delta_criterion
        self.res.loc["cp", "allowance"] = CP_ALLOWANCE

        self.res.loc["tdi", "estimator"] = self._tdi.estimator
        self.res.loc["tdi", "criterion"] = self._pi_criterion
        self.res.loc["tdi", "allowance"] = TDI_ALLOWANCE
        print(self.res)