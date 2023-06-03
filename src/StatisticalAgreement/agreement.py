# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatiscalAgreement.

# Based on Lin, L. I.-K. (2000). 
## Total deviation index for measuring individual agreement with applications in 
## laboratory performance and bioequivalence. Statistics in Medicine, 19(2), 255–270

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, shapiro
from .classutils import Indices, FlagData, TransformFunc, ConfidentLimit, TransformedEstimator, Estimator

DEFAULT_ALPHA = 0.05
CP_DELTA = 0.5
TDI_PI = 0.9
CP_ALLOWANCE = 0.9
TDI_ALLOWANCE = 10
WITHIN_SAMPLE_DEVIATION = 0.15

def rbs_allowance(cp_allowance: float) -> float:
    if cp_allowance == 0.75:
        return 1/2
    if cp_allowance == 0.8:
        return 8
    if cp_allowance == 0.85:
        return 2
    if cp_allowance == 0.9:
        return 1
    if cp_allowance == 0.95:
        return 1/2
    return np.nan

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

def _precision(x, y, alpha: float) -> TransformedEstimator:
    n = len(x)
    s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(x, y, bias=True).flatten()

    sqr_var = np.sqrt(s_sq_hat_biased_x * s_sq_hat_biased_y)
    rho_hat = s_hat_biased_xy / sqr_var

    var_rho_hat = (1 - rho_hat**2/2)/(n-3)
    rho = TransformedEstimator(
        estimate=rho_hat, 
        variance=var_rho_hat, 
        transformed_function=TransformFunc.Z,
        alpha=alpha, 
        confident_limit=ConfidentLimit.Lower, 
        n=n
    )
    return rho

def _accuracy(x, y, precision: TransformedEstimator, alpha: float) -> TransformedEstimator:
    n = len(x)
    mu_d = np.mean(x - y)
    rho_hat = precision.estimate

    s_sq_hat_biased_x, _, _, s_sq_hat_biased_y = np.cov(x, y, bias=True).flatten()
    sqr_var = np.sqrt(s_sq_hat_biased_x * s_sq_hat_biased_y)
    nu_sq_hat = mu_d**2 / sqr_var
    omega_hat = np.sqrt(s_sq_hat_biased_x / s_sq_hat_biased_y)

    acc_hat = 2 * sqr_var / (s_sq_hat_biased_x + s_sq_hat_biased_y + mu_d**2)
    var_acc_hat = (
        acc_hat**2*nu_sq_hat*(omega_hat + 1/omega_hat - 2*rho_hat) +
        0.5*acc_hat**2*(omega_hat**2+1/omega_hat**2+2*rho_hat**2) +
        (1+rho_hat**2)*(acc_hat*nu_sq_hat-1)
        ) / ((n-2)*(1-acc_hat)**2)
    
    acc = TransformedEstimator(
        estimate=acc_hat, 
        variance=var_acc_hat, 
        transformed_function=TransformFunc.Logit,
        alpha=alpha, 
        confident_limit=ConfidentLimit.Lower, 
        n=n
    )
    return acc
    
def _ccc_lin(x, y, 
             precision: TransformedEstimator, 
             accuracy: TransformedEstimator, 
             alpha: float, 
             allowance_whitin_sample_deviation: float) -> TransformedEstimator:
    n = len(x)
    mu_d = np.mean(x - y)
    rho_hat = precision.estimate

    s_sq_hat_biased_x, _, _, s_sq_hat_biased_y = np.cov(x, y, bias=True).flatten()
    sqr_var = np.sqrt(s_sq_hat_biased_x * s_sq_hat_biased_y)
    nu_sq_hat = mu_d**2 / sqr_var

    ccc_hat = rho_hat * accuracy.estimate
    var_ccc_hat = 1 / (n - 2) * ((1-rho_hat**2)*ccc_hat**2/rho_hat**2
                                 + 2*ccc_hat**3*(1-ccc_hat)*nu_sq_hat / (rho_hat*(1-ccc_hat**2))
                                 - ccc_hat**4 * nu_sq_hat**2 / (2*rho_hat**2*(1-ccc_hat**2)))
    var_z_hat = var_ccc_hat / (1-ccc_hat**2)

    ccc = TransformedEstimator(
        estimate=ccc_hat, 
        variance=var_ccc_hat, 
        transformed_variance=var_z_hat,
        transformed_function=TransformFunc.Z,
        allowance=1-allowance_whitin_sample_deviation**2,
        alpha=alpha, 
        confident_limit=ConfidentLimit.Lower, 
        n=n
    )
    return ccc

def _ccc_ustat(x, y, alpha: float, allowance_whitin_sample_deviation: float) -> TransformedEstimator:
    n = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    ssx = np.sum(x**2)
    ssy = np.sum(y**2)

    xy = x * y
    sxy = np.sum(xy)

    u1 = -4/n * sxy
    u2 = 2/n * (ssx + ssy)
    u3 = -4/(n*(n-1)) * (sx * sy - sxy)

    h = (n-1) * (u3 - u1)
    g = u1 + n*u2 + (n-1)*u3

    psi1 = (n-2)*xy + sxy/n
    psi2 = (n-2)*(x**2 - ssx / n + y**2 - ssy / n)
    psi3 = x * sy + y * sx - sx * sy / n + 2*xy + sxy/n

    v_u1 = 64*np.sum(psi1**2)/(n**2*(n-1)**2)
    v_u2 = 4*np.sum(psi2**2)/(n**2*(n-1)**2)
    v_u3 = 64*np.sum(psi3**2)/(n**2*(n-1)**2)
    cov_u1_u2 = -16*np.sum(psi1*psi2)/(n**2*(n-1)**2)
    cov_u1_u3 = 64*np.sum(psi1*psi3)/(n**2*(n-1)**2)   
    cov_u2_u3 = -16*np.sum(psi2*psi3)/(n**2*(n-1)**2)
    
    v_h = (n-1)**2 * (v_u3 + v_u1 - 2 * cov_u1_u3)
    v_g = v_u1 + n**2*v_u2 + (n-1)**2*v_u3 + 2*n*cov_u1_u2 + 2*(n-1)*cov_u1_u3 + 2*n*(n-1)*cov_u2_u3
    cov_h_g = (n-1)*(-(n-2)*cov_u1_u3 + n*cov_u2_u3 + (n-1)*v_u3 - v_u1 - n*cov_u1_u2)
    
    ccc_hat = h/g
    var_ccc_hat = ccc_hat**2 * (v_h / h**2 - 2*cov_h_g / (h*g) + v_g / g**2)
    var_z_hat = var_ccc_hat / (1 - ccc_hat**2)**2
    
    ccc = TransformedEstimator(
        estimate=ccc_hat, 
        variance=var_ccc_hat, 
        transformed_variance=var_z_hat,
        transformed_function=TransformFunc.Z,
        allowance=1-allowance_whitin_sample_deviation**2,
        robust=True,
        alpha=alpha, 
        confident_limit=ConfidentLimit.Lower, 
        n=n
    )
    return ccc
    
def _msd(x, y, alpha: float) -> TransformedEstimator:
    n = len(x)
    D = x - y
    mu_d = np.mean(D)

    eps_sq_hat = np.sum(D**2) / (n - 1)
    var_esp_hat = 2 / (n - 2) * ( 1 - mu_d**4 / eps_sq_hat**2)

    msd = TransformedEstimator(
        estimate=eps_sq_hat, 
        variance=var_esp_hat, 
        transformed_function=TransformFunc.Log,
        alpha=alpha, 
        confident_limit=ConfidentLimit.Upper, 
        n=n
    )
    return msd

def _rbs(x, y, cp_allowance: float) -> TransformedEstimator:
    n = len(x)
    D = x - y
    s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(x, y, bias=True).flatten()
    s_sq_hat_d = n / (n - 3) * (s_sq_hat_biased_x + s_sq_hat_biased_y - 2 * s_hat_biased_xy)
    mu_d = np.mean(D)

    rbs_hat = mu_d**2 / s_sq_hat_d
    rbs = TransformedEstimator(
        estimate=rbs_hat,
        allowance=rbs_allowance(cp_allowance)
    )
    return rbs

def _cp_approx(x, y, msd: TransformedEstimator, alpha: float, delta_criterion: float, cp_allowance: float) -> TransformedEstimator:
    n = len(x)
    D = x - y
    s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(x, y, bias=True).flatten()
    s_sq_hat_d = n / (n - 3) * (s_sq_hat_biased_x + s_sq_hat_biased_y - 2 * s_hat_biased_xy)
    mu_d = np.mean(D)

    delta_plus = (delta_criterion + mu_d) / np.sqrt(s_sq_hat_d)
    n_delta_plus = norm.pdf(-delta_plus)
    delta_minus = (delta_criterion - mu_d) / np.sqrt(s_sq_hat_d)
    n_delta_minus = norm.pdf(delta_minus)

    cp_hat = chi2.cdf(delta_criterion**2 / msd.estimate, df=1)
    var_cp_hat = 1/(n-3) * ((n_delta_plus - n_delta_minus)**2 + 0.5*(delta_minus*n_delta_minus + delta_plus*n_delta_plus))
    var_transform_cp_hat = var_cp_hat / ((1-cp_hat)**2*cp_hat**2)
    
    cp = TransformedEstimator(
        estimate=cp_hat, 
        variance=var_cp_hat, 
        transformed_variance=var_transform_cp_hat,
        transformed_function=TransformFunc.Logit,
        allowance=cp_allowance,
        alpha=alpha, 
        confident_limit=ConfidentLimit.Lower, 
        n=n
    )
    return cp

def _cp_exact(x, y, alpha: float, delta_criterion: float, cp_allowance: float) -> TransformedEstimator:
    n = len(x)
    D = x - y
    s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(x, y, bias=True).flatten()
    s_sq_hat_d = n / (n - 3) * (s_sq_hat_biased_x + s_sq_hat_biased_y - 2 * s_hat_biased_xy)
    mu_d = np.mean(D)

    delta_plus = (delta_criterion + mu_d) / np.sqrt(s_sq_hat_d)
    n_delta_plus = norm.pdf(-delta_plus)
    delta_minus = (delta_criterion - mu_d) / np.sqrt(s_sq_hat_d)
    n_delta_minus = norm.pdf(delta_minus)

    cp_hat = norm.cdf(delta_minus) - norm.cdf(-delta_plus)
    var_cp_hat = 1/(n-3) * ((n_delta_plus - n_delta_minus)**2 + 0.5*(delta_minus*n_delta_minus + delta_plus*n_delta_plus))
    var_transform_cp_hat = var_cp_hat / ((1-cp_hat)**2*cp_hat**2)
    
    cp = TransformedEstimator(
        estimate=cp_hat, 
        variance=var_cp_hat, 
        transformed_variance=var_transform_cp_hat,
        transformed_function=TransformFunc.Logit,
        allowance=cp_allowance,
        robust=True,
        alpha=alpha, 
        confident_limit=ConfidentLimit.Lower, 
        n=n
    )
    return cp

def _tdi_approx(msd: TransformedEstimator, pi_criterion: float, tdi_allowance: float)  -> TransformedEstimator:
    coeff_tdi = norm.ppf(1 - (1 - pi_criterion) / 2)
    tdi = TransformedEstimator(
        estimate=coeff_tdi * np.sqrt(msd.estimate), 
        limit=coeff_tdi * np.sqrt(msd.limit),
        allowance=tdi_allowance,
        robust=False,
        confident_limit=ConfidentLimit.Upper
    )
    return tdi
    
class agreement_index:
    def __init__(self, name: Indices):
        self._name = name

    def __call__(self, x, y, method="approx", alpha=DEFAULT_ALPHA, criterion=0.0, allowance=0.0) -> Estimator:
        '''
        Compute index estimate and its confident interval

        Parameters
        ----------
        x : array_like of float
            Target values.
        y : array_like of float
            Observation values, should have the same length as x.
        method : str, default: approx
            Method used to compute the index.
        alpha : float, default: 0.05
            Confident level used in confident interval computation.
        criterion : float, optional
            Criterion used in some index computation (CP and TDI).
        allowance : float, optional
            Allowance level to assert agreement.

        Returns
        -------
        Estimator
            dataclass storing estimate of index, its confident limit and allowance if given.

        Raises
        ------
        ValueError
            If wrong method is given.

        Examples
        --------
        >>> X = np.array([12, 10, 13, 10])
        >>> Y = np.array([11, 12, 16, 9])
        >>> sa.ccc(X, Y, method='approx', alpha=0.05, allowance=0.10)
        Estimator(estimate=0.5714285714285715, limit=-0.4247655971444191, allowance=0.99)
        '''
        
        if self._name == Indices.ccc:
            if method == "approx":
                # Lin LI. A concordance correlation coefficient to evaluate reproducibility. 
                # Biometrics. 1989 Mar;45(1):255-68. PMID: 2720055.
                rho = _precision(x, y, alpha)
                acc = _accuracy(x, y, rho, alpha)
                index = _ccc_lin(x, y, rho, acc, alpha, allowance)
            elif method == "ustat":
                # King TS, Chinchilli VM. 
                # Robust estimators of the concordance correlation coefficient. 
                # J Biopharm Stat. 2001;11(3):83-105. doi: 10.1081/BIP-100107651. PMID: 11725932.
                index = _ccc_ustat(x, y, alpha, allowance)
            else:
                raise ValueError("Wrong method called for ccc computation, current possible methods are approx or ustat.")
        elif self._name == Indices.cp:
            msd = _msd(x, y, alpha)
            if method == "approx":
                index = _cp_approx(x, y, msd, alpha, criterion, allowance)
            elif method == "exact":
                index = _cp_exact(x, y, alpha, criterion, allowance)
            else:
                raise ValueError("Wrong method called for cp computation, current possible methods are approx or exact.")
        elif self._name == Indices.tdi:
            if method == "approx":
                msd = _msd(x, y, alpha)
                index = _tdi_approx(msd, criterion, allowance)
            else:
                raise ValueError("Wrong method called for tdi computation, current possible methods are approx.")
        return index.as_estimator()

ccc = agreement_index(name=Indices.ccc)
cp = agreement_index(name=Indices.cp)
tdi = agreement_index(name=Indices.tdi)

def agreement(x, y, delta_criterion, pi_criterion, alpha=DEFAULT_ALPHA,
              allowance_whitin_sample_deviation=WITHIN_SAMPLE_DEVIATION,
              cp_allowance=CP_ALLOWANCE,
              tdi_allowance=TDI_ALLOWANCE,
              log=False,
              display=False):
    
    flag = FlagData.Data_Ok

    if log:
        if np.sum(x<=0) + np.sum(y<=0) > 0:
            flag = FlagData.Negative
            raise ValueError("Input data are not positive for a log transformation")
        else:
            x=np.log(x)
            y=np.log(y)
            delta_criterion=np.log(1+delta_criterion/100)

    res = pd.DataFrame(columns=["estimate", "limit", "variance", "transformed_function",
                                "transformed_estimate", "transformed_variance", "allowance",
                                "robust"])
    
    if np.var(x)==0 or np.var(y)==0:
        flag = FlagData.Constant
        warnings.warn("Input values are constant, can't compute ccc-related indexes")

    if flag != FlagData.Constant:
        rho = _precision(x, y, alpha)
        acc = _accuracy(x, y, rho, alpha)
        ccc_lin = _ccc_lin(x, y, rho, acc, alpha, allowance_whitin_sample_deviation)
        ccc_ustat = _ccc_ustat(x, y, alpha, allowance_whitin_sample_deviation)

    msd = _msd(x, y, alpha)
    rbs = _rbs(x, y, cp_allowance)
    cp_approx = _cp_approx(x, y, msd, alpha, delta_criterion, cp_allowance)
    cp = _cp_exact(x, y, alpha, delta_criterion, cp_allowance)
    tdi = _tdi_approx(msd, pi_criterion, tdi_allowance)

    res.loc["acc", :] = acc.to_series()
    res.loc["rho", :] = rho.to_series()
    res.loc["ccc", :] = ccc_lin.to_series()
    res.loc["ccc_robust", :] = ccc_ustat.to_series()
    res.loc["msd", :] = msd.to_series()
    res.loc["rbs", :] = rbs.to_series()

    if flag != FlagData.Negative:
        res.loc["cp_approx", :] = cp_approx.to_series()
        res.loc["cp", :] = cp.to_series()
        res.loc["tdi", :] = tdi.to_series()
        res.loc["cp_approx", "criterion"] = delta_criterion
        res.loc["cp", "criterion"] = delta_criterion
        res.loc["tdi", "criterion"] = pi_criterion
    
    if display:
        print(res)
        print(shapiro(x - y))

    return flag, res