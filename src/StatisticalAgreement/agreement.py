# Copyright (C) 2023 Rémy Cases
# See LICENSE file for extended copyright information.
# This file is part of StatisticalAgreement project from https://github.com/remyCases/StatiscalAgreement.

# Based on Lin, L. I.-K. (2000). 
## Total deviation index for measuring individual agreement with applications in 
## laboratory performance and bioequivalence. Statistics in Medicine, 19(2), 255–270

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, shapiro
from .classutils import TransformFunc, ConfidentLimit, TransformedEstimator, Estimator

ALPHA = 0.05
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
        estimator=rho_hat, 
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
    rho_hat = precision.estimator

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
        estimator=acc_hat, 
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
    rho_hat = precision.estimator

    s_sq_hat_biased_x, _, _, s_sq_hat_biased_y = np.cov(x, y, bias=True).flatten()
    sqr_var = np.sqrt(s_sq_hat_biased_x * s_sq_hat_biased_y)
    nu_sq_hat = mu_d**2 / sqr_var

    ccc_hat = rho_hat * accuracy.estimator
    var_ccc_hat = 1 / (n - 2) * ((1-rho_hat**2)*ccc_hat**2/rho_hat**2
                                 + 2*ccc_hat**3*(1-ccc_hat)*nu_sq_hat / (rho_hat*(1-ccc_hat**2))
                                 - ccc_hat**4 * nu_sq_hat**2 / (2*rho_hat**2*(1-ccc_hat**2)))
    var_z_hat = var_ccc_hat / (1-ccc_hat**2)

    ccc = TransformedEstimator(
        estimator=ccc_hat, 
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
        estimator=ccc_hat, 
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
    s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(x, y, bias=True).flatten()
    s_sq_hat_d = n / (n - 3) * (s_sq_hat_biased_x + s_sq_hat_biased_y - 2 * s_hat_biased_xy)
    mu_d = np.mean(D)

    eps_sq_hat = np.sum(D**2) / (n - 1)
    var_esp_hat = 2 / (n - 2) * ( 1 - mu_d**4 / eps_sq_hat**2)

    msd = TransformedEstimator(
        estimator=eps_sq_hat, 
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
        estimator=rbs_hat,
        allowance=rbs_allowance(cp_allowance)
    )
    return rbs

def _cp_approx(x, y, msd: TransformedEstimator, alpha: float, delta_criterion: float, cp_allowance: float):
    n = len(x)
    D = x - y
    s_sq_hat_biased_x, s_hat_biased_xy, _, s_sq_hat_biased_y = np.cov(x, y, bias=True).flatten()
    s_sq_hat_d = n / (n - 3) * (s_sq_hat_biased_x + s_sq_hat_biased_y - 2 * s_hat_biased_xy)
    mu_d = np.mean(D)

    delta_plus = (delta_criterion + mu_d) / np.sqrt(s_sq_hat_d)
    n_delta_plus = norm.pdf(-delta_plus)
    delta_minus = (delta_criterion - mu_d) / np.sqrt(s_sq_hat_d)
    n_delta_minus = norm.pdf(delta_minus)

    cp_hat = chi2.cdf(delta_criterion**2 / msd.estimator, df=1)
    var_cp_hat = 1/(n-3) * ((n_delta_plus - n_delta_minus)**2 + 0.5*(delta_minus*n_delta_minus + delta_plus*n_delta_plus))
    var_transform_cp_hat = var_cp_hat / ((1-cp_hat)**2*cp_hat**2)
    
    cp = TransformedEstimator(
        estimator=cp_hat, 
        variance=var_cp_hat, 
        transformed_variance=var_transform_cp_hat,
        transformed_function=TransformFunc.Logit,
        allowance=cp_allowance,
        alpha=alpha, 
        confident_limit=ConfidentLimit.Lower, 
        n=n
    )
    return cp

def _cp_exact(x, y, alpha: float, delta_criterion: float, cp_allowance: float):
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
        estimator=cp_hat, 
        variance=var_cp_hat, 
        transformed_variance=var_transform_cp_hat,
        transformed_function=TransformFunc.Logit,
        allowance=cp_allowance,
        alpha=alpha, 
        confident_limit=ConfidentLimit.Lower, 
        n=n
    )
    return cp

def _tdi_approx(x, y, msd: TransformedEstimator, pi_criterion: float, tdi_allowance: float):
    coeff_tdi = norm.ppf(1 - (1 - pi_criterion) / 2)
    tdi = Estimator(
        estimator=coeff_tdi * np.sqrt(msd.estimator),
        limit=coeff_tdi * np.sqrt(msd.limit),
        allowance=tdi_allowance
    )
    return tdi
    
def ccc(x, y, method="approx", alpha=ALPHA, allowance_whitin_sample_deviation=WITHIN_SAMPLE_DEVIATION) -> Estimator:
    if method == "approx":
        # Lin LI. A concordance correlation coefficient to evaluate reproducibility. 
        # Biometrics. 1989 Mar;45(1):255-68. PMID: 2720055.
        rho = _precision(x, y, alpha)
        acc = _accuracy(x, y, rho, alpha)
        ccc = _ccc_lin(x, y, rho, acc, alpha, allowance_whitin_sample_deviation)
        return ccc.as_estimator()
    elif method == "ustat":
        # King TS, Chinchilli VM. 
        # Robust estimators of the concordance correlation coefficient. 
        # J Biopharm Stat. 2001;11(3):83-105. doi: 10.1081/BIP-100107651. PMID: 11725932.
        ccc = _ccc_ustat(x, y, alpha, allowance_whitin_sample_deviation)
        return ccc.as_estimator()
    else:
        raise("Wrong method called for ccc computation")

def cp(x, y, delta_criterion: float, method="approx", alpha=ALPHA, cp_allowance=CP_ALLOWANCE) -> Estimator:
    msd = _msd(x, y, alpha)
    if method == "approx":
        cp = _cp_approx(x, y, msd, alpha, delta_criterion, cp_allowance).as_estimator()
        return cp
    elif method == "exact":
        cp = _cp_exact(x, y, alpha, delta_criterion, cp_allowance).as_estimator()
        return cp
    else:
        raise("Wrong method called for cp computation")

def tdi(x, y, pi_criterion: float, alpha=ALPHA, tdi_allowance=TDI_ALLOWANCE) -> Estimator:
    msd = _msd(x, y, alpha)
    print(msd)
    return _tdi_approx(x, y, msd, pi_criterion, tdi_allowance)

def agreement(x, y, delta_criterion, pi_criterion, alpha=ALPHA,
              allowance_whitin_sample_deviation=WITHIN_SAMPLE_DEVIATION,
              cp_allowance=CP_ALLOWANCE,
              tdi_allowance=TDI_ALLOWANCE,
              display=False):
    
    res = pd.DataFrame(columns=["estimator", "limit", "variance", "transformed_function",
                                "transformed_estimator", "transformed_variance", "allowance",
                                "robust"])
    
    rho = _precision(x, y, alpha)
    acc = _accuracy(x, y, rho, alpha)
    ccc_lin = _ccc_lin(x, y, rho, acc, alpha, allowance_whitin_sample_deviation)
    ccc_ustat = _ccc_ustat(x, y, alpha, allowance_whitin_sample_deviation)

    msd = _msd(x, y, alpha)
    rbs = _rbs(x, y, cp_allowance)
    cp_approx = _cp_approx(x, y, msd, alpha, delta_criterion, cp_allowance).as_estimator()
    cp = _cp_exact(x, y, alpha, delta_criterion, cp_allowance).as_estimator()
    tdi = _tdi_approx(x, y, msd, pi_criterion, tdi_allowance)

    res.loc["acc", :] = acc.to_series()
    res.loc["rho", :] = rho.to_series()
    res.loc["ccc", :] = ccc_lin.to_series()
    res.loc["ccc_robust", :] = ccc_ustat.to_series()
    res.loc["msd", :] = msd.to_series()
    res.loc["rbs", :] = rbs.to_series()
    res.loc["cp_approx", :] = cp_approx.to_series()
    res.loc["cp", :] = cp.to_series()
    res.loc["tdi", :] = tdi.to_series()

    res.loc["cp_approx", "criterion"] = delta_criterion
    res.loc["cp", "criterion"] = delta_criterion
    res.loc["tdi", "criterion"] = pi_criterion
    
    if display:
        print(res)
        print(shapiro(x - y))