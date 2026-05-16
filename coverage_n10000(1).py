"""
coverage_n10000.py
==================

Coverage-Studie fuer Algorithmus 1 bei n=10000 mit 50 Replikationen.
Berechnet parallel IF-SE und Sandwich-SE.

Drittes n in der Asymptotik-Sequenz (nach n=2000 und n=5000). Erwartung:
  - Sandwich-Coverage bleibt nahe 0.95
  - Ratio empirical_SD / IF_SE faellt weiter (Trend: 1.85 -> 1.62 -> ?)
  - Bias schrumpft weiter (Trend: -0.025 -> -0.019 -> ?)

Laufzeit auf Colab CPU: ca. 50-75 Min.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


# =====================================================================
# DGP A
# =====================================================================
def generate_data(n, seed, alpha_prop=np.array([0.8, -0.5, 0.4])):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 3))
    prop_true = 1.0 / (1.0 + np.exp(-(Z @ alpha_prop)))
    A = rng.binomial(1, prop_true).astype(int)
    beta_Z = np.array([1.0, -0.5, 0.7])
    beta_A = 1.2
    Y = Z @ beta_Z + beta_A * A + 0.5 * rng.standard_normal(n)
    X = np.hstack([Z, A[:, None]]).astype(float)
    return Z, A, Y, X


# =====================================================================
# Referenz mu* via Oracle bei sehr grossem n
# =====================================================================
def reference_mu_star(n_ref=200_000, seed=999, eps=0.30, T=20_000,
                      lr_p=0.003, lr_d=0.02):
    Z, A, Y, X = generate_data(n_ref, seed)
    D = X.shape[1]
    p1 = float(A.mean())
    E_X = np.concatenate([Z.mean(0), [p1]])
    E_do = np.concatenate([Z.mean(0), [1.0]])
    theta = np.zeros(D)
    mu = 0.0
    mu_trace = []
    for _ in range(T):
        f = X @ theta
        grad_pred = -2.0 / n_ref * X.T @ (Y - f)
        d = theta @ (E_X - E_do)
        s = np.sign(d) if abs(d) > 1e-12 else 1.0
        grad_phi = s * (E_X - E_do)
        theta = theta - lr_p * (grad_pred + mu * grad_phi)
        mu = max(0.0, mu + lr_d * (abs(d) - eps))
        mu_trace.append(mu)
    return float(np.mean(mu_trace[T // 2:]))


# =====================================================================
# Algorithmus 1: AIPW-korrigierter Primal-Dual (linear)
# =====================================================================
def aipw_solver_linear(Z, A, Y, X, eps=0.30,
                       T=6_000, T0=1_000, lr_p=0.003, lr_d=0.02,
                       K=5, seed=0, clip=(0.02, 0.98)):
    n, D = X.shape

    # Phase 1: Warm-up
    theta = np.zeros(D)
    for _ in range(T0):
        f = X @ theta
        grad_pred = -2.0 / n * X.T @ (Y - f)
        theta = theta - lr_p * grad_pred

    # Phase 2: cross-fitted Nuisances
    e_hat = np.zeros(n)
    nabla_m = np.zeros((n, D))
    kf = KFold(K, shuffle=True, random_state=seed)
    for tr_idx, te_idx in kf.split(Z):
        sub = (A[tr_idx] == 1)
        ps = LogisticRegression(max_iter=500)
        ps.fit(Z[tr_idx], A[tr_idx])
        e_hat[te_idx] = ps.predict_proba(Z[te_idx])[:, 1]
        rf_g = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=15,
            n_jobs=-1, random_state=seed)
        rf_g.fit(Z[tr_idx][sub], X[tr_idx][sub])
        nabla_m[te_idx] = rf_g.predict(Z[te_idx])

    e_c = np.clip(e_hat, clip[0], clip[1])
    ind = A.astype(float)
    aipw_grad_i = (ind / e_c)[:, None] * (X - nabla_m) + nabla_m
    g_dr = X.mean(0) - aipw_grad_i.mean(0)

    # Phase 3: Refinement
    mu = 0.0
    theta_trace, mu_trace = [], []
    for _ in range(T - T0):
        f = X @ theta
        grad_pred = -2.0 / n * X.T @ (Y - f)
        m_f = nabla_m @ theta
        aipw_f = (ind / e_c) * (f - m_f) + m_f
        D_hat = f.mean() - aipw_f.mean()
        s_t = np.sign(D_hat) if abs(D_hat) > 1e-12 else 1.0
        phi_t = abs(D_hat)
        theta = theta - lr_p * (grad_pred + mu * s_t * g_dr)
        mu = max(0.0, mu + lr_d * (phi_t - eps))
        theta_trace.append(theta.copy())
        mu_trace.append(mu)

    half = (T - T0) // 2
    bar_theta = np.mean(theta_trace[half:], axis=0)
    bar_mu = float(np.mean(mu_trace[half:]))

    return {
        "bar_theta": bar_theta,
        "bar_mu": bar_mu,
        "e_hat": e_hat,
        "nabla_m": nabla_m,
    }


# =====================================================================
# Beide SEs (IF + Sandwich)
# =====================================================================
def both_ses(theta, mu_hat, Z, A, X, Y, reuse_nuisances, eps=0.30,
             clip=(0.02, 0.98)):
    n, D = X.shape
    f_eval = X @ theta
    e_hat, nabla_m = reuse_nuisances
    e_c = np.clip(e_hat, clip[0], clip[1])
    ind = A.astype(float)

    U_i = -2.0 * (Y - f_eval)[:, None] * X
    aipw_grad_i = (ind / e_c)[:, None] * (X - nabla_m) + nabla_m
    V_i = X - aipw_grad_i
    m_f = nabla_m @ theta
    aipw_f_i = (ind / e_c) * (f_eval - m_f) + m_f
    D_i = f_eval - aipw_f_i

    ell_hat = U_i.mean(0)
    g_hat = V_i.mean(0)
    D_bar = D_i.mean()
    s_hat = np.sign(D_bar) if abs(D_bar) > 1e-12 else 1.0
    norm_g_sq = float(g_hat @ g_hat)

    mu_dr = float(-s_hat * (ell_hat @ g_hat) / norm_g_sq)

    # IF-SE (Oracle-Form, v9 eq. (15))
    term1 = (U_i - ell_hat) @ g_hat / norm_g_sq
    mix = ell_hat + 2.0 * s_hat * mu_dr * g_hat
    term2 = (V_i - g_hat) @ mix / norm_g_sq
    psi_if = -s_hat * (term1 + term2)
    if_se = float(psi_if.std(ddof=1)) / np.sqrt(n)

    # Sandwich-SE (Joint M-Estimator, v9 eq. (12)-(14))
    H_L = (2.0 / n) * (X.T @ X)
    h = np.linalg.solve(H_L, g_hat)
    k = float(g_hat @ h)
    if abs(k) < 1e-12:
        sand_se = float("nan")
    else:
        psi_theta_i = U_i + mu_hat * s_hat * V_i
        psi_mu_i = D_i - s_hat * eps
        psi_theta_c = psi_theta_i - psi_theta_i.mean(0)
        psi_mu_c = psi_mu_i - psi_mu_i.mean()
        Sigma_tt = (psi_theta_c.T @ psi_theta_c) / n
        Sigma_tm = (psi_theta_c.T @ psi_mu_c) / n
        Sigma_mm = float((psi_mu_c ** 2).mean())
        var_joint = (h @ Sigma_tt @ h - 2.0 * (h @ Sigma_tm) + Sigma_mm) / (k * k)
        sand_se = float(np.sqrt(max(var_joint, 0.0) / n))

    return mu_dr, if_se, sand_se


# =====================================================================
# Coverage-Studie
# =====================================================================
def coverage_study(n, n_reps, mu_star, T=6_000, T0=1_000, K=5):
    print("=" * 72)
    print(f"Coverage-Studie: n={n}, n_reps={n_reps}")
    print(f"Referenz mu* = {mu_star:.4f}")
    print("=" * 72)

    cov_if = 0
    cov_sand = 0
    mus, if_ses, sand_ses, biases = [], [], [], []
    fails = 0

    for r in range(n_reps):
        try:
            Z, A, Y, X = generate_data(n, seed=10_000 + r)
            sol = aipw_solver_linear(Z, A, Y, X, T=T, T0=T0, K=K, seed=r)
            mu_dr, if_se, sand_se = both_ses(
                sol["bar_theta"], sol["bar_mu"], Z, A, X, Y,
                reuse_nuisances=(sol["e_hat"], sol["nabla_m"]),
            )

            lo_if, hi_if = mu_dr - 1.96 * if_se, mu_dr + 1.96 * if_se
            lo_s, hi_s = mu_dr - 1.96 * sand_se, mu_dr + 1.96 * sand_se
            cov_if += int(lo_if <= mu_star <= hi_if)
            cov_sand += int(lo_s <= mu_star <= hi_s)

            mus.append(mu_dr)
            if_ses.append(if_se)
            sand_ses.append(sand_se)
            biases.append(mu_dr - mu_star)

            if (r + 1) % 5 == 0:
                print(f"  rep {r+1}/{n_reps}: cov(IF)={cov_if/(r+1):.3f}, "
                      f"cov(Sand)={cov_sand/(r+1):.3f}, "
                      f"bias={np.mean(biases):+.4f}", flush=True)
        except Exception as e:
            fails += 1
            print(f"  rep {r} failed: {e}")
            continue

    n_ok = n_reps - fails
    cov_if_final = cov_if / max(n_ok, 1)
    cov_sand_final = cov_sand / max(n_ok, 1)
    mean_bias = float(np.mean(biases))
    rmse = float(np.sqrt(np.mean(np.array(biases) ** 2)))
    mean_if = float(np.mean(if_ses))
    mean_sand = float(np.mean(sand_ses))
    emp_sd = float(np.std(mus, ddof=1))

    print()
    print(f"--- Ergebnisse ueber {n_ok} erfolgreiche Reps ---")
    print(f"  Coverage IF-SE       (Ziel 0.95): {cov_if_final:.3f}")
    print(f"  Coverage Sandwich-SE (Ziel 0.95): {cov_sand_final:.3f}")
    print(f"  Mean bias (mu_hat - mu*):         {mean_bias:+.4f}")
    print(f"  RMSE:                             {rmse:.4f}")
    print(f"  Mean IF SE:                       {mean_if:.4f}")
    print(f"  Mean Sandwich SE:                 {mean_sand:.4f}")
    print(f"  Empirical SD across reps:         {emp_sd:.4f}")
    print(f"  Ratio emp_SD / IF_SE:             {emp_sd / mean_if:.3f}")
    print(f"  Ratio emp_SD / Sandwich_SE:       {emp_sd / mean_sand:.3f}")
    print()
    print(f"Trend ueber n=2000, 5000, 10000:")
    print(f"  Erwartung: IF-Coverage waechst langsam, bleibt unter 0.95;")
    print(f"             Sandwich-Coverage bleibt nahe 0.95;")
    print(f"             Bias und IF-Ratio fallen monoton.")

    return {
        "n": n, "cov_if": cov_if_final, "cov_sand": cov_sand_final,
        "mean_bias": mean_bias, "rmse": rmse,
        "mean_if": mean_if, "mean_sand": mean_sand, "emp_sd": emp_sd,
    }


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print("Coverage-Studie bei n=10000 (drittes n in der Sequenz)\n")
    print("Schritt 1: Referenz mu* via Oracle bei n=200000 ...")
    mu_star = reference_mu_star()
    print(f"  mu* = {mu_star:.4f}\n")

    print("Schritt 2: Coverage-Studie bei n=10000 mit 50 Replikationen...")
    res = coverage_study(n=10_000, n_reps=50, mu_star=mu_star,
                          T=6_000, T0=1_000, K=5)
