import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

SEED = 42


def generate_data(alpha_prop, beta_Z, beta_A, N=20_000, D_Z=3, y_noise=0.5, seed=SEED):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, D_Z))
    prop_true = 1.0 / (1.0 + np.exp(-(Z @ alpha_prop)))
    A = rng.binomial(1, prop_true).astype(int)
    Y = Z @ beta_Z + beta_A * A + y_noise * rng.standard_normal(N)
    X = np.hstack([Z, A[:, None]]).astype(float)
    return Z, A, Y, X, prop_true


def train(X, Y, A, D_Z, eps, T=8000, lr_p=0.003, lr_d=0.02):
    N, D  = X.shape
    p1    = float(A.mean())
    theta = np.zeros(D)
    mu    = 0.0

    E_Z    = X[:, :D_Z].mean(axis=0)
    E_X    = np.concatenate([E_Z, [p1]])
    E_do_X = np.concatenate([E_Z, [1.0]])

    for _ in range(T):
        f = X @ theta
        grad_pred = -2.0 / N * X.T @ (Y - f)
        discrep   = theta @ (E_X - E_do_X)
        phi       = abs(discrep)
        s_star    = np.sign(discrep) if abs(discrep) > 1e-12 else 1.0
        grad_phi  = s_star * (E_X - E_do_X)
        theta = theta - lr_p * (grad_pred + mu * grad_phi)
        mu    = max(0.0, mu + lr_d * (phi - eps))

    return theta, mu


def aipw_diagnostic(X, A, Y, theta, Z_use, mu_opt,
                    propensity="hgb", clip=(0.02, 0.98), name="",
                    bootstrap=True, B=100):
    n, D = X.shape
    f_eval = X @ theta

    ell_i   = -2.0 * (Y - f_eval)[:, None] * X
    ell_hat = ell_i.mean(axis=0)
    grad_tr = X.copy()

    e_hat   = np.zeros(n)
    nabla_m = np.zeros((n, D))
    m_f     = np.zeros(n)

    kf = KFold(5, shuffle=True, random_state=SEED)
    for tr_idx, te_idx in kf.split(Z_use):
        Z_tr_f, Z_te_f = Z_use[tr_idx], Z_use[te_idx]
        A_tr_f         = A[tr_idx]
        sub            = (A_tr_f == 1)

        if propensity == "hgb":
            ps = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
            ps.fit(Z_tr_f, A_tr_f)
            e_hat[te_idx] = ps.predict_proba(Z_te_f)[:, 1]
        elif propensity == "constant":
            e_hat[te_idx] = A_tr_f.mean()
        else:
            raise ValueError(propensity)

        rf_g = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=15,
            n_jobs=-1, random_state=SEED,
        )
        rf_g.fit(Z_tr_f[sub], grad_tr[tr_idx][sub])
        nabla_m[te_idx] = rf_g.predict(Z_te_f)

        rf_f = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=15,
            n_jobs=-1, random_state=SEED,
        )
        rf_f.fit(Z_tr_f[sub], f_eval[tr_idx][sub])
        m_f[te_idx] = rf_f.predict(Z_te_f)

    n_clipped = int(((e_hat < clip[0]) | (e_hat > clip[1])).sum())
    e_c       = np.clip(e_hat, clip[0], clip[1])
    ind       = A.astype(float)

    aipw_grad_i = (ind / e_c)[:, None] * (grad_tr - nabla_m) + nabla_m
    aipw_f_i    = (ind / e_c) * (f_eval - m_f) + m_f

    g_hat  = grad_tr.mean(0) - aipw_grad_i.mean(0)
    diff_f = f_eval.mean()   - aipw_f_i.mean()
    s_star = np.sign(diff_f) if abs(diff_f) > 1e-12 else 1.0

    norm_g_sq = float(g_hat @ g_hat)
    mu_dr     = -s_star * (ell_hat @ g_hat) / norm_g_sq

    U_i   = ell_i
    V_i   = grad_tr - aipw_grad_i
    term1 = (U_i - ell_hat) @ g_hat / norm_g_sq
    mix   = ell_hat + 2.0 * s_star * mu_dr * g_hat
    term2 = (V_i - g_hat) @ mix / norm_g_sq
    psi_i = -s_star * (term1 + term2)
    if_se = float(psi_i.std(ddof=1)) / np.sqrt(n)
    z     = (mu_opt - mu_dr) / if_se

    bs_se = bootstrap_se(X, A, Y, theta, Z_use, propensity, clip, B) if bootstrap else None

    return {
        "name":   name,
        "mu_opt": mu_opt,
        "mu_dr":  mu_dr,
        "if_se":  if_se,
        "bs_se":  bs_se,
        "z":      z,
        "clip":   clip,
        "n_clipped": n_clipped,
        "e_min":  float(e_hat.min()),
        "e_max":  float(e_hat.max()),
    }


def bootstrap_se(X, A, Y, theta, Z_use, propensity, clip, B):
    n, D = X.shape
    boot_mus = []
    rng = np.random.default_rng(SEED + 1000)

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        X_b, A_b, Y_b, Z_b = X[idx], A[idx], Y[idx], Z_use[idx]
        sub = (A_b == 1)
        if not sub.any() or sub.all():
            continue

        if propensity == "hgb":
            ps = LogisticRegression(max_iter=500, C=1.0)
            ps.fit(Z_b, A_b)
            e_b = ps.predict_proba(Z_b)[:, 1]
        elif propensity == "constant":
            e_b = np.full(n, A_b.mean())
        else:
            raise ValueError(propensity)

        f_b    = X_b @ theta
        grad_b = X_b

        rg = Ridge(alpha=1.0)
        rg.fit(Z_b[sub], grad_b[sub])
        nabla_m_b = rg.predict(Z_b)

        rf = Ridge(alpha=1.0)
        rf.fit(Z_b[sub], f_b[sub])
        m_f_b = rf.predict(Z_b)

        e_c = np.clip(e_b, clip[0], clip[1])
        ind = A_b.astype(float)
        aipw_grad_b = (ind / e_c)[:, None] * (grad_b - nabla_m_b) + nabla_m_b
        aipw_f_b    = (ind / e_c) * (f_b - m_f_b) + m_f_b

        g_b    = grad_b.mean(0) - aipw_grad_b.mean(0)
        diff_b = f_b.mean()     - aipw_f_b.mean()
        s_b    = np.sign(diff_b) if abs(diff_b) > 1e-12 else 1.0

        ell_b     = -2.0 * (Y_b - f_b)[:, None] * X_b
        ell_hat_b = ell_b.mean(0)

        norm_sq = float(g_b @ g_b)
        if norm_sq < 1e-12:
            continue
        boot_mus.append(-s_b * (ell_hat_b @ g_b) / norm_sq)

    return float(np.std(boot_mus, ddof=1)) if len(boot_mus) > 1 else float("nan")


# ============================================================
# DGP A
# ============================================================
print("=" * 70)
print("DGP A: alpha_prop=[0.8, -0.5, 0.4], good overlap")
print("=" * 70)

Z_a, A_a, Y_a, X_a, prop_a = generate_data(
    alpha_prop=np.array([0.8, -0.5, 0.4]),
    beta_Z=np.array([1.0, -0.5, 0.7]),
    beta_A=1.2,
)
print(f"P(A=1)={A_a.mean():.3f}")
print(f"true propensity range: [{prop_a.min():.4f}, {prop_a.max():.4f}]")

theta_a, mu_opt_a = train(X_a, Y_a, A_a, D_Z=3, eps=0.30)
print(f"theta_A*={theta_a[3]:+.4f}, mu_opt={mu_opt_a:.4f}")

results = []
results.append(aipw_diagnostic(X_a, A_a, Y_a, theta_a, Z_a, mu_opt_a,
                                propensity="hgb",      clip=(0.02, 0.98),
                                name="(1) correct"))
results.append(aipw_diagnostic(X_a, A_a, Y_a, theta_a, Z_a[:, [0, 2]], mu_opt_a,
                                propensity="hgb",      clip=(0.02, 0.98),
                                name="(2) Z_1 omitted"))
results.append(aipw_diagnostic(X_a, A_a, Y_a, theta_a, Z_a, mu_opt_a,
                                propensity="constant", clip=(0.02, 0.98),
                                name="(3) propensity misspec"))


# ============================================================
# DGP B
# ============================================================
print("\n" + "=" * 70)
print("DGP B: alpha_prop=[3.5, -2.5, 1.8], overlap problem")
print("=" * 70)

Z_b, A_b, Y_b, X_b, prop_b = generate_data(
    alpha_prop=np.array([3.5, -2.5, 1.8]),
    beta_Z=np.array([1.0, -0.5, 0.7]),
    beta_A=1.2,
)
print(f"P(A=1)={A_b.mean():.3f}")
print(f"true propensity range: [{prop_b.min():.4f}, {prop_b.max():.4f}]")
print(f"true propensity outside [0.02, 0.98]: "
      f"{int(((prop_b < 0.02) | (prop_b > 0.98)).sum())}/{len(prop_b)}")
print(f"true propensity outside [0.001, 0.999]: "
      f"{int(((prop_b < 0.001) | (prop_b > 0.999)).sum())}/{len(prop_b)}")

theta_b, mu_opt_b = train(X_b, Y_b, A_b, D_Z=3, eps=0.30)
print(f"theta_A*={theta_b[3]:+.4f}, mu_opt={mu_opt_b:.4f}")

results.append(aipw_diagnostic(X_b, A_b, Y_b, theta_b, Z_b, mu_opt_b,
                                propensity="hgb", clip=(0.02, 0.98),
                                name="(4) overlap trimmed"))
results.append(aipw_diagnostic(X_b, A_b, Y_b, theta_b, Z_b, mu_opt_b,
                                propensity="hgb", clip=(0.001, 0.999),
                                name="(5) overlap full"))


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 95)
print("SUMMARY (Table 1)")
print("=" * 95)
hdr = (f"{'Condition':<27} {'mu_opt':>8} {'mu_DR':>8} "
       f"{'IF SE':>7} {'BS SE':>7} {'Z':>8}  {'clipped':>9}  Flag")
print(hdr)
print("-" * 95)
for r in results:
    flag = "Yes" if abs(r["z"]) > 1.96 else "No"
    if abs(r["z"]) > 5:
        flag = "Yes (strong)"
    bs = f"{r['bs_se']:.4f}" if r['bs_se'] is not None and not np.isnan(r['bs_se']) else "  n/a"
    print(f"{r['name']:<27} {r['mu_opt']:>+8.4f} {r['mu_dr']:>+8.4f} "
          f"{r['if_se']:>7.4f} {bs:>7} {r['z']:>+8.3f}  "
          f"{r['n_clipped']:>9d}  {flag}")
