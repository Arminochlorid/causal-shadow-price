"""
algorithm1_validation.py
========================

Empirical validation of Algorithm 1 (AIPW-corrected primal-dual) from
"The Causal Shadow Price" (Yousefi 2026, v8).

This script complements the original synthetic_diagnostic(1).py, which
validates the AIPW DIAGNOSTIC under oracle training. Here we validate
the AIPW SOLVER itself: training-time gradients computed via AIPW with
cross-fitted nuisances, not closed form.

Three experiments:
  EXP A: Identity check. Does bar_mu (solver) ≈ mu_hat_DR(bar_theta)
         (data-driven post-hoc estimator)? Tests Proposition 4 in v8.
  EXP B: Coverage study over many seeds. Does the IF-based 95% CI
         attain nominal coverage of the population mu*?
  EXP C: Nonlinear variant. Small 2-layer MLP with periodic refits of
         the gradient regression. Single-seed sanity check.

All defaults are tuned for a modest Colab run (~30-60 minutes).
Scale up n / n_reps for stronger evidence; scale down for quick checks.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# =====================================================================
# DGP -- identical to Section 10, DGP A (good overlap)
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
# Reference mu*: oracle solver at very large n
# =====================================================================
def reference_mu_star(n_ref=200_000, seed=999, eps=0.30, T=20_000,
                      lr_p=0.003, lr_d=0.02):
    """Population mu* estimated by running the oracle primal-dual at large n.

    Uses closed-form interventional means (linear DGP), so deviation from
    the true population mu* is O(n_ref^{-1/2}) ≈ 0.002 at n_ref=2e5.
    """
    Z, A, Y, X = generate_data(n_ref, seed)
    D = X.shape[1]
    p1 = float(A.mean())
    E_X = np.concatenate([Z.mean(0), [p1]])
    E_do = np.concatenate([Z.mean(0), [1.0]])
    theta = np.zeros(D)
    mu = 0.0
    mu_trace = []
    for t in range(T):
        f = X @ theta
        grad_pred = -2.0 / n_ref * X.T @ (Y - f)
        d = theta @ (E_X - E_do)
        s = np.sign(d) if abs(d) > 1e-12 else 1.0
        grad_phi = s * (E_X - E_do)
        theta = theta - lr_p * (grad_pred + mu * grad_phi)
        mu = max(0.0, mu + lr_d * (abs(d) - eps))
        mu_trace.append(mu)
    # average over second half (past transient)
    return float(np.mean(mu_trace[T // 2:]))


# =====================================================================
# Algorithm 1 -- AIPW-corrected primal-dual (linear model)
# =====================================================================
def aipw_solver_linear(Z, A, Y, X, eps=0.30,
                       T=8_000, T0=1_000, lr_p=0.003, lr_d=0.02,
                       K=5, seed=0, clip=(0.02, 0.98),
                       propensity="hgb"):
    """
    Algorithm 1 for the linear model f_theta(X) = theta^T X.

    Phase 1: unconstrained warm-up to obtain theta^(0).
    Phase 2: K-fold cross-fit nuisances (e_hat, nabla_m, m_f-template)
             at theta^(0). For linear f, nabla theta f = X is theta-free,
             so the gradient regression is also theta-free; m_a(z; theta)
             is theta-LINEAR so we can store nabla_m once and compute
             m_f(z; theta_t) = nabla_m(z) @ theta_t on the fly.
    Phase 3: constrained refinement using AIPW gradients.
    """
    n, D = X.shape

    # ---- Phase 1 ----
    theta = np.zeros(D)
    for _ in range(T0):
        f = X @ theta
        grad_pred = -2.0 / n * X.T @ (Y - f)
        theta = theta - lr_p * grad_pred
    theta_warm = theta.copy()

    # ---- Phase 2 ----
    e_hat = np.zeros(n)
    nabla_m = np.zeros((n, D))         # E[X | Z, A=1] per observation
    kf = KFold(K, shuffle=True, random_state=seed)
    for tr_idx, te_idx in kf.split(Z):
        sub = (A[tr_idx] == 1)
        if propensity == "hgb":
            ps = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        else:
            ps = LogisticRegression(max_iter=500)
        ps.fit(Z[tr_idx], A[tr_idx])
        e_hat[te_idx] = ps.predict_proba(Z[te_idx])[:, 1]

        rf_g = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=15,
            n_jobs=-1, random_state=seed
        )
        rf_g.fit(Z[tr_idx][sub], X[tr_idx][sub])
        nabla_m[te_idx] = rf_g.predict(Z[te_idx])

    e_c = np.clip(e_hat, clip[0], clip[1])
    ind = A.astype(float)
    # AIPW per-obs gradient summand (theta-free for linear f)
    aipw_grad_i = (ind / e_c)[:, None] * (X - nabla_m) + nabla_m
    aipw_grad_est = aipw_grad_i.mean(0)
    emp_X_mean = X.mean(0)
    g_dr = emp_X_mean - aipw_grad_est       # constant in theta for linear f

    # ---- Phase 3 ----
    mu = 0.0
    theta_trace, mu_trace, phi_trace = [], [], []
    for t in range(T - T0):
        f = X @ theta
        grad_pred = -2.0 / n * X.T @ (Y - f)
        # m_f(z; theta) = nabla_m(z) @ theta  (linear f)
        m_f = nabla_m @ theta
        aipw_f = (ind / e_c) * (f - m_f) + m_f
        D_hat = f.mean() - aipw_f.mean()
        s_t = np.sign(D_hat) if abs(D_hat) > 1e-12 else 1.0
        phi_t = abs(D_hat)
        # updates
        theta = theta - lr_p * (grad_pred + mu * s_t * g_dr)
        mu = max(0.0, mu + lr_d * (phi_t - eps))
        theta_trace.append(theta.copy())
        mu_trace.append(mu)
        phi_trace.append(phi_t)

    # average over second half (past transient)
    half = (T - T0) // 2
    bar_theta = np.mean(theta_trace[half:], axis=0)
    bar_mu = float(np.mean(mu_trace[half:]))

    return {
        "bar_theta": bar_theta,
        "bar_mu": bar_mu,
        "theta_warm": theta_warm,
        "mu_trace": np.array(mu_trace),
        "phi_trace": np.array(phi_trace),
        "e_hat": e_hat,
        "nabla_m": nabla_m,
        "g_dr": g_dr,
    }


# =====================================================================
# AIPW diagnostic at a given theta (same idea as Section 10's diagnostic
# but parameterised by theta so we can plug in either the trained
# bar_theta or an externally supplied one).
# =====================================================================
def diagnostic_at(theta, Z, A, X, Y, K=5, seed=0, clip=(0.02, 0.98),
                  reuse_nuisances=None):
    """
    Compute mu_hat_DR(theta) and IF SE.

    reuse_nuisances=None -> fresh cross-fit
    reuse_nuisances=(e_hat, nabla_m) -> reuse precomputed cross-fits
    """
    n, D = X.shape
    f_eval = X @ theta
    ell_i = -2.0 * (Y - f_eval)[:, None] * X
    ell_hat = ell_i.mean(0)

    if reuse_nuisances is None:
        e_hat = np.zeros(n)
        nabla_m = np.zeros((n, D))
        m_f = np.zeros(n)
        kf = KFold(K, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(Z):
            sub = (A[tr_idx] == 1)
            ps = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
            ps.fit(Z[tr_idx], A[tr_idx])
            e_hat[te_idx] = ps.predict_proba(Z[te_idx])[:, 1]
            rf_g = RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=15,
                n_jobs=-1, random_state=seed)
            rf_g.fit(Z[tr_idx][sub], X[tr_idx][sub])
            nabla_m[te_idx] = rf_g.predict(Z[te_idx])
            rf_f = RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=15,
                n_jobs=-1, random_state=seed)
            rf_f.fit(Z[tr_idx][sub], f_eval[tr_idx][sub])
            m_f[te_idx] = rf_f.predict(Z[te_idx])
    else:
        e_hat, nabla_m = reuse_nuisances
        m_f = nabla_m @ theta   # linear-f shortcut

    e_c = np.clip(e_hat, clip[0], clip[1])
    ind = A.astype(float)
    aipw_grad_i = (ind / e_c)[:, None] * (X - nabla_m) + nabla_m
    aipw_f_i = (ind / e_c) * (f_eval - m_f) + m_f

    g_hat = X.mean(0) - aipw_grad_i.mean(0)
    D_hat = f_eval.mean() - aipw_f_i.mean()
    s_hat = np.sign(D_hat) if abs(D_hat) > 1e-12 else 1.0
    norm_g_sq = float(g_hat @ g_hat)
    mu_dr = float(-s_hat * (ell_hat @ g_hat) / norm_g_sq)

    U_i = ell_i
    V_i = X - aipw_grad_i
    term1 = (U_i - ell_hat) @ g_hat / norm_g_sq
    mix = ell_hat + 2.0 * s_hat * mu_dr * g_hat
    term2 = (V_i - g_hat) @ mix / norm_g_sq
    psi_i = -s_hat * (term1 + term2)
    if_se = float(psi_i.std(ddof=1)) / np.sqrt(n)

    return mu_dr, if_se


# =====================================================================
# EXP A -- identity check on a single replicate
# =====================================================================
def exp_A(n=20_000, seed=42, T=8_000, T0=1_000):
    print("=" * 72)
    print(f"EXP A: identity check at n={n}, seed={seed}")
    print("=" * 72)
    Z, A, Y, X = generate_data(n, seed)

    sol = aipw_solver_linear(Z, A, Y, X, T=T, T0=T0, seed=seed)
    # diagnostic at bar_theta using the SAME nuisances (Proposition 4 form)
    mu_dr_same, se_same = diagnostic_at(
        sol["bar_theta"], Z, A, X, Y, seed=seed,
        reuse_nuisances=(sol["e_hat"], sol["nabla_m"]),
    )
    # and with FRESH nuisances (sample-split sanity)
    mu_dr_fresh, se_fresh = diagnostic_at(
        sol["bar_theta"], Z, A, X, Y, seed=seed + 10_000,
    )

    print(f"  bar_mu (solver):                {sol['bar_mu']:+.5f}")
    print(f"  mu_hat_DR(bar_theta), reused:   {mu_dr_same:+.5f}  "
          f"(diff = {sol['bar_mu'] - mu_dr_same:+.6f}, IF SE = {se_same:.5f})")
    print(f"  mu_hat_DR(bar_theta), fresh CV: {mu_dr_fresh:+.5f}  "
          f"(diff = {sol['bar_mu'] - mu_dr_fresh:+.6f}, IF SE = {se_fresh:.5f})")
    print(f"  |diff| / IF SE (reused):        "
          f"{abs(sol['bar_mu'] - mu_dr_same) / se_same:.3f}")
    print(f"  (Proposition 4 predicts diff = o_p(n^-1/2); reused-CV diff")
    print(f"   should be very small, fresh-CV diff bounded by O_p(n^-1/2).)")
    return sol


# =====================================================================
# EXP B -- coverage study
# =====================================================================
def exp_B(n=2_000, n_reps=100, mu_star=None,
          T=4_000, T0=500, K=5, light=True):
    """
    Coverage of the 95% IF-based CI for mu*.

    light=True uses LogisticRegression + smaller RF to keep runtime manageable;
    set light=False for the production HGB + larger RF configuration.
    """
    print("=" * 72)
    print(f"EXP B: coverage study, n={n}, reps={n_reps}, light={light}")
    print("=" * 72)
    if mu_star is None:
        print("  computing reference mu* ...", flush=True)
        mu_star = reference_mu_star()
    print(f"  reference mu* = {mu_star:.4f}")

    covered = 0
    mus, ses, biases = [], [], []
    fails = 0

    for r in range(n_reps):
        try:
            Z, A, Y, X = generate_data(n, seed=10_000 + r)
            propensity = "logistic" if light else "hgb"
            sol = aipw_solver_linear(
                Z, A, Y, X, T=T, T0=T0, K=K, seed=r, propensity=propensity)
            mu_dr, if_se = diagnostic_at(
                sol["bar_theta"], Z, A, X, Y, seed=r,
                reuse_nuisances=(sol["e_hat"], sol["nabla_m"]),
            )
            # Two-sided 95% IF CI around mu_dr
            lo, hi = mu_dr - 1.96 * if_se, mu_dr + 1.96 * if_se
            cov = (lo <= mu_star <= hi)
            covered += int(cov)
            mus.append(mu_dr); ses.append(if_se); biases.append(mu_dr - mu_star)
            if (r + 1) % 10 == 0:
                print(f"  rep {r+1}/{n_reps}: rolling coverage = "
                      f"{covered/(r+1):.3f}, mean bias = "
                      f"{np.mean(biases):+.4f}", flush=True)
        except Exception as e:
            fails += 1
            print(f"  rep {r} failed: {e}")
            continue

    coverage = covered / max(n_reps - fails, 1)
    print(f"\n  --- Coverage results over {n_reps - fails} successful reps ---")
    print(f"  Empirical 95%-CI coverage:   {coverage:.3f}  (target 0.95)")
    print(f"  Mean bias (mu_hat - mu*):    {np.mean(biases):+.4f}")
    print(f"  RMSE:                        {np.sqrt(np.mean(np.array(biases)**2)):.4f}")
    print(f"  Mean IF SE:                  {np.mean(ses):.4f}")
    print(f"  Empirical SD across reps:    {np.std(mus, ddof=1):.4f}")
    print(f"  (IF SE ≈ empirical SD if asymptotic variance is well-estimated.)")
    return {
        "coverage": coverage, "mu_dr": np.array(mus),
        "se": np.array(ses), "biases": np.array(biases),
        "mu_star": mu_star,
    }


# =====================================================================
# EXP C -- nonlinear (2-layer MLP, manual autodiff in NumPy)
# =====================================================================
class TinyMLP:
    """2-layer MLP, scalar output, tanh activation, manual backprop."""
    def __init__(self, d_in, d_hidden=8, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((d_in, d_hidden)) / np.sqrt(d_in)
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.standard_normal(d_hidden) / np.sqrt(d_hidden)
        self.b2 = 0.0
        self.d_hidden = d_hidden
        self.d_in = d_in

    @property
    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + 1

    def pack(self):
        return np.concatenate([self.W1.ravel(), self.b1,
                               self.W2.ravel(), [self.b2]])

    def unpack(self, vec):
        p = 0
        self.W1 = vec[p:p + self.W1.size].reshape(self.W1.shape); p += self.W1.size
        self.b1 = vec[p:p + self.b1.size]; p += self.b1.size
        self.W2 = vec[p:p + self.W2.size]; p += self.W2.size
        self.b2 = float(vec[p])

    def forward(self, X):
        H = np.tanh(X @ self.W1 + self.b1)
        return H @ self.W2 + self.b2, H

    def grad_theta(self, X):
        """∇θ f(X) for each row: shape (n, n_params)."""
        n = X.shape[0]
        _, H = self.forward(X)
        # dH/dpre = 1 - tanh^2(pre), output = H @ W2 + b2
        pre = X @ self.W1 + self.b1
        dH = 1.0 - np.tanh(pre) ** 2     # (n, d_hidden)
        # df/dW1[i,j] = X[:, i] * dH[:, j] * W2[j]
        d_W1 = np.einsum('ni,nj,j->nij', X, dH, self.W2).reshape(n, -1)
        d_b1 = dH * self.W2              # (n, d_hidden)
        d_W2 = H                          # (n, d_hidden)
        d_b2 = np.ones((n, 1))
        return np.hstack([d_W1, d_b1, d_W2, d_b2])


def exp_C(n=10_000, seed=42, T=6_000, T0=1_000, refit_every=500, eps=0.30,
          lr_p=5e-3, lr_d=2e-2, K=5, clip=(0.02, 0.98)):
    """
    Nonlinear sanity check: small MLP with periodic refits of the gradient
    regression. Single-seed.
    """
    print("=" * 72)
    print(f"EXP C: nonlinear (MLP) variant, n={n}, seed={seed}")
    print("=" * 72)
    Z, A, Y, X = generate_data(n, seed)
    d = X.shape[1]
    net = TinyMLP(d_in=d, d_hidden=8, seed=seed)
    p_dim = net.n_params

    # ---- Phase 1: unconstrained warm-up ----
    theta = net.pack()
    for t in range(T0):
        f, _ = net.forward(X)
        # MSE gradient wrt theta
        G_i = net.grad_theta(X)        # (n, p_dim)
        resid = (Y - f)
        grad_pred = -2.0 / n * (G_i * resid[:, None]).sum(0)
        theta = theta - lr_p * grad_pred
        net.unpack(theta)
    print(f"  Phase 1 done; warm-start MSE = "
          f"{np.mean((Y - net.forward(X)[0])**2):.4f}")

    # ---- Phase 2: initial cross-fitted nuisances at theta^(0) ----
    def fit_nuisances(net):
        """Cross-fitted (e_hat, nabla_m_arr) at the current network state."""
        n_ = X.shape[0]
        e_hat = np.zeros(n_)
        nabla_m = np.zeros((n_, p_dim))
        kf = KFold(K, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(Z):
            sub = (A[tr_idx] == 1)
            ps = LogisticRegression(max_iter=500)
            ps.fit(Z[tr_idx], A[tr_idx])
            e_hat[te_idx] = ps.predict_proba(Z[te_idx])[:, 1]
            G_tr = net.grad_theta(X[tr_idx])
            rf = RandomForestRegressor(
                n_estimators=150, max_depth=10, min_samples_leaf=15,
                n_jobs=-1, random_state=seed)
            rf.fit(Z[tr_idx][sub], G_tr[sub])
            nabla_m[te_idx] = rf.predict(Z[te_idx])
        return e_hat, nabla_m

    print("  fitting nuisances at theta^(0) ...", flush=True)
    e_hat, nabla_m = fit_nuisances(net)
    e_c = np.clip(e_hat, clip[0], clip[1])
    ind = A.astype(float)

    # ---- Phase 3: constrained refinement with periodic refits ----
    mu = 0.0
    mu_trace, phi_trace = [], []
    for t in range(T - T0):
        # refit nuisances every refit_every iterations (matches one-step DML idea)
        if t > 0 and t % refit_every == 0:
            e_hat, nabla_m = fit_nuisances(net)
            e_c = np.clip(e_hat, clip[0], clip[1])

        f, _ = net.forward(X)
        G_i = net.grad_theta(X)             # (n, p_dim)
        resid = (Y - f)
        grad_pred = -2.0 / n * (G_i * resid[:, None]).sum(0)

        # AIPW constraint gradient
        aipw_grad_i = (ind / e_c)[:, None] * (G_i - nabla_m) + nabla_m
        g_dr = G_i.mean(0) - aipw_grad_i.mean(0)

        # AIPW constraint VALUE: needs m_f(z; theta) = E[f(X;theta) | Z, A=1].
        # For nonlinear f we either (a) refit a regressor of f on Z|A=1 each
        # iteration (expensive) or (b) use the linearisation
        # m_f(z; theta) ≈ nabla_m(z) @ theta. The latter is exact for linear f
        # and a first-order approximation for nonlinear f -- it is the same
        # one-step approximation that DML uses for nuisance updates.
        m_f = nabla_m @ theta
        aipw_f = (ind / e_c) * (f - m_f) + m_f
        D_hat = f.mean() - aipw_f.mean()
        s_t = np.sign(D_hat) if abs(D_hat) > 1e-12 else 1.0
        phi_t = abs(D_hat)

        theta = theta - lr_p * (grad_pred + mu * s_t * g_dr)
        net.unpack(theta)
        mu = max(0.0, mu + lr_d * (phi_t - eps))
        mu_trace.append(mu); phi_trace.append(phi_t)

    half = (T - T0) // 2
    bar_mu = float(np.mean(mu_trace[half:]))
    bar_phi = float(np.mean(phi_trace[half:]))
    final_mse = float(np.mean((Y - net.forward(X)[0]) ** 2))
    print(f"  Phase 3 done.")
    print(f"  bar_mu          = {bar_mu:+.4f}")
    print(f"  bar_Phi (last half avg) = {bar_phi:.4f}  (target eps = {eps})")
    print(f"  final MSE       = {final_mse:.4f}")
    print(f"  (Sanity: solver converges; bar_Phi ≈ eps confirms constraint")
    print(f"   binds, bar_mu finite and positive confirms multiplier exists.")
    print(f"   Full statistical inference for nonlinear case is left to")
    print(f"   future work -- this only checks the solver runs.)")
    return {"bar_mu": bar_mu, "bar_phi": bar_phi, "mse": final_mse,
            "mu_trace": np.array(mu_trace), "phi_trace": np.array(phi_trace)}


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print("Algorithm 1 validation -- causal shadow price\n")

    print("Step 1: reference mu* via large-n oracle ...")
    mu_star = reference_mu_star()
    print(f"  mu* ≈ {mu_star:.4f}\n")

    # EXP A
    sol = exp_A(n=20_000, seed=42)
    print()

    # EXP B  -- defaults are quick-run; production should use n=10000, n_reps=500
    res_B = exp_B(n=2_000, n_reps=100, mu_star=mu_star, light=True)
    print()

    # EXP C
    res_C = exp_C(n=10_000, seed=42)
