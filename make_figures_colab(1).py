import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.0,
})
import matplotlib.pyplot as plt
import numpy as np

SEED = 42

# =====================================================================
# Figure 1: convergence of the primal-dual solver on DGP A
# =====================================================================
rng = np.random.default_rng(SEED)
N = 20000
Z = rng.standard_normal((N, 3))
prop_true = 1.0 / (1.0 + np.exp(-(Z @ np.array([0.8, -0.5, 0.4]))))
A = rng.binomial(1, prop_true).astype(int)
Y = Z @ np.array([1.0, -0.5, 0.7]) + 1.2 * A + 0.5 * rng.standard_normal(N)
X = np.hstack([Z, A[:, None]]).astype(float)

D = X.shape[1]
p1 = A.mean()
theta = np.zeros(D)
mu = 0.0
E_X = np.concatenate([Z.mean(0), [p1]])
E_do_X = np.concatenate([Z.mean(0), [1.0]])

mu_hist, phi_hist = [], []
T, eps = 8000, 0.30
for t in range(T):
    f = X @ theta
    grad_pred = -2.0 / N * X.T @ (Y - f)
    discrep = theta @ (E_X - E_do_X)
    phi = abs(discrep)
    s_star = np.sign(discrep) if abs(discrep) > 1e-12 else 1.0
    grad_phi = s_star * (E_X - E_do_X)
    theta = theta - 0.003 * (grad_pred + mu * grad_phi)
    mu = max(0.0, mu + 0.02 * (phi - eps))
    mu_hist.append(mu)
    phi_hist.append(phi)

fig, ax = plt.subplots(figsize=(6.5, 2.8))
its = np.arange(T)
ax.plot(its, mu_hist, color='black', linewidth=0.9,
        label=r'$\mu_t$ (dual variable)')
ax.plot(its, phi_hist, color='black', linestyle='--', linewidth=0.9,
        label=r'$\Phi_{\mathrm{INT}}(f_{\theta_t})$ (constraint value)')
ax.axhline(eps, color='gray', linestyle=':', linewidth=0.7)
ax.text(T * 0.99, eps + 0.025, r'$\epsilon = 0.30$',
        ha='right', va='bottom', color='gray', fontsize=9)
ax.axhline(1.0822, color='gray', linestyle=':', linewidth=0.7)
ax.text(T * 0.99, 1.0822 + 0.025, r'$\mu^*_{\mathrm{opt}} = 1.082$',
        ha='right', va='bottom', color='gray', fontsize=9)
ax.set_xlabel('Primal--dual iteration')
ax.set_ylabel('value')
ax.legend(loc='center right', frameon=False, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, T)
ax.set_ylim(-0.05, 1.35)
plt.tight_layout()
plt.savefig('fig_convergence.pdf', bbox_inches='tight')
plt.close()

# =====================================================================
# Figure 2: forest plot of mu_DR with 95% CIs across five conditions
# =====================================================================
results = [
    {"name": r"(1) Correct",                 "mu_opt": 1.0822, "mu_dr": 1.0822, "if_se": 0.0184, "dgp": "A"},
    {"name": r"(2) $Z_1$ omitted",           "mu_opt": 1.0822, "mu_dr": 0.8965, "if_se": 0.0199, "dgp": "A"},
    {"name": r"(3) Propensity misspec.",     "mu_opt": 1.0822, "mu_dr": 1.0753, "if_se": 0.0187, "dgp": "A"},
    {"name": r"(4) Overlap trimmed",         "mu_opt": 0.8605, "mu_dr": 0.7656, "if_se": 0.0214, "dgp": "B"},
    {"name": r"(5) Overlap full",            "mu_opt": 0.8605, "mu_dr": 0.7919, "if_se": 0.0252, "dgp": "B"},
]

fig, axs = plt.subplots(1, 2, figsize=(8.5, 2.6),
                        gridspec_kw={'width_ratios': [3, 2]})

# Panel A
ax = axs[0]
dgp_a = [r for r in results if r["dgp"] == "A"]
for i, r in enumerate(reversed(dgp_a)):
    ax.errorbar(r["mu_dr"], i, xerr=1.96 * r["if_se"],
                fmt='o', color='black', markersize=5,
                capsize=3, elinewidth=0.8, capthick=0.8)
ax.axvline(1.0822, color='gray', linestyle='--', linewidth=0.7)
ax.set_yticks(list(range(len(dgp_a))))
ax.set_yticklabels([r["name"] for r in reversed(dgp_a)])
ax.set_xlabel(r'$\hat{\mu}^*_{\mathrm{DR}}$ with 95\% CI')
ax.set_title(r'DGP A ($\mu^*_{\mathrm{opt}} = 1.082$, good overlap)',
             fontsize=10)
ax.set_xlim(0.83, 1.13)
ax.set_ylim(-0.6, len(dgp_a) - 0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B
ax = axs[1]
dgp_b = [r for r in results if r["dgp"] == "B"]
for i, r in enumerate(reversed(dgp_b)):
    ax.errorbar(r["mu_dr"], i, xerr=1.96 * r["if_se"],
                fmt='o', color='black', markersize=5,
                capsize=3, elinewidth=0.8, capthick=0.8)
ax.axvline(0.8605, color='gray', linestyle='--', linewidth=0.7)
ax.set_yticks(list(range(len(dgp_b))))
ax.set_yticklabels([r["name"] for r in reversed(dgp_b)])
ax.set_xlabel(r'$\hat{\mu}^*_{\mathrm{DR}}$ with 95\% CI')
ax.set_title(r'DGP B ($\mu^*_{\mathrm{opt}} = 0.861$, overlap violation)',
             fontsize=10)
ax.set_xlim(0.69, 0.93)
ax.set_ylim(-0.6, len(dgp_b) - 0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_forest.pdf', bbox_inches='tight')
plt.close()

print("figures saved")
