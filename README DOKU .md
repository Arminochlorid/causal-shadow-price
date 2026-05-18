Reproduction Materials

Reproduction code and paper for **"The Causal Shadow Price: Identification, Doubly Robust Estimation, and Semiparametric Efficiency for the Lagrange Multiplier of Interventional Fairness Constraints"** (Seyed Yousefi, IT & Science GmbH, 2026).

## Contents

```
.
├── README Doku.md
├── LICENSE
├── requirements.txt                         # Pinned package versions
├── .gitignore                               # Standard Python ignores
├── paper/
│   ├── v14.tex                              # LaTeX source
│   └── v14.pdf                              # Compiled paper
├── synthetic/
│   ├── synthetic_diagnostic.py              # Section 8: oracle vs AIPW diagnostic
│   ├── algorithm1_validation.py             # Section 10.2: Algorithm 1 validation (Fig. 1, Fig. 2)
│   ├── coverage_n2000_sandwich_500reps.py   # Table 2, column n=2000
│   ├── coverage_n5000_sandwich_500reps.py   # Table 2, column n=5000
│   ├── coverage_n10000_sandwich_500reps.py  # Table 2, column n=10000
│   └── make_figures_colab.py                # Generates fig_convergence.pdf and fig_forest.pdf
├── compas/
│   ├── compas_step1.ipynb                   # COMPAS adjustment-set diagnostics
│   └── compas_step2.ipynb                   # COMPAS solver + Sandwich CI (Section 10.3)
├── figures/
│   ├── fig_convergence.pdf                  # Figure 1
│   └── fig_forest.pdf                       # Figure 2
└── expected_results/                        # JSON snapshots for verification
    ├── n2000_500reps.json
    ├── n5000_500reps.json
    └── n10000_500reps.json
```

## Requirements

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

## Reproducing the Paper

### Synthetic Experiments (Section 8 and 10.2)

The synthetic experiments are deterministic given fixed seeds.

```bash
# Section 8: oracle vs AIPW diagnostic (~5 minutes)
python synthetic/synthetic_diagnostic.py

# Section 10.2: Algorithm 1 validation (~15 minutes)
python synthetic/algorithm1_validation.py
```

### Coverage Table (Table 2)

The three coverage scripts each estimate one column of Table 2 with B=500 replications. Seeds 10,000 through 10,499.

```bash
# Run in parallel if cores permit, sequentially otherwise.
python synthetic/coverage_n2000_sandwich_500reps.py   > log_n2000.txt 2>&1    # ~30-45 min
python synthetic/coverage_n5000_sandwich_500reps.py   > log_n5000.txt 2>&1    # ~60-90 min
python synthetic/coverage_n10000_sandwich_500reps.py  > log_n10000.txt 2>&1   # ~120-180 min
```

Each script writes a JSON result file (e.g. `results_n2000_500reps.json`) and prints intermediate progress every 25 replications.

### Figures (Figures 1 and 2)

```bash
python synthetic/make_figures_colab.py
```

Outputs `fig_convergence.pdf` and `fig_forest.pdf`.

### COMPAS Real-Data Demonstration (Section 10.3)

The COMPAS analysis is split into two notebooks. Run Step 1 first, then Step 2 in the same Colab/Jupyter session.

**Step 1** (`compas_step1.ipynb`): downloads the ProPublica COMPAS cohort from GitHub, applies the standard filter (n=5278), and diagnoses three back-door adjustment-set candidates via cross-fitted propensity. No local data file is needed; the notebook fetches data from:

```
https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
```

**Step 2** (`compas_step2.ipynb`): runs the AIPW-corrected primal-dual solver, computes Sandwich CIs for the shadow price `mu_bar`, and verifies activity at the unconstrained model. Depends on the variables `df`, `RESULTS`, `CANDIDATES` defined in Step 1.

Expected output:
- `mu_bar = 0.7217` under specification A (minimal: age, priors)
- `mu_bar = 0.7214` under specification B (comprehensive)
- 95% Sandwich CI: approximately [0.671, 0.772] under both specifications
- Race coefficient exactly halved: 0.4969 (unconstrained) -> 0.2484 (constrained), matching the theoretical prediction `epsilon / |g_A| = 0.099 / 0.398 = 0.2487`
- Activity verification: |D| at theta_unc = 0.198, 14 standard errors above epsilon = 0.099

## Citation

```bibtex
@article{yousefi2026causalshadowprice,
  title  = {The Causal Shadow Price: Identification, Doubly Robust Estimation,
            and Semiparametric Efficiency for the Lagrange Multiplier of
            Interventional Fairness Constraints},
  author = {Yousefi, Seyed},
  year   = {2026},
  note   = {IT \& Science GmbH, Switzerland}
}
```

## License

Code: MIT (see `LICENSE`). Paper: All rights reserved by the author.

## Contact

Open an issue on this repository for reproduction questions or bug reports.
