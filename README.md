causal-shadow-price


Reproduction code and paper for "The Causal Shadow Price: Identification, Doubly Robust Estimation, and Semiparametric Efficiency for the Lagrange Multiplier of Interventional Fairness Constraints" by Seyed A. Yousefi, IT and Science GmbH, 2026.

Paper on Zenodo:
https://doi.org/10.5281/zenodo.20241478


Contents

    README.md
    LICENSE
    requirements.txt
    synthetic_diagnostic.py     Section 8, oracle vs AIPW diagnostic
    algorithm1_validation.py    Section 10.2, Algorithm 1 validation, Figures 1 and 2
    coverage_n2000_sandwich.py  Table 2, column n equals 2000
    coverage_n10000.py          Table 2, column n equals 10000
    coverage_sandwich.py        Sandwich variance helpers used by the coverage scripts
    make_figures_colab.py       Generates fig_convergence.pdf and fig_forest.pdf
    compas_step2.ipynb          Section 10.3, COMPAS solver and Sandwich CI


Requirements

Python 3.9 or later.

Install the dependencies with:

    pip install -r requirements.txt

Direct dependencies are numpy, scipy, scikit-learn, pandas and matplotlib. The notebook also needs jupyter and an internet connection to download the COMPAS dataset.


Reproducing the paper


Synthetic experiments, Sections 8 and 10.2

Results are deterministic for fixed seeds.

Section 8, oracle vs AIPW diagnostic, about 5 minutes:

    python synthetic_diagnostic.py

Section 10.2, Algorithm 1 validation, about 15 minutes:

    python algorithm1_validation.py


Coverage table, Table 2

Each script estimates one column of Table 2 with 500 replications, using seeds 10000 to 10499. Run in parallel if cores permit, sequentially otherwise.

    python coverage_n2000_sandwich.py
    python coverage_n10000.py

The run for n equals 2000 takes about 30 to 45 minutes. The run for n equals 10000 takes about 120 to 180 minutes. Each script writes a JSON result file and prints progress every 25 replications.


Figures, Figures 1 and 2

    python make_figures_colab.py

Outputs fig_convergence.pdf and fig_forest.pdf.


COMPAS real data demonstration, Section 10.3

compas_step2.ipynb runs the AIPW corrected primal dual solver on the ProPublica COMPAS cohort, n equals 5278 after the standard filter. It computes Sandwich confidence intervals for the shadow price mu_bar and verifies that the constraint is active at the unconstrained model.

Data is fetched directly from:

    https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv

Expected output:

    mu_bar equals 0.7217 under specification A, the minimal adjustment set with age and priors.
    mu_bar equals 0.7214 under specification B, the comprehensive adjustment set.
    The 95 percent Sandwich confidence interval is roughly 0.671 to 0.772 in both specifications.
    The race coefficient is halved, 0.4969 unconstrained becomes 0.2484 constrained, matching the theoretical prediction epsilon divided by absolute value of g_A, that is 0.099 divided by 0.398 equals 0.2487.
    Activity check: absolute value of D at theta_unc is 0.198, which is 14 standard errors above epsilon equals 0.099.


Citation

Yousefi, Seyed A. (2026). The Causal Shadow Price: Identification, Doubly Robust Estimation, and Semiparametric Efficiency for the Lagrange Multiplier of Interventional Fairness Constraints. IT and Science GmbH, Switzerland. Preprint. DOI 10.5281/zenodo.20241478.


License

Code is MIT, see the LICENSE file. The paper is licensed CC-BY-4.0 on Zenodo.


Status

Preprint, currently under peer review.


Contact

Open an issue on this repository.
