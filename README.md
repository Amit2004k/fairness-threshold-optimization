# ⚖️ Operationalizing Fairness: Post-Hoc Threshold Optimization Under Hard Resource Limits

[![arXiv](https://img.shields.io/badge/arXiv-2602.22560-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2602.22560)
[![Journal](https://img.shields.io/badge/Neurocomputing-Major%20Revision-orange?style=flat-square)](https://www.sciencedirect.com/journal/neurocomputing)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Amit2004k/fairness-threshold-optimization?style=flat-square)](https://github.com/Amit2004k/fairness-threshold-optimization/stargazers)

> **Under Major Revision at Neurocomputing (Elsevier)**
> Preprint: [arXiv:2602.22560](https://arxiv.org/abs/2602.22560)

---

## 🧠 What is this about?

Deploying fair ML in the real world isn't just about satisfying fairness metrics — it requires operating **within hard constraints** like limited intervention budgets, fixed decision quotas, and regulatory compliance thresholds.

This paper proposes a **post-hoc threshold optimization framework** that:
1. Takes any pre-trained binary classifier
2. Optimizes group-specific decision thresholds
3. Simultaneously satisfies fairness constraints (Demographic Parity, Equalized Odds, etc.)
4. **Without exceeding hard resource limits** (e.g., "approve at most K% of applicants")

---

## 🔥 Key Contributions

- ✅ **Constrained threshold optimization** — formulated as a resource-bounded fairness problem
- ✅ **Multi-metric fairness** — supports DP, EO, PP, and custom constraints simultaneously
- ✅ **Hard resource limits** — budget constraints encoded directly into the optimization
- ✅ **Post-hoc** — works on top of any existing classifier (model-agnostic)
- ✅ **COMPAS benchmark** — evaluated on recidivism prediction, the canonical fairness testbed
- ✅ **Robustness analysis** — tested under distribution shift and threshold sensitivity

---

## 📊 Results Summary

### COMPAS Recidivism Dataset

| Method | Accuracy | DP Gap (↓) | EO Gap (↓) | Within Budget |
|--------|----------|------------|------------|---------------|
| Unconstrained Baseline | 67.2% | 0.241 | 0.198 | ✅ |
| Fairness Constraints Only | 65.1% | 0.042 | 0.031 | ❌ (violates) |
| **Ours (Hard Budget)** | **66.4%** | **0.038** | **0.027** | **✅** |

> Our method achieves near-parity fairness **while respecting hard resource limits**, sacrificing only 0.8% accuracy vs. the unconstrained baseline.

---

## 🏗️ Framework Overview

```
Pre-trained Classifier
        │
        ▼
  Predict Probabilities
  P(Y=1 | X) for all groups
        │
        ▼
┌───────────────────────────┐
│  Threshold Optimizer      │
│                           │
│  Objective:               │
│    min  Σ_g loss(τ_g)     │
│                           │
│  Subject to:              │
│    |DP(τ)| ≤ ε_dp         │  ← Demographic Parity
│    |EO(τ)| ≤ ε_eo         │  ← Equalized Odds
│    Σ_g P(Ŷ=1|g) ≤ Budget  │  ← Hard Resource Limit
└───────────────────────────┘
        │
        ▼
  Group-Specific Thresholds
  {τ_A, τ_B, ...}
        │
        ▼
  Fair + Budget-Compliant Decisions
```

---

## 📁 Repository Structure

```
📦 fairness-threshold-optimization
├── 📂 src/
│   ├── threshold_optimizer.py     # Core constrained optimization
│   ├── fairness_metrics.py        # DP, EO, PP, calibration metrics
│   ├── resource_constraints.py    # Hard budget constraint handlers
│   └── evaluation.py              # Full evaluation pipeline
├── 📂 notebooks/
│   ├── 01_compas_baseline.ipynb
│   ├── 02_threshold_optimization.ipynb
│   ├── 03_robustness_analysis.ipynb
│   └── 04_ablation_study.ipynb
├── 📂 data/
│   └── README.md                  # How to obtain COMPAS dataset
├── 📂 results/
│   ├── 📂 figures/                # All paper figures
│   └── 📂 tables/                 # LaTeX-ready result tables
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Amit2004k/fairness-threshold-optimization.git
cd fairness-threshold-optimization
pip install -r requirements.txt
```

### Run threshold optimization on COMPAS:

```python
from src.threshold_optimizer import FairnessThresholdOptimizer
from src.fairness_metrics import demographic_parity_gap, equalized_odds_gap

# Load your pre-trained model predictions
# y_prob: predicted probabilities, sensitive_attr: protected group labels

optimizer = FairnessThresholdOptimizer(
    fairness_metric="equalized_odds",
    epsilon=0.05,           # max allowed fairness gap
    budget=0.35,            # approve at most 35% of applicants
)
thresholds = optimizer.fit(y_prob, y_true, sensitive_attr)
y_pred_fair = optimizer.predict(y_prob, sensitive_attr)

print(f"DP Gap: {demographic_parity_gap(y_pred_fair, sensitive_attr):.4f}")
print(f"EO Gap: {equalized_odds_gap(y_pred_fair, y_true, sensitive_attr):.4f}")
```

---

## 🧩 Fairness Metrics Implemented

| Metric | Definition | Formula |
|--------|-----------|---------|
| **Demographic Parity (DP)** | Equal positive rates across groups | `|P(Ŷ=1\|A=0) - P(Ŷ=1\|A=1)|` |
| **Equalized Odds (EO)** | Equal TPR and FPR across groups | `max(ΔTPR, ΔFPR)` |
| **Predictive Parity (PP)** | Equal precision across groups | `|PPV_A - PPV_B|` |
| **Calibration** | Predicted probabilities match true rates | ECE per group |

---

## 📂 Dataset

**COMPAS Recidivism Dataset** (ProPublica):
```bash
# Download from ProPublica GitHub
wget https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv -P data/
```
See `data/README.md` for preprocessing steps.

---

## 🔬 Related Work

This builds on and extends:
- Hardt et al. (2016) — Equality of Opportunity in Supervised Learning
- Chouldechova (2017) — Fair Prediction with Disparate Impact
- Menon & Williamson (2018) — Cost-sensitive fairness

---

## 📖 Citation

```bibtex
@article{kalita2026fairness,
  title   = {Operationalizing Fairness: Post-Hoc Threshold Optimization Under Hard Resource Limits},
  author  = {Kalita, Amit and others},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Under Major Revision},
  url     = {https://arxiv.org/abs/2602.22560}
}
```

---

## 🙋 Author

**Amit Kalita**
B.Tech CSE, BIT Mesra (Dibrugarh Campus)
[GitHub](https://github.com/Amit2004k) | [arXiv](https://arxiv.org/abs/2602.22560)

> 📌 *Part of a series of published ML research repos. See also: [Breast Cancer Classification](https://github.com/Amit2004k/decision-aware-breast-cancer-classification), DDI Prediction, Alzheimer's Detection, Fraud Detection, and more.*

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

⭐ **Star this repo if you work on algorithmic fairness — helps others find it!**
