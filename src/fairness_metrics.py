"""
fairness_metrics.py
-------------------
Fairness evaluation metrics: Demographic Parity, Equalized Odds,
Predictive Parity, and per-group calibration (ECE).
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def demographic_parity_gap(y_pred, sensitive_attr):
    """
    Demographic Parity Gap: |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|
    Lower is fairer.
    """
    groups = np.unique(sensitive_attr)
    rates = [y_pred[sensitive_attr == g].mean() for g in groups]
    return max(rates) - min(rates)


def equalized_odds_gap(y_pred, y_true, sensitive_attr):
    """
    Equalized Odds Gap: max(ΔTPR, ΔFPR) across groups.
    Lower is fairer.
    """
    groups = np.unique(sensitive_attr)
    tprs, fprs = [], []
    for g in groups:
        mask = sensitive_attr == g
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0,1]).ravel()
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    return max(max(tprs) - min(tprs), max(fprs) - min(fprs))


def predictive_parity_gap(y_pred, y_true, sensitive_attr):
    """
    Predictive Parity Gap: |PPV_A - PPV_B|
    """
    groups = np.unique(sensitive_attr)
    ppvs = []
    for g in groups:
        mask = (sensitive_attr == g) & (y_pred == 1)
        if mask.sum() == 0:
            ppvs.append(0.0)
        else:
            ppvs.append(y_true[mask].mean())
    return max(ppvs) - min(ppvs)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE) for a single group.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() * abs(acc - conf)
    return ece / max(len(y_true), 1)


def group_calibration_gap(y_true, y_prob, sensitive_attr, n_bins=10):
    """
    Maximum ECE gap across groups.
    """
    groups = np.unique(sensitive_attr)
    eces = [
        expected_calibration_error(y_true[sensitive_attr == g],
                                   y_prob[sensitive_attr == g], n_bins)
        for g in groups
    ]
    return max(eces) - min(eces)


def fairness_report(y_pred, y_true, y_prob, sensitive_attr):
    """
    Print a full fairness report.
    """
    dp  = demographic_parity_gap(y_pred, sensitive_attr)
    eo  = equalized_odds_gap(y_pred, y_true, sensitive_attr)
    pp  = predictive_parity_gap(y_pred, y_true, sensitive_attr)
    cal = group_calibration_gap(y_true, y_prob, sensitive_attr)

    print("=" * 45)
    print("        FAIRNESS REPORT")
    print("=" * 45)
    print(f"  Demographic Parity Gap : {dp:.4f}")
    print(f"  Equalized Odds Gap     : {eo:.4f}")
    print(f"  Predictive Parity Gap  : {pp:.4f}")
    print(f"  Group Calibration Gap  : {cal:.4f}")
    print("=" * 45)
    print("  (Lower values = fairer model)")
    return {"dp_gap": dp, "eo_gap": eo, "pp_gap": pp, "cal_gap": cal}
