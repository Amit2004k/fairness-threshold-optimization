"""
threshold_optimizer.py
----------------------
Post-hoc group-specific threshold optimizer with hard resource constraints.
Supports Demographic Parity, Equalized Odds, and Predictive Parity.
"""

import numpy as np
from scipy.optimize import minimize
from .fairness_metrics import demographic_parity_gap, equalized_odds_gap


class FairnessThresholdOptimizer:
    """
    Optimize group-specific classification thresholds to satisfy
    fairness constraints under hard resource limits.

    Parameters
    ----------
    fairness_metric : str
        One of 'demographic_parity', 'equalized_odds', 'predictive_parity'
    epsilon : float
        Maximum allowed fairness gap (e.g. 0.05 = 5%)
    budget : float or None
        Max allowed positive-rate across all groups (e.g. 0.35 = approve 35% max).
        None means no budget constraint.
    """

    def __init__(self, fairness_metric="equalized_odds", epsilon=0.05, budget=None):
        self.fairness_metric = fairness_metric
        self.epsilon = epsilon
        self.budget = budget
        self.thresholds_ = None
        self.groups_ = None

    def fit(self, y_prob, y_true, sensitive_attr):
        """
        Fit group-specific thresholds.

        Parameters
        ----------
        y_prob : array-like, shape (n,)
            Predicted probabilities from a pre-trained model
        y_true : array-like, shape (n,)
            Ground truth labels
        sensitive_attr : array-like, shape (n,)
            Sensitive attribute (group membership)

        Returns
        -------
        thresholds : dict {group: threshold}
        """
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        sensitive_attr = np.array(sensitive_attr)

        self.groups_ = np.unique(sensitive_attr)
        n_groups = len(self.groups_)

        # Initial thresholds: 0.5 for all groups
        x0 = np.full(n_groups, 0.5)

        def objective(thresholds):
            # Minimize total misclassification
            total_loss = 0
            for i, g in enumerate(self.groups_):
                mask = sensitive_attr == g
                y_pred_g = (y_prob[mask] >= thresholds[i]).astype(int)
                total_loss += np.mean(y_pred_g != y_true[mask])
            return total_loss

        constraints = []

        # Fairness constraint
        def fairness_constraint(thresholds):
            y_pred_all = np.zeros(len(y_prob), dtype=int)
            for i, g in enumerate(self.groups_):
                mask = sensitive_attr == g
                y_pred_all[mask] = (y_prob[mask] >= thresholds[i]).astype(int)
            if self.fairness_metric == "demographic_parity":
                gap = demographic_parity_gap(y_pred_all, sensitive_attr)
            elif self.fairness_metric == "equalized_odds":
                gap = equalized_odds_gap(y_pred_all, y_true, sensitive_attr)
            else:
                gap = demographic_parity_gap(y_pred_all, sensitive_attr)
            return self.epsilon - gap  # must be >= 0

        constraints.append({"type": "ineq", "fun": fairness_constraint})

        # Hard budget constraint
        if self.budget is not None:
            def budget_constraint(thresholds):
                y_pred_all = np.zeros(len(y_prob), dtype=int)
                for i, g in enumerate(self.groups_):
                    mask = sensitive_attr == g
                    y_pred_all[mask] = (y_prob[mask] >= thresholds[i]).astype(int)
                return self.budget - y_pred_all.mean()  # must be >= 0

            constraints.append({"type": "ineq", "fun": budget_constraint})

        bounds = [(0.01, 0.99)] * n_groups

        result = minimize(
            objective, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9}
        )

        self.thresholds_ = {g: result.x[i] for i, g in enumerate(self.groups_)}
        print(f"Optimization {'succeeded' if result.success else 'failed'}: {result.message}")
        print(f"Group thresholds: {self.thresholds_}")
        return self.thresholds_

    def predict(self, y_prob, sensitive_attr):
        """
        Apply fitted group-specific thresholds to generate predictions.
        """
        if self.thresholds_ is None:
            raise RuntimeError("Call fit() before predict()")
        y_prob = np.array(y_prob)
        sensitive_attr = np.array(sensitive_attr)
        y_pred = np.zeros(len(y_prob), dtype=int)
        for g, t in self.thresholds_.items():
            mask = sensitive_attr == g
            y_pred[mask] = (y_prob[mask] >= t).astype(int)
        return y_pred

    def predict_proba_adjusted(self, y_prob, sensitive_attr):
        """
        Return the threshold-adjusted decision scores (for soft evaluation).
        """
        if self.thresholds_ is None:
            raise RuntimeError("Call fit() before predict_proba_adjusted()")
        y_prob = np.array(y_prob)
        sensitive_attr = np.array(sensitive_attr)
        adjusted = np.zeros(len(y_prob))
        for g, t in self.thresholds_.items():
            mask = sensitive_attr == g
            adjusted[mask] = y_prob[mask] / t  # normalized margin
        return np.clip(adjusted, 0, 1)
