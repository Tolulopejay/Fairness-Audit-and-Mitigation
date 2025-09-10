

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def group_fairness_report(y_true, y_pred, sensitive_features, positive_label=1):
    """
    Compute group fairness metrics by sensitive attribute.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    sensitive_features : pandas.Series
        Sensitive attribute (e.g. sex, race, age_group).
    positive_label : int
        Which class is considered the "positive" outcome (default = 1).
    """

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": sensitive_features})
    report = []

    for g in df["group"].unique():
        g_df = df[df["group"] == g]
        tn, fp, fn, tp = confusion_matrix(g_df["y_true"], g_df["y_pred"], labels=[0, 1]).ravel()

        selection_rate = (tp + fp) / len(g_df)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        report.append({
            "group": g,
            "n": len(g_df),
            "selection_rate": round(selection_rate, 3),
            "TPR": round(tpr, 3),
            "FPR": round(fpr, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })

    return pd.DataFrame(report)
