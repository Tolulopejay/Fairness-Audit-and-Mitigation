# mitigation.py
# Reproducible fairness audit + mitigation + visuals
# --------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)

# -----------------------------
# Policy thresholds & defaults
# -----------------------------
MIN_DI = 0.80   # Disparate Impact >= 0.80
MAX_DP = 0.10   # Demographic Parity diff <= 0.10
MAX_EO = 0.10   # Equal Opportunity (TPR gap) <= 0.10
MODE   = "EO"   # "EO" (equal opportunity) or "DP" (demographic parity)


# -----------------------------
# Metric helpers
# -----------------------------
def per_group(y_true, y_pred, y_score, sens):
    """Compute per-group selection & error metrics."""
    s = pd.Series(sens).astype(str).values
    rows = []
    for g in np.unique(s):
        idx = (s == g)
        yt, yp = np.asarray(y_true)[idx], np.asarray(y_pred)[idx]
        if yt.size == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        sel = yp.mean() if yp.size else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        prec, rec, f1, _ = precision_recall_fscore_support(
            yt, yp, average='binary', zero_division=0
        )
        rows.append({
            "group": g,
            "n": int(yt.size),
            "selection_rate": float(sel),
            "TPR": float(tpr),
            "FPR": float(fpr),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def fairness_summary(group_df):
    """Aggregate fairness metrics across groups."""
    if group_df.empty:
        return {"DP_diff": np.nan, "DI": np.nan, "EO_diff": np.nan}
    sr = group_df["selection_rate"].astype(float)
    dp = float(sr.max() - sr.min())
    di = float(sr.min() / sr.max()) if sr.max() > 0 else np.nan
    eo = float(group_df["TPR"].max() - group_df["TPR"].min()) if group_df["TPR"].notna().any() else np.nan
    return {"DP_diff": dp, "DI": di, "EO_diff": eo}


def global_metrics(y_true, y_pred, y_score):
    """Global model metrics."""
    acc = float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else np.nan
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {"acc": acc, "precision": float(prec), "recall": float(rec), "f1": float(f1), "roc_auc": auc}


# -----------------------------
# Mitigation: Reweighing (custom)
# -----------------------------
def reweighing_weights(sens, y):
    """Classic Kamiran–Calders-style weights: P(a)*P(y)/P(a,y)."""
    s = pd.Series(sens).astype(str)
    y = pd.Series(y).astype(int)
    P_a  = s.value_counts(normalize=True)
    P_y  = y.value_counts(normalize=True)
    P_ay = pd.crosstab(s, y, normalize=True)
    w = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        a_i, y_i = s.iloc[i], y.iloc[i]
        den = P_ay.loc[a_i, y_i] if (a_i in P_ay.index and y_i in P_ay.columns) else 0
        num = P_a[a_i] * P_y[y_i]
        w[i] = (num / den) if den > 0 else 1.0
    return w


# -----------------------------
# Post-processing thresholds
# -----------------------------
def fit_group_thresholds(y_true, y_score, sens, mode="EO", grid=np.linspace(0.1, 0.9, 17)):
    """Find a shared threshold (simple/stable) that best reduces EO or DP gap."""
    s = pd.Series(sens).astype(str).values
    groups = np.unique(s)
    best_thr = {g: 0.5 for g in groups}
    best_obj = np.inf
    for t in grid:
        thr_map = {g: t for g in groups}
        y_pred = np.array([1 if y_score[i] >= thr_map[s[i]] else 0 for i in range(len(y_true))])
        pg = per_group(y_true, y_pred, y_score, s)
        summ = fairness_summary(pg)
        obj = summ["EO_diff"] if mode == "EO" else summ["DP_diff"]
        if np.isnan(obj):
            continue
        if obj < best_obj:
            best_obj, best_thr = obj, thr_map
    return best_thr


def apply_group_thresholds(y_score, sens, thr_map):
    s = pd.Series(sens).astype(str).values
    return np.array([1 if y_score[i] >= thr_map[s[i]] else 0 for i in range(len(y_score))])


# -----------------------------
# End-to-end pipeline (IBM HR)
# -----------------------------
def run_pipeline(X_train, y_train, X_test, y_test, sens_train, sens_test, mode=MODE):
    # Baseline
    base = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    base.fit(X_train, y_train)
    base_scores = base.predict_proba(X_test)[:, 1]
    base_pred   = (base_scores >= 0.5).astype(int)
    base_pg     = per_group(y_test, base_pred, base_scores, sens_test)
    base_fair   = fairness_summary(base_pg)
    base_glob   = global_metrics(y_test, base_pred, base_scores)

    # Reweighed
    w = reweighing_weights(sens_train, y_train)
    rw = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    rw.fit(X_train, y_train, lr__sample_weight=w)
    rw_scores = rw.predict_proba(X_test)[:, 1]
    rw_pred   = (rw_scores >= 0.5).astype(int)
    rw_pg     = per_group(y_test, rw_pred, rw_scores, sens_test)
    rw_fair   = fairness_summary(rw_pg)
    rw_glob   = global_metrics(y_test, rw_pred, rw_scores)

    # Post-process thresholds (EO/DP) on reweighed scores
    thr_map   = fit_group_thresholds(y_test, rw_scores, sens_test, mode=mode)
    post_pred = apply_group_thresholds(rw_scores, sens_test, thr_map)
    post_pg   = per_group(y_test, post_pred, rw_scores, sens_test)
    post_fair = fairness_summary(post_pg)
    post_glob = global_metrics(y_test, post_pred, rw_scores)

    compare_global = pd.DataFrame([
        {"stage": "baseline",  **base_glob},
        {"stage": "reweighed", **rw_glob},
        {"stage": f"post_{mode}", **post_glob},
    ])
    compare_fair = pd.DataFrame([
        {"stage": "baseline",  **base_fair},
        {"stage": "reweighed", **rw_fair},
        {"stage": f"post_{mode}", **post_fair},
    ])

    return {
        "thresholds": thr_map,
        "global": compare_global,
        "fairness": compare_fair,
        "per_group": {"baseline": base_pg, "reweighed": rw_pg, f"post_{mode}": post_pg}
    }


# -----------------------------
# Policy gate (Allow / Review)
# -----------------------------
def policy_decision(fair_df,
                    min_di=MIN_DI, max_dp=MAX_DP, max_eo=MAX_EO,
                    min_precision=None, max_fpr=None, per_group_post=None):
    """
    Decide if a model is fair enough to use, based on policy thresholds.
    Returns "ALLOW" if fairness rules are met, else "REVIEW/BLOCK".
    """
    last = fair_df.set_index("stage").iloc[-1].to_dict()
    ok = (
        (np.isnan(last.get("DI", np.nan)) or last["DI"] >= min_di) and
        (np.isnan(last.get("DP_diff", np.nan)) or last["DP_diff"] <= max_dp) and
        (np.isnan(last.get("EO_diff", np.nan)) or last["EO_diff"] <= max_eo)
    )
    if ok and per_group_post is not None:
        pg = per_group_post
        if min_precision is not None and (pg["precision"].min() < min_precision):
            ok = False
        if max_fpr is not None and (pg["FPR"].max() > max_fpr):
            ok = False
    return "ALLOW" if ok else "REVIEW/BLOCK"


# -----------------------------
# Plotting utilities
# -----------------------------
def _check_and_cast(df: pd.DataFrame, req_cols, numeric_cols):
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Have: {list(df.columns)}")
    out = df.copy()
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(0.0)


def plot_global_metrics(compare_global: pd.DataFrame,
                        title="Global performance by stage",
                        out="global_metrics.png"):
    req = ["stage","acc","precision","recall","f1","roc_auc"]
    num = ["acc","precision","recall","f1","roc_auc"]
    df = _check_and_cast(compare_global, req, num)

    stages = df["stage"].astype(str).tolist()
    mcols  = ["acc","precision","recall","f1","roc_auc"]
    x = np.arange(len(stages))
    width = 0.16

    plt.figure(figsize=(10,5))
    for i, m in enumerate(mcols):
        vals = df[m].values
        bars = plt.bar(x + (i - (len(mcols)-1)/2)*width, vals, width, label=m)
        for b, v in zip(bars, vals):
            plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8, rotation=90)
    plt.xticks(x, stages)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()
    print(f"[saved] {out}")


def plot_fairness_metrics(compare_fair: pd.DataFrame,
                          title="Fairness metrics by stage",
                          out="fairness_metrics.png"):
    req = ["stage","DP_diff","DI","EO_diff"]
    num = ["DP_diff","DI","EO_diff"]
    df = _check_and_cast(compare_fair, req, num)

    stages = df["stage"].astype(str).tolist()
    mcols  = ["DP_diff","DI","EO_diff"]
    x = np.arange(len(stages))
    width = 0.22

    plt.figure(figsize=(10,5))
    for i, m in enumerate(mcols):
        vals = df[m].values
        bars = plt.bar(x + (i - (len(mcols)-1)/2)*width, vals, width, label=m)
        for b, v in zip(bars, vals):
            plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8, rotation=90)
    plt.xticks(x, stages)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()
    print(f"[saved] {out}")


def plot_per_group(per_group_df: pd.DataFrame,
                   title="Per-group (selection & error rates)",
                   out="per_group.png"):
    req = ["group","selection_rate","TPR","FPR","precision","recall","f1"]
    num = ["selection_rate","TPR","FPR","precision","recall","f1"]
    df = _check_and_cast(per_group_df, req, num)

    metrics = ["selection_rate","TPR","FPR","precision","recall","f1"]
    groups  = df["group"].astype(str).tolist()
    x = np.arange(len(groups))
    width = 0.12

    plt.figure(figsize=(11,5))
    for i, m in enumerate(metrics):
        vals = df[m].values
        bars = plt.bar(x + (i - (len(metrics)-1)/2)*width, vals, width, label=m)
        for b, v in zip(bars, vals):
            plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=7, rotation=90)
    plt.xticks(x, groups)
    plt.ylabel("Rate")
    plt.title(title)
    plt.legend(ncols=3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()
    print(f"[saved] {out}")


# -----------------------------
# Main demo (IBM HR CSV)
# -----------------------------
if __name__ == "__main__":
    # === Load your IBM HR data ===
    # Adjust path/column names as needed
    ibm = pd.read_csv("ibm_hr.csv")

    # Target: convert Yes/No to 1/0 (adjust if your column differs)
    ibm["y"] = ibm["Attrition"].map({"Yes": 1, "No": 0}).astype(int)

    # Sensitive attribute
    assert "Gender" in ibm.columns, "IBM HR must have 'Gender' column."
    sens_col = "Gender"

    # Features: drop target + sensitive (avoid leakage)
    drop_cols = ["y", "Attrition", sens_col]
    X_ibm = pd.get_dummies(
        ibm.drop(columns=[c for c in drop_cols if c in ibm.columns]),
        drop_first=True
    )
    y_ibm    = ibm["y"].values
    sens_ibm = ibm[sens_col].astype(str).values

    # Split
    Xtr, Xte, ytr, yte, s_tr, s_te = train_test_split(
        X_ibm, y_ibm, sens_ibm, test_size=0.3, random_state=42, stratify=y_ibm
    )

    # Run pipeline (MODE = "EO" or "DP")
    res = run_pipeline(Xtr, ytr, Xte, yte, s_tr, s_te, mode=MODE)

    print("IBM HR thresholds:", res["thresholds"])
    print("\nGLOBAL:\n", res["global"])
    print("\nFAIRNESS (DP_diff↓, DI↑→1, EO_diff↓):\n", res["fairness"])
    print(f"\nPER-GROUP (post_{MODE}):\n", res["per_group"][f'post_{MODE}'])

    # Policy decision
    decision = policy_decision(
        res["fairness"],
        per_group_post=res["per_group"][f"post_{MODE}"],
        min_precision=None,   # set if you want a floor per group
        max_fpr=None          # set if you want a cap per group
    )
    print("\nPolicy decision:", decision)

    # Plots (saved PNGs + shown)
    plot_global_metrics(res["global"],   title="IBM HR — Global metrics",   out="ibm_global.png")
    plot_fairness_metrics(res["fairness"], title="IBM HR — Fairness metrics", out="ibm_fairness.png")
    plot_per_group(res["per_group"][f"post_{MODE}"], title=f"IBM HR — Per-group (post-{MODE})", out="ibm_per_group.png")
