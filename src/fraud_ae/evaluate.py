"""End-to-end evaluation of the frozen model. No training of the autoencoder happens here.

    python -m fraud_ae.evaluate        (from repo root with PYTHONPATH=src)

Writes: models/threshold.json, reports/metrics.json, docs/figures/*.png
Design of the evaluation is explained in docs/REPORT.md section 6.
"""
import json
import warnings

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from . import config, thresholds
from .features import Preprocessor
from .model import anomaly_score, feature_errors, load_autoencoder

warnings.filterwarnings("ignore")
SURFACE, INK, INK2, NORMAL_C, FRAUD_C = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 10})
TARGET_FPR = 0.01
ASSUMED_PREVALENCE = 0.01  # for a realistic precision estimate (the test set itself is 20% fraud)


def split():
    train = pd.read_csv(config.TRAIN_CSV)
    fraud = pd.read_csv(config.FRAUD_CSV)
    idx_tr, idx_val = train_test_split(np.arange(len(train)), test_size=config.VAL_FRACTION,
                                       random_state=config.SPLIT_SEED)  # identical rows to the notebook split
    idx_cal, idx_test = train_test_split(idx_val, test_size=0.5, random_state=0)
    return train, train.iloc[idx_tr], train.iloc[idx_cal], train.iloc[idx_test], fraud


def metrics_at(thr, s_norm, s_fraud):
    fpr, rec = float((s_norm > thr).mean()), float((s_fraud > thr).mean())
    p = ASSUMED_PREVALENCE
    prec = rec * p / (rec * p + fpr * (1 - p)) if rec + fpr > 0 else float("nan")
    return {"threshold": thr, "fpr_normal_test": fpr, "recall_fraud": rec,
            f"precision_at_{p:.0%}_prevalence": prec}


def legacy_refit_transform(df):
    """What the hackathon api.py did: refit encoders + scaler on the incoming batch (the bug)."""
    X = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns]).copy()
    for c in config.CATEGORICAL_COLS:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return MinMaxScaler().fit_transform(X.astype(float)).astype("float32")


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(exist_ok=True)
    train, tr, cal, test, fraud = split()
    pre, model = Preprocessor.load(), load_autoencoder()
    X = {k: pre.transform(d) for k, d in {"train": tr.sample(20_000, random_state=0), "cal": cal,
                                          "test": test, "fraud": fraud}.items()}
    s = {k: anomaly_score(model, v) for k, v in X.items()}
    y = np.r_[np.zeros(len(s["test"])), np.ones(len(s["fraud"]))]
    yscore = np.r_[s["test"], s["fraud"]]
    out = {"data": {"train_rows": len(tr), "calibration_normals": len(cal), "test_normals": len(test),
                    "test_fraud": len(fraud)},
           "score_summary": {k: {"median": float(np.median(v)), "p99": float(np.quantile(v, .99)),
                                 "max": float(v.max())} for k, v in s.items()},
           "roc_auc": roc_auc_score(y, yscore), "average_precision": average_precision_score(y, yscore)}

    # 1. thresholds: every rule fitted on calibration normals only
    out["thresholds"] = {name: metrics_at(rule(s["cal"]), s["test"], s["fraud"]) for name, rule in thresholds.RULES.items()}
    adopted = thresholds.quantile(s["cal"], TARGET_FPR)
    config.THRESHOLD_PATH.write_text(json.dumps({
        "value": adopted, "rule": f"quantile(1 - {TARGET_FPR}) of calibration-normal scores",
        "calibration_rows": len(cal), "model": config.MODEL_PATH.name}, indent=2))

    # 2. checkpoint comparison (same preprocessing; see caveat in docs)
    ckpts = {"night_4.h5 (shipped)": model}
    for p in sorted((config.MODEL_DIR / "legacy_checkpoints").glob("*.h5")):
        ckpts[p.name] = load_autoencoder(p)
    out["checkpoints"] = {}
    for name, m in ckpts.items():
        sc, st, sf = (anomaly_score(m, X[k]) for k in ("cal", "test", "fraud"))
        out["checkpoints"][name] = {"params": int(m.count_params()), "val_mse_normal": float(st.mean()),
                                    "roc_auc": roc_auc_score(y, np.r_[st, sf]),
                                    **metrics_at(thresholds.quantile(sc, TARGET_FPR), st, sf)}

    # 3. baseline: IsolationForest on the exact same features
    iso = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(X["train"])
    out["baseline_isolation_forest_roc_auc"] = roc_auc_score(y, -iso.score_samples(np.r_[X["test"], X["fraud"]]))

    # 4. single-signal recall: only feature group g is fraudulent, all else copied from a normal user
    rng = np.random.default_rng(0)
    donors = X["test"][rng.integers(0, len(X["test"]), len(X["fraud"]))]
    feats = pre.features
    out["single_group_recall"] = {}
    for g, cols in config.FEATURE_GROUPS.items():
        idx = [feats.index(c) for c in cols]
        mix = donors.copy()
        mix[:, idx] = X["fraud"][:, idx]
        out["single_group_recall"][g] = float((anomaly_score(model, mix) > adopted).mean())
    out["single_feature_recall"] = {}
    for i, c in enumerate(feats):  # finer grain: which lone features are enough to trigger an alert
        mix = donors.copy()
        mix[:, i] = X["fraud"][:, i]
        out["single_feature_recall"][c] = float((anomaly_score(model, mix) > adopted).mean())
    out["single_feature_recall"] = dict(sorted(out["single_feature_recall"].items(), key=lambda kv: -kv[1]))
    no_scores = donors.copy()
    keep = [i for i, c in enumerate(feats) if c not in config.FEATURE_GROUPS["precomputed_scores"]]
    no_scores[:, keep] = X["fraud"][:, keep]
    out["recall_without_precomputed_scores"] = float((anomaly_score(model, no_scores) > adopted).mean())

    # 5. per-feature attribution: where does the reconstruction error of fraud come from?
    fe_f, fe_n = feature_errors(model, X["fraud"]).mean(0), feature_errors(model, X["test"]).mean(0)
    attr = pd.Series(fe_f / fe_f.sum(), index=feats).sort_values(ascending=False)
    out["top_error_features_fraud"] = attr.head(10).round(4).to_dict()

    # 6. train/serve skew in the original API (posting fraud_night_1.csv as call.py did)
    legacy = anomaly_score(model, legacy_refit_transform(fraud))
    out["legacy_api_on_fraud_file"] = {"flagged_share_p96_rule": float((legacy > np.percentile(legacy, 96)).mean()),
                                       "flagged_share_frozen_pipeline": float((s["fraud"] > adopted).mean())}
    mixed = pd.concat([test.sample(2000, random_state=1), fraud.drop(columns="Unnamed: 0").head(50)])
    lm = anomaly_score(model, legacy_refit_transform(mixed))
    out["legacy_api_on_mixed_batch"] = {"fraud_rows": 50, "rows": len(mixed),
                                        "fraud_caught_p96_refit": int((lm[-50:] > np.percentile(lm, 96)).sum()),
                                        "fraud_caught_frozen": int((anomaly_score(model, pre.transform(mixed))[-50:] > adopted).sum())}

    config.REPORT_DIR.joinpath("metrics.json").write_text(json.dumps(out, indent=2, default=float))
    figures(s, adopted, y, yscore, attr, out)
    print(json.dumps({k: out[k] for k in ("roc_auc", "average_precision", "baseline_isolation_forest_roc_auc",
                                         "recall_without_precomputed_scores")}, indent=2))
    print(pd.DataFrame(out["thresholds"]).T.round(4).to_string())
    print(pd.DataFrame(out["checkpoints"]).T.round(4).to_string())
    print(out["single_group_recall"], out["legacy_api_on_fraud_file"], out["legacy_api_on_mixed_batch"])


def figures(s, thr, y, yscore, attr, out):
    # score distributions, log-x because fraud errors are orders of magnitude larger
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(np.log10(min(s["test"].min(), s["fraud"].min())), np.log10(s["fraud"].max()), 80)
    ax.hist(s["test"], bins=bins, color=NORMAL_C, alpha=.85)
    ax.hist(s["fraud"], bins=bins, color=FRAUD_C, alpha=.85)
    ax.axvline(thr, color=INK, lw=1.2, ls="--")
    top = ax.get_ylim()[1]
    ax.text(thr * 1.1, top * .55, f"threshold {thr:.4f}\n(1% FPR on calibration)", color=INK, fontsize=9)
    ax.text(np.quantile(s["test"], .02), top * .6, f"normal, held-out\n(n={len(s['test']):,})", color=INK2, fontsize=9, ha="right")
    ax.text(np.median(s["fraud"]), top * .4, f"fraud\n(n={len(s['fraud']):,})", color=INK2, fontsize=9, ha="center")
    ax.set(xscale="log", xlabel="reconstruction MSE (log scale)", ylabel="users", title="Anomaly score by population")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "score_distribution.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    f, t, _ = roc_curve(y, yscore)
    axes[0].plot(f, t, color=NORMAL_C, lw=2); axes[0].plot([0, 1], [0, 1], color=INK2, lw=.8, ls=":")
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title=f"ROC (AUC {out['roc_auc']:.4f})")
    p, r, _ = precision_recall_curve(y, yscore)
    axes[1].plot(r, p, color=NORMAL_C, lw=2)
    axes[1].set(xlabel="recall", ylabel="precision", title=f"Precision-recall (AP {out['average_precision']:.4f})", ylim=(0, 1.02))
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "roc_pr.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    top = attr.head(15)[::-1]
    ax.barh(top.index, top.values, color=FRAUD_C, height=.7)
    for i, v in enumerate(top.values):
        ax.text(v, i, f" {v:.1%}", va="center", fontsize=8, color=INK2)
    ax.set(xlabel="share of total fraud reconstruction error", title="Which features make fraud look anomalous")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "feature_attribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    g = pd.Series(out["single_group_recall"]).sort_values()
    ax.barh(g.index, g.values, color=NORMAL_C, height=.7)
    for i, v in enumerate(g.values):
        ax.text(v, i, f" {v:.1%}", va="center", fontsize=8, color=INK2)
    ax.set(xlim=(0, 1.12), xlabel="recall at adopted threshold",
           title="Recall when only ONE feature group is fraudulent")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "single_group_recall.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
