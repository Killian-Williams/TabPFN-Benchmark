# CLI uses revised A-v2 protocol; functions below retained for legacy imports/tests. See ../README.md.
r"""A-v2 CLI: this entry point delegates to ../benchmark_a.py.
See ../README.md for current commands, protocol, and output locations.
The historical description below applies ONLY to retained legacy functions
used by older tests and analysis scripts, not to this file's current CLI.

Experiment A: Olist canceled-versus-delivered orders under data scarcity.

From the project root:
    .\.venv\Scripts\python.exe "experiment A\orders\olist_orders_A.py" --run
    .\.venv\Scripts\python.exe "experiment A\orders\olist_orders_A.py" --smoke
    .\.venv\Scripts\python.exe "experiment A\orders\olist_orders_A.py" --plot-only

Self-contained; reads data\olist_orders_dataset.csv without changing it.
Class 1=canceled, 0=delivered; exclude all other (including unfinished)
statuses. This is a retrospective comparison conditioned on these outcomes,
not a calibrated cancellation forecast for all orders.

Exclude IDs, raw dates, status and post-purchase approval/actual shipping/
delivery timestamps, including their missingness. Use purchase year, month,
weekday, hour and estimated-delivery minus purchase interval in days.
The estimate is assumed to represent the purchase-time promise, but the
file has no timestamped snapshots proving it was unchanged. No external
customer/items/payment files are joined. Features are limited calendar/
promise information, not comprehensive order characteristics.

Remove exact raw duplicates, reject missing/duplicated order IDs, invalid
purchase dates, malformed estimates and estimates preceding purchase.
Missing estimates remain missing and are imputed using training data.
Split orders 80/20 stratified with seed 42. Select a 2,000-order training
pool and fixed 1,000-order test. Shared nested stratified subsets at
100/50/25/10/5/1% retain at least one example of each class without
replacement. This distorts tiny subsets' natural prevalence; actual counts
and rates are recorded. No customer-level generalization is claimed.

Reuse the seven untuned retail A configurations. All features are numeric,
so no categorical encoder or categorical TabPFN indices are needed.
Fresh median imputation per fit; scaling only for logistic regression.
TabPFN uses one v2 CPU estimator, memory saving, explicit large-CPU-data
override and retained pretraining limits. No tuning, reweighting,
oversampling, calibration or threshold optimization.

Metrics: accuracy, macro F1, log loss, cancellation F1/precision/recall,
AP and ROC-AUC from class-1 probabilities; fixed >=0.5 threshold.
Undefined precision=0 with explicit flag/warning. Rare cancellations make
accuracy misleading. Single-seed results have no uncertainty estimates.
Save 42 attempts, failures, membership/hashes, configuration/versions,
and CSV-derived PNG/SVG beside this script. Smoke uses 200/100 orders
in a separate smoke folder. Failures prevent incomplete success graphs.
"""

import argparse
import hashlib
import json
import logging
import platform
import warnings
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator, PercentFormatter
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.inference_config import InferenceConfig
from tabpfn.settings import settings
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
PERCENTAGES = (100, 50, 25, 10, 5, 1)
FEATURES = ("purchase_year", "purchase_month", "purchase_weekday", "purchase_hour", "estimated_delivery_days")
EXCLUDED = ("order_id", "customer_id", "order_status", "order_purchase_timestamp",
            "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date", "order_estimated_delivery_date")
METRICS = ("accuracy", "macro_f1", "log_loss", "f1", "precision", "recall", "average_precision", "roc_auc")
PLOT_METRICS = ("accuracy", "macro_f1", "average_precision")
STYLES = {
    "TabPFN v2": ("#0072B2", "o", "-"), "XGBoost": ("#D55E00", "s", "-"),
    "Logistic regression": ("#CC79A7", "P", "--"), "Random forest": ("#8064A2", "^", ":"),
    "LightGBM": ("#999999", "v", "-."), "CatBoost": ("#009E73", "D", "--"),
    "Dummy (prior)": ("#444444", "x", ":"),
}
LOG = logging.getLogger("olist_A")


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if not set(EXCLUDED).issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(set(EXCLUDED).difference(raw.columns))}")
    if not raw.index.is_unique:
        raise ValueError("Source indices must be unique")
    profile = {
        "source_rows": len(raw), "missing_values": raw.isna().sum().to_dict(),
        "exact_duplicates_removed": int(raw.duplicated().sum()),
        "status_counts": raw.order_status.value_counts(dropna=False).to_dict(),
    }
    frame = raw.drop_duplicates().copy()
    if frame.order_id.isna().any() or frame.order_id.astype("string").str.strip().eq("").any():
        raise ValueError("Missing order IDs are not supported")
    if frame.order_id.duplicated().any():
        raise ValueError("Conflicting duplicate order IDs; resolve them before benchmarking")
    eligible = frame.order_status.isin(["canceled", "delivered"])
    profile["excluded_status_rows"] = int((~eligible).sum())
    frame = frame.loc[eligible].copy()
    frame["is_canceled"] = frame.order_status.eq("canceled").astype(int)
    purchase = pd.to_datetime(frame.order_purchase_timestamp, errors="coerce")
    estimate = pd.to_datetime(frame.order_estimated_delivery_date, errors="coerce")
    if purchase.isna().any():
        raise ValueError("Eligible orders contain invalid/missing purchase timestamps")
    if (frame.order_estimated_delivery_date.notna() & estimate.isna()).any():
        raise ValueError("Eligible orders contain malformed estimated delivery dates")
    interval = (estimate - purchase).dt.total_seconds() / 86400
    if interval.lt(0).any():
        raise ValueError("Estimated delivery precedes purchase for eligible orders")
    frame["purchase_year"] = purchase.dt.year
    frame["purchase_month"] = purchase.dt.month
    frame["purchase_weekday"] = purchase.dt.dayofweek
    frame["purchase_hour"] = purchase.dt.hour
    frame["estimated_delivery_days"] = interval
    if set(frame.is_canceled.unique()) != {0, 1}:
        raise ValueError("Both canceled and delivered classes are required")
    profile.update(eligible_rows=len(frame), date_min=str(purchase.min()), date_max=str(purchase.max()),
                   canceled_rows=int(frame.is_canceled.sum()), cancellation_rate=float(frame.is_canceled.mean()),
                   feature_missing_values=frame[list(FEATURES)].isna().sum().to_dict())
    return frame[["order_id", "is_canceled", *FEATURES]], profile


def stratified_sample(frame: pd.DataFrame, rows: int, seed: int = 42) -> pd.DataFrame:
    counts = frame.is_canceled.value_counts()
    if rows < 2 or rows > len(frame) or set(counts.index) != {0, 1}:
        raise ValueError(f"Cannot sample {rows} distinct two-class orders from {len(frame)}")
    if rows == len(frame):
        return frame.copy()
    positives = int(np.clip(round(rows * counts[1] / len(frame)), 1, rows - 1))
    if positives > counts[1] or rows - positives > counts[0]:
        raise ValueError("Insufficient orders in a class; no replacement allowed")
    return pd.concat([
        frame.loc[frame.is_canceled == 0].sample(n=rows - positives, random_state=seed),
        frame.loc[frame.is_canceled == 1].sample(n=positives, random_state=seed),
    ]).sample(frac=1, random_state=seed)


def make_pools(frame: pd.DataFrame, training_rows: int, test_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not frame.index.is_unique or frame.order_id.duplicated().any():
        raise ValueError("Orders and source indices must be unique")
    train_pool, test_pool = train_test_split(frame, test_size=0.2, stratify=frame.is_canceled, random_state=42)
    train = stratified_sample(train_pool, training_rows)
    test = stratified_sample(test_pool, test_rows)
    if set(train.order_id).intersection(test.order_id):
        raise ValueError("Order overlap between training and test")
    return train, test


def build_models() -> list[tuple[str, Pipeline]]:
    tabpfn = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=1)
    tabpfn.set_params(categorical_features_indices=[], memory_saving_mode=True)
    classifiers = [
        ("TabPFN v2", tabpfn),
        ("Logistic regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
        ("XGBoost", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", n_estimators=100, max_depth=3,
            learning_rate=0.1, tree_method="hist", device="cpu", random_state=42, n_jobs=1,
        )),
        ("LightGBM", LGBMClassifier(
            objective="binary", n_estimators=100, max_depth=3, num_leaves=7, learning_rate=0.1,
            device_type="cpu", random_state=42, n_jobs=1, verbosity=-1,
        )),
        ("CatBoost", CatBoostClassifier(
            iterations=100, depth=3, learning_rate=0.1, loss_function="Logloss", task_type="CPU",
            random_seed=42, thread_count=1, verbose=False, allow_writing_files=False,
        )),
        ("Dummy (prior)", DummyClassifier(strategy="prior", random_state=42)),
    ]
    return [
        (name, make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            *([StandardScaler()] if name == "Logistic regression" else []), classifier,
        ))
        for name, classifier in classifiers
    ]


def evaluate(labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict:
    if (
        labels.ndim != 1 or classes.shape != (2,) or set(classes) != {0, 1}
        or probabilities.shape != (len(labels), 2) or set(np.unique(labels)) != {0, 1}
        or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Expected both binary classes and valid two-class probabilities")
    scores = probabilities[:, np.flatnonzero(classes == 1)[0]]
    predictions = (scores >= 0.5).astype(int)
    if not predictions.any():
        LOG.warning("No cancellations predicted at 0.5; undefined precision recorded as zero")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(labels, probabilities[:, np.argsort(classes)], labels=[0, 1])),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)), "precision_defined": bool(predictions.any()),
        "predicted_cancellations": int(predictions.sum()),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def sample_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.index.to_numpy(dtype=np.int64).tobytes()).hexdigest()


def run_benchmark(frame: pd.DataFrame, output: Path, training_rows: int, test_rows: int) -> pd.DataFrame:
    selected, test = make_pools(frame, training_rows, test_rows)
    models = build_models()
    results, membership = [], []

    def record(role, percentage, data):
        membership.append(pd.DataFrame({
            "role": role, "training_percentage": percentage, "source_index": data.index,
            "order_id": data.order_id.to_numpy(), "is_canceled": data.is_canceled.to_numpy(),
        }))

    record("test", 0, test)
    for percentage in PERCENTAGES:
        selected = stratified_sample(selected, max(2, training_rows * percentage // 100))
        record("train", percentage, selected)
        LOG.info("%s%%: %s training orders (%s canceled), %s fixed test (%s canceled)",
                 percentage, len(selected), int(selected.is_canceled.sum()), len(test), int(test.is_canceled.sum()))
        for name, template in models:
            LOG.info("Evaluating %s", name)
            row = {
                "model": name, "training_percentage": percentage, "training_rows": len(selected),
                "test_rows": len(test), "training_pool_rows": training_rows,
                "train_canceled_rows": int(selected.is_canceled.sum()), "test_canceled_rows": int(test.is_canceled.sum()),
                "train_cancellation_rate": float(selected.is_canceled.mean()),
                "test_cancellation_rate": float(test.is_canceled.mean()),
                "training_sample_hash": sample_hash(selected), "test_sample_hash": sample_hash(test),
                "threshold": 0.5, "positive_class": 1, "status": "failed", "error": "", "warnings": "",
                **{key: np.nan for key in (*METRICS, "fitting_time_s", "prediction_time_s")},
            }
            stage = "fit"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    start = perf_counter()
                    model = clone(template)
                    model.fit(selected[list(FEATURES)], selected.is_canceled)
                    row["fitting_time_s"] = perf_counter() - start
                    stage = "predict"
                    start = perf_counter()
                    probabilities = model.predict_proba(test[list(FEATURES)])
                    row["prediction_time_s"] = perf_counter() - start
                    stage = "metrics"
                    row.update(evaluate(test.is_canceled.to_numpy(), np.asarray(probabilities), np.asarray(model.classes_)))
                    row["status"] = "success"
                except (ValueError, RuntimeError, TypeError, MemoryError, OSError, CatBoostError, LightGBMError) as error:
                    if stage in ("fit", "predict"):
                        row["fitting_time_s" if stage == "fit" else "prediction_time_s"] = perf_counter() - start
                    row["error"] = f"{stage}: {type(error).__name__}: {error}"
                    LOG.exception("Recorded failed run: %s", row["error"])
            row["warnings"] = " | ".join(str(w.message) for w in caught)
            if row["warnings"]:
                LOG.warning("%s: %s", name, row["warnings"])
            results.append(row)
            pd.DataFrame(results).to_csv(output / "olist_benchmark_results.csv.tmp", index=False)
    pd.concat(membership, ignore_index=True).to_csv(output / "sample_membership.csv", index=False)
    (output / "olist_benchmark_results.csv.tmp").replace(output / "olist_benchmark_results.csv")
    result = pd.DataFrame(results)
    result.loc[result.status != "success"].to_csv(output / "failures.csv", index=False)
    return result


def create_plot(results: pd.DataFrame) -> Figure:
    required = {"model", "status", "training_percentage", "training_rows", "training_pool_rows",
                "test_rows", "training_sample_hash", "test_sample_hash", *PLOT_METRICS}
    if not required.issubset(results.columns):
        raise ValueError(f"Missing plotting columns: {sorted(required.difference(results.columns))}")
    if (
        len(results) != 42 or not results.status.eq("success").all() or set(results.model) != set(STYLES)
        or results.duplicated(["model", "training_percentage"]).any()
    ):
        raise ValueError("Expected all 42 successful model/percentage runs; inspect failures.csv")
    for column in ("training_pool_rows", "test_rows", "test_sample_hash"):
        if results[column].isna().any() or results[column].nunique() != 1:
            raise ValueError(f"Expected fixed {column}")
    if results.training_sample_hash.isna().any() or not results.groupby("training_percentage").training_sample_hash.nunique().eq(1).all():
        raise ValueError("Models must share training samples")
    pool = int(results.training_pool_rows.iloc[0])
    for name, group in results.groupby("model"):
        group = group.sort_values("training_percentage")
        if (
            group.training_percentage.tolist() != sorted(PERCENTAGES)
            or group.training_rows.tolist() != [max(2, pool * p // 100) for p in sorted(PERCENTAGES)]
        ):
            raise ValueError(f"Incorrect training percentages or counts for {name}")
    scores = results[list(PLOT_METRICS)].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Plot scores must be finite and in [0,1]")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for name, group in results.groupby("model", sort=False):
        group = group.sort_values("training_percentage")
        color, marker, linestyle = STYLES[name]
        for ax, metric in zip(axes, PLOT_METRICS):
            ax.plot(group.training_percentage, group[metric], label=name, color=color,
                    marker=marker, linestyle=linestyle,
                    linewidth=2.8 if name in ("TabPFN v2", "XGBoost") else 1.6)
    for ax, label in zip(axes, ("Test accuracy", "Macro F1", "Average precision (AP)")):
        ax.set_xscale("log")
        ax.set_xticks(sorted(PERCENTAGES),
                      labels=[f"{p}%\n({max(2, pool * p // 100)} orders)" for p in sorted(PERCENTAGES)])
        ax.tick_params(axis="x", labelsize=8, labelrotation=25)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set(xlabel="Training pool used (log scale)", ylabel=label, title=label, ylim=(0, 1.03))
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(loc="best", title="Models", fontsize=8)
    fig.suptitle(f"Experiment A: Olist canceled vs delivered ({pool:,}-order training pool)")
    fig.text(0.5, 0.025, f"Fixed {int(results.test_rows.iloc[0]):,} test orders | "
             "At least one training order per class | Single run per point | Limited calendar/promise features",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def plot_results(output: Path, *, show: bool = False) -> None:
    fig = create_plot(pd.read_csv(output / "olist_benchmark_results.csv"))
    for extension in ("png", "svg"):
        path = output / f"olist_performance.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        LOG.info("Graph saved to %s", path)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--smoke", action="store_true", help="Use 200/100 orders in separate smoke folder")
    modes.add_argument("--plot-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not (args.run or args.smoke or args.plot_only):
        parser.print_help()
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output = Path(__file__).resolve().parent / ("smoke" if args.smoke else "")
    output.mkdir(exist_ok=True)
    if not args.plot_only:
        training_rows, test_rows = (200, 100) if args.smoke else (2000, 1000)
        limits = InferenceConfig.get_default("multiclass", ModelVersion.V2)
        if training_rows > limits.MAX_NUMBER_OF_SAMPLES or len(FEATURES) > limits.MAX_NUMBER_OF_FEATURES:
            raise ValueError("Samples exceed installed TabPFN v2 pretraining limits")
        source = ROOT / "data" / "olist_orders_dataset.csv"
        frame, profile = prepare_data(pd.read_csv(source))
        LOG.info("Dataset profile: %s", profile)
        configuration = {
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "experiment": "A", "smoke": args.smoke, "training_pool_rows": training_rows, "test_rows": test_rows,
            "percentages": PERCENTAGES, "seed": 42, "threshold": 0.5, "positive_class": 1,
            "target": "canceled=1 vs delivered=0; all other statuses excluded", "features": FEATURES,
            "excluded": EXCLUDED, "profile": profile, "python": platform.python_version(),
            "minimum_training_rows_per_class": 1, "cpu_large_dataset_override": training_rows > 1000,
            "estimated_delivery_assumption": "Purchase-time promise; unchanged snapshot not verified",
            "versions": {name: version(name) for name in (
                "tabpfn", "scikit-learn", "pandas", "numpy", "xgboost", "lightgbm", "catboost", "matplotlib",
            )},
            "model_parameters": {name: model.get_params() for name, model in build_models()},
            "status": "running",
        }
        config_path = output / "olist_config.json"
        config_path.write_text(json.dumps(configuration, indent=2, default=str), encoding="utf-8")
        previous = settings.tabpfn.allow_cpu_large_dataset
        settings.tabpfn.allow_cpu_large_dataset = training_rows > 1000
        try:
            results = run_benchmark(frame, output, training_rows, test_rows)
        finally:
            settings.tabpfn.allow_cpu_large_dataset = previous
        failures = int(results.status.ne("success").sum())
        configuration.update(status="completed_with_failures" if failures else "completed", failures=failures)
        config_path.write_text(json.dumps(configuration, indent=2, default=str), encoding="utf-8")
        if failures:
            raise SystemExit(f"{failures} runs failed; inspect {output / 'failures.csv'}")
    plot_results(output, show=args.show)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark_a import main as revised_main
    revised_main('orders')