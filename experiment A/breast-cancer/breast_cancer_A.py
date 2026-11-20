# CLI uses revised A-v2 protocol; functions below retained for legacy imports/tests. See ../README.md.
r"""A-v2 CLI: this entry point delegates to ../benchmark_a.py.
See ../README.md for current commands, protocol, and output locations.
The historical description below applies ONLY to retained legacy functions
used by older tests and analysis scripts, not to this file's current CLI.

Experiment A: benign versus malignant classification under data scarcity.

From the project root:
    .\.venv\Scripts\python.exe "experiment A\breast-cancer\breast_cancer_A.py" --run
    .\.venv\Scripts\python.exe "experiment A\breast-cancer\breast_cancer_A.py" --smoke
    .\.venv\Scripts\python.exe "experiment A\breast-cancer\breast_cancer_A.py" --plot-only

Standalone; reads data\breast-cancer.csv unchanged. diagnosis M -> 1
(malignant), B -> 0 (benign). All 30 mean/SE/worst measurements are numeric
predictors; id and diagnosis excluded. No test-driven feature selection.
Exact raw duplicates removed keeping first; conflicting duplicate IDs,
missing IDs, invalid labels and malformed/infinite measurements rejected.
Missing predictor measurements, if present, remain for training-only imputation.

Stratified 80/20 split, seed 42: 455 training / 114 fixed test records for
the supplied 569-row file. Shared nested subsets at 100/50/25/10/5/1% of
the TRAINING pool, flooring sizes: 455/227/113/45/22/4. Both classes retained,
no replacement; tiny-subset prevalence changes due to rounding are recorded.
Every model receives the same rows at each level and the same test rows.

Seven untuned settings match existing Experiment A: TabPFN v2, logistic
regression, random forest, XGBoost, LightGBM, CatBoost, dummy prior.
Fresh median imputation fitted on each selected training subset; scaling
only for logistic regression. All inputs numeric: no category encoders.
TabPFN uses public v2 weights, one CPU estimator, memory saving and no
categorical indices, retaining pretraining and default CPU-size limits.
No tuning, weighting, oversampling, calibration or threshold optimization.

Class-1 probabilities for ROC-AUC/AP; fixed >=0.5 classification threshold.
Report accuracy, Macro F1, log loss, malignant F1/precision/recall,
AP, ROC-AUC, confusion counts and separate fit/predict times including
preprocessing. Undefined precision=0 with warning and explicit flag.

Save 42 attempts, failures, selected membership/hashes, configuration and
versions, CSV-derived accuracy/Macro F1/ROC-AUC PNG/SVG beside this script.
Incremental .csv.tmp replaces final CSV after all attempts. Failed attempts
remain recorded, cause nonzero exit and prevent incomplete performance plots.
Smoke uses 100 training records from the same split in a separate directory.
No mode prints help. Single-seed exploratory benchmark: not clinical
validation, not a diagnostic tool, no uncertainty or superiority claims.
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
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
PERCENTAGES = (100, 50, 25, 10, 5, 1)
MEASUREMENTS = ("radius", "texture", "perimeter", "area", "smoothness", "compactness",
                "concavity", "concave points", "symmetry", "fractal_dimension")
FEATURES = tuple(f"{measurement}_{suffix}" for suffix in ("mean", "se", "worst") for measurement in MEASUREMENTS)
EXCLUDED = ("id", "diagnosis")
METRICS = ("accuracy", "macro_f1", "log_loss", "f1", "precision", "recall", "average_precision", "roc_auc")
PLOT_METRICS = ("accuracy", "macro_f1", "roc_auc")
STYLES = {
    "TabPFN v2": ("#0072B2", "o", "-"), "XGBoost": ("#D55E00", "s", "-"),
    "Logistic regression": ("#CC79A7", "P", "--"), "Random forest": ("#8064A2", "^", ":"),
    "LightGBM": ("#999999", "v", "-."), "CatBoost": ("#009E73", "D", "--"),
    "Dummy (prior)": ("#444444", "x", ":"),
}
LOG = logging.getLogger("breast_cancer_A")


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {*EXCLUDED, *FEATURES}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(required.difference(raw.columns))}")
    if not raw.index.is_unique:
        raise ValueError("Source row indices must be unique")
    profile = {
        "source_rows": len(raw), "missing_values": raw.isna().sum().to_dict(),
        "exact_duplicates_removed": int(raw.duplicated().sum()),
    }
    frame = raw.drop_duplicates().copy()
    if frame.id.isna().any() or frame.id.astype("string").str.strip().eq("").any():
        raise ValueError("Record IDs must not be missing")
    if frame.id.duplicated().any():
        raise ValueError("Conflicting duplicate record IDs")
    if frame.diagnosis.isna().any() or set(frame.diagnosis.unique()) != {"B", "M"}:
        raise ValueError("diagnosis must contain both B/M labels and no missing labels")
    frame["is_malignant"] = frame.diagnosis.map({"B": 0, "M": 1}).astype(int)
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite {column} values are not supported")
    profile.update(
        eligible_rows=len(frame), benign_rows=int((frame.is_malignant == 0).sum()),
        malignant_rows=int(frame.is_malignant.sum()), malignant_rate=float(frame.is_malignant.mean()),
        feature_missing_values=frame[list(FEATURES)].isna().sum().to_dict(),
    )
    return frame[["id", "is_malignant", *FEATURES]], profile


def stratified_sample(frame: pd.DataFrame, rows: int, seed: int = 42) -> pd.DataFrame:
    counts = frame.is_malignant.value_counts()
    if rows < 2 or rows > len(frame) or set(counts.index) != {0, 1}:
        raise ValueError(f"Cannot select {rows} distinct two-class records from {len(frame)}")
    if rows == len(frame):
        return frame.copy()
    positive = int(np.clip(round(rows * counts[1] / len(frame)), 1, rows - 1))
    if positive > counts[1] or rows - positive > counts[0]:
        raise ValueError("Insufficient records in a class; no replacement allowed")
    return pd.concat([
        frame.loc[frame.is_malignant == 0].sample(n=rows - positive, random_state=seed),
        frame.loc[frame.is_malignant == 1].sample(n=positive, random_state=seed),
    ]).sample(frac=1, random_state=seed)


def make_pools(frame: pd.DataFrame, smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not frame.index.is_unique or frame.id.duplicated().any():
        raise ValueError("Record IDs and source indices must be unique")
    train, test = train_test_split(frame, test_size=0.2, stratify=frame.is_malignant, random_state=42)
    if smoke:
        train = stratified_sample(train, min(100, len(train)))
    if set(train.id).intersection(test.id):
        raise ValueError("Record overlap between training and test")
    if set(train.is_malignant) != {0, 1} or set(test.is_malignant) != {0, 1}:
        raise ValueError("Both split populations must contain both target classes")
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
        LOG.warning("No malignant cases predicted at 0.5; undefined precision recorded as zero")
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
        "predicted_malignant": int(predictions.sum()),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def sample_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.index.to_numpy(dtype=np.int64).tobytes()).hexdigest()


def run_benchmark(frame: pd.DataFrame, output: Path, *, smoke: bool = False) -> pd.DataFrame:
    selected, test = make_pools(frame, smoke)
    pool_rows = len(selected)
    limits = InferenceConfig.get_default("multiclass", ModelVersion.V2)
    if pool_rows > min(1000, limits.MAX_NUMBER_OF_SAMPLES) or len(FEATURES) > limits.MAX_NUMBER_OF_FEATURES:
        raise ValueError("Data exceed TabPFN v2 pretraining/default CPU sample limits")
    models = build_models()
    results, membership = [], []

    def record(role, percentage, data):
        membership.append(pd.DataFrame({
            "role": role, "training_percentage": percentage, "source_index": data.index,
            "id": data.id.to_numpy(), "is_malignant": data.is_malignant.to_numpy(),
        }))

    record("test", 0, test)
    for percentage in PERCENTAGES:
        selected = stratified_sample(selected, max(2, pool_rows * percentage // 100))
        record("train", percentage, selected)
        LOG.info("%s%%: %s training records (%s malignant), %s fixed test (%s malignant)",
                 percentage, len(selected), int(selected.is_malignant.sum()), len(test), int(test.is_malignant.sum()))
        for name, template in models:
            LOG.info("Evaluating %s", name)
            row = {
                "model": name, "training_percentage": percentage, "training_rows": len(selected),
                "test_rows": len(test), "training_pool_rows": pool_rows,
                "train_malignant_rows": int(selected.is_malignant.sum()),
                "test_malignant_rows": int(test.is_malignant.sum()),
                "train_malignant_rate": float(selected.is_malignant.mean()),
                "test_malignant_rate": float(test.is_malignant.mean()),
                "training_sample_hash": sample_hash(selected), "test_sample_hash": sample_hash(test),
                "positive_class": 1, "threshold": 0.5, "status": "failed", "error": "", "warnings": "",
                **{key: np.nan for key in (*METRICS, "fitting_time_s", "prediction_time_s")},
            }
            stage = "fit"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    start = perf_counter()
                    model = clone(template)
                    model.fit(selected[list(FEATURES)], selected.is_malignant)
                    row["fitting_time_s"] = perf_counter() - start
                    stage = "predict"
                    start = perf_counter()
                    probabilities = model.predict_proba(test[list(FEATURES)])
                    row["prediction_time_s"] = perf_counter() - start
                    stage = "metrics"
                    row.update(evaluate(test.is_malignant.to_numpy(), np.asarray(probabilities), np.asarray(model.classes_)))
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
            pd.DataFrame(results).to_csv(output / "breast_cancer_benchmark_results.csv.tmp", index=False)
    pd.concat(membership, ignore_index=True).to_csv(output / "sample_membership.csv", index=False)
    (output / "breast_cancer_benchmark_results.csv.tmp").replace(output / "breast_cancer_benchmark_results.csv")
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
    for ax, label in zip(axes, ("Test accuracy", "Macro F1", "ROC-AUC")):
        ax.set_xscale("log")
        ax.set_xticks(sorted(PERCENTAGES),
                      labels=[f"{p}%\n({max(2, pool * p // 100)} rows)" for p in sorted(PERCENTAGES)])
        ax.tick_params(axis="x", labelsize=8, labelrotation=25)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set(xlabel="Training pool used (log scale)", ylabel=label, title=label, ylim=(0, 1.03))
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(loc="best", title="Models", fontsize=8)
    fig.suptitle(f"Experiment A: benign vs malignant ({pool:,}-record training pool)")
    fig.text(0.5, 0.025, f"Fixed {int(results.test_rows.iloc[0]):,} test records | Positive class: malignant | "
             "Nested shared training rows | Single run per point | Not clinical validation",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def plot_results(output: Path, *, show: bool = False) -> None:
    fig = create_plot(pd.read_csv(output / "breast_cancer_benchmark_results.csv"))
    for extension in ("png", "svg"):
        path = output / f"breast_cancer_performance.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        LOG.info("Graph saved to %s", path)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
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
        source = ROOT / "data" / "breast-cancer.csv"
        frame, profile = prepare_data(pd.read_csv(source))
        LOG.info("Dataset profile: %s", profile)
        train, test = make_pools(frame, args.smoke)
        configuration = {
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "experiment": "A", "smoke": args.smoke, "training_pool_rows": len(train), "test_rows": len(test),
            "percentages": PERCENTAGES, "seed": 42, "test_fraction": 0.2, "threshold": 0.5, "positive_class": 1,
            "target": "diagnosis: M=1 (malignant), B=0 (benign)", "features": FEATURES, "excluded": EXCLUDED,
            "profile": profile, "python": platform.python_version(), "minimum_training_rows_per_class": 1,
            "cpu_large_dataset_override": False,
            "versions": {name: version(name) for name in (
                "tabpfn", "scikit-learn", "pandas", "numpy", "xgboost", "lightgbm", "catboost", "matplotlib",
            )},
            "model_parameters": {name: model.get_params() for name, model in build_models()}, "status": "running",
        }
        config_path = output / "breast_cancer_config.json"
        config_path.write_text(json.dumps(configuration, indent=2, default=str), encoding="utf-8")
        results = run_benchmark(frame, output, smoke=args.smoke)
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
    revised_main('breast-cancer')