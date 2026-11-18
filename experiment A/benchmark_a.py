"""Auditable, repeated Experiment A. See README.md for the frozen protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, log_loss, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCE = "TabPFN v2"
PERCENTAGES = (100, 50, 25, 10, 5, 1)
SPECS = {
    "titanic": ("titanic/titanic Test/titanic_test_A.py", "Titanic-Dataset.csv", "Survived", "dependency_group", "log_loss"),
    "breast-cancer": ("breast-cancer/breast_cancer_A.py", "breast-cancer.csv", "is_malignant", None, "log_loss"),
    "orders": ("orders/olist_orders_A.py", "olist_orders_dataset.csv", "is_canceled", None, "average_precision"),
    "shoppers": ("shoppers/shoppers_A.py", "online_shoppers_intention.csv", "Revenue", None, "log_loss"),
    "hotel": ("hotel/hotel_book_test_A.py", "hotel_bookings.csv", "is_canceled", None, "log_loss"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def adapter(dataset):
    spec = importlib.util.spec_from_file_location("adapter_" + dataset.replace("-", "_"), HERE / SPECS[dataset][0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset(dataset):
    module = adapter(dataset)
    source = ROOT / "data" / SPECS[dataset][1]
    raw = pd.read_csv(source)
    if dataset == "hotel" and (raw.hotel.isna().any() or set(raw.hotel.unique()) != {"Resort Hotel"}):
        raise ValueError("The active hotel cohort must contain Resort Hotel only")
    frame, profile = module.prepare_data(raw)
    profile.update(dataset_label="Resort Hotel" if dataset == "hotel" else ("Online shoppers" if dataset == "shoppers" else dataset), dataset_revision="20261109-resort-shoppers")
    target, group = SPECS[dataset][2:4]
    frame = frame.copy()
    frame["_y"] = frame[target].astype(int)
    frame["_group"] = frame[group].astype(str) if group else frame.index.astype(str)
    if not frame.index.is_unique or frame._group.isna().any():
        raise ValueError("Unique source rows and nonmissing evaluation groups required")
    return module, frame, profile


def sample(frame, size, seed):
    """Training only: proportional without replacement, both classes retained."""
    if size < 2 or size > len(frame) or set(frame._y) != {0, 1}:
        raise ValueError("Infeasible two-class training sample")
    counts = frame._y.value_counts()
    positive = int(np.clip(round(size * frame._y.mean()), max(1, size - counts[0]), min(size - 1, counts[1])))
    return pd.concat([frame[frame._y.eq(0)].sample(size - positive, random_state=seed),
                      frame[frame._y.eq(1)].sample(positive, random_state=seed)]).sample(frac=1, random_state=seed)


def split_data(frame, dataset, seed, smoke=False):
    if SPECS[dataset][3]:
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(frame, groups=frame._group))
        train, test = frame.iloc[left], frame.iloc[right]
    else:
        train, test = train_test_split(frame, test_size=.2, random_state=seed, stratify=frame._y)
    if smoke and len(test) > 500:
        # Select complete groups, without looking at labels. Smoke is never research evidence.
        groups = pd.Series(test._group.unique()).sample(frac=1, random_state=seed)
        sizes = test.groupby("_group").size().reindex(groups)
        selected = groups.iloc[:max(1, int((sizes.cumsum() <= 500).sum()))]
        test = test[test._group.isin(selected)]
    if set(train._group) & set(test._group) or set(train._y) != {0, 1} or set(test._y) != {0, 1}:
        raise ValueError("Split must be disjoint and two-class; do not search for a favorable replacement seed")
    return train, test


def nested_samples(train, seed, smoke=False):
    pool = sample(train, min(len(train), 100 if smoke else 2000), seed)
    current = pool
    for percentage in ((100, 10) if smoke else PERCENTAGES):
        current = sample(current, max(2, len(pool) * percentage // 100), seed)
        yield percentage, current


def scores(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if set(y) != {0, 1} or p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Require two-class labels and finite probabilities in [0,1]")
    pred = p >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(log_loss=float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
                average_precision=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)),
                accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                recall=float(tp / (tp+fn)), specificity=float(tn / (tn+fp)),
                precision=float(tp / (tp+fp)) if tp+fp else 0., precision_defined=bool(tp+fp),
                true_negatives=int(tn), false_positives=int(fp), false_negatives=int(fn), true_positives=int(tp))


def positive_probabilities(model, features, batch_size=512):
    classes = np.asarray(model.classes_)
    if classes.shape != (2,) or set(classes) != {0, 1}:
        raise ValueError("Estimator must expose binary classes")
    result = []
    for start in range(0, len(features), batch_size):
        batch = features.iloc[start:start+batch_size]
        p = np.asarray(model.predict_proba(batch), dtype=np.float64)
        if (p.shape != (len(batch), 2) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
                or not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0)):
            raise ValueError("Invalid predict_proba output")
        result.extend(p[:, np.flatnonzero(classes == 1)[0]])
    return np.asarray(result)


def models_for(module, variant="main"):
    models = module.build_models()
    # Explicit and fixed algorithm RNG: sample variability is measured separately.
    for _, model in models:
        params = model.get_params(deep=True)
        model.set_params(**{k: 42 for k in params if k.endswith("random_state") or k.endswith("random_seed")})
    if variant == "onehot-trees":
        # Controlled explanatory ablation; only conventional tree encoders change.
        from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
        for name, model in models:
            if name in ("Random forest", "XGBoost", "LightGBM", "CatBoost"):
                replacements = {k: OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                                for k, v in model.get_params(deep=True).items() if isinstance(v, OrdinalEncoder)}
                model.set_params(**replacements)
    return models


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return str(item)
        return item
    Path(path).write_text(json.dumps(clean(value), indent=2, default=str, allow_nan=False), encoding="utf-8")


def membership(frame, role, split_seed, sample_seed=-1, percentage=-1):
    return pd.DataFrame(dict(source_index=frame.index, actual=frame._y.to_numpy(), group=frame._group.to_numpy(),
                             role=role, split_seed=split_seed, sample_seed=sample_seed, training_percentage=percentage))


def run_dataset(dataset, output, split_seeds, sample_seeds, smoke=False, variant="main", bootstrap=2000):
    from tabpfn.settings import settings
    from threadpoolctl import threadpool_limits
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    module, frame, profile = load_dataset(dataset)
    models = models_for(module, variant)
    config = dict(protocol="A-v3-resort-shoppers", dataset=dataset, smoke=smoke, variant=variant,
                  created_utc=datetime.now(timezone.utc).isoformat(), split_seeds=list(split_seeds), sample_seeds=list(sample_seeds),
                  primary_metric=SPECS[dataset][4], practical_threshold=.01, bootstrap=bootstrap,
                  percentages=[100, 10] if smoke else list(PERCENTAGES), model_seed=42, threads=1, batch_size=512,
                  source_sha256=digest(ROOT / "data" / SPECS[dataset][1]), profile=profile,
                  code_sha256={str(p.relative_to(HERE)): digest(p) for p in HERE.rglob("*.py") if "__pycache__" not in p.parts},
                  versions={p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "tabpfn", "torch", "xgboost", "lightgbm", "catboost")},
                  python=sys.version, platform=platform.platform(), processor=platform.processor(),
                  model_parameters={name: model.get_params(deep=True) for name, model in models}, status="running")
    write_json(output / "configuration.json", config)
    results, members = [], []
    previous = settings.tabpfn.allow_cpu_large_dataset
    previous_threads = torch.get_num_threads()
    settings.tabpfn.allow_cpu_large_dataset = True
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            for split_seed in split_seeds:
                train, test = split_data(frame, dataset, split_seed, smoke)
                members.append(membership(test, "test", split_seed))
                for sample_seed in sample_seeds:
                    for percentage, subset in nested_samples(train, sample_seed, smoke):
                        members.append(membership(subset, "train", split_seed, sample_seed, percentage))
                        pd.concat(members).to_csv(output / "membership.csv", index=False)
                        for name, template in models:
                            row = dict(dataset=dataset, split_seed=split_seed, sample_seed=sample_seed,
                                       training_percentage=percentage, training_rows=len(subset), training_positives=int(subset._y.sum()),
                                       test_rows=len(test), test_positives=int(test._y.sum()), test_groups=test._group.nunique(),
                                       model=name, status="failed", error="")
                            print(f"{dataset} split={split_seed} sample={sample_seed} n={len(subset)} {name}", flush=True)
                            try:
                                model = clone(template)
                                started = perf_counter()
                                model.fit(subset[list(module.FEATURES)], subset._y)
                                row["fit_seconds"] = perf_counter() - started
                                if name == REFERENCE and "checkpoint_sha256" not in config:
                                    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
                                    checkpoint = getattr(estimator, "model_path", None)
                                    paths = checkpoint if isinstance(checkpoint, (list, tuple)) else [checkpoint]
                                    config["checkpoint_sha256"] = {str(p): digest(p) for p in paths
                                                                   if isinstance(p, (str, Path)) and Path(p).is_file()}
                                started = perf_counter()
                                p = positive_probabilities(model, test[list(module.FEATURES)])
                                row["predict_seconds"] = perf_counter() - started
                                row["total_seconds"] = row["fit_seconds"] + row["predict_seconds"]
                                row.update(scores(test._y, p), status="ok")
                                pred = membership(test, "test", split_seed, sample_seed, percentage).drop(columns="role")
                                pred["model"], pred["probability"] = name, p
                                pred.to_csv(output / "predictions.csv", mode="a", header=not (output / "predictions.csv").exists(), index=False)
                            except Exception as exc:
                                row["error"] = f"{type(exc).__name__}: {exc}"
                                with (output / "errors.log").open("a", encoding="utf-8") as f:
                                    traceback.print_exc(file=f)
                            results.append(row)
                            pd.DataFrame(results).to_csv(output / "results.csv", index=False)
    finally:
        settings.tabpfn.allow_cpu_large_dataset = previous
        torch.set_num_threads(previous_threads)
        config["status"] = "complete" if len(results) == len(split_seeds)*len(sample_seeds)*len(config["percentages"])*len(models) and all(r["status"] == "ok" for r in results) else "incomplete"
        pd.DataFrame([r for r in results if r["status"] != "ok"], columns=pd.DataFrame(results).columns).to_csv(output / "failures.csv", index=False)
        config["output_sha256"] = {p.name: digest(p) for p in output.glob("*.csv")}
        write_json(output / "configuration.json", config)
    if config["status"] != "complete":
        raise RuntimeError(f"Incomplete benchmark: inspect {output / 'failures.csv'}")
    analyze(output)
    return pd.DataFrame(results)


def primary_value(y, p, metric, weights=None):
    if metric == "log_loss":
        return log_loss(y, np.column_stack([1-p, p]), labels=[0, 1], sample_weight=weights)
    return average_precision_score(y, p, sample_weight=weights)


def bootstrap_statistic(y, p, metric):
    """Cache fixed-prediction work; each resample changes only cluster weights."""
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if metric == "log_loss":
        p = np.clip(p, np.finfo(np.float64).eps, 1-np.finfo(np.float64).eps)
        losses = np.where(y == 1, -np.log(p), -np.log1p(-p))
        return lambda w: float(np.dot(w, losses) / np.sum(w))
    if metric != "average_precision":
        raise ValueError("Unsupported primary metric")
    order = np.argsort(-p, kind="stable")
    ranked_y = y[order]
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p)-1]

    def average_precision(weights):
        w = np.asarray(weights)[order]
        positives = np.cumsum(w * ranked_y)[ends]
        totals = np.cumsum(w)[ends]
        precision = np.divide(positives, totals, out=np.zeros(len(ends), dtype=float), where=totals > 0)
        return float(np.dot(np.diff(np.r_[0, positives]), precision) / positives[-1])

    return average_precision


def paired_interval(y, reference, baseline, groups, metric, draws, seed=20261104, alpha=.05, family_size=1):
    """Whole-group paired bootstrap. Positive effects favor TabPFN."""
    y, reference, baseline = map(np.asarray, (y, reference, baseline))
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    if draws < 100 or len(unique) < 2:
        raise ValueError("Require >=100 bootstrap draws and >=2 evaluation groups")
    direction = -1 if metric == "log_loss" else 1
    effect = direction * (primary_value(y, reference, metric) - primary_value(y, baseline, metric))
    reference_stat = bootstrap_statistic(y, reference, metric)
    baseline_stat = bootstrap_statistic(y, baseline, metric)
    rng = np.random.default_rng(seed)
    effects = []
    for _ in range(draws):
        weights = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[inverse]
        if len(np.unique(y[weights > 0])) != 2:
            continue
        effects.append(direction * (reference_stat(weights) - baseline_stat(weights)))
    if len(effects) < .95 * draws:
        return dict(effect=effect, ci_low=np.nan, ci_high=np.nan, familywise_ci_low=np.nan,
                    familywise_ci_high=np.nan, valid_draws=len(effects), inference_status="insufficient_two_class_resamples")
    low, high = np.quantile(effects, [alpha/2, 1-alpha/2])
    family_low, family_high = np.quantile(effects, [alpha/(2*family_size), 1-alpha/(2*family_size)])
    return dict(effect=effect, ci_low=float(low), ci_high=float(high), familywise_ci_low=float(family_low),
                familywise_ci_high=float(family_high), valid_draws=len(effects), inference_status="ok")


def analyze(output):
    """Read saved evidence only; never silently analyze incomplete or altered runs."""
    output = Path(output)
    config = json.loads((output / "configuration.json").read_text())
    if config["status"] != "complete":
        raise ValueError("Incomplete runs cannot produce summary figures")
    for name, expected in config["output_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
    results = pd.read_csv(output / "results.csv")
    keys = ["split_seed", "sample_seed", "training_percentage"]
    if results.duplicated(keys + ["model"]).any() or not results.status.eq("ok").all():
        raise ValueError("Invalid attempt table")
    metric = config["primary_metric"]
    # Preselected first split AND first sample: no pooling repeated people and no best-seed selection.
    selected_parts = []
    for chunk in pd.read_csv(output / "predictions.csv", dtype={"group": str}, chunksize=100000):
        selected_parts.append(chunk[(chunk.split_seed == config["split_seeds"][0]) &
                                    (chunk.sample_seed == config["sample_seeds"][0]) & chunk.training_percentage.eq(100)])
    selected = pd.concat(selected_parts, ignore_index=True)
    ref = selected[selected.model.eq(REFERENCE)].set_index("source_index").sort_index()
    comparisons = []
    baselines = [m for m in results.model.unique() if m != REFERENCE]
    for name in baselines:
        other = selected[selected.model.eq(name)].set_index("source_index").sort_index()
        if not ref.index.is_unique or not other.index.equals(ref.index) or not other[["actual", "group"]].equals(ref[["actual", "group"]]):
            raise ValueError("Predictions are not paired on identical evaluation rows")
        interval = paired_interval(ref.actual, ref.probability, other.probability, ref.group, metric, config["bootstrap"], family_size=len(baselines))
        comparisons.append(dict(model=name, primary_metric=metric, **interval,
                                practical_threshold=config["practical_threshold"],
                                practically_supported=bool(interval["familywise_ci_low"] > config["practical_threshold"]),
                                family=f"baselines within dataset, {config['variant']} variant, full training, first predefined split/sample"))
    pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False)
    summary = results.groupby(["model", "training_percentage"]).agg(
        training_rows_mean=("training_rows", "mean"), metric_mean=(metric, "mean"), metric_sd=(metric, "std"),
        total_seconds_mean=("total_seconds", "mean"), attempts=(metric, "count")).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    # Within-split variability separates training selection from changing test samples.
    results.groupby(["split_seed", "model", "training_percentage"])[metric].agg(["mean", "std", "count"]).to_csv(output / "training_sample_variability.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, rows in summary.groupby("model", sort=False):
        rows = rows.sort_values("training_rows_mean")
        axes[0].errorbar(rows.training_rows_mean, rows.metric_mean, yerr=rows.metric_sd.fillna(0), marker="o", capsize=2, label=name)
        axes[1].plot(rows.total_seconds_mean, rows.metric_mean, marker="o", label=name)
    curve_title = "Mean ± SD across shared runs (not confidence intervals)" if summary.attempts.min() > 1 else "Single-run values (SD unavailable)"
    axes[0].set(xscale="log", xlabel="Mean absolute training rows", ylabel=metric, title=curve_title)
    axes[1].set(xscale="log", xlabel="Mean fit + batched prediction seconds", ylabel=metric, title="Performance versus compute")
    axes[0].legend(fontsize=8)
    fig.suptitle(config.get("profile", {}).get("dataset_label", config["dataset"]) + (" — SMOKE: execution check only" if config["smoke"] else " — exploratory extension"))
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"learning_curve.{ext}", dpi=160)
    plt.close(fig)


def main(dataset_default=None):
    if dataset_default == "retail":
        raise SystemExit("Online Retail II is retired from the active suite; use --dataset shoppers on the canonical runner. Historical retail evidence is unchanged.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*SPECS, "all"], default=dataset_default or "all")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--analyze", type=Path)
    parser.add_argument("--output", type=Path, help="New run directory; existing dataset directories are refused")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--variant", choices=["main", "onehot-trees"], default="main")
    args = parser.parse_args()
    if args.analyze:
        analyze(args.analyze)
        return
    if args.bootstrap < 100 or len(set(args.split_seeds)) != len(args.split_seeds) or len(set(args.sample_seeds)) != len(args.sample_seeds):
        parser.error("Require >=100 bootstrap draws and distinct seeds")
    output = args.output or HERE / "runs" / (("smoke_" if args.smoke else "full_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    datasets = list(SPECS) if args.dataset == "all" else [args.dataset]
    failed = []
    for dataset in datasets:
        try:
            run_dataset(dataset, output / dataset, args.split_seeds[:1] if args.smoke else args.split_seeds,
                        args.sample_seeds[:1] if args.smoke else args.sample_seeds, args.smoke, args.variant, args.bootstrap)
        except Exception:
            traceback.print_exc()
            failed.append(dataset)
    if failed:
        raise SystemExit("Failed datasets: " + ", ".join(failed))
    print(f"Saved verified run bundles to {output}")


if __name__ == "__main__":
    main()
