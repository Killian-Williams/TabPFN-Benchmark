"""Shared, fixed seven-model configuration for new dataset adapters."""
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

def build_models(X: pd.DataFrame) -> list[tuple[str, Pipeline]]:
    """Build unfitted pipelines using the existing experiments' untuned settings."""
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    ordinal_preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric_columns),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
                categorical_columns,
            ),
        ]
    )
    logistic_preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                ),
                numeric_columns,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical_columns,
            ),
        ]
    )
    tabpfn = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device="cpu", n_estimators=1
    )
    tabpfn.set_params(
        categorical_features_indices=list(
            range(len(numeric_columns), len(numeric_columns) + len(categorical_columns))
        ),
        memory_saving_mode=True,
    )
    classifiers = [
        ("TabPFN v2", tabpfn),
        ("Logistic regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
        (
            "XGBoost",
            XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                n_estimators=100, max_depth=3, learning_rate=0.1,
                tree_method="hist", device="cpu", random_state=42, n_jobs=1,
            ),
        ),
        (
            "LightGBM",
            LGBMClassifier(
                objective="binary", n_estimators=100, max_depth=3,
                num_leaves=7, learning_rate=0.1, device_type="cpu",
                random_state=42, n_jobs=1, verbosity=-1,
            ),
        ),
        (
            "CatBoost",
            CatBoostClassifier(
                iterations=100, depth=3, learning_rate=0.1,
                loss_function="Logloss", task_type="CPU", random_seed=42,
                thread_count=1, verbose=False, allow_writing_files=False,
            ),
        ),
        ("Dummy (prior)", DummyClassifier(strategy="prior", random_state=42)),
    ]
    models = []
    for name, classifier in classifiers:
        preprocessor = logistic_preprocessor if name == "Logistic regression" else ordinal_preprocessor
        models.append((name, make_pipeline(clone(preprocessor), classifier)))
    return models

