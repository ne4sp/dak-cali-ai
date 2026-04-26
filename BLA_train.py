"""
Обучение CatBoostClassifier по матрице признаков BLA_dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from BLA_dataset import load_logs_to_xy


def train_classifier(
    good_paths: tuple[Path, ...],
    bad_paths: tuple[Path, ...],
    levels: tuple[float, ...],
    *,
    model_out: Path,
    dip_min_delta: float = 25.0,
    dip_min_y: float = 0.0,
    dip_smooth_window: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.1,
    depth: int = 6,
    auto_class_weights: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    X, y, feature_names = load_logs_to_xy(
        good_paths,
        bad_paths,
        levels,
        dip_min_delta=dip_min_delta,
        dip_min_y=dip_min_y,
        dip_smooth_window=dip_smooth_window,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_pool = Pool(X_train, label=y_train, feature_names=feature_names)
    test_pool = Pool(X_test, label=y_test, feature_names=feature_names)

    clf_kw: dict[str, Any] = {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "loss_function": "Logloss",
        "random_seed": random_state,
        "verbose": 100 if verbose else False,
        "allow_writing_files": False,
    }
    if auto_class_weights:
        clf_kw["auto_class_weights"] = auto_class_weights

    model = CatBoostClassifier(**clf_kw)
    model.fit(train_pool, eval_set=test_pool)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test).astype(int).ravel()

    out: dict[str, Any] = {
        "n_total": int(X.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_good": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
        "roc_auc": float("nan"),
        "feature_names": feature_names,
        "feature_importance": dict(
            zip(feature_names, model.get_feature_importance().tolist())
        ),
        "model": model,
        "y_test": y_test,
        "y_pred": pred,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_test, proba))
    except ValueError:
        pass

    model.save_model(str(model_out))

    if verbose:
        print(classification_report(y_test, pred, target_names=["bad (0)", "good (1)"]))
        print(f"Accuracy: {out['accuracy']:.4f}  F1 (good): {out['f1_good']:.4f}  ROC-AUC: {out['roc_auc']}")
        print(f"Важности: {out['feature_importance']}")
        print(f"Модель сохранена: {model_out}")

    return out
