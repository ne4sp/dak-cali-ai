"""
Оценка обученного классификатора на «смешанном паке»:
папка, внутри которой есть две подпапки: good/ и bad/ (имена задаются).

Каждая подпапка содержит *.log, из которых извлекаются линии через Train.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from BLA_dataset import load_logs_to_xy


def _list_logs(folder: Path) -> list[Path]:
    folder = folder.resolve()
    if not folder.is_dir():
        return []
    return sorted([p for p in folder.rglob("*.log") if p.is_file()])


def eval_on_mixed_pack(
    model: CatBoostClassifier,
    *,
    pack_root: Path,
    good_subdir: str = "good",
    bad_subdir: str = "bad",
    levels: Sequence[float],
    dip_min_delta: float,
    dip_min_y: float,
    dip_smooth_window: int,
    verbose: bool = True,
) -> dict[str, Any]:
    pack_root = Path(pack_root).resolve()
    good_dir = pack_root / good_subdir
    bad_dir = pack_root / bad_subdir

    good_paths = _list_logs(good_dir)
    bad_paths = _list_logs(bad_dir)
    if not good_paths and not bad_paths:
        raise FileNotFoundError(
            f"В паке нет логов *.log. Ожидаются подпапки {good_dir} и {bad_dir}."
        )

    X, y, feature_names = load_logs_to_xy(
        good_paths=good_paths,
        bad_paths=bad_paths,
        levels=tuple(levels),
        dip_min_delta=dip_min_delta,
        dip_min_y=dip_min_y,
        dip_smooth_window=dip_smooth_window,
    )

    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X).astype(int).ravel()

    out: dict[str, Any] = {
        "pack_root": str(pack_root),
        "n_total": int(X.shape[0]),
        "n_good_files": int(len(good_paths)),
        "n_bad_files": int(len(bad_paths)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_good": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float("nan"),
        "feature_names": feature_names,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, proba))
    except ValueError:
        pass

    if verbose:
        print()
        print(f"=== Mixed pack eval: {pack_root} ===")
        print(f"Files: good={len(good_paths)} bad={len(bad_paths)} | lines={len(y)}")
        print(classification_report(y, pred, target_names=["bad (0)", "good (1)"]))
        print(
            f"Accuracy: {out['accuracy']:.4f}  F1 (good): {out['f1_good']:.4f}  ROC-AUC: {out['roc_auc']}"
        )

    return out


def load_model(model_path: Path) -> CatBoostClassifier:
    m = CatBoostClassifier()
    m.load_model(str(Path(model_path).resolve()))
    return m

