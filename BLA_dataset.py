"""
Сбор обучающей выборки из логов через класс Train (DatasetGenerator).
Два признака на линию:
  - k: максимальный наклон (по модулю);
  - accuracy_of_line: сколько «провалов» на линии.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from DatasetGenerator import Train


def _count_dips(
    y: Sequence[float],
    *,
    dip_min_delta: float = 25.0,
    dip_min_y: float = 0.0,
    smooth_window: int = 1,
) -> int:
    """
    «Провал» = локальный минимум: y[i] < y[i-1] и y[i] < y[i+1],
    с глубиной не меньше dip_min_delta относительно меньшего из соседей.
    Значения ниже dip_min_y игнорируются (часто это шум около нуля).
    """
    n = len(y)
    if n < 3:
        return 0

    yy = np.asarray(y, dtype=float)
    w = int(smooth_window)
    if w > 1:
        # Простое сглаживание скользящим средним, чтобы не считать шум «провалами».
        # mode="same" сохраняет длину.
        kernel = np.ones(w, dtype=float) / float(w)
        yy = np.convolve(yy, kernel, mode="same")

    dips = 0
    for i in range(1, n - 1):
        yi = float(yy[i])
        if yi < dip_min_y:
            continue
        yl = float(yy[i - 1])
        yr = float(yy[i + 1])
        if yi < yl and yi < yr:
            depth = min(yl - yi, yr - yi)
            if depth >= dip_min_delta:
                dips += 1
    return dips


def _line_features(
    line,
    *,
    dip_min_delta: float,
    dip_min_y: float,
    dip_smooth_window: int,
) -> tuple[float, float]:
    """
    Признаки:
      1) k — максимальный по модулю наклон профиля;
      2) accuracy_of_line — число провалов (локальных минимумов) на линии.
    """
    k = float(line.k) if line.k is not None else np.nan
    dips = _count_dips(
        line.lv,
        dip_min_delta=dip_min_delta,
        dip_min_y=dip_min_y,
        smooth_window=dip_smooth_window,
    )
    accuracy_of_line = float(dips)
    return (k, accuracy_of_line)


def load_logs_to_xy(
    good_paths: Sequence[Path],
    bad_paths: Sequence[Path],
    levels: Sequence[float],
    *,
    dip_min_delta: float = 25.0,
    dip_min_y: float = 0.0,
    dip_smooth_window: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Возвращает X формы (n_samples, 2), y из {0,1} (0 — плохие, 1 — хорошие),
    список имён признаков для отчёта.
    """
    rows: list[tuple[float, float]] = []
    labels: list[int] = []

    for p in good_paths:
        p = Path(p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Нет файла (хорошие): {p}")
        tr = Train(str(p), levels=levels)
        for line in tr.lines:
            rows.append(
                _line_features(
                    line,
                    dip_min_delta=dip_min_delta,
                    dip_min_y=dip_min_y,
                    dip_smooth_window=dip_smooth_window,
                )
            )
            labels.append(1)

    for p in bad_paths:
        p = Path(p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Нет файла (плохие): {p}")
        tr = Train(str(p), levels=levels)
        for line in tr.lines:
            rows.append(
                _line_features(
                    line,
                    dip_min_delta=dip_min_delta,
                    dip_min_y=dip_min_y,
                    dip_smooth_window=dip_smooth_window,
                )
            )
            labels.append(0)

    if not rows:
        raise ValueError("Пустой датасет: нет ни одной линии в указанных логах.")

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=np.int32)
    feature_names = ["k", "accuracy_of_line"]
    return X, y, feature_names
