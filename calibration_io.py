"""Сборка секции conc для calibration.json (как в dak/calibration (19).json)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from DatasetGenerator import Train, iter_labeled_lines

DEFAULT_LEVELS_TEXT = "3275, 3300, 3325, 3350, 3375, 3400, 3425"


def concentration_calibration_stats(
    trains_dir: str,
    include_relative: set[str] | None,
    levels: list[float],
) -> list[tuple[float, int, int]]:
    """
    По обучающим логам: для каждой уникальной концентрации
    (ключ round(conc, 6)) — сколько всего записей и сколько с полными пересечениями на всех уровнях
    (попадут в calibration points при выборе этой density).
    """
    total: dict[float, int] = defaultdict(int)
    valid: dict[float, int] = defaultdict(int)
    for line in iter_labeled_lines(trains_dir, include_relative, levels):
        if line.conc is None:
            continue
        key = round(float(line.conc), 6)
        total[key] += 1
        if all(v is not None for v in line.level_intersections):
            valid[key] += 1
    return [(k, total[k], valid[k]) for k in sorted(total.keys())]


def parse_levels_text(text: str) -> list[float]:
    raw = text.replace(";", ",").replace("\n", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Укажите хотя бы один уровень (числа через запятую).")
    return [float(x) for x in parts]


def parse_densities_text(text: str) -> list[float]:
    """Одна концентрация на строку; в строке можно несколько чисел через запятую."""
    out: list[float] = []
    for line in text.splitlines():
        for part in line.split(","):
            s = part.strip()
            if not s:
                continue
            try:
                out.append(float(s))
            except ValueError as e:
                raise ValueError(f"Не число: «{s}»") from e
    return out


def _json_level(x: float) -> int | float:
    if float(x).is_integer():
        return int(round(float(x)))
    return float(x)


def _json_density_out(key: float) -> int | float:
    d = float(key)
    return int(d) if float(d).is_integer() else d


def _collect_valid_buckets(
    trains_dir: str,
    include_relative: set[str] | None,
    levels: list[float],
) -> dict[float, list[list[float]]]:
    buckets: dict[float, list[list[float]]] = defaultdict(list)
    for line in iter_labeled_lines(trains_dir, include_relative, levels):
        if line.conc is None:
            continue
        if any(v is None for v in line.level_intersections):
            continue
        arr = [float(v) for v in line.level_intersections]
        key = round(float(line.conc), 6)
        buckets[key].append(arr)
    return buckets


def reference_zero_intersections(
    reference_log: str,
    levels: list[float],
) -> list[float] | None:
    """Абсолютные пересечения нулевой линии (первая запись) из train.log на всех уровнях."""
    train = Train(reference_log, levels=levels)
    if not train.lines:
        return None
    ref = train.lines[0]
    if any(v is None for v in ref.level_intersections):
        return None
    return [float(v) for v in ref.level_intersections]


def train_log_abs_arrays_ordered(
    train_log_path: str,
    levels: list[float],
) -> list[tuple[float, list[float]]]:
    """
    Концентрации в порядке появления в train.log и усреднённый array пересечений
    (если одна концентрация встречается в файле несколько раз).
    """
    train = Train(train_log_path, levels=levels)
    file_buck: dict[float, list[list[float]]] = defaultdict(list)
    order: list[float] = []
    for line in train.lines:
        if line.conc is None:
            continue
        if any(v is None for v in line.level_intersections):
            continue
        key = round(float(line.conc), 6)
        arr = [float(v) for v in line.level_intersections]
        if key not in file_buck:
            order.append(key)
        file_buck[key].append(arr)
    out: list[tuple[float, list[float]]] = []
    for key in order:
        arrs = file_buck[key]
        mean_arr = np.mean(np.asarray(arrs, dtype=float), axis=0).tolist()
        out.append((key, mean_arr))
    return out


def compare_trainlog_vs_training_corpus(
    train_log_path: str,
    trains_dir: str,
    include_relative: set[str] | None,
    levels: list[float],
    stats: list[tuple[float, int, int]] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """
    Для каждой density из train.log: «синтетическая» точка = среднее array по корпусу trains/,
    эталон = array из train.log. Возвращает таблицу, матрицу |Δ| размера (n_conc × n_levels),
    подписи концентраций и уровней (для heatmap).
    """
    stats_map: dict[float, tuple[int, int]] = {}
    if stats:
        for k, nt, nv in stats:
            stats_map[k] = (nt, nv)

    file_ordered = train_log_abs_arrays_ordered(train_log_path, levels)
    buckets = _collect_valid_buckets(trains_dir, include_relative, levels)

    rows: list[dict[str, Any]] = []
    n = len(file_ordered)
    m = len(levels)
    Z = np.full((n, m), np.nan, dtype=float)
    x_labels: list[str] = []
    y_level_lbl = [str(_json_level(float(L))) for L in levels]

    for i, (key, arr_file) in enumerate(file_ordered):
        d_out = _json_density_out(key)
        x_labels.append(str(d_out))
        nt, nv = stats_map.get(key, (0, 0))
        row: dict[str, Any] = {
            "density": d_out,
            "в обучении (всего)": nt,
            "готовых к array": nv,
        }
        if key not in buckets:
            row["MAE (корпус vs train.log)"] = None
            row["max |Δ| по уровням"] = None
            row["статус"] = "нет в корпусе trains/"
            rows.append(row)
            continue

        arr_train = np.mean(np.asarray(buckets[key], dtype=float), axis=0)
        arr_f = np.asarray(arr_file, dtype=float)
        deltas = np.abs(arr_train - arr_f)
        Z[i, :] = deltas
        row["MAE (корпус vs train.log)"] = round(float(np.mean(deltas)), 6)
        row["max |Δ| по уровням"] = round(float(np.max(deltas)), 6)
        row["статус"] = "ok"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, Z, x_labels, y_level_lbl


def build_conc_calibration(
    trains_dir: str,
    include_relative: set[str] | None,
    levels: list[float],
    *,
    density_keys: set[float] | None = None,
    density_order: list[float] | None = None,
) -> dict[str, Any]:
    """
    conc: levels + points[{array, density}], zero.
    density_order: порядок точек как в списке (только ключи, для которых есть данные).
    density_keys: множество ключей, порядок — по возрастанию ключа (как раньше).
    Оба None — все концентрации из логов, порядок по возрастанию ключа.
    """
    buckets = _collect_valid_buckets(trains_dir, include_relative, levels)

    if density_order is not None:
        seen: set[float] = set()
        key_sequence: list[float] = []
        for d in density_order:
            k = round(float(d), 6)
            if k not in seen:
                seen.add(k)
                key_sequence.append(k)
    elif density_keys is not None:
        key_sequence = sorted(d for d in buckets if d in density_keys)
    else:
        key_sequence = sorted(buckets.keys())

    points: list[dict[str, Any]] = []
    for dens in key_sequence:
        if dens not in buckets:
            continue
        arrs = buckets[dens]
        mean_arr = np.mean(np.asarray(arrs, dtype=float), axis=0).tolist()
        points.append({"array": mean_arr, "density": _json_density_out(dens)})

    return {
        "levels": [_json_level(x) for x in levels],
        "points": points,
        "zero": 0,
    }


def calibration_manual_export(
    trains_dir: str,
    include_relative: set[str] | None,
    levels: list[float],
    densities_ordered: list[float],
    stats: list[tuple[float, int, int]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Экспорт conc.points по списку density (порядок = ввод).
    Таблица: сколько строк в корпусе, без сравнения с train.log.
    """
    stats_map: dict[float, tuple[int, int]] = {}
    if stats:
        for k, nt, nv in stats:
            stats_map[k] = (nt, nv)

    buckets = _collect_valid_buckets(trains_dir, include_relative, levels)

    rows: list[dict[str, Any]] = []
    seen: set[float] = set()
    order_keys: list[float] = []
    for d in densities_ordered:
        k = round(float(d), 6)
        if k in seen:
            continue
        seen.add(k)
        order_keys.append(k)

    points: list[dict[str, Any]] = []
    for k in order_keys:
        nt, nv = stats_map.get(k, (0, 0))
        row: dict[str, Any] = {
            "density": _json_density_out(k),
            "в обучении (всего)": nt,
            "готовых к array": nv,
        }
        if k not in buckets:
            row["статус"] = "нет данных для array"
            rows.append(row)
            continue

        mean_arr = np.mean(np.asarray(buckets[k], dtype=float), axis=0).tolist()
        row["статус"] = "ok"
        rows.append(row)
        points.append({"array": mean_arr, "density": _json_density_out(k)})

    conc = {
        "levels": [_json_level(x) for x in levels],
        "points": points,
        "zero": 0,
    }
    df = pd.DataFrame(rows)
    return conc, df


def merge_calibration_with_template(
    conc: dict[str, Any],
    template_path: str | Path | None,
) -> dict[str, Any]:
    """Подменяет только conc; EC/NTC/pH из шаблона сохраняются."""
    if template_path and Path(template_path).is_file():
        with open(template_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
        data["conc"] = conc
        return data
    return {"conc": conc}


def calibration_to_json_bytes(data: dict[str, Any], *, indent: str = "\t") -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=indent).encode("utf-8")
