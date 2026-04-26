"""
Разбор pickle с объектом Train (старое сохранение из main.Train / main.Line)
и запись логов *.log в папки good/ и bad/ по тем же признакам, что и BLA_dataset:
k и accuracy_of_line (число провалов).

Режимы: dips | k_median | k_dips_rules (по умолчанию из BLA_config: k < 140 и провалов ≤ 10 → good).

Запуск из корня проекта:
    python BLA_pkl_to_logs.py

Пути и порог — в BLA_config.py.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import BLA_config as cfg
from BLA_dataset import _line_features
from DatasetGenerator import Line, Train


def _print_ascii(msg: str) -> None:
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))


def _patch_main_for_pickle() -> None:
    """Pickle ссылается на main.Train / main.Line — подставляем классы из DatasetGenerator."""
    import main as m

    m.Train = Train
    m.Line = Line


def _load_train_from_pkl(path: Path) -> Any:
    _patch_main_for_pickle()
    with open(path, "rb") as f:
        return pickle.load(f)


def _ensure_line_features(line: Line) -> None:
    """После pickle.load у Line может не быть levels / k — привести к актуальной схеме и пересчитать."""
    if not getattr(line, "levels", None):
        line.levels = tuple(float(x) for x in cfg.LEVELS)
    line.create_features()


def _format_conc(conc: Any) -> float:
    try:
        if conc is None:
            return 0.0
        c = float(conc)
        if np.isnan(c):
            return 0.0
        return c
    except (TypeError, ValueError):
        return 0.0


def line_to_log_record(probe_idx: int, line: Line, cols_per_row: int) -> str:
    """Один блок как в train.log: Probe + Concentration + строки отсчётов + пустая строка."""
    conc = _format_conc(line.conc)
    header = (
        "**************************************************************\n"
        f"Probe number {probe_idx} \n"
        f"Concentration entered by User {conc:.2f}%\n"
    )
    lv = list(line.lv)
    rows: list[str] = []
    for i in range(0, len(lv), cols_per_row):
        chunk = lv[i : i + cols_per_row]
        rows.append("\t".join(f"{float(x):.2f}" for x in chunk))
    body = "\n".join(rows)
    return header + body + "\n\n"


def export_pkl_to_folders(
    *,
    pkl_path: Path,
    good_dir: Path,
    bad_dir: Path,
    good_name: str,
    bad_name: str,
    good_max_dips: int,
    cols_per_row: int,
    mode: str = "dips",
    k_good_lt: float = 140.0,
    dips_good_max: int = 10,
) -> tuple[int, int, Path, Path, Counter[int]]:
    obj = _load_train_from_pkl(Path(pkl_path).resolve())
    if not hasattr(obj, "lines"):
        raise TypeError("В pickle ожидается объект с атрибутом lines (Train).")

    good_dir = Path(good_dir).resolve()
    bad_dir = Path(bad_dir).resolve()
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[float, float, Line]] = []
    dip_hist: Counter[int] = Counter()

    for line in obj.lines:
        if len(line.lv) < 2:
            continue
        _ensure_line_features(line)
        k, acc = _line_features(
            line,
            dip_min_delta=cfg.DIP_MIN_DELTA,
            dip_min_y=cfg.DIP_MIN_Y,
            dip_smooth_window=cfg.DIP_SMOOTH_WINDOW,
        )
        if np.isnan(k):
            continue
        dips = int(round(acc))
        dip_hist[dips] += 1
        rows.append((float(k), float(acc), line))

    good_lines: list[Line] = []
    bad_lines: list[Line] = []

    if mode == "dips":
        for k, acc, line in rows:
            dips = int(round(acc))
            if dips <= good_max_dips:
                good_lines.append(line)
            else:
                bad_lines.append(line)
    elif mode == "k_median":
        if not rows:
            good_lines, bad_lines = [], []
        else:
            rows.sort(key=lambda t: t[0])
            half = len(rows) // 2
            for i, (_, __, line) in enumerate(rows):
                (good_lines if i < half else bad_lines).append(line)
    elif mode == "k_dips_rules":
        # good: k < k_good_lt и провалов <= dips_good_max; иначе bad
        for k, acc, line in rows:
            dips = int(round(acc))
            if k < k_good_lt and dips <= dips_good_max:
                good_lines.append(line)
            else:
                bad_lines.append(line)
    else:
        raise ValueError(
            f"Unknown mode: {mode!r} (use dips, k_median, or k_dips_rules)"
        )

    good_parts = [
        line_to_log_record(i, line, cols_per_row) for i, line in enumerate(good_lines)
    ]
    bad_parts = [
        line_to_log_record(i, line, cols_per_row) for i, line in enumerate(bad_lines)
    ]
    n_good, n_bad = len(good_lines), len(bad_lines)

    good_path = good_dir / good_name
    bad_path = bad_dir / bad_name
    good_path.write_text("".join(good_parts), encoding="utf-8")
    bad_path.write_text("".join(bad_parts), encoding="utf-8")

    return n_good, n_bad, good_path, bad_path, dip_hist


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Train pickle to good/bad .log files")
    parser.add_argument("--pkl", type=str, default=str(cfg.PKL_INPUT_PATH))
    parser.add_argument("--good-dir", type=str, default=str(cfg.PKL_EXPORT_GOOD_DIR))
    parser.add_argument("--bad-dir", type=str, default=str(cfg.PKL_EXPORT_BAD_DIR))
    parser.add_argument("--max-dips-good", type=int, default=cfg.PKL_EXPORT_GOOD_MAX_DIPS)
    parser.add_argument("--cols", type=int, default=cfg.PKL_EXPORT_COLS_PER_ROW)
    parser.add_argument(
        "--mode",
        type=str,
        default=cfg.PKL_EXPORT_MODE,
        choices=("dips", "k_median", "k_dips_rules"),
        help="dips | k_median | k_dips_rules (k and dips thresholds from config/args)",
    )
    parser.add_argument(
        "--k-good-lt",
        type=float,
        default=cfg.PKL_RULE_K_GOOD_LT,
        help="k_dips_rules: good requires k < this",
    )
    parser.add_argument(
        "--dips-good-max",
        type=int,
        default=cfg.PKL_RULE_DIPS_GOOD_MAX,
        help="k_dips_rules: good requires dips <= this (>10 bad means set 10)",
    )
    args = parser.parse_args()

    n_good, n_bad, gp, bp, dip_hist = export_pkl_to_folders(
        pkl_path=Path(args.pkl),
        good_dir=Path(args.good_dir),
        bad_dir=Path(args.bad_dir),
        good_name=cfg.PKL_EXPORT_GOOD_LOG_NAME,
        bad_name=cfg.PKL_EXPORT_BAD_LOG_NAME,
        good_max_dips=int(args.max_dips_good),
        cols_per_row=int(args.cols),
        mode=str(args.mode),
        k_good_lt=float(args.k_good_lt),
        dips_good_max=int(args.dips_good_max),
    )
    top = dip_hist.most_common(12)
    _print_ascii(f"lines written: good={n_good} -> {gp}")
    _print_ascii(f"lines written: bad={n_bad} -> {bp}")
    if args.mode == "dips":
        _print_ascii(f"mode=dips: accuracy_of_line <= {args.max_dips_good} -> good, else bad")
    elif args.mode == "k_median":
        _print_ascii("mode=k_median: sort lines by k, lower half -> good, upper half -> bad")
    else:
        _print_ascii(
            f"mode=k_dips_rules: good if k < {args.k_good_lt} and dips <= {args.dips_good_max}; else bad"
        )
    _print_ascii(f"dip count histogram (top): {top}")

    # Проверка: Train() читает созданные файлы
    tr_g = Train(str(gp), levels=cfg.LEVELS)
    _print_ascii(f"Train() check good.lines={len(tr_g.lines)}")
    if n_bad > 0:
        tr_b = Train(str(bp), levels=cfg.LEVELS)
        _print_ascii(f"Train() check bad.lines={len(tr_b.lines)}")
    else:
        _print_ascii("Train() skip bad (empty file)")


if __name__ == "__main__":
    main()
