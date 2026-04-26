"""Запуск из корня проекта: python main.py"""

from pathlib import Path

import pandas as pd

from analysis import (
    history_comparison_df,
    load_training_data,
    print_cli_report,
    train_model,
)
from calibration_io import DEFAULT_LEVELS_TEXT, parse_levels_text

DEFAULT_REF = Path(__file__).resolve().parent / "dak" / "train.log"
DEFAULT_HIST = Path(__file__).resolve().parent / "dak" / "conc_history_2025-10-14.log"


def main():
    levels = parse_levels_text(DEFAULT_LEVELS_TEXT)
    print("Загрузка датасета из trains/…", f"уровни: {levels}")
    X, Y = load_training_data(levels=levels)
    print("Начало обучения CatBoost…")
    model, metrics, _ = train_model(X, Y, verbose=True)
    print("Обучение завершено.")

    if not DEFAULT_REF.is_file() or not DEFAULT_HIST.is_file():
        print(
            f"Предупреждение: нет файлов сравнения ({DEFAULT_REF.name} / {DEFAULT_HIST.name})."
        )
        print_cli_report(metrics, pd.DataFrame())
        return

    df = history_comparison_df(
        model, str(DEFAULT_REF), str(DEFAULT_HIST), levels=levels
    )
    print_cli_report(metrics, df)


if __name__ == "__main__":
    main()
