"""Обучение CatBoost и сравнение предсказаний с логами (общая логика для CLI и UI)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from DatasetGenerator import Train, generate


def load_training_data(
    trains_dir: str = "trains",
    include_relative: set[str] | None = None,
    levels: Sequence[float] | None = None,
):
    if levels is None:
        levels = (3400.0,)
    fulldata = generate(
        trains_dir=trains_dir,
        include_relative=include_relative,
        levels=levels,
    )
    X = [d[0] for d in fulldata]
    Y = [d[1] for d in fulldata]
    return X, Y


def train_model(
    X,
    Y,
    *,
    test_size=0.1,
    random_state=42,
    iterations=600,
    learning_rate=0.1,
    verbose=False,
):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=100 if verbose else False,
    )
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    y_true = np.asarray(Y_test, dtype=float).ravel()
    y_hat = np.asarray(Y_pred, dtype=float).ravel()
    mse = mean_squared_error(y_true, y_hat)
    rmse = float(np.sqrt(mse))
    mae_test = float(mean_absolute_error(y_true, y_hat))
    r2 = float(r2_score(y_true, y_hat)) if len(y_true) > 1 else None
    metrics = {
        "mse": float(mse),
        "rmse": rmse,
        "mae_test": mae_test,
        "r2": r2,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    bundle = {"y_test": y_true, "y_pred": y_hat}
    return model, metrics, bundle


def history_comparison_df(
    model,
    reference_log: str,
    history_log: str,
    *,
    levels: Sequence[float] | None = None,
) -> pd.DataFrame:
    if levels is None:
        levels = (3400.0,)
    train_ref = Train(reference_log, levels=levels)
    log = Train(history_log, levels=levels)
    if not train_ref.lines:
        raise ValueError("В эталонном логе нет записей (нулевая линия).")
    ref_line = train_ref.lines[0]
    rows = []
    for line in log.lines:
        line.to_compare = ref_line
        line.create_o()
        real = float(line.conc)
        feat = line.execute_data()[0]
        pred = float(np.asarray(model.predict(feat)).ravel()[0])
        rows.append(
            {
                "№": len(rows) + 1,
                "Реальная концентрация": round(real, 4),
                "Предсказание": round(pred, 4),
                "Ошибка": round(pred - real, 4),
                "|Ошибка|": round(abs(pred - real), 4),
            }
        )
    return pd.DataFrame(rows)


def print_cli_report(metrics: dict, df: pd.DataFrame) -> None:
    print(f"RMSE (тест): {metrics['rmse']:.4f}")
    print(f"MAE (тест): {metrics['mae_test']:.4f}")
    if metrics.get("r2") is not None:
        print(f"R² (тест): {metrics['r2']:.4f}")
    if len(df) == 0:
        return
    print()
    print(f"{'REAL':^12} | {'AI':^12}")
    print("-" * 29)
    for _, row in df.iterrows():
        print(f"{row['Реальная концентрация']:^12.2f} | {row['Предсказание']:^12.2f}")
    print()
    print("avg real =", round(df["Реальная концентрация"].mean(), 4))
    print("avg ai   =", round(df["Предсказание"].mean(), 4))
    print("MAE (история) =", round(df["|Ошибка|"].mean(), 4))
