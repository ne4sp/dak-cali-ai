"""Генерация блок-схем для диплома (рис. 3.5, 4.1, 4.2). Запуск: python диплом/generate_figures.py"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

BOX_KW = dict(boxstyle="round,pad=0.35", linewidth=1.2, facecolor="#E8F4FC", edgecolor="#2E6DA4")
BOX_OK = dict(boxstyle="round,pad=0.35", linewidth=1.2, facecolor="#E8F8E8", edgecolor="#3D8B3D")
BOX_WARN = dict(boxstyle="round,pad=0.35", linewidth=1.2, facecolor="#FFF4E5", edgecolor="#C77D00")


def _box(ax, xy, w, h, text, style=BOX_KW, fs=9):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, **style)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, p1, p2):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#333333"
        )
    )


def fig_3_5_algorithm():
    fig, ax = plt.subplots(figsize=(7.5, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("Рисунок 3.5 — Блок-схема двухэтапной обработки данных DAK", fontsize=11, pad=12)

    w, h = 7.5, 1.0
    x0 = 1.25
    y = 12.5
    _box(ax, (x0, y), w, h, "Файл log: кривая y, концентрация c")
    y -= 1.5
    _arrow(ax, (5, y + 1.0), (5, y + 0.35))
    _box(ax, (x0, y), w, h, "Разбор и извлечение признаков:\nk, n_dip, z_m")
    y -= 1.5
    _arrow(ax, (5, y + 1.0), (5, y + 0.35))
    _box(ax, (x0, y), w, h, "Классификатор CatBoost\n(g(φ) ≥ τ → пригодная?)", style=BOX_WARN)
    y -= 1.5
    _arrow(ax, (5, y + 1.0), (5, y + 0.35))
    _box(ax, (x0, y), w, h, "Отбор S_good", style=BOX_OK)
    y -= 1.5
    _arrow(ax, (5, y + 1.0), (5, y + 0.35))
    _box(ax, (x0, y), w, h, "Регрессор CatBoost\nf(z) → c")
    y -= 1.5
    _arrow(ax, (5, y + 1.0), (5, y + 0.35))
    _box(ax, (x0, y), w, h, "Метрики RMSE, MAE, R²;\nδ_mean, δ_max")
    y -= 1.5
    _arrow(ax, (5, y + 1.0), (5, y + 0.35))
    _box(ax, (x0, y), w, h, "JSON калибровки + отчёт", style=BOX_OK)

    ax.text(0.3, 9.2, "Нет", color="#A00", fontsize=9)
    _arrow(ax, (1.0, 9.7), (0.6, 8.8))
    bad_style = {**BOX_WARN, "facecolor": "#FFE8E8"}
    _box(ax, (0.2, 7.6), 2.2, 0.9, "Брак", style=bad_style)

    fig.tight_layout()
    path = OUT / "fig_3_5_algorithm.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def _arrow_labeled(ax, p1, p2, label=None, color="#333333", style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=1.1,
            color=color,
            linestyle=ls,
        )
    )
    if label:
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=7, color="#222222")


BOX_IO = dict(boxstyle="round,pad=0.3", linewidth=1.2, facecolor="#F5F5F5", edgecolor="#555555")
BOX_CFG = dict(boxstyle="round,pad=0.3", linewidth=1.0, facecolor="#FFFDE7", edgecolor="#9E9E00", linestyle="--")


def fig_4_1_architecture():
    """Рис. 4.1 по диплом/block-schemas.md (блоки B1–B5)."""
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title(
        "Рисунок 4.1 — Архитектура программного комплекса (блоки B1–B5)",
        fontsize=12,
        pad=14,
    )

    # Верхний ряд: B1–B4
    bw, bh = 2.55, 1.75
    y_top = 5.35
    xs = [0.55, 3.45, 6.35, 9.25]
    titles = [
        "B1\nРазбор данных\nDatasetGenerator",
        "B2\nПризнаки\nx_m, z, φ",
        "B3\nКлассификация\nBLA_* / CatBoost",
        "B4\nРегрессия и калибровка\nanalysis, calibration_io",
    ]
    for x, t in zip(xs, titles):
        _box(ax, (x, y_top), bw, bh, t, fs=7.5)

    # B5 — представление
    x5, w5 = 4.35, 5.3
    _box(
        ax,
        (x5, 2.35),
        w5,
        1.55,
        "B5  Представление результатов\napp.py (Streamlit, Plotly)  ·  main.py  ·  BLA_cli",
        fs=7.5,
        style=BOX_OK,
    )

    # Вход / выход
    _box(ax, (0.4, 0.35), 2.1, 1.05, "Вход (IN)\nкаталог *.log", style=BOX_IO, fs=8)
    _box(ax, (11.35, 0.35), 2.25, 1.05, "Выход (OUT)\nJSON, cbm, PNG,\nотчёт", style=BOX_IO, fs=7.5)

    # CFG
    _box(ax, (4.9, 7.55), 4.2, 0.85, "Параметры пользователя (CFG): пути, L_m, τ, CatBoost", style=BOX_CFG, fs=7.5)

    cy = y_top + bh / 2
    # IN → B1
    _arrow_labeled(ax, (1.45, 1.4), (1.82, y_top), "*.log")
    # B1 → B2 → B3 → B4
    _arrow_labeled(ax, (xs[0] + bw, cy), (xs[1], cy), "Train, Line")
    _arrow_labeled(ax, (xs[1] + bw, cy), (xs[2], cy), "φ")
    _arrow_labeled(ax, (xs[2] + bw, cy), (xs[3], cy), "ω̂")
    # B2 → B4 (z, c)
    _arrow_labeled(ax, (4.72, y_top + 0.15), (9.4, y_top + 0.15), "z, c, x_m")
    # B3, B4 → B5
    _arrow_labeled(ax, (7.62, y_top), (6.2, 3.92), "метрики CLS")
    _arrow_labeled(ax, (10.52, y_top), (8.5, 3.92), "JSON, REG")
    # B2 → B5 (графики кривых)
    _arrow_labeled(ax, (4.72, y_top - 0.05), (5.5, 3.9), "y")
    # B5 → OUT
    _arrow_labeled(ax, (9.65, 2.35), (12.0, 1.4))
    # CFG → blocks (пунктир)
    cfg_y = 7.55
    for xc in [1.82, 4.72, 7.62, 10.52, 7.0]:
        _arrow_labeled(ax, (xc, cfg_y), (xc, y_top + bh + 0.02), ls="--", style="-|>", color="#8A8A00")

    ax.text(
        0.35,
        8.75,
        "Источник: диплом/block-schemas.md",
        fontsize=6.5,
        color="#666666",
        style="italic",
    )

    fig.tight_layout()
    path = OUT / "fig_4_1_architecture.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def fig_4_2_pipeline():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Рисунок 4.2 — Конвейер обработки log-файлов в приложении", fontsize=11, pad=10)

    steps = [
        (0.3, 2.2, "Ввод путей\nи L_m"),
        (2.5, 2.2, "Сканирование\nкаталога"),
        (4.7, 2.2, "Парсинг\nTrain / Line"),
        (6.9, 2.2, "φ, z"),
        (9.1, 2.2, "Фильтр\nω=1"),
        (11.3, 2.2, "Обучение /\nпредсказание"),
    ]
    for i, (x, y, t) in enumerate(steps):
        _box(ax, (x, y), 1.8, 1.5, t, fs=8)
        if i < len(steps) - 1:
            xn = steps[i + 1][0]
            _arrow(ax, (x + 1.8, y + 0.75), (xn, y + 0.75))

    _box(ax, (5.5, 0.3), 3.5, 1.0, "Отчёт: метрики, графики, calibration.json", style=BOX_OK, fs=8)
    _arrow(ax, (12.2, 2.2), (7.2, 1.3))

    fig.tight_layout()
    path = OUT / "fig_4_2_pipeline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    fig_3_5_algorithm()
    fig_4_1_architecture()
    fig_4_2_pipeline()
