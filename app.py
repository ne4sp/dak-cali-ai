"""
Веб-интерфейс: обучение модели и визуализация результатов.
Запуск из корня проекта: streamlit run app.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from DatasetGenerator import list_training_log_relpaths
from analysis import history_comparison_df, load_training_data, train_model
from calibration_io import (
    DEFAULT_LEVELS_TEXT,
    calibration_manual_export,
    calibration_to_json_bytes,
    compare_trainlog_vs_training_corpus,
    concentration_calibration_stats,
    merge_calibration_with_template,
    parse_densities_text,
    parse_levels_text,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_ZERO = ROOT / "dak" / "train.log"
DEFAULT_VAL = ROOT / "dak" / "conc_history_2025-10-14.log"
DEFAULT_TRAINS_DIR = "trains"
DEFAULT_CALIB_TEMPLATE = ROOT / "dak" / "calibration (19).json"


def _resolve_path(raw: str) -> Path:
    p = Path(raw.strip())
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p.resolve()


def _open_in_file_manager(folder: Path) -> bool:
    folder = folder.resolve()
    if not folder.is_dir():
        return False
    try:
        if sys.platform == "win32":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)], check=False)
        else:
            subprocess.run(["xdg-open", str(folder)], check=False)
        return True
    except OSError:
        return False


def _fig_test_scatter(y_test: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    y_plot = np.round(np.asarray(y_pred, dtype=float), 1)
    df = pd.DataFrame({"Факт": y_test, "Модель": y_plot})
    lim_min = float(min(float(y_test.min()), float(y_plot.min())))
    lim_max = float(max(float(y_test.max()), float(y_plot.max())))
    pad = (lim_max - lim_min) * 0.05 + 1e-6
    lim_min -= pad
    lim_max += pad
    fig = px.scatter(
        df,
        x="Факт",
        y="Модель",
        opacity=0.75,
        title="Тестовая выборка CatBoost: факт vs предсказание модели (округление до 0,1)",
    )
    fig.add_trace(
        go.Scatter(
            x=[lim_min, lim_max],
            y=[lim_min, lim_max],
            mode="lines",
            name="Идеал (y = x)",
            line=dict(dash="dash", width=2, color="rgba(255,120,80,0.9)"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,17,22,1)",
        plot_bgcolor="rgba(22,25,32,1)",
        font=dict(family="Segoe UI, sans-serif", size=14),
        height=420,
        xaxis_title="Реальная концентрация (тест)",
        yaxis_title="Предсказание модели (до десятых)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _format_density_option(key: float, n_total: int, n_valid: int) -> str:
    if float(key).is_integer():
        disp = str(int(round(float(key))))
    else:
        disp = str(key).rstrip("0").rstrip(".")
    return f"{disp} — всего {n_total}, в калибровку {n_valid}"


def _fig_concentration_counts(
    stats: list[tuple[float, int, int]],
) -> go.Figure:
    if not stats:
        fig = go.Figure()
        fig.update_layout(title="Нет данных по концентрациям", height=320)
        return fig
    labels = [_format_density_option(s[0], s[1], s[2]) for s in stats]
    tot = [s[1] for s in stats]
    ok = [s[2] for s in stats]
    fig = go.Figure(
        data=[
            go.Bar(name="Всего записей", x=labels, y=tot, marker_color="rgba(90, 160, 220, 0.85)"),
            go.Bar(
                name="С полным пересечением (попадут в points)",
                x=labels,
                y=ok,
                marker_color="rgba(80, 200, 140, 0.9)",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Концентрации в обучающих логах: сколько раз встречается density",
        template="plotly_dark",
        paper_bgcolor="rgba(15,17,22,1)",
        plot_bgcolor="rgba(22,25,32,1)",
        font=dict(family="Segoe UI, sans-serif", size=13),
        height=max(360, min(520, 120 + 28 * len(stats))),
        xaxis=dict(title="Концентрация (density)", tickangle=-35),
        yaxis=dict(title="Количество записей"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _fig_trainlog_vs_corpus_mae(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["MAE (корпус vs train.log)"])
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title="MAE: нет пар «корпус ↔ train.log»",
            template="plotly_dark",
            paper_bgcolor="rgba(15,17,22,1)",
            plot_bgcolor="rgba(22,25,32,1)",
            height=280,
        )
        return fig
    n = len(sub)
    # Спокойный градиент сине-бирюзовый (без «ошибочного» красного/оранжевого)
    base = np.linspace(0.35, 0.92, n)
    bar_colors = [f"rgba(70, {int(140 + 80 * t)}, {int(200 + 40 * t)}, 0.92)" for t in base]

    fig = go.Figure(
        data=[
            go.Bar(
                x=sub["density"].astype(str),
                y=sub["MAE (корпус vs train.log)"],
                marker=dict(
                    color=bar_colors,
                    line=dict(width=0),
                    cornerradius=4,
                ),
                name="MAE",
            )
        ]
    )
    fig.update_layout(
        title="Средняя модульная разница по уровням: корпус trains/ и train.log",
        template="plotly_dark",
        paper_bgcolor="rgba(15,17,22,1)",
        plot_bgcolor="rgba(22,25,32,1)",
        font=dict(family="Segoe UI, sans-serif", size=13, color="#e2e8f0"),
        height=380,
        xaxis_title="density из train.log",
        yaxis_title="MAE по уровням",
        showlegend=False,
        xaxis=dict(gridcolor="rgba(80,100,130,0.25)", zeroline=False),
        yaxis=dict(gridcolor="rgba(80,100,130,0.25)", zeroline=False),
    )
    return fig


# Теплокарта сравнения: тёмно-синий → бирюза → мягкий лавандовый (нейтрально, не «ошибка»)
_TRAINLOG_HEATMAP_SCALE = [
    [0.0, "rgb(18, 32, 52)"],
    [0.25, "rgb(35, 85, 130)"],
    [0.5, "rgb(55, 145, 168)"],
    [0.75, "rgb(110, 190, 205)"],
    [1.0, "rgb(175, 205, 235)"],
]


def _fig_trainlog_delta_heatmap(
    Z: np.ndarray,
    x_labels: list[str],
    y_level_lbl: list[str],
) -> go.Figure:
    if Z.size == 0 or not np.any(np.isfinite(Z)):
        fig = go.Figure()
        fig.update_layout(
            title="Теплокарта: нет данных",
            template="plotly_dark",
            paper_bgcolor="rgba(15,17,22,1)",
            height=280,
        )
        return fig
    zt = np.where(np.isfinite(Z), Z, np.nan).T
    fig = go.Figure(
        data=go.Heatmap(
            x=x_labels,
            y=y_level_lbl,
            z=zt,
            colorscale=_TRAINLOG_HEATMAP_SCALE,
            colorbar=dict(
                title=dict(
                    text="Модуль<br>разницы",
                    side="right",
                    font=dict(color="#e2e8f0", size=12),
                ),
                tickfont=dict(color="#cbd5e1"),
            ),
            hovertemplate="density %{x}<br>уровень %{y}<br>|Δ| %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Разница по уровням и density (|корпус − train.log|)",
        template="plotly_dark",
        paper_bgcolor="rgba(15,17,22,1)",
        plot_bgcolor="rgba(22,25,32,1)",
        font=dict(family="Segoe UI, sans-serif", size=13, color="#e2e8f0"),
        height=max(340, 48 + 24 * len(y_level_lbl)),
        xaxis_title="density (как в train.log)",
        yaxis_title="Уровень засветки",
    )
    return fig


def _summary_block_html(
    mean_hist: float | None,
    mae_test: float,
    rmse: float,
    r2: float | None,
) -> str:
    h = "—" if mean_hist is None else f"{mean_hist:.4f}"
    r2s = "—" if r2 is None else f"{r2:.4f}"
    return f"""
    <div style="
        background: linear-gradient(145deg, rgba(30, 75, 120, 0.55) 0%, rgba(12, 28, 48, 0.92) 100%);
        border: 1px solid rgba(100, 190, 255, 0.55);
        border-radius: 16px;
        padding: 22px 26px 20px 26px;
        margin: 6px 0 22px 0;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.06) inset, 0 8px 32px rgba(40, 120, 200, 0.18);
    ">
        <div style="
            font-size: 1.35rem;
            font-weight: 700;
            color: #d4ecff;
            margin-bottom: 6px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        ">Сводка отклонений</div>
        <div style="font-size: 0.9rem; color: #9ec8e8; margin-bottom: 18px;">
            Краткий итог по проверочному логу и отложенной тестовой выборке CatBoost
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between;">
            <div style="min-width: 140px; flex: 1;">
                <div style="font-size: 0.75rem; color: #7eb8e0; text-transform: uppercase; letter-spacing: 0.08em;">
                    Среднее |отклонение|</div>
                <div style="font-size: 1.65rem; font-weight: 700; color: #fff; line-height: 1.2;">{h}</div>
                <div style="font-size: 0.8rem; color: #8ab4d9;">проверочный лог</div>
            </div>
            <div style="min-width: 120px; flex: 1;">
                <div style="font-size: 0.75rem; color: #7eb8e0; text-transform: uppercase; letter-spacing: 0.08em;">MAE</div>
                <div style="font-size: 1.65rem; font-weight: 700; color: #fff; line-height: 1.2;">{mae_test:.4f}</div>
                <div style="font-size: 0.8rem; color: #8ab4d9;">тест CatBoost</div>
            </div>
            <div style="min-width: 120px; flex: 1;">
                <div style="font-size: 0.75rem; color: #7eb8e0; text-transform: uppercase; letter-spacing: 0.08em;">RMSE</div>
                <div style="font-size: 1.65rem; font-weight: 700; color: #fff; line-height: 1.2;">{rmse:.4f}</div>
                <div style="font-size: 0.8rem; color: #8ab4d9;">тест CatBoost</div>
            </div>
            <div style="min-width: 100px; flex: 1;">
                <div style="font-size: 0.75rem; color: #7eb8e0; text-transform: uppercase; letter-spacing: 0.08em;">R²</div>
                <div style="font-size: 1.65rem; font-weight: 700; color: #fff; line-height: 1.2;">{r2s}</div>
                <div style="font-size: 0.8rem; color: #8ab4d9;">тест CatBoost</div>
            </div>
        </div>
    </div>
    """


ERR_OK = 0.2
ERR_BAD = 0.5


def _abs_error_bar_color(v: float) -> str:
    """≤0.2 — допустимо (зелёный), ≥0.5 — красный, между — плавный переход."""
    g = (46, 180, 100)
    r = (220, 55, 55)
    if v <= ERR_OK:
        return f"rgb({g[0]},{g[1]},{g[2]})"
    if v >= ERR_BAD:
        return f"rgb({r[0]},{r[1]},{r[2]})"
    t = (v - ERR_OK) / (ERR_BAD - ERR_OK)
    rr = int(g[0] + t * (r[0] - g[0]))
    gg = int(g[1] + t * (r[1] - g[1]))
    bb = int(g[2] + t * (r[2] - g[2]))
    return f"rgb({rr},{gg},{bb})"


def _fig_history_errors(df: pd.DataFrame) -> go.Figure:
    y = df["|Ошибка|"].astype(float)
    colors = [_abs_error_bar_color(float(v)) for v in y]
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["№"],
                y=y,
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate="№ %{x}<br>|ошибка| %{y:.4f}<extra></extra>",
            )
        ]
    )
    ymax = max(float(y.max()), ERR_BAD * 1.05, 0.01)
    fig.add_hline(
        y=ERR_OK,
        line_dash="dot",
        line_width=1.5,
        line_color="rgba(100, 220, 140, 0.75)",
        annotation_text=f"допустимо ≤ {ERR_OK}",
        annotation_position="right",
        annotation_font_size=11,
        annotation_font_color="rgb(140, 220, 170)",
    )
    fig.add_hline(
        y=ERR_BAD,
        line_dash="dot",
        line_width=1.5,
        line_color="rgba(255, 100, 100, 0.9)",
        annotation_text=f"критично ≥ {ERR_BAD}",
        annotation_position="right",
        annotation_font_size=11,
        annotation_font_color="rgb(255, 150, 150)",
    )
    fig.update_layout(
        title=(
            "Абсолютная ошибка по проверочному логу "
            f"(зелёный ≤ {ERR_OK}, красный ≥ {ERR_BAD})"
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(15,17,22,1)",
        plot_bgcolor="rgba(22,25,32,1)",
        font=dict(family="Segoe UI, sans-serif", size=14),
        height=400,
        showlegend=False,
        yaxis=dict(title="|Ошибка|", range=[0, ymax]),
        xaxis=dict(title="№ записи"),
    )
    return fig


def main():
    st.set_page_config(
        page_title="Калибровка концентрации",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; max-width: 1280px; }
        div[data-testid="stMetricValue"] { font-size: 1.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Оценка концентрации по кривой засветки")
    st.caption(
        "Обучение: выбранные `.log`. Признаки — относительные пересечения с **несколькими** уровнями засветки "
        "(дельта к нулевой линии файла). Формат `calibration.json` — как в `conc`: `levels` + `points[].array` + `density`."
    )

    st.subheader("1. Данные для обучения")
    trains_raw = st.text_input(
        "Каталог с логами обучения",
        value=DEFAULT_TRAINS_DIR,
        help="Относительно корня проекта или абсолютный путь.",
    )
    trains_root = _resolve_path(trains_raw)
    st.code(str(trains_root), language=None)

    c_open, c_hint = st.columns([1, 3])
    with c_open:
        if st.button("Открыть папку в проводнике", key="open_trains"):
            if _open_in_file_manager(trains_root):
                st.toast("Папка открыта.", icon="📂")
            else:
                st.warning("Каталог не найден или не удалось открыть.")

    with c_hint:
        st.caption("Отредактируйте или добавьте `.log` в этой папке, затем обновите страницу (R), если список файлов не совпал.")

    log_relpaths = list_training_log_relpaths(str(trains_root))
    if not log_relpaths:
        st.warning("В каталоге нет файлов `*.log`.")
        selected_logs: list[str] = []
    else:
        selected_logs = st.multiselect(
            "Файлы для обучения (снимите галочку, чтобы **исключить** файл)",
            options=log_relpaths,
            default=log_relpaths,
            help="Используются только отмеченные файлы. Пути относительно каталога выше.",
        )

    st.subheader("2. Эталонный train.log (нулевая линия)")
    zero_raw = st.text_input(
        "Путь к train.log",
        value=str(DEFAULT_ZERO),
        help="Из этого файла берётся только первая запись (ноль) для расчёта относительных признаков на проверочном логе.",
    )
    zero_p = _resolve_path(zero_raw)

    st.subheader("3. Лог проверки (с реальными концентрациями)")
    val_raw = st.text_input(
        "Путь к проверочному логу",
        value=str(DEFAULT_VAL),
        help="Концентрации из этого файла сравниваются с предсказаниями модели (эталон — нулевая линия из п. 2).",
    )
    val_p = _resolve_path(val_raw)

    st.subheader("4. Уровни засветки (пересечения)")
    levels_text = st.text_area(
        "Список уровней через запятую (или с новой строки)",
        value=DEFAULT_LEVELS_TEXT,
        height=88,
        help="Столько же чисел в каждом `array` в calibration.json. Как в прошивке: несколько горизонталей по кривой.",
    )
    cal_template_raw = st.text_input(
        "Шаблон полного calibration.json (EC/NTC/pH)",
        value=str(DEFAULT_CALIB_TEMPLATE) if DEFAULT_CALIB_TEMPLATE.is_file() else "",
        help="Если файл есть — в выгрузке подставится только секция `conc`, остальное из шаблона.",
    )
    cal_template_p = _resolve_path(cal_template_raw) if cal_template_raw.strip() else None

    with st.sidebar:
        st.header("Модель CatBoost")
        iterations = st.slider("Итерации", 100, 1200, 600, 50)
        lr = st.slider("Learning rate", 0.01, 0.3, 0.1, 0.01)
        test_frac = st.slider("Доля теста", 0.05, 0.35, 0.1, 0.05)
        train_btn = st.button("Обучить модель", type="primary", use_container_width=True)

    if train_btn:
        if not selected_logs:
            st.error("Выберите хотя бы один файл для обучения.")
        elif not zero_p.is_file():
            st.error(f"Эталонный лог не найден: `{zero_p}`")
        elif not val_p.is_file():
            st.error(f"Проверочный лог не найден: `{val_p}`")
        elif not trains_root.is_dir():
            st.error(f"Каталог обучения не найден: `{trains_root}`")
        else:
            try:
                levels_list = parse_levels_text(levels_text)
            except ValueError as e:
                st.error(str(e))
                return
            include_set = set(selected_logs)
            with st.spinner("Загрузка датасета и обучение…"):
                try:
                    X, Y = load_training_data(
                        trains_dir=str(trains_root),
                        include_relative=include_set,
                        levels=levels_list,
                    )
                except Exception as e:
                    st.error(f"Не удалось собрать датасет: {e}")
                    return
                if not X:
                    st.error("Датасет пустой: проверьте выбранные файлы и формат логов.")
                    return
                model, metrics, bundle = train_model(
                    X,
                    Y,
                    test_size=test_frac,
                    iterations=iterations,
                    learning_rate=lr,
                    verbose=False,
                )
                try:
                    cdf = history_comparison_df(
                        model,
                        str(zero_p),
                        str(val_p),
                        levels=levels_list,
                    )
                    compare_err = None
                except Exception as e:
                    cdf = None
                    compare_err = str(e)

                st.session_state["model"] = model
                st.session_state["metrics"] = metrics
                st.session_state["bundle"] = bundle
                st.session_state["compare_df"] = cdf
                st.session_state["compare_error"] = compare_err
                st.session_state["trains_root"] = str(trains_root)
                st.session_state["include_set"] = include_set
                st.session_state["levels_list"] = levels_list
                st.session_state["cal_template_path"] = (
                    str(cal_template_p) if cal_template_p and cal_template_p.is_file() else ""
                )
                st.session_state["zero_log_path"] = str(zero_p)

            st.success("Готово.")

    if "model" not in st.session_state:
        st.info("Заполните пути, отметьте файлы обучения и нажмите **«Обучить модель»** в боковой панели.")
        return

    m = st.session_state["metrics"]
    cdf = st.session_state.get("compare_df")
    err = st.session_state.get("compare_error")
    mean_hist = None
    if cdf is not None and len(cdf):
        mean_hist = float(cdf["|Ошибка|"].mean())

    st.markdown(
        _summary_block_html(mean_hist, m["mae_test"], m["rmse"], m.get("r2")),
        unsafe_allow_html=True,
    )

    st.subheader("Калибровка для датчика (`calibration.json`)")
    lv = st.session_state.get("levels_list") or []
    if not lv:
        st.info("Перезапустите обучение — в сессии нет списка уровней для выгрузки.")
    else:
        st.caption(
            f"Уровни: **{lv}**. Ниже: для каждой density из **train.log** сравниваем `array` из файла "
            "с **усреднённым** `array` по корпусу **trains/** (то, что ушло бы в синтетическую калибровку)."
        )
        tpl = st.session_state.get("cal_template_path") or ""
        ref_path = st.session_state.get("zero_log_path") or ""
        if not ref_path and DEFAULT_ZERO.is_file():
            ref_path = str(DEFAULT_ZERO)
        try:
            stats = concentration_calibration_stats(
                st.session_state["trains_root"],
                st.session_state.get("include_set"),
                list(lv),
            )
            st.plotly_chart(
                _fig_concentration_counts(stats),
                use_container_width=True,
            )

            if ref_path and Path(ref_path).is_file():
                st.subheader("Сравнение: train.log vs усреднённый корпус trains/")
                st.caption(
                    "Точки берутся по **всем концентрациям из train.log** (порядок как в файле). "
                    "Для каждой density: эталон — пересечения кривой **из train.log**; «корпус» — среднее "
                    "тех же пересечений по всем выбранным логам в `trains/`."
                )
                cmp_df, Z_cmp, xl, yl = compare_trainlog_vs_training_corpus(
                    ref_path,
                    st.session_state["trains_root"],
                    st.session_state.get("include_set"),
                    list(lv),
                    stats=stats,
                )
                st.dataframe(
                    cmp_df,
                    use_container_width=True,
                    height=min(460, 48 + 36 * len(cmp_df)),
                )
                st.plotly_chart(
                    _fig_trainlog_vs_corpus_mae(cmp_df),
                    use_container_width=True,
                )
                st.plotly_chart(
                    _fig_trainlog_delta_heatmap(Z_cmp, xl, yl),
                    use_container_width=True,
                )
            else:
                st.warning("Укажите существующий train.log в п. 2, чтобы построить сравнение.")

            st.subheader("Экспорт `calibration.json`")
            st.text_area(
                "Список density для `conc.points` (ориентир по графику выше)",
                value="0\n1\n5\n10",
                height=120,
                help="Числа по одному в строке или через запятую. Повторы одной концентрации игнорируются.",
                key="cal_density_lines",
            )
            density_text = st.session_state.get("cal_density_lines", "")
            try:
                densities_list = parse_densities_text(density_text)
            except ValueError as e:
                st.error(str(e))
                densities_list = []

            if not densities_list:
                st.info("Введите хотя бы одну концентрацию для выгрузки JSON.")
            else:
                conc, export_df = calibration_manual_export(
                    st.session_state["trains_root"],
                    st.session_state.get("include_set"),
                    list(lv),
                    densities_list,
                    stats=stats,
                )
                st.dataframe(
                    export_df,
                    use_container_width=True,
                    height=min(280, 48 + 36 * len(export_df)),
                )
                if not conc.get("points"):
                    st.warning(
                        "Ни одна из введённых концентраций не дала валидных данных в корпусе "
                        "(нет полных пересечений на всех уровнях)."
                    )
                full_cal = merge_calibration_with_template(conc, tpl if tpl else None)
                st.download_button(
                    label="Скачать calibration.json",
                    data=calibration_to_json_bytes(full_cal),
                    file_name="calibration.json",
                    mime="application/json",
                    key="dl_cal",
                    disabled=len(conc.get("points", [])) == 0,
                )
                st.caption(
                    f"В JSON попадёт **{len(conc.get('points', []))}** точек "
                    f"(уникальных в списке: **{len(export_df)}**)."
                )
        except Exception as e:
            st.warning(f"Не удалось собрать calibration.json: {e}")

    tab_a, tab_b = st.tabs(["Проверочный лог", "Тестовая выборка CatBoost"])

    with tab_a:
        if err:
            st.warning(err)
        elif cdf is not None and len(cdf):
            st.dataframe(
                cdf.style.format(
                    {
                        "Реальная концентрация": "{:.4f}",
                        "Предсказание": "{:.4f}",
                        "Ошибка": "{:.4f}",
                        "|Ошибка|": "{:.4f}",
                    }
                ),
                use_container_width=True,
                height=min(420, 42 + 35 * len(cdf)),
            )
            st.plotly_chart(_fig_history_errors(cdf), use_container_width=True)
        else:
            st.info("Нет данных сравнения.")

    with tab_b:
        b = st.session_state["bundle"]
        st.plotly_chart(
            _fig_test_scatter(b["y_test"], b["y_pred"]),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
