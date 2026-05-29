# -*- coding: utf-8 -*-
"""
app.py — интерактивный дашборд Монте-Карло оценки девелоперского проекта.

Прикладное продолжение курсовой работы «Монте-Карло симуляция в оценке
девелоперских проектов: моделирование совместного распределения рисков».

Приложение содержит только пользовательский интерфейс и визуализацию;
вся расчётная логика вынесена в модули model.py и utils.py.

Запуск:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model
import utils

# ---------------------------------------------------------------------------
# Конфигурация страницы
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Монте-Карло оценка девелоперского проекта",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# фирменная палитра (спокойные корпоративные цвета)
BLUE = "#2f5d8a"
LIGHT_BLUE = "#7ba3c9"
RED = "#b5482e"
GREY = "#8a8a8a"
GREEN = "#3c7a57"
BG_CARD = "#f4f7fa"

PLOTLY_FONT = dict(family="Georgia, 'Times New Roman', serif", size=13, color="#23303d")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1180px;}
      h1, h2, h3 {color: #23303d;}
      .metric-card {
          background: #f4f7fa; border: 1px solid #e1e8ef; border-radius: 10px;
          padding: 14px 16px; height: 100%;
      }
      .metric-card .lbl {font-size: 0.82rem; color: #5a6b7b; margin-bottom: 4px;}
      .metric-card .val {font-size: 1.35rem; font-weight: 700; color: #23303d;}
      .verdict-box {border-radius: 12px; padding: 18px 22px; margin: 6px 0 4px 0;}
      .small-note {color: #6b7a88; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Вспомогательные функции отображения
# ---------------------------------------------------------------------------
def metric_card(label: str, value: str, color: str = "#23303d") -> str:
    """Вернуть HTML карточки-показателя."""
    return (
        f'<div class="metric-card"><div class="lbl">{label}</div>'
        f'<div class="val" style="color:{color}">{value}</div></div>'
    )


def fig_histogram(npv: np.ndarray, metrics: dict, title: str) -> go.Figure:
    """Гистограмма распределения NPV с линиями NPV=0, среднего и VaR."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=npv, nbinsx=60, marker_color=BLUE, opacity=0.85,
        marker_line_color="white", marker_line_width=0.4, name="NPV",
    ))
    for x, color, dash, label in [
        (0, RED, "dash", "NPV = 0"),
        (metrics["mean"], "#23303d", "solid", f"Средняя = {utils.fmt_num(metrics['mean'])}"),
        (metrics["var"], GREY, "dot", f"VaR (5%) = {utils.fmt_num(metrics['var'])}"),
    ]:
        fig.add_vline(x=x, line_color=color, line_dash=dash, line_width=2,
                      annotation_text=label, annotation_position="top",
                      annotation_font_size=11)
    fig.update_layout(
        title=title, bargap=0.02, font=PLOTLY_FONT, height=430,
        xaxis_title="NPV, млн руб.", yaxis_title="Частота (число итераций)",
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        margin=dict(l=60, r=30, t=70, b=50),
    )
    fig.update_xaxes(gridcolor="#eef2f6", zeroline=False)
    fig.update_yaxes(gridcolor="#eef2f6")
    return fig


def fig_cdf(npv_corr: np.ndarray, npv_ind: np.ndarray) -> go.Figure:
    """Кумулятивные функции распределения NPV для двух моделей."""
    fig = go.Figure()
    for npv, color, name in [
        (npv_ind, GREY, "Независимые риски"),
        (npv_corr, BLUE, "Коррелированные риски"),
    ]:
        xs = np.sort(npv)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name,
                                 line=dict(color=color, width=2.4)))
    fig.add_vline(x=0, line_color=RED, line_dash="dash", line_width=1.6)
    fig.update_layout(
        title="Кумулятивная функция распределения NPV  P(NPV ≤ x)",
        font=PLOTLY_FONT, height=430, xaxis_title="NPV, млн руб.",
        yaxis_title="P(NPV ≤ x)", plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    fig.update_xaxes(gridcolor="#eef2f6")
    fig.update_yaxes(gridcolor="#eef2f6", range=[0, 1])
    return fig


def fig_tornado(sensitivity: list) -> go.Figure:
    """Торнадо-диаграмма чувствительности NPV к факторам риска."""
    s = sorted(sensitivity, key=lambda d: abs(d["corr"]))
    names = [d["factor"] for d in s]
    vals = [d["corr"] for d in s]
    colors = [BLUE if v >= 0 else RED for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker_color=colors,
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#23303d", line_width=1)
    fig.update_layout(
        title="Чувствительность NPV к факторам риска (корреляция фактора с NPV)",
        font=PLOTLY_FONT, height=430, xaxis_title="Коэффициент корреляции с NPV",
        plot_bgcolor="white", paper_bgcolor="white", xaxis_range=[-1, 1],
        margin=dict(l=60, r=40, t=70, b=50),
    )
    fig.update_xaxes(gridcolor="#eef2f6")
    return fig


def fig_heatmap(matrix: np.ndarray) -> go.Figure:
    """Тепловая карта корреляционной матрицы факторов риска."""
    labels = model.RISK_LABELS
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu",
        reversescale=True, text=[[f"{v:.2f}" for v in row] for row in matrix],
        texttemplate="%{text}", textfont={"size": 12},
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(
        title="Корреляционная структура факторов риска",
        font=PLOTLY_FONT, height=480, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=80, r=30, t=70, b=80),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def fig_compare_hist(npv_corr: np.ndarray, npv_ind: np.ndarray) -> go.Figure:
    """Наложенные гистограммы NPV двух моделей."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=npv_ind, nbinsx=60, name="Независимые риски",
                               marker_color=GREY, opacity=0.55))
    fig.add_trace(go.Histogram(x=npv_corr, nbinsx=60, name="Коррелированные риски",
                               marker_color=BLUE, opacity=0.65))
    fig.add_vline(x=0, line_color=RED, line_dash="dash", line_width=1.6)
    fig.update_layout(
        barmode="overlay", title="Сравнение распределений NPV",
        font=PLOTLY_FONT, height=430, xaxis_title="NPV, млн руб.",
        yaxis_title="Частота", plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    fig.update_xaxes(gridcolor="#eef2f6")
    fig.update_yaxes(gridcolor="#eef2f6")
    return fig


# ---------------------------------------------------------------------------
# Заголовок
# ---------------------------------------------------------------------------
st.title("Монте-Карло оценка девелоперского проекта: NPV и корреляция рисков")
st.markdown(
    "<p style='font-size:1.05rem;color:#4a5a6a;margin-top:-8px'>"
    "Интерактивная модель оценки влияния строительных, рыночных и "
    "регуляторных рисков на NPV проекта</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Боковая панель — параметры проекта, факторы риска, симуляция
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.header("Параметры модели")

sb.subheader("Базовые параметры проекта")
initial_investment = sb.number_input("Начальные инвестиции I₀, млн руб.",
                                      value=500.0, min_value=0.0, step=10.0)
base_revenue = sb.number_input("Базовая выручка (GDV), млн руб.",
                               value=1850.0, min_value=0.0, step=50.0)
base_construction = sb.number_input("Базовые строительные затраты, млн руб.",
                                    value=550.0, min_value=0.0, step=10.0)
opex = sb.number_input("Операционные расходы, млн руб.",
                       value=50.0, min_value=0.0, step=5.0)
discount_rate = sb.slider("Ставка дисконтирования, %",
                          min_value=5.0, max_value=30.0, value=16.0, step=0.5) / 100.0
horizon = sb.slider("Горизонт проекта, лет", min_value=3, max_value=6, value=4)

sb.subheader("Параметры симуляции")
n_iter = sb.select_slider("Число итераций",
                          options=[1000, 2000, 5000, 10000, 20000, 50000],
                          value=10000)
seed = sb.number_input("Зерно генератора (seed)", value=42, min_value=0, step=1)

sb.markdown("---")
run = sb.button("▶  Запустить симуляцию", type="primary", use_container_width=True)
sb.caption("Левшин Д. Д. · Дисциплина «Инвестиционная оценка» · 2026")

# профили потоков по умолчанию (адаптируются под горизонт в model._profile)
project = model.ProjectParams(
    initial_investment=initial_investment,
    base_revenue=base_revenue,
    base_construction=base_construction,
    opex=opex,
    discount_rate=discount_rate,
    horizon=int(horizon),
)


# ===========================================================================
# 1. ОПИСАНИЕ МОДЕЛИ
# ===========================================================================
st.header("1. Описание модели")
st.markdown(
    "Модель оценивает чистую приведённую стоимость (**NPV**) девелоперского "
    "проекта методом **Монте-Карло**. Вместо одной точечной оценки строится "
    "распределение нескольких тысяч возможных исходов проекта, учитывающее "
    "одновременное действие пяти факторов риска. Ключевая особенность модели — "
    "учёт **корреляции** между факторами: в девелопменте перерасход бюджета, "
    "задержки, снижение цен и регуляторные ограничения, как правило, наступают "
    "не независимо, а совместно."
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Целевая аудитория модели")
    st.info(
        "Dashboard предназначен для девелоперов, инвесторов, банков проектного "
        "финансирования, оценщиков и консультантов, которым необходимо оценить "
        "не только базовую доходность проекта, но и распределение возможных "
        "исходов с учетом совместного наступления строительных, рыночных и "
        "регуляторных рисков. В учебном контексте модель демонстрирует "
        "практическое применение метода Монте-Карло в инвестиционной оценке."
    )
with c2:
    st.markdown("##### Практический смысл модели")
    st.success(
        "Модель позволяет перейти от вопроса «чему равна NPV проекта?» к вопросу "
        "«с какой вероятностью проект создаст или разрушит стоимость?». Это "
        "особенно важно для девелопмента, где перерасход бюджета, задержка "
        "строительства, изменение цен продажи и регуляторные ограничения часто "
        "реализуются не независимо, а совместно."
    )

st.markdown(
    "**Пять факторов риска модели:** рост строительных затрат, изменение цены "
    "продажи, задержка строительства, дополнительные регуляторные расходы и "
    "темп продаж (absorption). Случайные значения факторов генерируются "
    "методом **гауссовой копулы** (разложение Холецкого корреляционной матрицы) "
    "с заданными предельными распределениями."
)


# ===========================================================================
# 2. ВВОД ПАРАМЕТРОВ
# ===========================================================================
st.header("2. Ввод параметров проекта")
st.markdown(
    "Базовые параметры задаются на боковой панели слева. Ниже приведены "
    "текущие значения и рассчитанная по ним **детерминированная (базовая) NPV** "
    "— оценка без учёта неопределённости."
)

base_npv = model.calculate_base_npv(project)
params_df = pd.DataFrame({
    "Параметр": ["Начальные инвестиции I₀", "Базовая выручка (GDV)",
                 "Базовые строительные затраты", "Операционные расходы",
                 "Ставка дисконтирования", "Горизонт проекта"],
    "Значение": [utils.fmt_money(initial_investment), utils.fmt_money(base_revenue),
                 utils.fmt_money(base_construction), utils.fmt_money(opex),
                 utils.fmt_pct(discount_rate * 100), f"{int(horizon)} лет"],
})
cc1, cc2 = st.columns([1.4, 1])
with cc1:
    st.dataframe(params_df, hide_index=True, use_container_width=True)
with cc2:
    color = GREEN if base_npv >= 0 else RED
    st.markdown(metric_card("Базовая (детерминированная) NPV",
                            utils.fmt_money(base_npv), color), unsafe_allow_html=True)
    st.caption("Рассчитана при нейтральных значениях всех факторов риска.")


# ===========================================================================
# 3. НАСТРОЙКА ФАКТОРОВ РИСКА
# ===========================================================================
st.header("3. Настройка факторов риска")
st.markdown(
    "Каждый фактор риска задаётся предельным распределением. Непрерывные "
    "факторы моделируются **треугольным** распределением (минимум / наиболее "
    "вероятное значение / максимум), задержка строительства — **дискретным**."
)

rc1, rc2 = st.columns(2)
with rc1:
    st.markdown("**Рост строительных затрат (доли)**")
    cost_min = st.number_input("Затраты: минимум", value=-0.05, step=0.01, format="%.2f")
    cost_mode = st.number_input("Затраты: наиболее вероятно", value=0.10, step=0.01, format="%.2f")
    cost_max = st.number_input("Затраты: максимум", value=0.30, step=0.01, format="%.2f")

    st.markdown("**Изменение цены продажи (доли)**")
    price_min = st.number_input("Цена: минимум", value=-0.20, step=0.01, format="%.2f")
    price_mode = st.number_input("Цена: наиболее вероятно", value=0.05, step=0.01, format="%.2f")
    price_max = st.number_input("Цена: максимум", value=0.20, step=0.01, format="%.2f")

    st.markdown("**Темп продаж / absorption (индекс −1…+1)**")
    abs_min = st.number_input("Absorption: минимум", value=-1.0, step=0.1, format="%.1f")
    abs_mode = st.number_input("Absorption: наиболее вероятно", value=0.0, step=0.1, format="%.1f")
    abs_max = st.number_input("Absorption: максимум", value=1.0, step=0.1, format="%.1f")

with rc2:
    st.markdown("**Дополнительные регуляторные расходы (млн руб.)**")
    reg_min = st.number_input("Регул.: минимум", value=0.0, step=1.0, format="%.1f")
    reg_mode = st.number_input("Регул.: наиболее вероятно", value=20.0, step=1.0, format="%.1f")
    reg_max = st.number_input("Регул.: максимум", value=80.0, step=1.0, format="%.1f")

    st.markdown("**Задержка строительства (дискретное распределение)**")
    p0 = st.slider("P(задержка = 0 мес.)", 0.0, 1.0, 0.50, 0.05)
    p6 = st.slider("P(задержка = 6 мес.)", 0.0, 1.0, 0.35, 0.05)
    p12 = st.slider("P(задержка = 12 мес.)", 0.0, 1.0, 0.15, 0.05)
    psum = p0 + p6 + p12
    if abs(psum - 1.0) > 1e-6:
        st.caption(f"⚠ Сумма вероятностей = {psum:.2f}; будет нормирована к 1.")

risk = model.RiskParams(
    cost_growth=(cost_min, cost_mode, cost_max),
    price_change=(price_min, price_mode, price_max),
    delay_values=(0.0, 6.0, 12.0),
    delay_probs=(p0, p6, p12),
    reg_cost=(reg_min, reg_mode, reg_max),
    absorption=(abs_min, abs_mode, abs_max),
)


# ===========================================================================
# 4. КОРРЕЛЯЦИОННАЯ МАТРИЦА
# ===========================================================================
st.header("4. Корреляционная матрица")
st.markdown(
    "Матрица задаёт попарные коэффициенты корреляции между факторами риска. "
    "Её можно редактировать прямо в таблице. Перед симуляцией матрица "
    "проверяется на **положительную определённость**: если введённые значения "
    "делают её некорректной, автоматически используется ближайшая корректная "
    "матрица, о чём выводится предупреждение."
)

default_corr_df = utils.correlation_dataframe(model.DEFAULT_CORRELATION)
edited = st.data_editor(
    default_corr_df, use_container_width=True,
    column_config={c: st.column_config.NumberColumn(format="%.2f", min_value=-1.0,
                                                     max_value=1.0, step=0.05)
                   for c in default_corr_df.columns},
    key="corr_editor",
)
corr_input = edited.to_numpy(dtype=float)
corr_matrix, is_valid, corr_msg = utils.validate_correlation_matrix(
    corr_input, fallback=model.DEFAULT_CORRELATION)

if is_valid:
    st.success("✓ " + corr_msg)
else:
    st.warning("⚠ " + corr_msg)
    with st.expander("Показать скорректированную матрицу"):
        st.dataframe(utils.correlation_dataframe(corr_matrix).round(3),
                     use_container_width=True)

st.plotly_chart(fig_heatmap(corr_matrix), use_container_width=True)


# ===========================================================================
# 5. ЗАПУСК СИМУЛЯЦИИ
# ===========================================================================
st.header("5. Запуск симуляции")

if run:
    with st.spinner(f"Выполняется {n_iter:,} итераций Монте-Карло…".replace(",", " ")):
        try:
            results = model.run_both_models(project, risk, corr_matrix,
                                            n=int(n_iter), seed=int(seed))
            st.session_state["results"] = results
            st.session_state["meta"] = {"n": int(n_iter), "seed": int(seed)}
        except Exception as exc:  # защита от падения интерфейса
            st.error(f"Ошибка при расчёте: {exc}")

if "results" not in st.session_state:
    st.info(
        "Задайте параметры на боковой панели и нажмите **«Запустить симуляцию»**. "
        "Результаты появятся в разделах 6–9 ниже."
    )
    st.stop()

results = st.session_state["results"]
meta = st.session_state.get("meta", {"n": int(n_iter), "seed": int(seed)})
mc = results["correlated"]
mi = results["independent"]
m_corr = mc["metrics"]
m_ind = mi["metrics"]

st.success(
    f"Симуляция выполнена: {meta['n']:,} итераций, seed = {meta['seed']}. "
    "Результаты воспроизводимы при том же зерне генератора.".replace(",", " ")
)


# ===========================================================================
# 6. РЕЗУЛЬТАТЫ MONTE CARLO
# ===========================================================================
st.header("6. Результаты Monte Carlo")
st.markdown(
    "Распределение NPV по модели с **коррелированными** рисками — основной "
    "результат оценки. Линии отмечают NPV = 0, среднюю NPV и уровень потерь "
    "VaR (5-й перцентиль)."
)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(metric_card("Средняя NPV", utils.fmt_money(m_corr["mean"]),
                        GREEN if m_corr["mean"] >= 0 else RED), unsafe_allow_html=True)
k2.markdown(metric_card("Медианная NPV", utils.fmt_money(m_corr["median"])),
            unsafe_allow_html=True)
k3.markdown(metric_card("Вероятность NPV < 0", utils.fmt_pct(m_corr["prob_neg"]),
                        RED if m_corr["prob_neg"] > 30 else "#23303d"),
            unsafe_allow_html=True)
k4.markdown(metric_card("VaR (5%)", utils.fmt_money(m_corr["var"]), RED),
            unsafe_allow_html=True)

st.plotly_chart(fig_histogram(mc["npv"], m_corr,
                "Распределение NPV (коррелированные риски)"), use_container_width=True)

with st.expander("Полная таблица показателей риска (коррелированная модель)"):
    single = pd.DataFrame({
        "Показатель": ["Средняя NPV", "Медиана", "Стандартное отклонение",
                       "Минимум", "Максимум", "5-й перцентиль (VaR)",
                       "95-й перцентиль", "Ожидаемые потери (ES 5%)",
                       "Вероятность NPV < 0"],
        "Значение": [utils.fmt_money(m_corr["mean"]), utils.fmt_money(m_corr["median"]),
                     utils.fmt_money(m_corr["std"]), utils.fmt_money(m_corr["min"]),
                     utils.fmt_money(m_corr["max"]), utils.fmt_money(m_corr["var"]),
                     utils.fmt_money(m_corr["p95"]), utils.fmt_money(m_corr["es"]),
                     utils.fmt_pct(m_corr["prob_neg"])],
    })
    st.dataframe(single, hide_index=True, use_container_width=True)


# ===========================================================================
# 7. СРАВНЕНИЕ НЕЗАВИСИМЫХ / КОРРЕЛИРОВАННЫХ РИСКОВ
# ===========================================================================
st.header("7. Сравнение независимых и коррелированных рисков")
st.markdown(
    "Сопоставление двух моделей показывает **цену игнорирования корреляции**. "
    "Модель с независимыми рисками, как правило, недооценивает «левый хвост» "
    "распределения и вероятность убытка."
)

st.plotly_chart(fig_compare_hist(mc["npv"], mi["npv"]), use_container_width=True)
cmp1, cmp2 = st.columns(2)
with cmp1:
    st.plotly_chart(fig_cdf(mc["npv"], mi["npv"]), use_container_width=True)
with cmp2:
    st.markdown("**Сравнительная таблица показателей риска**")
    st.dataframe(utils.metrics_table(m_corr, m_ind), hide_index=True,
                 use_container_width=True)
    d_pneg = m_corr["prob_neg"] - m_ind["prob_neg"]
    st.markdown(
        f"<p class='small-note'>Разница вероятности убытка (коррел. − незав.): "
        f"<b>{d_pneg:+.1f} п.п.</b></p>", unsafe_allow_html=True)


# ===========================================================================
# 8. ЧУВСТВИТЕЛЬНОСТЬ NPV
# ===========================================================================
st.header("8. Чувствительность NPV к факторам риска")
st.markdown(
    "Чувствительность измеряется коэффициентом корреляции Пирсона между "
    "значением фактора и итоговой NPV. Чем длиннее столбец, тем сильнее фактор "
    "влияет на результат; цвет отражает направление влияния."
)

t1, t2 = st.columns([1.5, 1])
with t1:
    st.plotly_chart(fig_tornado(mc["sensitivity"]), use_container_width=True)
with t2:
    st.markdown("**Таблица чувствительности**")
    st.dataframe(utils.sensitivity_table(mc["sensitivity"]), hide_index=True,
                 use_container_width=True)


# ===========================================================================
# 9. ИНВЕСТИЦИОННЫЙ ВЫВОД
# ===========================================================================
st.header("9. Инвестиционный вывод")
verdict = utils.generate_investment_verdict(m_corr, m_ind)
vcolor = {"attractive": "#e6f3ea", "moderate": "#fdf3e0", "risky": "#fbe8e4"}[verdict["verdict"]]
vborder = {"attractive": GREEN, "moderate": "#c08a2e", "risky": RED}[verdict["verdict"]]

st.markdown(
    f'<div class="verdict-box" style="background:{vcolor};border-left:6px solid {vborder}">'
    f'<h3 style="margin-top:0">{verdict["headline"]}</h3>'
    f'<p style="margin-bottom:8px">{verdict["body"]}</p>'
    f'<p style="margin-bottom:0"><b>Эффект учёта корреляции:</b> {verdict["effect"]}</p>'
    f'</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Вывод формируется автоматически по показателям модели с коррелированными "
    "рисками. Это учебная модель; результаты не являются инвестиционной "
    "рекомендацией."
)


# ===========================================================================
# 10. МЕТОДОЛОГИЯ
# ===========================================================================
st.header("10. Методология")
st.markdown(
    """
**Денежные потоки и NPV.** Проект моделируется на горизонте в несколько лет.
Выручка, строительные и операционные расходы распределяются по годам согласно
заданным профилям и дисконтируются по ставке *r*. Чистая приведённая стоимость:
"""
)
st.latex(r"NPV = -I_0 + \sum_{t=1}^{T} \frac{CF_t}{(1+r)^{t}}")
st.markdown(
    """
где *I₀* — начальные инвестиции, *CFₜ* — чистый денежный поток года *t*,
*r* — ставка дисконтирования, *T* — горизонт проекта.

**Факторы риска и предельные распределения.** Пять факторов задаются
треугольными (затраты, цена, регуляторные расходы, темп продаж) и дискретным
(задержка) распределениями. Значения факторов влияют на потоки: цена
масштабирует выручку, рост затрат — строительные расходы, задержка сдвигает
поступление выручки и добавляет расходы на содержание, темп продаж
перераспределяет выручку между годами.

**Гауссова копула.** Для генерации коррелированных факторов используется
гауссова копула. Из независимых стандартных нормальных величин *Z* строятся
коррелированные *Z\\* = Z·Lᵀ*, где *L* — нижнетреугольный множитель Холецкого
корреляционной матрицы *Σ = L·Lᵀ*. Затем *Z\\** переводятся в равномерные
*U = Φ(Z\\*)* и через обратные предельные функции — в значения факторов риска.
"""
)
st.latex(r"\Sigma = L L^{\mathsf T}, \quad Z^{*} = Z L^{\mathsf T}, \quad U = \Phi(Z^{*})")
st.markdown(
    """
**Показатели риска.** VaR (5%) — 5-й перцентиль распределения NPV; Expected
Shortfall (ES) — средняя NPV в 5% худших сценариев; P(NPV<0) — вероятность
убытка. Чувствительность оценивается корреляцией Пирсона факторов с NPV.

**Корректность матрицы.** Перед симуляцией корреляционная матрица проверяется
на положительную определённость (разложение Холецкого). При некорректном вводе
строится ближайшая корректная матрица (проекция собственных значений).

**Ограничения.** Это учебная модель с условными параметрами. Распределения и
корреляции заданы экспертно и подлежат калибровке на реальных данных. Результаты
носят иллюстративный характер и **не являются инвестиционной рекомендацией**.
"""
)

st.divider()
st.caption(
    "Прикладное продолжение курсовой работы по дисциплине «Инвестиционная "
    "оценка». Автор: Левшин Даниил Дмитриевич, Москва, 2026."
)
