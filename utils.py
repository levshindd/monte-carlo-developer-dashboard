# -*- coding: utf-8 -*-
"""
utils.py — вспомогательные функции дашборда: форматирование чисел,
проверка корректности корреляционной матрицы (положительная
полуопределённость), построение таблиц для интерфейса и генерация
текстового инвестиционного вывода.

Модуль не зависит от Streamlit и может тестироваться отдельно.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from model import RISK_LABELS


# ---------------------------------------------------------------------------
# Форматирование чисел
# ---------------------------------------------------------------------------
def fmt_money(value: float, digits: int = 1) -> str:
    """
    Отформатировать денежную величину в млн руб. с разделителями разрядов.

    Пример: 1234.5 -> '1 234,5 млн руб.'
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    s = f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")
    return f"{s} млн руб."


def fmt_num(value: float, digits: int = 1) -> str:
    """Отформатировать число с пробелом-разделителем разрядов (без единиц)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def fmt_pct(value: float, digits: int = 1) -> str:
    """Отформатировать процент. На вход подаётся величина уже в процентах."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{value:.{digits}f}".replace(".", ",") + " %"


# ---------------------------------------------------------------------------
# Проверка и коррекция корреляционной матрицы
# ---------------------------------------------------------------------------
def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """
    Найти ближайшую положительно полуопределённую корреляционную матрицу
    (проекция через клиппинг собственных значений + нормировка диагонали к 1).
    """
    m = np.asarray(matrix, dtype=float)
    m = (m + m.T) / 2.0  # симметризация
    w, v = np.linalg.eigh(m)
    w = np.clip(w, 1e-8, None)  # убираем отрицательные собственные значения
    m_psd = v @ np.diag(w) @ v.T
    # нормировка диагонали к единице -> корректная корреляционная матрица
    d = np.sqrt(np.clip(np.diag(m_psd), 1e-12, None))
    m_psd = m_psd / np.outer(d, d)
    # числовая чистка симметрии и диагонали
    m_psd = (m_psd + m_psd.T) / 2.0
    np.fill_diagonal(m_psd, 1.0)
    return np.clip(m_psd, -1.0, 1.0)


def validate_correlation_matrix(matrix: np.ndarray,
                                fallback: np.ndarray | None = None
                                ) -> Tuple[np.ndarray, bool, str]:
    """
    Проверить корреляционную матрицу и при необходимости скорректировать её.

    Проверяется:
        * квадратная форма и симметричность;
        * единичная диагональ;
        * диапазон элементов [-1, 1];
        * положительная определённость (через разложение Холецкого).

    Если матрица некорректна, возвращается ближайшая положительно
    полуопределённая матрица (или переданная базовая `fallback`, если
    ближайшую построить не удалось).

    Возвращает кортеж (matrix_corrected, is_valid, message):
        matrix_corrected — гарантированно корректная матрица для симуляции;
        is_valid         — True, если исходная матрица уже была корректной;
        message          — поясняющее сообщение для интерфейса.
    """
    m = np.asarray(matrix, dtype=float)

    # форма
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        base = fallback if fallback is not None else np.eye(5)
        return np.asarray(base, dtype=float), False, (
            "Матрица не квадратная — использована базовая корреляционная матрица."
        )

    # диапазон значений
    if np.any(np.abs(m) > 1.0 + 1e-9):
        m = np.clip(m, -1.0, 1.0)

    # симметрия и диагональ
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 1.0)

    # проверка положительной определённости
    try:
        np.linalg.cholesky(m)
        eig_min = float(np.linalg.eigvalsh(m).min())
        if eig_min > 1e-10:
            return m, True, "Матрица корректна (положительно определена)."
    except np.linalg.LinAlgError:
        pass

    # коррекция
    try:
        m_fixed = _nearest_psd(m)
        np.linalg.cholesky(m_fixed)  # контроль успешности
        return m_fixed, False, (
            "Введённая матрица не является положительно определённой. "
            "Использована ближайшая корректная корреляционная матрица."
        )
    except Exception:
        base = fallback if fallback is not None else np.eye(m.shape[0])
        return np.asarray(base, dtype=float), False, (
            "Матрицу не удалось скорректировать — использована базовая матрица."
        )


def is_positive_definite(matrix: np.ndarray) -> bool:
    """Проверить положительную определённость матрицы (через Холецкого)."""
    try:
        np.linalg.cholesky(np.asarray(matrix, dtype=float))
        return True
    except np.linalg.LinAlgError:
        return False


# ---------------------------------------------------------------------------
# Таблицы для интерфейса
# ---------------------------------------------------------------------------
def metrics_table(metrics_corr: Dict[str, float],
                  metrics_ind: Dict[str, float]) -> pd.DataFrame:
    """
    Построить сравнительную таблицу показателей риска для двух моделей
    (коррелированные и независимые риски).
    """
    rows = [
        ("Средняя NPV", "mean"),
        ("Медианная NPV", "median"),
        ("Стандартное отклонение", "std"),
        ("Минимум", "min"),
        ("Максимум", "max"),
        ("5-й перцентиль (VaR 95%)", "p5"),
        ("95-й перцентиль", "p95"),
        ("Ожидаемые потери (ES 5%)", "es"),
        ("Вероятность NPV < 0, %", "prob_neg"),
    ]
    data = []
    for label, key in rows:
        c = metrics_corr.get(key, float("nan"))
        i = metrics_ind.get(key, float("nan"))
        if key == "prob_neg":
            data.append([label, fmt_pct(c), fmt_pct(i)])
        else:
            data.append([label, fmt_num(c), fmt_num(i)])
    return pd.DataFrame(
        data, columns=["Показатель", "Коррелированные риски", "Независимые риски"]
    )


def sensitivity_table(sensitivity: List[Dict]) -> pd.DataFrame:
    """Построить таблицу чувствительности NPV к факторам риска."""
    data = [
        [s["factor"], fmt_num(s["corr"], 3), s["direction"]]
        for s in sensitivity
    ]
    return pd.DataFrame(
        data, columns=["Фактор риска", "Коэффициент корреляции с NPV", "Направление влияния"]
    )


def correlation_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    """Преобразовать корреляционную матрицу в DataFrame с подписями факторов."""
    m = np.asarray(matrix, dtype=float)
    return pd.DataFrame(m, index=RISK_LABELS, columns=RISK_LABELS)


def scenario_table(scenarios: List[Tuple[str, float]]) -> pd.DataFrame:
    """Построить таблицу NPV по сценариям (название, значение)."""
    data = [[name, fmt_money(val)] for name, val in scenarios]
    return pd.DataFrame(data, columns=["Сценарий", "NPV"])


# ---------------------------------------------------------------------------
# Текстовый инвестиционный вывод
# ---------------------------------------------------------------------------
def generate_investment_verdict(metrics_corr: Dict[str, float],
                                metrics_ind: Dict[str, float]) -> Dict[str, str]:
    """
    Сформировать текстовый инвестиционный вывод на основе показателей риска
    модели с коррелированными рисками (более консервативная оценка).

    Возвращает словарь:
        'verdict'   — краткий вердикт ('attractive' / 'moderate' / 'risky');
        'headline'  — заголовок вывода;
        'body'      — развёрнутый текст;
        'effect'    — эффект учёта корреляции (сравнение двух моделей).
    """
    mean = metrics_corr["mean"]
    base = metrics_corr["base"]
    p_neg = metrics_corr["prob_neg"]
    var = metrics_corr["var"]
    es = metrics_corr["es"]

    # классификация привлекательности по вероятности убытка и средней NPV
    if mean > 0 and p_neg < 15:
        verdict = "attractive"
        headline = "Проект инвестиционно привлекателен"
    elif mean > 0 and p_neg < 35:
        verdict = "moderate"
        headline = "Проект умеренно привлекателен при контроле рисков"
    else:
        verdict = "risky"
        headline = "Проект высокорискованный"

    body = (
        f"Средняя ожидаемая NPV проекта с учётом корреляции рисков составляет "
        f"{fmt_money(mean)} при базовой (детерминированной) NPV {fmt_money(base)}. "
        f"Вероятность отрицательной NPV оценивается в {fmt_pct(p_neg)}. "
        f"С вероятностью 95 % потери не превысят уровня VaR = {fmt_money(var)}, "
        f"а в 5 % худших сценариев средняя NPV (ожидаемые потери, ES) составляет "
        f"{fmt_money(es)}."
    )

    if verdict == "attractive":
        body += (
            " Распределение NPV смещено в положительную область, и даже в "
            "неблагоприятных сценариях проект сохраняет приемлемый профиль риска. "
            "Проект может быть рекомендован к реализации при стандартном уровне "
            "контроля затрат и сроков."
        )
    elif verdict == "moderate":
        body += (
            " Математическое ожидание положительно, однако «левый хвост» "
            "распределения значим. Реализация целесообразна при условии "
            "управления ключевыми факторами риска (контроль сметы, сроков "
            "строительства и темпов продаж) и наличия резерва на покрытие потерь "
            "уровня VaR/ES."
        )
    else:
        body += (
            " Высокая вероятность убытка и значительные потери в неблагоприятных "
            "сценариях указывают на необходимость пересмотра параметров проекта "
            "(цена реализации, структура затрат, график) либо отказа от него в "
            "текущей конфигурации."
        )

    # эффект учёта корреляции
    d_pneg = metrics_corr["prob_neg"] - metrics_ind["prob_neg"]
    d_var = metrics_corr["var"] - metrics_ind["var"]
    if d_pneg > 0.5:
        effect = (
            f"Учёт корреляции рисков увеличивает оценку вероятности убытка на "
            f"{fmt_pct(abs(d_pneg))} (с {fmt_pct(metrics_ind['prob_neg'])} до "
            f"{fmt_pct(metrics_corr['prob_neg'])}) и ухудшает VaR на "
            f"{fmt_money(abs(d_var))}. Модель с независимыми рисками "
            f"систематически недооценивает риск проекта: совместное наступление "
            f"взаимосвязанных неблагоприятных событий (рост затрат, задержки, "
            f"снижение цены) делает «левый хвост» распределения тяжелее."
        )
    elif d_pneg < -0.5:
        effect = (
            f"В данной конфигурации учёт корреляции снижает оценку вероятности "
            f"убытка на {fmt_pct(abs(d_pneg))}: преобладают компенсирующие "
            f"(отрицательные) взаимосвязи между факторами риска."
        )
    else:
        effect = (
            "Учёт корреляции существенно не меняет интегральные показатели риска "
            "в данной конфигурации, однако влияет на форму распределения NPV."
        )

    return {"verdict": verdict, "headline": headline, "body": body, "effect": effect}
