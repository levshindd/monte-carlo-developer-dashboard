# -*- coding: utf-8 -*-
"""
model.py — расчётное ядро имитационной модели Монте-Карло для оценки
девелоперского проекта.

Модуль не зависит от Streamlit и может использоваться отдельно (в скриптах,
ноутбуках, тестах). Вся «математика» проекта сосредоточена здесь:

    * calculate_base_npv      — детерминированная (базовая) NPV;
    * generate_correlated_risks — генерация факторов риска (гауссова копула);
    * calculate_npv           — векторизованный расчёт NPV по факторам риска;
    * run_monte_carlo         — полная симуляция (независимая и коррелированная);
    * calculate_risk_metrics  — показатели риска (VaR, ES, P(NPV<0) и др.);
    * calculate_sensitivity   — чувствительность NPV к факторам риска.

Для функции нормального распределения используется SciPy, если он доступен;
при его отсутствии применяется точная численная аппроксимация на NumPy,
поэтому модуль работоспособен даже без установленного SciPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Функции стандартного нормального распределения (SciPy с NumPy-fallback)
# ---------------------------------------------------------------------------
try:  # предпочтительно — SciPy (требование стека проекта)
    from scipy.stats import norm as _scipy_norm

    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Функция распределения N(0, 1)."""
        return _scipy_norm.cdf(x)

    def _norm_ppf(p: np.ndarray) -> np.ndarray:
        """Обратная функция распределения (квантиль) N(0, 1)."""
        return _scipy_norm.ppf(p)

    HAS_SCIPY = True
except Exception:  # pragma: no cover - используется только без SciPy
    HAS_SCIPY = False

    _vec_erf = np.vectorize(math.erf)

    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Функция распределения N(0, 1) через math.erf (без SciPy)."""
        x = np.asarray(x, dtype=float)
        return 0.5 * (1.0 + _vec_erf(x / math.sqrt(2.0)))

    def _norm_ppf(p: np.ndarray) -> np.ndarray:
        """
        Квантиль N(0, 1) — рациональная аппроксимация Акклама (Acklam),
        точность ~1e-9 на интервале (0, 1). Используется как замена scipy.
        """
        p = np.asarray(p, dtype=float)
        # коэффициенты аппроксимации
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        out = np.zeros_like(p)
        # хвосты
        lo = p < plow
        hi = p > phigh
        mid = ~(lo | hi)
        # нижний хвост
        if np.any(lo):
            q = np.sqrt(-2 * np.log(p[lo]))
            out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        # верхний хвост
        if np.any(hi):
            q = np.sqrt(-2 * np.log(1 - p[hi]))
            out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                       ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        # центральная область
        if np.any(mid):
            q = p[mid] - 0.5
            r = q * q
            out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                       (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        return out


# ---------------------------------------------------------------------------
# Параметры проекта и факторов риска
# ---------------------------------------------------------------------------
RISK_LABELS = [
    "Строительные затраты",
    "Цена продажи",
    "Задержка строительства",
    "Регуляторные расходы",
    "Темп продаж",
]

# базовая корреляционная матрица (порядок факторов — как в RISK_LABELS)
DEFAULT_CORRELATION = np.array([
    [1.00, 0.30, 0.50, 0.25, -0.15],
    [0.30, 1.00, -0.20, -0.20, 0.55],
    [0.50, -0.20, 1.00, 0.45, -0.30],
    [0.25, -0.20, 0.45, 1.00, -0.20],
    [-0.15, 0.55, -0.30, -0.20, 1.00],
])


@dataclass
class ProjectParams:
    """Базовые (детерминированные) параметры девелоперского проекта."""

    initial_investment: float = 500.0   # I0, млн руб.
    base_revenue: float = 1850.0         # базовая выручка (GDV), млн руб.
    base_construction: float = 550.0     # базовые строительные затраты, млн руб.
    opex: float = 50.0                   # операционные/админ. расходы, млн руб.
    discount_rate: float = 0.16          # ставка дисконтирования (доли)
    horizon: int = 4                     # горизонт проекта, лет
    # профили распределения потоков по годам (для горизонта 4)
    construction_profile: Tuple[float, ...] = (0.40, 0.40, 0.20, 0.00)
    revenue_profile: Tuple[float, ...] = (0.00, 0.30, 0.40, 0.30)
    opex_profile: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    delay_carry_per_year: float = 40.0   # доп. расходы на содержание за год задержки


@dataclass
class RiskParams:
    """Параметры предельных распределений факторов риска."""

    # рост строительных затрат (доли): треугольное (min, mode, max)
    cost_growth: Tuple[float, float, float] = (-0.05, 0.10, 0.30)
    # изменение цены продажи (доли): треугольное
    price_change: Tuple[float, float, float] = (-0.20, 0.05, 0.20)
    # задержка строительства (мес.): дискретное — значения и вероятности
    delay_values: Tuple[float, ...] = (0.0, 6.0, 12.0)
    delay_probs: Tuple[float, ...] = (0.50, 0.35, 0.15)
    # дополнительные регуляторные расходы (млн руб.): треугольное
    reg_cost: Tuple[float, float, float] = (0.0, 20.0, 80.0)
    # темп продаж / absorption (индекс смещения выручки): треугольное
    absorption: Tuple[float, float, float] = (-1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Вспомогательные обратные функции распределений (inverse-CDF / PPF)
# ---------------------------------------------------------------------------
def _triangular_ppf(u: np.ndarray, a: float, c: float, b: float) -> np.ndarray:
    """
    Обратная функция треугольного распределения Tri(a=min, c=mode, b=max).

    Параметры
    ---------
    u : np.ndarray
        Значения квантилей в (0, 1).
    a, c, b : float
        Минимум, мода и максимум распределения.
    """
    if not (a <= c <= b) or a == b:
        # вырожденный случай — возвращаем моду
        return np.full_like(np.asarray(u, dtype=float), c)
    u = np.asarray(u, dtype=float)
    fc = (c - a) / (b - a)
    left = a + np.sqrt(u * (b - a) * (c - a))
    right = b - np.sqrt((1.0 - u) * (b - a) * (b - c))
    return np.where(u < fc, left, right)


def _discrete_ppf(u: np.ndarray, values: Tuple[float, ...],
                  probs: Tuple[float, ...]) -> np.ndarray:
    """
    Обратная функция дискретного распределения (для задержки строительства).

    u : квантили в (0, 1); values : исходы; probs : их вероятности.
    """
    u = np.asarray(u, dtype=float)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()  # нормировка на случай неточного ввода
    edges = np.cumsum(probs)
    out = np.full_like(u, values[-1])
    # назначаем исход по интервалам кумулятивной вероятности
    assigned = np.zeros_like(u, dtype=bool)
    for val, edge in zip(values, edges):
        mask = (~assigned) & (u <= edge)
        out[mask] = val
        assigned |= mask
    return out


# ---------------------------------------------------------------------------
# Базовая (детерминированная) NPV
# ---------------------------------------------------------------------------
def calculate_base_npv(params: ProjectParams) -> float:
    """
    Рассчитать детерминированную NPV проекта при «наиболее вероятных»
    (нейтральных) значениях факторов риска.

    Возвращает NPV в млн руб.
    """
    base = calculate_npv(
        cost_growth=np.array([0.0]),
        price_change=np.array([0.0]),
        delay_months=np.array([0.0]),
        reg_cost=np.array([0.0]),
        absorption=np.array([0.0]),
        params=params,
    )
    return float(base[0])


# ---------------------------------------------------------------------------
# Векторизованный расчёт NPV
# ---------------------------------------------------------------------------
def calculate_npv(cost_growth: np.ndarray, price_change: np.ndarray,
                  delay_months: np.ndarray, reg_cost: np.ndarray,
                  absorption: np.ndarray, params: ProjectParams) -> np.ndarray:
    """
    Векторизованный расчёт NPV для массива реализаций факторов риска.

    Логика денежных потоков:
        * фактор цены масштабирует выручку: REV = base_revenue * (1 + price_change);
        * фактор cost overrun масштабирует строительные затраты;
        * регуляторные расходы добавляются к затратам (в 1-й год);
        * задержка сдвигает поступления выручки на более поздний период
          (ухудшая дисконтирование) и добавляет расходы на содержание;
        * темп продаж (absorption) перераспределяет выручку между годами.

    Все входные массивы должны быть одинаковой длины N. Возвращает массив NPV.
    """
    r = params.discount_rate
    T = params.horizon
    cons_frac = _profile(params.construction_profile, T, front=True)
    rev_frac = _profile(params.revenue_profile, T, front=False)
    opex_frac = _profile(params.opex_profile, T, front=None)

    cost_growth = np.asarray(cost_growth, dtype=float)
    price_change = np.asarray(price_change, dtype=float)
    delay_months = np.asarray(delay_months, dtype=float)
    reg_cost = np.asarray(reg_cost, dtype=float)
    absorption = np.asarray(absorption, dtype=float)

    delay_yr = delay_months / 12.0
    rev_total = params.base_revenue * (1.0 + price_change)
    cons_total = params.base_construction * (1.0 + cost_growth)

    # перераспределение выручки между ранним (2-й год) и поздним (T-й год)
    # периодами в зависимости от темпа продаж
    rev_frac_mat = np.tile(rev_frac, (len(cost_growth), 1)).astype(float)
    if T >= 4:
        shift = 0.12 * absorption
        rev_frac_mat[:, 1] = rev_frac[1] + shift          # 2-й год
        rev_frac_mat[:, T - 1] = rev_frac[T - 1] - shift  # последний год
    # нормировка (на случай нестандартных профилей)
    row_sums = rev_frac_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    rev_frac_mat = rev_frac_mat / row_sums

    npv = np.full(len(cost_growth), -params.initial_investment, dtype=float)
    for t in range(1, T + 1):
        idx = t - 1
        rev_part = rev_total * rev_frac_mat[:, idx]
        cost_part = cons_total * cons_frac[idx] + params.opex * opex_frac[idx]
        # выручка дисконтируется с учётом задержки, затраты — по своему графику
        npv += rev_part / (1.0 + r) ** (t + delay_yr)
        npv += (-cost_part) / (1.0 + r) ** t

    # регуляторные расходы и расходы на содержание из-за задержки (1-й год)
    extra = reg_cost + params.delay_carry_per_year * delay_yr
    npv += (-extra) / (1.0 + r) ** 1
    return npv


def _profile(profile: Tuple[float, ...], T: int, front) -> np.ndarray:
    """
    Привести профиль распределения по годам к длине T.

    Если длина профиля совпадает с T — используется как есть. Иначе профиль
    перестраивается равномерно: строительство (front=True) — на годы 1..T-1,
    выручка (front=False) — на годы 2..T, прочее (front=None) — равномерно.
    """
    profile = np.asarray(profile, dtype=float)
    if len(profile) == T:
        return profile
    arr = np.zeros(T, dtype=float)
    if front is True:
        k = max(1, T - 1)
        arr[:k] = 1.0 / k
    elif front is False:
        arr[1:] = 1.0 / max(1, T - 1)
    else:
        arr[:] = 1.0 / T
    return arr


# ---------------------------------------------------------------------------
# Генерация факторов риска (гауссова копула + Холецкий)
# ---------------------------------------------------------------------------
def generate_correlated_risks(corr_matrix: np.ndarray, risk: RiskParams,
                              n: int, seed: int,
                              use_correlation: bool = True) -> Dict[str, np.ndarray]:
    """
    Сгенерировать N реализаций пяти факторов риска.

    Алгоритм (гауссова копула):
        1. сгенерировать независимые стандартные нормальные Z ~ N(0, I);
        2. при use_correlation=True применить разложение Холецкого
           corr = L·Lᵀ и получить Z* = Z·Lᵀ (коррелированные нормальные);
        3. перевести в равномерные U = Φ(Z*);
        4. через обратные функции предельных распределений получить
           значения факторов риска.

    Возвращает словарь массивов длины N по ключам:
        'cost', 'price', 'delay', 'reg', 'absorption', а также матрицу
        'matrix' (N x 5) в порядке RISK_LABELS.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 5))

    if use_correlation:
        # матрица должна быть положительно определённой; вызывающий код
        # обязан передать корректную (см. utils.validate_correlation_matrix)
        L = np.linalg.cholesky(corr_matrix)
        z = z @ L.T

    u = _norm_cdf(z)
    # числовая стабилизация краёв (чтобы PPF не уходил в бесконечность)
    eps = 1e-10
    u = np.clip(u, eps, 1 - eps)

    cost = _triangular_ppf(u[:, 0], *risk.cost_growth)
    price = _triangular_ppf(u[:, 1], *risk.price_change)
    delay = _discrete_ppf(u[:, 2], risk.delay_values, risk.delay_probs)
    reg = _triangular_ppf(u[:, 3], *risk.reg_cost)
    absorption = _triangular_ppf(u[:, 4], *risk.absorption)

    matrix = np.column_stack([cost, price, delay, reg, absorption])
    return {"cost": cost, "price": price, "delay": delay, "reg": reg,
            "absorption": absorption, "matrix": matrix}


# ---------------------------------------------------------------------------
# Полная симуляция Монте-Карло
# ---------------------------------------------------------------------------
def run_monte_carlo(params: ProjectParams, risk: RiskParams,
                    corr_matrix: np.ndarray, n: int = 10000,
                    seed: int = 42, use_correlation: bool = True) -> Dict:
    """
    Выполнить симуляцию Монте-Карло для одной конфигурации рисков.

    Возвращает словарь:
        'npv'    — массив NPV длины N;
        'factors'— матрица факторов риска (N x 5);
        'base_npv' — детерминированная NPV.
    """
    n = int(max(100, n))
    risks = generate_correlated_risks(corr_matrix, risk, n, seed, use_correlation)
    npv = calculate_npv(risks["cost"], risks["price"], risks["delay"],
                        risks["reg"], risks["absorption"], params)
    return {"npv": npv, "factors": risks["matrix"],
            "base_npv": calculate_base_npv(params)}


def run_both_models(params: ProjectParams, risk: RiskParams,
                    corr_matrix: np.ndarray, n: int = 10000,
                    seed: int = 42) -> Dict:
    """
    Рассчитать обе модели: с независимыми и с коррелированными рисками.

    Для независимой модели используется отдельный seed (seed + 1), чтобы
    выборки не совпадали тождественно. Возвращает словарь с результатами
    обеих моделей и их показателями риска.
    """
    base_npv = calculate_base_npv(params)
    corr = run_monte_carlo(params, risk, corr_matrix, n, seed, use_correlation=True)
    indep = run_monte_carlo(params, risk, corr_matrix, n, seed + 1, use_correlation=False)

    return {
        "base_npv": base_npv,
        "correlated": {
            "npv": corr["npv"],
            "factors": corr["factors"],
            "metrics": calculate_risk_metrics(corr["npv"], base_npv),
            "sensitivity": calculate_sensitivity(corr["factors"], corr["npv"]),
        },
        "independent": {
            "npv": indep["npv"],
            "factors": indep["factors"],
            "metrics": calculate_risk_metrics(indep["npv"], base_npv),
            "sensitivity": calculate_sensitivity(indep["factors"], indep["npv"]),
        },
    }


# ---------------------------------------------------------------------------
# Показатели риска
# ---------------------------------------------------------------------------
def calculate_risk_metrics(npv: np.ndarray, base_npv: float,
                           alpha: float = 0.05) -> Dict[str, float]:
    """
    Рассчитать сводные показатели риска по массиву NPV.

    Возвращает словарь:
        mean, median, std, min, max, p5, p95, prob_neg (в %),
        var (VaR на уровне alpha, в терминах NPV),
        es  (Expected Shortfall — средняя NPV худших alpha сценариев),
        base (базовая NPV).
    """
    npv = np.asarray(npv, dtype=float)
    npv_sorted = np.sort(npv)
    n = len(npv_sorted)
    p5 = float(np.percentile(npv, alpha * 100))
    p95 = float(np.percentile(npv, (1 - alpha) * 100))
    k = max(1, int(alpha * n))
    es = float(npv_sorted[:k].mean())
    return {
        "mean": float(np.mean(npv)),
        "median": float(np.median(npv)),
        "std": float(np.std(npv, ddof=1)),
        "min": float(np.min(npv)),
        "max": float(np.max(npv)),
        "p5": p5,
        "p95": p95,
        "prob_neg": float(np.mean(npv < 0) * 100.0),
        "var": p5,
        "es": es,
        "base": float(base_npv),
    }


# ---------------------------------------------------------------------------
# Чувствительность NPV к факторам риска
# ---------------------------------------------------------------------------
def calculate_sensitivity(factors: np.ndarray, npv: np.ndarray) -> List[Dict]:
    """
    Оценить чувствительность NPV к каждому фактору риска через коэффициент
    корреляции Пирсона между значением фактора и итоговой NPV.

    Возвращает список словарей (по убыванию |корреляции|):
        {'factor': имя, 'corr': коэффициент, 'direction': '+'/'−'}.
    """
    factors = np.asarray(factors, dtype=float)
    npv = np.asarray(npv, dtype=float)
    result = []
    for j, label in enumerate(RISK_LABELS):
        col = factors[:, j]
        if np.std(col) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col, npv)[0, 1])
        result.append({
            "factor": label,
            "corr": corr,
            "direction": "Прямое (+)" if corr >= 0 else "Обратное (−)",
        })
    result.sort(key=lambda d: abs(d["corr"]), reverse=True)
    return result


# ---------------------------------------------------------------------------
# Демонстрационные сценарии (для таблицы в интерфейсе)
# ---------------------------------------------------------------------------
def scenario_npv(params: ProjectParams, cost_growth: float, price_change: float,
                 delay_months: float, reg_cost: float, absorption: float) -> float:
    """Рассчитать NPV для одного заданного сценария (скалярные значения)."""
    val = calculate_npv(
        np.array([cost_growth]), np.array([price_change]),
        np.array([delay_months]), np.array([reg_cost]),
        np.array([absorption]), params,
    )
    return float(val[0])


if __name__ == "__main__":
    # быстрый самотест: воспроизводит результаты курсовой работы
    p = ProjectParams()
    rk = RiskParams()
    print("SciPy доступен:", HAS_SCIPY)
    print("Базовая NPV:", round(calculate_base_npv(p), 2))
    res = run_both_models(p, rk, DEFAULT_CORRELATION, n=10000, seed=20260529)
    mc = res["correlated"]["metrics"]
    mi = res["independent"]["metrics"]
    print("Коррелир.: mean=%.1f  P(neg)=%.1f%%  VaR=%.1f  ES=%.1f" %
          (mc["mean"], mc["prob_neg"], mc["var"], mc["es"]))
    print("Независ. : mean=%.1f  P(neg)=%.1f%%  VaR=%.1f  ES=%.1f" %
          (mi["mean"], mi["prob_neg"], mi["var"], mi["es"]))
    for s in res["correlated"]["sensitivity"]:
        print("  ", s["factor"], round(s["corr"], 3))
