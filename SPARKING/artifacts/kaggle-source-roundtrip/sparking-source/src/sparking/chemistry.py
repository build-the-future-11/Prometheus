"""Dependency-light composition parsing and fixed-schema descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
import re

import numpy as np


ELEMENTS = tuple(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc "
    "Lv Ts Og".split()
)
ATOMIC_NUMBER = {element: index + 1 for index, element in enumerate(ELEMENTS)}


def _period_group() -> tuple[dict[str, int], dict[str, int]]:
    rows: list[list[tuple[str, int]]] = [
        [("H", 1), ("He", 18)],
        [("Li", 1), ("Be", 2), ("B", 13), ("C", 14), ("N", 15), ("O", 16), ("F", 17), ("Ne", 18)],
        [("Na", 1), ("Mg", 2), ("Al", 13), ("Si", 14), ("P", 15), ("S", 16), ("Cl", 17), ("Ar", 18)],
        [("K", 1), ("Ca", 2), ("Sc", 3), ("Ti", 4), ("V", 5), ("Cr", 6), ("Mn", 7), ("Fe", 8), ("Co", 9), ("Ni", 10), ("Cu", 11), ("Zn", 12), ("Ga", 13), ("Ge", 14), ("As", 15), ("Se", 16), ("Br", 17), ("Kr", 18)],
        [("Rb", 1), ("Sr", 2), ("Y", 3), ("Zr", 4), ("Nb", 5), ("Mo", 6), ("Tc", 7), ("Ru", 8), ("Rh", 9), ("Pd", 10), ("Ag", 11), ("Cd", 12), ("In", 13), ("Sn", 14), ("Sb", 15), ("Te", 16), ("I", 17), ("Xe", 18)],
        [("Cs", 1), ("Ba", 2)] + [(e, 3) for e in ELEMENTS[56:71]] + [("Hf", 4), ("Ta", 5), ("W", 6), ("Re", 7), ("Os", 8), ("Ir", 9), ("Pt", 10), ("Au", 11), ("Hg", 12), ("Tl", 13), ("Pb", 14), ("Bi", 15), ("Po", 16), ("At", 17), ("Rn", 18)],
        [("Fr", 1), ("Ra", 2)] + [(e, 3) for e in ELEMENTS[88:103]] + [("Rf", 4), ("Db", 5), ("Sg", 6), ("Bh", 7), ("Hs", 8), ("Mt", 9), ("Ds", 10), ("Rg", 11), ("Cn", 12), ("Nh", 13), ("Fl", 14), ("Mc", 15), ("Lv", 16), ("Ts", 17), ("Og", 18)],
    ]
    periods: dict[str, int] = {}
    groups: dict[str, int] = {}
    for period, row in enumerate(rows, 1):
        for element, group in row:
            periods[element] = period
            groups[element] = group
    if set(periods) != set(ELEMENTS):
        raise RuntimeError("periodic table descriptor is incomplete")
    return periods, groups


PERIOD, GROUP = _period_group()
_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\)|(?:\d+(?:\.\d*)?|\.\d+))")


class FormulaError(ValueError):
    """Raised when a formula cannot be parsed without guessing."""


@dataclass
class _Parser:
    tokens: list[str]
    index: int = 0

    def parse(self, stop_at_close: bool = False) -> dict[str, float]:
        result: dict[str, float] = {}
        while self.index < len(self.tokens):
            token = self.tokens[self.index]
            if token == ")":
                if not stop_at_close:
                    raise FormulaError("unexpected closing parenthesis")
                self.index += 1
                return result
            if token == "(":
                self.index += 1
                group = self.parse(stop_at_close=True)
                multiplier = self._number()
                for element, count in group.items():
                    result[element] = result.get(element, 0.0) + count * multiplier
                continue
            if token not in ATOMIC_NUMBER:
                raise FormulaError(f"unexpected token {token!r}")
            self.index += 1
            result[token] = result.get(token, 0.0) + self._number()
        if stop_at_close:
            raise FormulaError("unclosed parenthesis")
        return result

    def _number(self) -> float:
        if self.index >= len(self.tokens):
            return 1.0
        token = self.tokens[self.index]
        if token in ATOMIC_NUMBER or token in {"(", ")"}:
            return 1.0
        self.index += 1
        value = float(token)
        if not np.isfinite(value) or value <= 0:
            raise FormulaError("stoichiometric multipliers must be positive and finite")
        return value


def parse_formula(formula: str) -> dict[str, float]:
    compact = formula.replace(" ", "")
    tokens = _TOKEN.findall(compact)
    if not compact or "".join(tokens) != compact:
        raise FormulaError(f"unsupported formula syntax: {formula!r}")
    parsed = _Parser(tokens).parse()
    if not parsed:
        raise FormulaError("formula contains no elements")
    return parsed


def element_fractions(formula: str) -> dict[str, float]:
    counts = parse_formula(formula)
    total = sum(counts.values())
    return {element: value / total for element, value in counts.items()}


def canonical_formula(formula: str) -> str:
    fractions = element_fractions(formula)
    return "|".join(
        f"{element}:{fractions[element]:.12g}"
        for element in sorted(fractions, key=ATOMIC_NUMBER.__getitem__)
    )


GLOBAL_FEATURE_NAMES = (
    "n_elements",
    "fraction_entropy",
    "fraction_min",
    "fraction_max",
    "fraction_l2",
    "z_mean",
    "z_std",
    "z_min",
    "z_max",
    "period_mean",
    "period_std",
    "group_mean",
    "group_std",
    "group_range",
)


def featurize(formulas: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return deterministic elemental fractions plus descriptor moments."""

    matrix = np.zeros((len(formulas), len(ELEMENTS) + len(GLOBAL_FEATURE_NAMES)), dtype=np.float64)
    for row, formula in enumerate(formulas):
        fractions = element_fractions(formula)
        for element, fraction in fractions.items():
            matrix[row, ATOMIC_NUMBER[element] - 1] = fraction
        elements = tuple(fractions)
        weights = np.asarray([fractions[e] for e in elements], dtype=float)
        z = np.asarray([ATOMIC_NUMBER[e] for e in elements], dtype=float)
        period = np.asarray([PERIOD[e] for e in elements], dtype=float)
        group = np.asarray([GROUP[e] for e in elements], dtype=float)

        def moments(values: np.ndarray) -> tuple[float, float]:
            mean = float(np.sum(weights * values))
            std = float(np.sqrt(np.sum(weights * (values - mean) ** 2)))
            return mean, std

        z_mean, z_std = moments(z)
        period_mean, period_std = moments(period)
        group_mean, group_std = moments(group)
        global_values = (
            float(len(elements)),
            float(-sum(w * log(w) for w in weights)),
            float(weights.min()),
            float(weights.max()),
            float(np.sqrt(np.sum(weights**2))),
            z_mean,
            z_std,
            float(z.min()),
            float(z.max()),
            period_mean,
            period_std,
            group_mean,
            group_std,
            float(group.max() - group.min()),
        )
        matrix[row, len(ELEMENTS) :] = global_values
    if not np.all(np.isfinite(matrix)):
        raise ValueError("featurizer produced a non-finite value")
    return matrix


def feature_names() -> tuple[str, ...]:
    return tuple(f"fraction_{element}" for element in ELEMENTS) + GLOBAL_FEATURE_NAMES

