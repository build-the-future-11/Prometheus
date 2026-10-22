"""Small, dependency-free composition parser and invariant representation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
import re


ELEMENTS = tuple(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
    .split()
)
ATOMIC_NUMBER = {symbol: index + 1 for index, symbol in enumerate(ELEMENTS)}
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")


class FormulaError(ValueError):
    """Raised when a chemical formula cannot be parsed without guessing."""


@dataclass
class _Parser:
    text: str
    pos: int = 0

    def parse(self) -> dict[str, Fraction]:
        if not isinstance(self.text, str) or not self.text:
            raise FormulaError("formula must be a non-empty string")
        values = self._group(stop=None)
        if self.pos != len(self.text):
            raise FormulaError(f"unexpected token at position {self.pos}: {self.text!r}")
        if not values:
            raise FormulaError("formula contains no elements")
        return values

    def _group(self, stop: str | None) -> dict[str, Fraction]:
        result: dict[str, Fraction] = {}
        while self.pos < len(self.text):
            token = self.text[self.pos]
            if token == ")":
                if stop != ")":
                    raise FormulaError(f"unmatched ')' at position {self.pos}")
                break
            if token == "(":
                self.pos += 1
                nested = self._group(stop=")")
                if self.pos >= len(self.text) or self.text[self.pos] != ")":
                    raise FormulaError("unclosed parenthesis")
                self.pos += 1
                multiplier = self._number()
                for element, amount in nested.items():
                    result[element] = result.get(element, Fraction()) + amount * multiplier
                continue
            if not token.isupper():
                raise FormulaError(f"expected element or '(' at position {self.pos}")
            start = self.pos
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].islower():
                self.pos += 1
            element = self.text[start : self.pos]
            if element not in ATOMIC_NUMBER:
                raise FormulaError(f"unknown element {element!r}")
            result[element] = result.get(element, Fraction()) + self._number()
        return result

    def _number(self) -> Fraction:
        match = _NUMBER.match(self.text, self.pos)
        if match is None:
            return Fraction(1)
        self.pos = match.end()
        value = Fraction(match.group(0))
        if value <= 0:
            raise FormulaError("stoichiometric amounts must be positive")
        return value


def parse_formula(formula: str) -> dict[str, Fraction]:
    """Parse elements, parentheses, integer counts, and finite decimal counts."""

    return _Parser(formula).parse()


def canonical_composition(formula: str) -> str:
    """Return a scale- and order-invariant integer-ratio composition key."""

    composition = parse_formula(formula)
    denominator = reduce(lcm, (value.denominator for value in composition.values()), 1)
    integers = [int(value * denominator) for value in composition.values()]
    divisor = reduce(gcd, integers)
    normalized = {
        element: int(composition[element] * denominator) // divisor
        for element in composition
    }
    return "|".join(
        f"{element}:{normalized[element]}"
        for element in sorted(normalized, key=ATOMIC_NUMBER.__getitem__)
    )


def element_fractions(formula: str) -> dict[str, float]:
    """Return atom fractions; representation is invariant to formula scaling."""

    composition = parse_formula(formula)
    total = sum(composition.values(), Fraction())
    return {element: float(amount / total) for element, amount in composition.items()}


def convert_band_gap(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between the only energy units admitted by this project."""

    if from_unit == to_unit:
        return float(value)
    if (from_unit, to_unit) == ("eV", "meV"):
        return float(value) * 1000.0
    if (from_unit, to_unit) == ("meV", "eV"):
        return float(value) / 1000.0
    raise ValueError(f"unsupported band-gap conversion: {from_unit!r} -> {to_unit!r}")

