from fractions import Fraction

import pytest

from ml4materials.chemistry import (
    FormulaError,
    canonical_composition,
    convert_band_gap,
    element_fractions,
    parse_formula,
)


def test_parentheses_and_decimal_formula_parsing() -> None:
    assert parse_formula("Ag(AuS)2") == {
        "Ag": Fraction(1),
        "Au": Fraction(2),
        "S": Fraction(2),
    }
    assert parse_formula("Ag0.5Ge1Pb1.75S4")["Pb"] == Fraction(7, 4)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Fe2O3", "Fe4O6"),
        ("Ag(AuS)2", "AgAu2S2"),
        ("Te0.5Se0.5", "TeSe"),
        ("In1.5Cu0.5Se2.5", "In3CuSe5"),
    ],
)
def test_canonical_composition_is_scale_order_and_parenthesis_invariant(
    left: str, right: str
) -> None:
    assert canonical_composition(left) == canonical_composition(right)
    assert element_fractions(left) == element_fractions(right)


@pytest.mark.parametrize("formula", ["", "Xx2O", "Fe(OH", "Fe)O(", "2Fe", "Fe0O"])
def test_malformed_formula_fails_closed(formula: str) -> None:
    with pytest.raises(FormulaError):
        parse_formula(formula)


def test_unit_conversion_round_trip() -> None:
    value_ev = 1.234
    assert convert_band_gap(convert_band_gap(value_ev, "eV", "meV"), "meV", "eV") == pytest.approx(value_ev)
    with pytest.raises(ValueError):
        convert_band_gap(1.0, "eV", "J")

