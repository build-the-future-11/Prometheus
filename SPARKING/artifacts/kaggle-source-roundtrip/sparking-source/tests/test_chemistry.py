import numpy as np
import pytest

from sparking.chemistry import FormulaError, canonical_formula, element_fractions, featurize, parse_formula


def test_nested_formula_and_fraction_invariance():
    parsed = parse_formula("Ag(W3Br7)2")
    assert parsed == {"Ag": 1.0, "W": 6.0, "Br": 14.0}
    assert canonical_formula("Li2O") == canonical_formula("Li4O2")
    assert sum(element_fractions("Ag0.5Ge1Pb1.75S4").values()) == pytest.approx(1.0)


def test_formula_parser_rejects_unsupported_or_unbalanced_input():
    with pytest.raises(FormulaError):
        parse_formula("CuSO4·5H2O")
    with pytest.raises(FormulaError):
        parse_formula("Ag(W3Br7")
    with pytest.raises(FormulaError):
        parse_formula("Xx2O")


def test_featurizer_is_fixed_finite_and_scale_invariant():
    matrix = featurize(["Li2O", "Li4O2", "SiO2"])
    assert matrix.shape == (3, 132)
    assert np.all(np.isfinite(matrix))
    np.testing.assert_allclose(matrix[0], matrix[1])

