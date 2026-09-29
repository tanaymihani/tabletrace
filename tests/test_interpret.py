"""Table interpretation on small synthetic grids that mimic real layouts in the corpus."""

import pytest

from tabletrace.grid import Cell, TableGrid, Word
from tabletrace.interpret import interpret, repair_alloy_name
from tabletrace.schema import PropertyKind as P


def grid(rows: list[list[str]], caption: str = "", source: str = "text_layer") -> TableGrid:
    cells = []
    for r, row in enumerate(rows):
        out = []
        for c, text in enumerate(row):
            words = [Word(t, c * 50 + 1, r * 10 + 1, c * 50 + 9, r * 10 + 8) for t in text.split()] if text else []
            out.append(Cell(r, c, (c * 50, r * 10, c * 50 + 50, r * 10 + 10), words))
        cells.append(out)
    return TableGrid("p", 1, (0, 0, 50 * len(rows[0]), 10 * len(rows)), cells, source, "test", caption, 3)


def values(g: TableGrid):
    return {(v.material, v.property.value, v.test_temperature_c): v.value for v in interpret(g, "p", "test").values}


def test_normal_orientation_with_units_and_missing_cells():
    g = grid([["Alloy", "Rp0.2 [MPa]", "Rm [MPa]", "A [%]", "E [GPa]", "HV30"],
              ["NbTaTi", "620", "683", "18.5", "143", "246"],
              ["TaTiZr", "-", "284", "0", "157", "485"]])
    v = values(g)
    assert v[("NbTaTi", "yield_strength", None)] == 620
    assert v[("TaTiZr", "elongation", None)] == 0
    assert ("TaTiZr", "yield_strength", None) not in v  # "-" is missing, not zero
    assert not any(k[1] == "youngs_modulus" for k in v)  # E is not a target property


def test_transposed_table_with_calc_exp_pairs():
    g = grid([["Metal", "Mo", "MoNbTaV (Calc/Exp)"],
              ["a (Å)", "3.1468", "3.2036/3.208"],
              ["Hv (MPa)", "1530", "1097/4947"],
              ["σ0.2 (MPa)", "438", "333/1525"]])
    v = values(g)
    assert v[("MoNbTaV", "yield_strength", None)] == 1525  # experimental half of the pair
    assert v[("MoNbTaV", "vickers_hardness", None)] == pytest.approx(504.45, abs=0.01)  # MPa -> HV
    assert v[("Mo", "vickers_hardness", None)] == pytest.approx(156.02, abs=0.01)
    assert len(v) == 4  # the lattice parameter row is ignored


def test_two_level_header_with_temperatures_as_rows():
    g = grid([["", "UTS in MPa", "", "εf in %", ""],
              ["", "CC", "DS", "CC", "DS"],
              ["~23 °C", "696 ± 33", "1197 ± 6", "49 ± 0", "27 ± 1"],
              ["600 °C", "561 ± 50", "1006 ± 16", "11 ± 8", "12 ± 2"]], caption="Ultimate tensile strength and strain to failure")
    res = interpret(g, "p", "test")
    v = {(x.material, x.property.value, x.test_temperature_c): (x.value, x.uncertainty) for x in res.values}
    assert v[("DS", "ultimate_strength", 600.0)] == (1006, 16)
    assert v[("CC", "elongation", 23.0)] == (49, 0)
    assert all(x.loading == "tension" for x in res.values)


def test_property_named_only_in_the_caption():
    g = grid([["Temperature (°C)", "HfMoNbTaTiZr", "HfMoNbTaTi"],
              ["800", "1007", "822"],
              ["1400", "N. A.", "367"]],
             caption="Table 4. The elevated temperature compressive yield strength (MPa) of the alloy series.")
    res = interpret(g, "p", "test")
    v = {(x.material, x.test_temperature_c): x.value for x in res.values}
    assert v == {("HfMoNbTaTiZr", 800.0): 1007, ("HfMoNbTaTi", 800.0): 822, ("HfMoNbTaTi", 1400.0): 367}
    assert all(x.loading == "compression" and x.property is P.yield_strength for x in res.values)


def test_predicted_columns_and_summary_rows_are_skipped():
    g = grid([["Alloy", "Hardness (HV)", "Hardness (HV) predicted"],
              ["A", "1084 ± 37", "825"],
              ["Averages:", "900", "800"]])
    v = values(g)
    assert v == {("A", "vickers_hardness", None): 1084}


def test_dual_unit_columns_keep_one_value():
    g = grid([["Nominal composition", "UTS MN/m2", "UTS ksi", "Elongation, percent"],
              ["Ti-25Nb", "820", "119", "8"],
              ["Ti-50Nb", "515", "75", "26"],
              ["Ti-25V", "695", "101", "13"]])
    res = interpret(g, "p", "test")
    uts = [x for x in res.values if x.property is P.ultimate_strength]
    assert sorted(x.value for x in uts) == [515, 695, 820]  # ksi twins are not extra records
    assert all(x.features.get("unit_agreement") == 1.0 for x in uts)


def test_unit_recovered_from_value_ratio_when_ocr_lost_it():
    g = grid([["Alloy", "Ultimate tensile strength", "Ultimate tensile strength"],
              ["Ti-25Nb", "820", "119"], ["Ti-50Nb", "515", "75"], ["Ti-25V", "695", "101"]], source="ocr")
    uts = sorted(x.value for x in interpret(g, "p", "test").values)
    assert uts == [515, 695, 820]


def test_quarantine_reasons():
    g = grid([["Alloy", "HV", "Elongation (%)", "Yield strength"],
              ["A", "abc", "140", "500"]])
    res = interpret(g, "p", "test")
    reasons = sorted(q.reason for q in res.quarantine)
    assert reasons == ["bad unit", "non-numeric value", "value outside physical range"]


def test_alloy_name_repair():
    assert repair_alloy_name("Ti-SONb") == "Ti-50Nb"
    assert repair_alloy_name("Ti-50V-l.5Si") == "Ti-50V-1.5Si"
    assert repair_alloy_name("Al0.5CoCrFeNi") == "Al0.5CoCrFeNi"
