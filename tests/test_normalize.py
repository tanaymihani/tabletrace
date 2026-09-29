import pytest

from tabletrace.normalize import (
    UnitError,
    clean_text,
    detect_decimal_comma,
    find_unit,
    parse_number,
    repair_ocr_cell,
    split_pair,
    temperature_c,
    to_canonical,
)
from tabletrace.schema import PropertyKind as P


@pytest.mark.parametrize(
    "text, value",
    [
        ("1,234", 1234.0),
        ("1,234.5", 1234.5),
        ("1 234,5", 1234.5),
        ("1.234,5", 1234.5),
        ("0,33", 0.33),
        ("1.2e3", 1200.0),
        ("1.2 × 10³", 1200.0),
        ("−1.26", -1.26),
        ("–423", -423.0),
        ("Hv1,049", 1049.0),
    ],
)
def test_plain_numbers(text, value):
    assert parse_number(text).value == pytest.approx(value)


def test_uncertainty():
    p = parse_number("412 ± 8")
    assert (p.value, p.uncertainty) == (412.0, 8.0)
    assert parse_number("412±8").uncertainty == 8.0
    assert parse_number("412 +/- 8").uncertainty == 8.0
    assert parse_number("534 ± 04").uncertainty == 4.0  # leading zero as printed in Sci. Rep. tables


def test_range():
    p = parse_number("5–50")
    assert p.qualifier == "range" and (p.low, p.high) == (5.0, 50.0) and p.value == 27.5
    assert parse_number("-423").qualifier == "="  # a negative number is not a range


def test_qualifiers_and_footnotes():
    assert parse_number("> 50").qualifier == ">"
    assert parse_number("~23").qualifier == "~"
    for text, value in [("9.1*", 9.1), ("199.0†", 199.0), ("*1190", 1190.0), ("929 [26]", 929.0)]:
        p = parse_number(text)
        assert p.value == value and p.footnote


def test_units_inside_cells():
    assert parse_number("Hv571").unit == "HV"
    assert parse_number("235 HV0.1").unit == "HV"
    assert parse_number("25%").unit == "%"


def test_missing_and_text_cells():
    for text in ["-", "---", "N. A.", "NA", "", "Ti-50V", "abc"]:
        assert parse_number(text) is None


def test_ocr_digit_repair_is_flagged():
    p = parse_number("6OO")
    assert p.value == 600.0 and p.repaired
    assert parse_number("O.3").value == pytest.approx(0.3)


def test_ocr_cell_repair():
    assert repair_ocr_cell("| 422+75") == "422 ± 75"
    assert repair_ocr_cell("575 £7") == "575 ± 7"
    assert repair_ocr_cell("Ti-5Al-2.5Sn") == "Ti-5Al-2.5Sn"


def test_decimal_convention_detection():
    assert detect_decimal_comma(["0,33", "12,5", "1155"]) is True
    assert detect_decimal_comma(["1,200", "25.62"]) is False
    assert detect_decimal_comma(["1,200", "1,250"]) is None  # ambiguous: read as thousands
    assert parse_number("1,200", decimal_comma=True).value == pytest.approx(1.2)


def test_calc_exp_pairs():
    assert split_pair("1097/4947") == ["1097", "4947"]
    assert split_pair("2528/–") == ["2528", "-"]
    assert split_pair("σmax/ρ") is None


def test_find_unit_prefers_brackets():
    assert find_unit("Hv (MPa)") == "MPa"  # hardness printed as a pressure
    assert find_unit("Rp0.2 [MPa]") == "MPa"
    assert find_unit("UTS in MPa") == "MPa"
    assert find_unit("HV30") == "HV"
    assert find_unit("Elongation, percent") == "%"
    assert find_unit("Ultimate tensile strength MN/m²") == "MN/m**2"
    assert find_unit("ρ (g/cm3)") is None


def test_unit_conversion():
    assert to_canonical(107.5, "ksi", P.ultimate_strength)[0] == pytest.approx(741.19, abs=0.01)
    assert to_canonical(820, "MN/m**2", P.ultimate_strength) == (820.0, False)
    hv, converted = to_canonical(4947, "MPa", P.vickers_hardness)  # via 1 HV = 1 kgf/mm^2
    assert hv == pytest.approx(504.45, abs=0.01) and converted


def test_unit_compatibility_is_enforced():
    with pytest.raises(UnitError):
        to_canonical(12, "%", P.vickers_hardness)
    with pytest.raises(UnitError):
        to_canonical(500, "MPa", P.elongation)
    with pytest.raises(UnitError):
        to_canonical(500, None, P.yield_strength)  # a strength without a unit is not guessed


def test_temperatures():
    assert temperature_c("800 °C") == 800
    assert temperature_c("~23 ◦C") == 23
    assert temperature_c("-423 °F") == pytest.approx(-252.8)
    assert temperature_c("Tested in LH2 (-423 °F)") == pytest.approx(-252.8)
    assert temperature_c("800") is None and temperature_c("800", "C") == 800
    assert temperature_c("Fe-0.5C") is None  # a composition is not a temperature
    assert temperature_c("room temperature") is None


def test_clean_text_degree_variants():
    assert clean_text("T ( ˝ C)") == "T ( °C)"
    assert clean_text("~23 ◦C") == "~23 °C"
