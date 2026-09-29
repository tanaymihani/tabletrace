import pytest
from pydantic import ValidationError

from tabletrace.mapping import caption_property, map_header
from tabletrace.schema import ExtractedValue
from tabletrace.schema import PropertyKind as P


@pytest.mark.parametrize(
    "header, prop",
    [
        ("HV", P.vickers_hardness),
        ("Vickers hardness", P.vickers_hardness),
        ("Hardness (HV0.5)", P.vickers_hardness),
        ("HV30", P.vickers_hardness),
        ("Rp0,2 [MPa]", P.yield_strength),
        ("σ y (MPa)", P.yield_strength),
        ("σ0.2 (MPa)", P.yield_strength),
        ("YS (ksi)", P.yield_strength),
        ("R m [MPa]", P.ultimate_strength),
        ("Ultimate ten- sile strength", P.ultimate_strength),
        ("σ max (MPa)", P.ultimate_strength),
        ("A [%]", P.elongation),
        ("εf in %", P.elongation),
        ("Fracture strain (%)", P.elongation),
        ("Elongation, percent", P.elongation),
        ("Percent EL 1 in", P.elongation),
        ("Yeild strength (MPa)", P.yield_strength),  # typo, fuzzy match
        ("Omax (MPa)", P.ultimate_strength),  # OCR read sigma as O
    ],
)
def test_headers_map_to_properties(header, prop):
    m = map_header(header)
    assert m.kind == "property" and m.property is prop


@pytest.mark.parametrize(
    "header",
    ["Reduction of area, percent", "Percent RA", "E [GPa]", "ρ (g/cm3)", "σmax/ρ", "a (Å)", "δ [%]", "ΔH mix [kJ/mol]",
     "σc (MPa)", "VEC"],
)
def test_lookalike_headers_do_not_map(header):
    assert map_header(header).kind != "property"


def test_calculated_columns_are_flagged():
    assert map_header("Hardness (HV) predicted").excluded
    assert map_header("A calc. [%]").excluded
    assert not map_header("Hardness (HV)").excluded


def test_descriptors_and_loading():
    assert map_header("Sample I.D.").descriptor == "id"
    assert map_header("Test Temp. (°F)").descriptor == "temperature"
    assert map_header("UTS (ksi)").loading == "tension"
    cap = caption_property("Table 4. The elevated temperature compressive yield strength (MPa) of the alloys.")
    assert cap.property is P.yield_strength and cap.loading == "compression" and cap.unit == "MPa"
    # a caption that names several properties does not decide any single cell
    assert caption_property("The yield strength, fracture strain, and hardness of the alloys") is None


def _value(**kw):
    base = dict(paper_id="p", page=1, table_index=1, material="Al0.5CoCrFeNi", property=P.vickers_hardness,
                value=450.0, unit="HV", raw_text="450", source="text_layer")
    base.update(kw)
    return ExtractedValue(**base)


def test_schema_accepts_a_clean_value():
    assert _value().confidence == 1.0


def test_schema_rejects_hardness_given_in_mpa():
    with pytest.raises(ValidationError, match="bad unit"):
        _value(unit="MPa", value=4413.0)


def test_schema_rejects_out_of_range_and_missing_material():
    with pytest.raises(ValidationError, match="physical range"):
        _value(property=P.elongation, unit="%", value=140.0)
    with pytest.raises(ValidationError, match="missing material"):
        _value(material="  ")
    with pytest.raises(ValidationError, match="physical range"):
        _value(value=-3.0)


def test_hot_ductility_above_100_percent_is_allowed():
    v = _value(property=P.elongation, unit="%", value=140.0, test_temperature_c=650.0)
    assert v.value == 140.0


def test_caption_lines_and_body_sentences():
    from tabletrace.captions import CAPTION, roman_to_int

    def number(t):
        m = CAPTION.match(t)
        return m.group(1) if m else None

    assert number("Table 3. The yield strength") == "3"
    assert number("TABLE I. - TENSILE TEST DATA") == "I" and roman_to_int("III") == 3 and roman_to_int("IV") == 4
    assert number("Table 2 The Vicker's hardness values") == "2"
    assert number("TABLE 1-continued") == "1"
    assert number("Table 5 shows the effect of minor element, boron") is None  # body text, not a caption
