import numpy as np
import pytest

from tabletrace.calibrate import apply_threshold, choose_threshold, clopper_pearson_upper, min_n_to_certify
from tabletrace.evaluate import match, material_similarity, score
from tabletrace.grid import Word, assign_words, cluster_rows, join_words
from tabletrace.tables_text import words_from_chars

# --------------------------------------------------------------------------- word-to-cell assignment

def _w(text, x0, top, w=10.0, h=8.0, conf=1.0):
    return Word(text, x0, top, x0 + w, top + h, conf)


def test_words_land_in_the_cell_containing_their_center():
    rows = [(0, 10), (10, 20), (20, 30)]
    cols = [(0, 50), (50, 100)]
    words = [_w("Alloy", 2, 1), _w("HV", 60, 1), _w("A1", 2, 11), _w("450", 60, 11), _w("B2", 2, 21), _w("512", 70, 21)]
    cells = assign_words(words, rows, cols, (0, 0, 100, 30))
    assert [[c.text for c in r] for r in cells] == [["Alloy", "HV"], ["A1", "450"], ["B2", "512"]]


def test_word_between_bands_goes_to_the_nearest_band():
    cells = assign_words([_w("x", 51, 12, w=0.5, h=0.5)], [(0, 10), (13, 20)], [(0, 50), (52, 100)], (0, 0, 100, 20))
    assert cells[1][1].text == "x"


def test_words_outside_the_table_are_ignored():
    cells = assign_words([_w("caption", 0, 40)], [(0, 10)], [(0, 50)], (0, 0, 50, 10))
    assert cells[0][0].text == ""


def test_multiline_cell_reads_top_to_bottom():
    assert join_words([_w("(HV)", 0, 10), _w("Hardness", 0, 0)]) == "Hardness (HV)"


def test_subscripts_join_their_line():
    # "Nb1.5Ta": base glyphs 9 pt, subscript digits 7 pt set 3 pt lower
    chars = [
        {"text": "N", "x0": 0, "x1": 7, "top": 100, "bottom": 109, "size": 9},
        {"text": "b", "x0": 7, "x1": 12, "top": 100, "bottom": 109, "size": 9},
        {"text": "1", "x0": 12, "x1": 15, "top": 103, "bottom": 110, "size": 7},
        {"text": ".", "x0": 15, "x1": 17, "top": 103, "bottom": 110, "size": 7},
        {"text": "5", "x0": 17, "x1": 20, "top": 103, "bottom": 110, "size": 7},
        {"text": "T", "x0": 20, "x1": 26, "top": 100, "bottom": 109, "size": 9},
        {"text": "a", "x0": 26, "x1": 31, "top": 100, "bottom": 109, "size": 9},
        {"text": "8", "x0": 60, "x1": 65, "top": 100, "bottom": 109, "size": 9},
    ]
    words = words_from_chars(chars)
    assert [w.text for w in sorted(words, key=lambda w: w.x0)] == ["Nb1.5Ta", "8"]


def test_a_label_between_two_header_lines_does_not_merge_them():
    # "HfMoNb" over "TaTiZr" with "Temperature" vertically centred between them (Tseng et al. Table 4)
    chars = []
    for i, ch in enumerate("HfMoNb"):
        chars.append({"text": ch, "x0": 100 + 5 * i, "x1": 105 + 5 * i, "top": 223.6, "bottom": 232.6, "size": 9})
    for i, ch in enumerate("TaTiZr"):
        chars.append({"text": ch, "x0": 100 + 5 * i, "x1": 105 + 5 * i, "top": 234.5, "bottom": 243.5, "size": 9})
    for i, ch in enumerate("Temp"):
        chars.append({"text": ch, "x0": 10 + 5 * i, "x1": 15 + 5 * i, "top": 229.1, "bottom": 238.1, "size": 9})
    texts = sorted(w.text for w in words_from_chars(chars))
    assert texts == ["HfMoNb", "TaTiZr", "Temp"]


def test_cluster_rows_keeps_rows_apart():
    words = [_w("a", 0, 0), _w("b", 20, 1), _w("c", 0, 12)]
    assert len(cluster_rows(words)) == 2


# --------------------------------------------------------------------------- Hungarian scoring

def _rec(material, prop, value, paper="p", temp=None):
    return {"paper_id": paper, "page": 1, "material": material, "property": prop, "value": value,
            "test_temperature_c": temp, "raw_text": str(value), "source": "text_layer"}


def test_hungarian_scoring_on_a_known_example():
    gold = [_rec("A", "vickers_hardness", 450), _rec("B", "vickers_hardness", 512), _rec("A", "yield_strength", 1200),
            _rec("B", "yield_strength", 1250)]
    pred = [
        _rec("A", "vickers_hardness", 451),  # within 1%: correct
        _rec("B", "vickers_hardness", 512),  # exact: correct
        _rec("A", "yield_strength", 1250),  # right property, wrong value (it is B's): false positive
        _rec("C", "elongation", 12.0),  # nothing like it in the labels: false positive
    ]
    res = score(pred, gold)
    assert res["strict"]["n_correct"] == 2
    assert res["strict"]["precision"] == pytest.approx(0.5)
    assert res["strict"]["recall"] == pytest.approx(0.5)
    assert res["strict"]["f1"] == pytest.approx(0.5)


def test_assignment_is_one_to_one_and_prefers_exact_matches():
    gold = [_rec("A", "yield_strength", 1000), _rec("A", "yield_strength", 1004)]
    pred = [_rec("A", "yield_strength", 1004), _rec("A", "yield_strength", 1000)]
    m = match(pred, gold)
    assert sorted((i, j) for i, j, _ in m.pairs) == [(0, 1), (1, 0)]
    assert m.pred_status == ["correct", "correct"]


def test_incompatible_property_or_temperature_never_matches():
    gold = [_rec("A", "yield_strength", 500, temp=800)]
    assert match([_rec("A", "ultimate_strength", 500, temp=800)], gold).pred_status == ["false_positive"]
    assert match([_rec("A", "yield_strength", 500, temp=1000)], gold).pred_status == ["false_positive"]
    assert match([_rec("A", "yield_strength", 500)], gold).pred_status == ["correct"]  # unknown temperature is allowed


def test_fuzzy_sample_names_are_partial_matches():
    gold = [_rec("Ti-25Nb", "ultimate_strength", 585)]
    res = score([_rec("Ti-25N", "ultimate_strength", 585)], gold)
    assert res["strict"]["n_correct"] == 0 and res["lenient"]["n_correct"] == 1
    assert material_similarity("Nb1.5TaTiZr0.5", "Nb1.5 TaTiZr0.5") == 1.0
    assert material_similarity("Ti-25Nb", "Ti-25Nb-25V") < 0.85  # neighbouring alloys are different samples


def test_zero_valued_labels_need_an_exact_zero():
    gold = [_rec("TaTiZr", "elongation", 0.0)]
    assert match([_rec("TaTiZr", "elongation", 0.0)], gold).pred_status == ["correct"]
    assert match([_rec("TaTiZr", "elongation", 0.33)], gold).pred_status == ["false_positive"]


# --------------------------------------------------------------------------- calibration

def test_clopper_pearson_bound():
    assert clopper_pearson_upper(0, 45, 0.1) <= 0.05  # 45 clean values certify 5% at delta 0.1
    assert clopper_pearson_upper(0, 44, 0.1) > 0.05
    assert min_n_to_certify(0.05, 0.1) == 45
    assert clopper_pearson_upper(3, 3, 0.1) == 1.0


def test_threshold_is_a_confidence_value_and_respects_the_bound():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 2000)
    correct = rng.uniform(0, 1, 2000) < conf  # higher confidence, more often right
    cal = choose_threshold(conf, correct, alpha=0.1, delta=0.1)
    assert 0.0 <= cal.threshold <= 1.0
    assert cal.upper_bound <= 0.1
    rep = apply_threshold(conf, correct, cal.threshold)
    assert rep.accepted_error_rate <= cal.upper_bound  # the observed rate never exceeds the certified bound


def test_threshold_accepts_nothing_when_nothing_can_be_certified():
    conf = np.linspace(0, 1, 30)
    correct = np.zeros(30, dtype=bool)
    cal = choose_threshold(conf, correct, alpha=0.05, delta=0.1)
    assert cal.threshold == float("inf") and cal.n_accepted == 0
    assert apply_threshold(conf, correct, cal.threshold).n_accepted == 0
