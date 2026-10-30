import numpy as np

from tabletrace.baselines import (
    bootstrap,
    caption_near,
    clean_cell,
    delatex,
    expand_spans,
    html_tables,
    records_from_text,
    tables_from_text,
)


def test_html_spans_are_repeated_in_every_slot():
    html = ('<table><tr><th rowspan="2">Alloy</th><th colspan="2">UTS (MPa)</th></tr>'
            '<tr><th>CC</th><th>DS</th></tr>'
            '<tr><td>Nb<sub>1.5</sub>TaTiZr<sub>0.5</sub></td><td>1 155 &plusmn; 8</td><td>9.1*</td></tr></table>')
    t = html_tables(html)[0]
    assert t["rows"] == [["Alloy", "UTS (MPa)", "UTS (MPa)"], ["Alloy", "CC", "DS"],
                         ["Nb1.5TaTiZr0.5", "1 155 ± 8", "9.1*"]]
    assert t["header_rows"] == 2


def test_unclosed_cells_and_rowspan_shift():
    rows = expand_spans([[("A", 1, 2), ("B", 1, 1)], [("C", 1, 1)]])
    assert rows == [["A", "B"], ["A", "C"]]
    t = html_tables("<table><tr><td>x<td>y<tr><td>z<td>w</table>")[0]
    assert t["rows"] == [["x", "y"], ["z", "w"]]


def test_latex_tabular_with_multirow_and_multicolumn():
    text = ("Table 3: Tensile data\n\\begin{tabular}{l c c} \\hline Alloy & $\\sigma_{y}$ (MPa) & "
            "\\multicolumn{2}{c}{$\\varepsilon$ (\\%)} \\\\ \\hline \\multirow{2}{*}{TiZr} & 1012$\\pm$15 & 12 & 13 \\\\ "
            "& 990 & 11 & 9 \\\\ \\hline \\end{tabular}")
    t = tables_from_text(text)[0]
    assert t["caption"] == "Table 3: Tensile data"
    assert t["rows"] == [["Alloy", "σy (MPa)", "ε (%)", "ε (%)"],
                         ["TiZr", "1012±15", "12", "13"], ["TiZr", "990", "11", "9"]]


def test_markdown_table_and_caption_below():
    text = "| Alloy | HV |\n|---|---|\n| **A1** | 452 |\n\nTable 2. Hardness of the alloys\n"
    t = tables_from_text(text)[0]
    assert t["rows"] == [["Alloy", "HV"], ["A1", "452"]]
    assert t["caption"] == "Table 2. Hardness of the alloys"
    assert caption_near("no captions here", 0, 5) is None


def test_cell_cleanup_keeps_footnote_marks():
    assert clean_cell("*1190") == "*1190"
    assert clean_cell("*italic*") == "italic"
    assert delatex(r"$\sigma_{0.2}$ at 25$^{\circ}$C") == "σ0.2 at 25°C"
    assert clean_cell(r"Nb\({}_{1.5}\)TaTiZr\({}_{0.5}\)") == "Nb1.5TaTiZr0.5"


def test_truncated_json_answer_keeps_complete_records():
    text = ('```json\n[{"material": "A1", "property": "vickers_hardness", "value": 452, "unit": "HV"},\n'
            ' {"material": "A2", "property": "elong')
    assert records_from_text(text) == [{"material": "A1", "property": "vickers_hardness", "value": 452, "unit": "HV"}]
    assert records_from_text("no JSON at all") == []


def test_document_bootstrap_resamples_documents():
    # [strict correct, lenient correct, predicted, gold] per document
    counts = {"a": np.array([10, 10, 10, 10]), "b": np.array([0, 0, 10, 10])}
    boot = bootstrap(counts, ["a", "b"], n_boot=2000, seed=1)
    assert boot.shape == (2000, 2)
    assert set(np.round(np.unique(boot[:, 0]), 3)) <= {0.0, 0.5, 1.0}
    diff = bootstrap(counts, ["a", "b"], n_boot=200, seed=1, other=counts)
    assert np.allclose(diff, 0.0)
