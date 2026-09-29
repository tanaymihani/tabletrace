"""Transcriptions of every labeled table, expanded into ground_truth.csv.

Each table below was typed from the rendered page image (300 DPI for scans),
not from pipeline output. Born-digital tables were checked against their text
layer after removing clipped ghost text; scanned tables were checked against
the vendor OCR layer and, for NASA TM X-3216, against the redundant ksi column.

Labeling policy (see labels/README.md for the long version):
  * Four properties: vickers_hardness, yield_strength, ultimate_strength
    (UTS or UCS), elongation (elongation / fracture strain / strain to failure).
  * Values marked calculated, predicted or theoretical are excluded, as are
    summary rows ("Averages"), missing cells, and other properties (E, RA, ...).
  * material is the printed label of the sample: the stub-column cell, or the
    column header in transposed tables. Footnote marks and citations are dropped
    and subscripts are written inline (Nb1.5TaTiZr0.5).
  * value is stored in canonical units (HV, MPa, %); raw_text keeps the cell.

Usage: python labels/transcriptions.py   (rewrites labels/ground_truth.csv)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Conversion to canonical units. Kept separate from the pipeline's pint-based
# normalizer on purpose, so label values do not inherit its bugs.
TO_CANONICAL = {
    "MPa": 1.0,
    "MN/m2": 1.0,
    "ksi": 6.894757,
    "HV": 1.0,
    "HV30": 1.0,
    "Hv": 1.0,
    "Hv (MPa)": 1.0 / 9.80665,  # Vickers number is kgf/mm2; 1 kgf/mm2 = 9.80665 MPa
    "%": 1.0,
}
CANONICAL_UNIT = {
    "vickers_hardness": "HV",
    "yield_strength": "MPa",
    "ultimate_strength": "MPa",
    "elongation": "%",
}
ROUTE_SOURCE = {"born_digital": "text_layer", "image_only": "ocr", "scanned": "ocr"}

NUM = r"[0-9][0-9,]*(?:\.[0-9]+)?"


def parse_cell(raw: str) -> tuple[float, float | None, str]:
    """Tiny parser for the cells typed below: returns (value, uncertainty, qualifier)."""
    s = raw.strip().replace("Hv", "").replace("*", "").replace("†", "").strip()
    qualifier = "="
    if s.startswith(">"):
        qualifier, s = ">", s[1:].strip()
    m = re.fullmatch(rf"({NUM})\s*(?:±\s*({NUM}))?", s)
    if not m:
        raise ValueError(f"cannot parse {raw!r}")
    value = float(m.group(1).replace(",", ""))
    unc = float(m.group(2).replace(",", "")) if m.group(2) else None
    return value, unc, qualifier


class Sheet:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, paper, page, table, material, prop, raw, raw_unit, *, temp_c=None, loading="",
            condition="", notes="", route):
        value, unc, qualifier = parse_cell(raw)
        k = TO_CANONICAL[raw_unit]
        self.rows.append(
            {
                "paper_id": paper,
                "page": page,
                "table_index": table,
                "material": material,
                "property": prop,
                "value": round(value * k, 4),
                "uncertainty": "" if unc is None else round(unc * k, 4),
                "unit": CANONICAL_UNIT[prop],
                "qualifier": qualifier,
                "raw_text": raw,
                "raw_unit": raw_unit,
                "test_temperature_c": "" if temp_c is None else temp_c,
                "loading": loading,
                "condition": condition,
                "source": ROUTE_SOURCE[route],
                "notes": notes,
            }
        )

    def grid(self, paper, page, table, props, rows, *, route, loading="", temp_c=None, notes_by_material=None):
        """Normal orientation: rows of (material, cell, cell, ...) under `props` columns.

        props is a list of (property, raw_unit) or None for columns that are not labeled.
        Cells that are None, '-', 'NA', 'N. A.' or '---' are skipped.
        """
        notes_by_material = notes_by_material or {}
        for material, *cells in rows:
            assert len(cells) == len(props), (paper, table, material)
            for spec, cell in zip(props, cells):
                if spec is None or cell in (None, "-", "NA", "N. A.", "---", "--"):
                    continue
                prop, unit = spec
                self.add(paper, page, table, material, prop, cell, unit, temp_c=temp_c, loading=loading,
                         notes=notes_by_material.get(material, ""), route=route)


S = Sheet()

# --------------------------------------------------------------------------- chen2014
# Table 3 (p7), transposed: alloys across the top, properties down the side.
alloys = ["Al0", "Al0.25", "Al0.5", "Al0.75", "Al1", "Al1.5"]
rows = {
    ("yield_strength", "MPa"): ["1,200", "1,250", "1,625", "1,260", "1,375", "500"],
    ("elongation", "%"): ["25.62", "12.91", "11.25", "7.5", "2.5", "1.3"],
    ("vickers_hardness", "HV"): ["440.7", "460.1", "486.5", "516.6", "536.6", "556.4"],
}
for (prop, unit), cells in rows.items():
    for alloy, cell in zip(alloys, cells):
        S.add("chen2014", 7, 3, alloy, prop, cell, unit, route="image_only",
              notes="fracture strain" if prop == "elongation" else "")

# --------------------------------------------------------------------------- yao2016
# Table 1 (p3), transposed. Pure-metal columns are literature values (ref [7]);
# the alloy column holds Calc/Exp pairs and only the experimental half is labeled.
metals = ["Mo", "Nb", "Ta", "V"]
for prop, unit, cells, exp_raw in [
    ("vickers_hardness", "Hv (MPa)", ["1530", "1320", "873", "628"], "1097/4947"),
    ("yield_strength", "MPa", ["438", "240", "345", "310"], "333/1525"),
]:
    for metal, cell in zip(metals, cells):
        S.add("yao2016", 3, 1, metal, prop, cell, unit, route="born_digital", notes="pure metal, literature value [7]")
    S.add("yao2016", 3, 1, "MoNbTaV", prop, exp_raw.split("/")[1], unit, route="born_digital",
          notes="Calc/Exp pair; experimental value labeled")
    S.rows[-1]["raw_text"] = exp_raw  # keep the whole cell as printed

# --------------------------------------------------------------------------- tseng2018
T_ALLOYS = ["HfMoNbTaTiZr", "HfNbTaTiZr", "HfMoTaTiZr", "HfMoNbTiZr", "HfMoNbTaZr", "HfMoNbTaTi"]
# Table 3 (p7): room temperature compression, transposed.
for prop, unit, cells in [
    ("yield_strength", "MPa", ["1512", "929", "1600", "1351", "1524", "1369"]),
    ("elongation", "%", ["12", "> 50", "4", "20", "16", "27"]),
]:
    for alloy, cell in zip(T_ALLOYS, cells):
        note = "values from ref [25]" if alloy == "HfNbTaTiZr" else ""
        if prop == "elongation":
            note = (note + "; fracture strain").strip("; ")
        S.add("tseng2018", 7, 3, alloy, prop, cell, unit, loading="compression", route="born_digital", notes=note)
# Table 4 (p7): elevated temperature compressive yield strength; temperatures are rows.
t4 = {
    800: ["1007", "535", "1045", "829", "1005", "822"],
    1000: ["814", "295", "855", "721", "927", "778"],
    1200: ["556", "92", "404", "301", "694", "699"],
    1400: ["N. A.", "N. A.", "N. A.", "N. A.", "278", "367"],
}
for temp, cells in t4.items():
    for alloy, cell in zip(T_ALLOYS, cells):
        if cell == "N. A.":
            continue
        S.add("tseng2018", 7, 4, alloy, "yield_strength", cell, "MPa", temp_c=temp, loading="compression",
              route="born_digital", notes="property given only in the caption")
# Table 7 (p11): only sigma_0.2 is measured; Gm, delta-sigma, sigma_m, sigma_c, Gcal are calculated.
for alloy, cell in zip(T_ALLOYS, ["1512", "929", "1600", "1351", "1524", "1369"]):
    S.add("tseng2018", 11, 7, alloy, "yield_strength", cell, "MPa", route="born_digital",
          notes="cell reads '929 [26]'" if alloy == "HfNbTaTiZr" else "")

# --------------------------------------------------------------------------- zyka2019
Z_PROPS = [("yield_strength", "MPa"), ("ultimate_strength", "MPa"), ("elongation", "%"), None, ("vickers_hardness", "HV30")]
S.grid("zyka2019", 4, 3, Z_PROPS, [
    ("HfNbTaTiZr", "1155", "1212", "12.3", "59", "359"),
    ("NbTaTiZr", "1144", "1205", "6.4", "98", "358"),
    ("NbTaTi", "620", "683", "18.5", "143", "246"),
    ("TaTiZr", "-", "284", "0", "157", "485"),
    ("NbTiZr", "956", "991", "14.2", "88", "295"),
    ("Nb1.5TaTiZr0.5", "822", "852", "0.33", "127", "294"),
    ("Nb0.5TaTiZr1.5", "-", "843", "0", "93", "489"),
], route="born_digital")
# Table 7 (p14): columns Ta+Zr | HV30 | Rp0.2 | Rm | A | A calc. | Ref. ; rows from ref [25] carry
# asterisks for values read off a figure in that reference.
Z7_PROPS = [None, ("vickers_hardness", "HV30"), ("yield_strength", "MPa"), ("ultimate_strength", "MPa"), ("elongation", "%"), None]
S.grid("zyka2019", 14, 7, Z7_PROPS, [
    ("HfNbTaTiZr", "0.4", "359", "1155", "1212", "12.3", "12.51"),
    ("NbTaTiZr", "0.5", "358", "1144", "1205", "6.4", "7.33"),
    ("NbTaTi", "0.333", "246", "620", "683", "18.5", "15.98"),
    ("TaTiZr", "0.666", "485", "-", "284", "0", "-1.26"),
    ("NbTiZr", "0.333", "295", "956", "991", "14.2", "15.98"),
    ("Nb1.5TaTiZr0.5", "0.375", "294", "822", "852", "0.33", "13.81"),
    ("Nb0.5TaTiZr1.5", "0.625", "489", "-", "843", "0", "0.86"),
], route="born_digital")
first_ref_row = len(S.rows)
S.grid("zyka2019", 14, 7, Z7_PROPS, [
    ("NbTaTiZr", "0.5", "-", "*1190", "*1190", "*0", "7.33"),
    ("NbTa0.8Ti1.2Zr", "0.45", "-", "*1100", "*1110", "*1", "9.92"),
    ("NbTa0.6Ti1.4Zr", "0.4", "-", "*1030", "*1070", "*2.5", "12.51"),
    ("NbTa0.4Ti1.6Zr", "0.35", "-", "910", "1040", "18", "15.10"),
    ("NbTa0.2Ti1.8Zr", "0.3", "-", "790", "920", "22", "17.69"),
], route="born_digital")
for r in S.rows[first_ref_row:]:
    r["notes"] = "ref [25]" + ("; asterisk: estimated from a figure in that reference" if r["raw_text"].startswith("*") else "")

# --------------------------------------------------------------------------- haas2019
# Table 2 (p6): rows are test temperatures; two-level header property > state (CC/DS).
haas = {
    23: ["696 ± 33", "1197 ± 6", "520 ± 19", "627 ± 9", "49 ± 0", "27 ± 1"],
    600: ["561 ± 50", "1006 ± 16", "470 ± 8", "594 ± 3", "11 ± 8", "12 ± 2"],
    700: ["560 ± 27", "840 ± 1", "461 ± 10", "547 ± 42", "9 ± 4", "17 ± 5"],
    800: ["536 ± 7", "575 ± 7", "445 ± 2", "399 ± 1", "6 ± 4", "20 ± 4"],
    900: ["312 ± 6", "319 ± 1", "219 ± 6", "243 ± 2", "8 ± 6", "34 ± 1"],
}
cols = [("ultimate_strength", "MPa", "CC"), ("ultimate_strength", "MPa", "DS"), ("yield_strength", "MPa", "CC"),
        ("yield_strength", "MPa", "DS"), ("elongation", "%", "CC"), ("elongation", "%", "DS")]
for temp, cells in haas.items():
    for (prop, unit, state), cell in zip(cols, cells):
        S.add("haas2019", 6, 2, state, prop, cell, unit, temp_c=temp, loading="tension", route="image_only",
              condition="conventionally cast" if state == "CC" else "directionally solidified",
              notes="row label '~23 °C'" if temp == 23 else "")

# --------------------------------------------------------------------------- rickman2019
# Table 2 (p5): measured HV with standard deviation; the 'predicted' column is excluded.
for alloy, cell in [
    ("Co33W07Al33Nb24Cr03", "1084 ± 37"),
    ("Ti18Ni24Ta12Cr22Co24", "1011 ± 20"),
    ("Co6W9Al36Mo38Ni11", "725 ± 47"),
    ("Ni47Co02Ta12Ti9Nb30", "815 ± 43"),
    ("Ti44Ni02Nb21Cr21Co12", "422 ± 13"),
    ("Ti32Nb9Ta01Cr19Co39", "856 ± 29"),
    ("Ti39W04Nb31Ta04Co22", "277 ± 12"),
]:
    S.add("rickman2019", 5, 2, alloy, "vickers_hardness", cell, "HV", route="born_digital")

# --------------------------------------------------------------------------- sanchez2019
# Table 3 (p9): rho | sigma_y | sigma_max | eps_max | E | sigma_max/rho ; compression at RT.
S.grid("sanchez2019", 9, 3, [None, ("yield_strength", "MPa"), ("ultimate_strength", "MPa"), ("elongation", "%"), None, None], [
    ("Al65Cu5Fe5Mg5Si15Zn5", "3.08", "422 ± 75", "482 ± 98", "1", "88.7 ± 04", "156"),
    ("Al65Cu5Mg5Ni5Si15Zn5", "3.15", "534 ± 04", "574 ± 32", "1", "107.8 ± 17", "182"),
    ("Al70Cr5Cu5Mg5Si10Zn5", "3.06", "490 ± 18", "608 ± 30", "6", "78.4 ± 03", "199"),
    ("Al70Cu5Mg5Mn5Si10Zn5", "2.98", "622 ± 15", "644 ± 13", "2", "114.1 ± 02", "216"),
    ("Al70Cu5Mg5Si10Zn5Zr5", "3.06", "565 ± 79", "633 ± 42", "4", "105.1 ± 27", "207"),
], route="image_only", loading="compression")

# --------------------------------------------------------------------------- yeh2002 (patent)
AS_CAST, HEAT = "as-cast", "heat-treated 1000 °C / 12 h, furnace cooled"
t1 = [
    ("1", "590", "600"), ("2", "800", "790"), ("3", "740", "760"), ("4", "620", "620"), ("5", "780", "790"),
    ("6", "630", "620"), ("7", "790", "800"), ("8", "790", "790"), ("9", "670", "690"), ("10", "780", "790"),
    ("11", "680", "680"), ("12", "780", "890"),
]
t1_cont = [
    ("13", "850", "850"), ("14", "720", "720"), ("15", "840", "870"), ("16", "670", "630"), ("17", "780", "800"),
    ("18", "830", "820"), ("19", "700", "630"), ("20", "840", "840"),
]
for page, block in [(3, t1), (4, t1_cont)]:
    for alloy, cast, ht in block:
        S.add("yeh2002", page, 1, alloy, "vickers_hardness", cast, "Hv", condition=AS_CAST, route="scanned")
        S.add("yeh2002", page, 1, alloy, "vickers_hardness", ht, "Hv", condition=HEAT, route="scanned")
# Table 4 (p5): a single transposed row with the unit printed inside each cell.
for alloy, cell in [("Alloy No. 21", "Hv571"), ("Alloy No. 22", "Hv1,049"), ("Alloy No. 23", "Hv760"), ("Alloy No. 24", "Hv666")]:
    S.add("yeh2002", 5, 4, alloy, "vickers_hardness", cell, "Hv", condition="rapidly solidified", route="scanned")
for table, page, block, minor in [
    (5, 5, [("25", "760"), ("26", "880"), ("27", "980"), ("28", "1030"), ("29", "1067")], "boron"),
    (6, 5, [("30", "480"), ("31", "500"), ("32", "520")], "silicon"),
    (6, 6, [("33", "540"), ("34", "570")], "silicon"),
    (7, 6, [("30", "760"), ("31", "750"), ("32", "730"), ("33", "690"), ("34", "630")], "copper"),
]:
    for alloy, cell in block:
        note = "alloy numbers repeat Table 6; the text calls these alloys 35 to 39" if table == 7 else ""
        S.add("yeh2002", page, table, alloy, "vickers_hardness", cell, "Hv", condition=f"minor element: {minor}",
              route="scanned", notes=note)

# --------------------------------------------------------------------------- gray1975 (NASA TM X-3216)
# Table I (p13). Per temperature: UTS MN/m2 | UTS ksi | RA % | elongation %. The ksi column is a
# duplicate of the same measurement, so only MN/m2 is labeled. Rows without a label are replicate
# tests of the alloy above. '---' means no specimen.
G = "---"
gray = [
    # material, [23 °C: MN, ksi, RA, EL], [500 °C], [650 °C]
    ("Ti-25Nb", ["820", "119", "11", "8"], ["585", "85", "46", "19"], ["165", "24", "96", "62"]),
    ("Ti-50Nb", ["515", "75", "84", "26"], ["235", "34", "85", "25"], ["165", "24", "96", "70"]),
    ("Ti-25Nb-25V", ["815", "118", "43", "16"], ["595", "86", "63", "26"], ["370", "54", "94", "105"]),
    ("Ti-25Nb-25Mo", ["905", "131", "46", "19"], ["640", "93", "74", "24"], ["495", "72", "84", "33"]),
    ("Ti-40Nb-2Si", ["705", "102", "12", "8"], ["285", "41", "62", "25"], ["165", "24", "86", "59"]),
    ("Ti-50Nb-2Si", [G] * 4, ["550", "80", "12", "8"], [G] * 4),
    ("Ti-50Nb-5Al", ["815", "118", "43", "78"], ["635", "92", "50", "21"], ["455", "66", "87", "39"]),
    ("Ti-25Mo", ["785", "114", "62", "23"], ["460", "67", "76", "20"], ["285", "41", "94", "93"]),
    ("Ti-40Mo", [G] * 4, ["690", "100", "52", "19"], [G] * 4),
    ("Ti-50Mo", [G] * 4, [G] * 4, [G] * 4),
    ("Ti-20Mo-25V", ["980", "142", "20", "6"], ["605", "88", "17", "5"], [G] * 4),
    ("Ti-25Mo-25V", [G] * 4, ["760", "110", "15", "5"], [G] * 4),
    ("Ti-40Mo-2Si", [G] * 4, ["310", "45", "1", "0"], [G] * 4),
    ("Ti-50Mo-2Si", [G] * 4, [G] * 4, [G] * 4),
    ("Ti-50Mo-5Al", [G] * 4, [G] * 4, [G] * 4),
    ("Ti-40Mo-5Al", [G] * 4, [G] * 4, [G] * 4),
    ("Ti-25V", ["695", "101", "27", "13"], ["415", "60", "85", "75"], ["170", "25", "96", "140"]),
    ("Ti-50V", ["815", "118", "45", "18"], ["670", "97", "41", "21"], ["490", "71", "86", "58"]),
    ("Ti-50V#2", [None] * 4, ["725", "105", "42", "20"], [None] * 4),
    ("Ti-50V-1Si", ["915", "133", "32", "15"], ["770", "112", "59", "28"], ["530", "79", "72", "32"]),
    ("Ti-50V-1.5Si", ["1060", "153", "15", "5"], ["850", "123", "41", "21"], [G] * 4),
    ("Ti-50V-2Si", ["1190", "173", "11", "3"], ["1010", "146", "19", "10"], ["475", "69", "99", "160"]),
    ("Ti-50V-2Si#2", [None] * 4, ["1030", "149", "25", "14"], [None] * 4),
    ("Ti-50V-3Si", ["1020", "148", "20", "13"], ["825", "120", "46", "22"], ["540", "78", "84", "52"]),
    ("Ti-50V-1Al", ["870", "126", "57", "20"], ["650", "94", "56", "17"], ["560", "81", "74", "42"]),
    ("Ti-50V-3Al", ["970", "141", "38", "11"], ["805", "117", "49", "21"], ["615", "89", "73", "32"]),
    ("Ti-50V-5Al", ["1080", "156", "28", "22"], ["945", "137", "27", "18"], ["580", "84", "79", "44"]),
    ("Ti-50V-5Al#2", [None] * 4, ["960", "139", "28", "20"], [None] * 4),
    ("Ti-50V-7Al", ["690", "100", "1", "1"], ["870", "126", "16", "7"], ["740", "107", "77", "57"]),
    ("Ti-35V-3Al-1.5Si", ["985", "143", "14", "13"], ["795", "115", "62", "33"], ["305", "44", "91", "79"]),
    ("Ti-42V-3Al-1.5Si", ["1030", "150", "11", "12"], ["875", "127", "34", "24"], ["430", "62", "89", "78"]),
    ("Ti-50V-3Al-1.5Si", ["1170", "169", "15", "10"], ["1030", "150", "18", "18"], [G] * 4),
    ("Ti-50V-3Al-1.0Si", ["1100", "159", "13", "10"], ["980", "142", "20", "20"], ["585", "85", "80", "68"]),
    ("Ti-50V-3Al-1.2Si", ["1140", "165", "16", "10"], ["1000", "145", "24", "18"], ["565", "82", "83", "56"]),
    ("Ti-50V-3Al-1.8Si", ["1190", "172", "11", "8"], ["1080", "156", "18", "19"], [G] * 4),
    ("Ti-50V-5Al-0.5Si", ["1030", "150", "1", "1"], ["910", "132", "40", "17"], ["585", "85", "78", "72"]),
    ("Ti-50V-5Al-1Si", ["1160", "168", "4", "1"], ["1010", "147", "21", "14"], ["785", "114", "54", "34"]),
    ("Ti-50V-5Al-2Si", ["1270", "184", "6", "5"], ["1120", "163", "18", "12"], ["660", "96", "68", "53"]),
    ("Ti-50V-5Al-3Si", [G] * 4, [G] * 4, [G] * 4),
    ("Ti-50V-1Al-2Si", ["1030", "150", "22", "14"], ["835", "121", "53", "23"], ["540", "78", "89", "53"]),
    ("Ti-50V-1.7Al-1.5Si", ["1060", "154", "9", "7"], ["930", "135", "24", "19"], [G] * 4),
    ("Ti-50V-2.3Al-1.5Si", ["1120", "162", "15", "10"], ["995", "144", "25", "19"], [G] * 4),
    ("Ti-50V-3.4Al-1.5Si", ["1210", "175", "11", "10"], ["1090", "158", "19", "16"], ["620", "90", "83", "64"]),
    ("Ti-50V-4.0Al-1.5Si", ["1220", "177", "9", "5"], ["1100", "160", "18", "12"], ["640", "93", "77", "51"]),
    ("Ti-50V-7Al-2Si", [G] * 4, [G] * 4, [G] * 4),
]
for material, *blocks in gray:
    name, _, rep = material.partition("#")
    for temp, block in zip((23, 500, 650), blocks):
        mn, ksi, _ra, el = block
        if mn in (None, G):
            continue
        # sanity check: the two UTS columns must agree (authors rounded MN/m2 to 5)
        ksi_mpa = float(ksi) * 6.894757
        note = "unlabeled replicate row" if rep else ""
        if abs(ksi_mpa - float(mn)) > 12:
            note = (note + f"; source inconsistency: ksi column reads {ksi} (= {ksi_mpa:.0f} MPa)").strip("; ")
        S.add("gray1975", 13, 1, name, "ultimate_strength", mn, "MN/m2", temp_c=temp, loading="tension",
              condition=f"replicate {rep}" if rep else "", route="scanned", notes=note)
        S.add("gray1975", 13, 1, name, "elongation", el, "%", temp_c=temp, loading="tension",
              condition=f"replicate {rep}" if rep else "", route="scanned",
              notes="unlabeled replicate row" if rep else "")

# --------------------------------------------------------------------------- billinghurst1992 (NASA TP-3288)
F70, F423 = round((70 - 32) / 1.8, 1), round((-423 - 32) / 1.8, 1)
t2 = [
    # id, material, media, temp_F, UTS, YS, RA, EL 1 in, EL 2 in
    ("2332", "Ti-5Al-2.5Sn", "Air", 70, "107.5", "99.4", "22.7", "15.4", "10.4"),
    ("2333", "Ti-5Al-2.5Sn", "Air", 70, "108.0", "100.0", "24.4", "5.5", "9.1*"),
    ("2334", "Ti-5Al-2.5Sn", "Air", 70, "108.9", "101.3", "26.7", "6.7", "7.5*"),
    ("2335", "Ti-5Al-2.5Sn", "LH2", -423, "186.9", "NA", "13.6", "10.0", "6.3"),
    ("2336", "Ti-5Al-2.5Sn", "LH2", -423, "187.7", "NA", "10.7", "7.4", "7.1*"),
    ("2337", "Ti-5Al-2.5Sn", "LH2", -423, "188.5", "NA", "12.1", "7.0", "7.3*"),
    ("2338", "Ti-5Al-2.5Sn", "LH2", -423, "174.8", "NA", "12.6", "2.5", "4.1*"),
    ("2331", "Ti-5Al-2.5Sn", "LH2", -423, "187.1", "156.9", "9.5", "10.9", "7.0"),
    ("2344", "Ti-6Al-4V", "Air", 70, "128.8", "118.4", "12.9", "10.5", "7.5"),
    ("2345", "Ti-6Al-4V", "Air", 70, "130.0", "119.6", "14.3", "9.4", "7.4"),
    ("2346", "Ti-6Al-4V", "Air", 70, "128.4", "117.8", "18.5", "12.5", "8.5"),
    ("2339", "Ti-6Al-4V", "LH2", -423, "214.1", "NA", "10.7", "2.8", "2.8*"),
    ("2340", "Ti-6Al-4V", "LH2", -423, "219.1", "NA", "8.2", "4.4", "3.4"),
    ("2341", "Ti-6Al-4V", "LH2", -423, "211.3", "NA", "8.2", "5.5", "2.9"),
    ("2342", "Ti-6Al-4V", "LH2", -423, "203.0", "NA", "11.6", "7.3", "4.0"),
    ("2343", "Ti-6Al-4V", "LH2", -423, "213.4", "199.0", "4.9", "8.1", "3.3"),
]
for sid, mat, media, tf, uts, ys, _ra, el1, el2 in t2:
    tc = F70 if tf == 70 else F423
    base = f"{mat}, tested in {media}"
    S.add("billinghurst1992", 13, 2, sid, "ultimate_strength", uts, "ksi", temp_c=tc, loading="tension", condition=base, route="scanned")
    if ys != "NA":
        S.add("billinghurst1992", 13, 2, sid, "yield_strength", ys, "ksi", temp_c=tc, loading="tension", condition=base, route="scanned")
    S.add("billinghurst1992", 13, 2, sid, "elongation", el1, "%", temp_c=tc, loading="tension", condition=base + "; 1 in gauge", route="scanned")
    S.add("billinghurst1992", 13, 2, sid, "elongation", el2, "%", temp_c=tc, loading="tension", condition=base + "; 2 in gauge", route="scanned",
          notes="asterisk: broke outside 1 in gauge marks" if el2.endswith("*") else "")
# The printed "Averages" rows of Table 2 are not labeled, but they are a free checksum on the
# per-sample readings above: every group mean must reproduce the printed average.
T2_PRINTED_AVERAGES = {
    ("Ti-5Al-2.5Sn", "Air"): {"uts": 108.5, "ys": 100.2, "el1": 9.2, "el2": 9.0},
    ("Ti-5Al-2.5Sn", "LH2"): {"uts": 185.0, "el1": 7.6, "el2": 6.4},
    ("Ti-6Al-4V", "Air"): {"uts": 129.1, "ys": 118.6, "el1": 17.4, "el2": 7.8},
    ("Ti-6Al-4V", "LH2"): {"uts": 212.2, "el1": 5.6, "el2": 3.3},
}


def check_t2_averages() -> list[str]:
    msgs = []
    for (mat, media), printed in T2_PRINTED_AVERAGES.items():
        group = [r for r in t2 if r[1] == mat and r[2] == media]
        cols = {"uts": 4, "ys": 5, "el1": 7, "el2": 8}
        for key, want in printed.items():
            vals = [float(r[cols[key]].rstrip("*")) for r in group if r[cols[key]] != "NA"]
            if key == "ys" and len(vals) < len(group):
                continue
            mean = sum(vals) / len(vals)
            status = "ok" if abs(mean - want) <= 0.051 else "MISMATCH"
            msgs.append(f"TP-3288 T2 {mat}/{media} {key}: mean of samples {mean:.2f}, printed {want} -> {status}")
    return msgs


# Table 3 (p14): averages per alloy and temperature.
for mat, tf, uts, ys, el1, el2 in [
    ("Ti-6Al-4V ELI", 70, "129.1", "118.6", "17.4", "7.8"),
    ("Ti-5Al-2.5Sn ELI", 70, "108.5", "100.2", "9.2", "9.0*"),
    ("Ti-6Al-4V ELI", -423, "212.2", "199.0†", "5.6", "3.3"),
    ("Ti-5Al-2.5Sn ELI", -423, "185.0", "156.9†", "7.6", "6.4*"),
]:
    tc = F70 if tf == 70 else F423
    S.add("billinghurst1992", 14, 3, mat, "ultimate_strength", uts, "ksi", temp_c=tc, loading="tension", route="scanned",
          notes="average of 3 to 5 samples")
    S.add("billinghurst1992", 14, 3, mat, "yield_strength", ys, "ksi", temp_c=tc, loading="tension", route="scanned",
          notes="dagger: one sample only" if ys.endswith("†") else "average")
    S.add("billinghurst1992", 14, 3, mat, "elongation", el1, "%", temp_c=tc, loading="tension", condition="1 in gauge", route="scanned",
          notes="17.4 disagrees with the mean of the Table 2 samples (10.8)" if el1 == "17.4" else "")
    S.add("billinghurst1992", 14, 3, mat, "elongation", el2, "%", temp_c=tc, loading="tension", condition="2 in gauge", route="scanned")
# Table 4 (p14): UTS in air and LH2 (reduction of area not labeled).
for mat, air, lh2 in [("Ti-5Al-2.5Sn ELI", "108.4", "185.0"), ("Ti-6Al-4V ELI", "129.1", "212.2")]:
    S.add("billinghurst1992", 14, 4, mat, "ultimate_strength", air, "ksi", temp_c=F70, loading="tension", condition="tested in air",
          route="scanned", notes="Table 3 gives 108.5 for the same average" if air == "108.4" else "")
    S.add("billinghurst1992", 14, 4, mat, "ultimate_strength", lh2, "ksi", temp_c=F423, loading="tension", condition="tested in LH2",
          route="scanned")


def main() -> None:
    fields = ["label_id", "paper_id", "page", "table_index", "material", "property", "value", "uncertainty", "unit",
              "qualifier", "raw_text", "raw_unit", "test_temperature_c", "loading", "condition", "source", "notes"]
    counters: dict[tuple, int] = {}
    out = HERE / "ground_truth.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in S.rows:
            key = (r["paper_id"], r["table_index"])
            counters[key] = counters.get(key, 0) + 1
            w.writerow({"label_id": f"{r['paper_id']}-T{r['table_index']}-{counters[key]:03d}", **r})
    by_paper: dict[str, int] = {}
    for r in S.rows:
        by_paper[r["paper_id"]] = by_paper.get(r["paper_id"], 0) + 1
    print(f"wrote {len(S.rows)} labels to {out}")
    for k, v in by_paper.items():
        print(f"  {k:18s} {v}")
    print("internal consistency checks:")
    n_ksi = sum(1 for r in S.rows if r["paper_id"] == "gray1975" and r["property"] == "ultimate_strength")
    bad_ksi = [r for r in S.rows if "ksi column reads" in r["notes"]]
    print(f"  TM X-3216: {n_ksi - len(bad_ksi)}/{n_ksi} MN/m2 labels agree with the ksi column; disagreements: "
          + ", ".join(f"{r['material']} {r['test_temperature_c']} C" for r in bad_ksi))
    for msg in check_t2_averages():
        print("  " + msg)


if __name__ == "__main__":
    main()
