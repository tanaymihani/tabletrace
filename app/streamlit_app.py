"""TableTrace demo: pick a page, see every value it reads there, boxed on the page.

Green boxes were accepted, amber ones go to a person (confidence under the review threshold,
which defaults to the one calibrated in reports/metrics.json). Click a row to find its cell.

Run: pip install -e ".[app]" && streamlit run app/streamlit_app.py
The built-in examples need `make data` first; any other PDF can be uploaded.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st
from PIL import ImageDraw

from tabletrace.confidence import score_values
from tabletrace.paths import ROOT, load_papers
from tabletrace.pipeline import Config, PaperResult, process_page
from tabletrace.render import page_count, render_page

DPI = 150
ACCEPT, REVIEW, PICKED = (15, 127, 88), (176, 110, 0), (42, 120, 214)

st.set_page_config(page_title="TableTrace", layout="wide")


def calibrated_threshold() -> float:
    metrics = ROOT / "reports" / "metrics.json"
    if metrics.exists():
        return float(json.loads(metrics.read_text())["calibration"]["alphas"]["0.1"]["threshold"])
    return 0.85


@st.cache_data(show_spinner="Reading the page...")
def extract(pdf: str, page: int, paper_id: str) -> tuple[list[dict], dict, list[dict]]:
    doc = SimpleNamespace(paper_id=paper_id, input_pdf=Path(pdf), raw_pdf=Path(pdf))
    cfg = Config(ocr_lang="eng_tt" if (ROOT / "models" / "tessdata" / "eng_tt.traineddata").exists() else "eng")
    res = PaperResult(paper_id)
    process_page(doc, page, cfg, res)
    route = res.routes[-1]
    has_second = {paper_id} if route["route"] == "text_layer" or route["vendor_ocr"] else set()
    score_values(res.values, res.secondary, has_second)
    records = [v.model_dump(mode="json", exclude={"features"}) for v in res.values]
    tables = [{"table": t.table_index, "bbox": t.bbox, "values": t.n_values, "orientation": t.orientation,
               "caption": t.caption} for t in res.tables]
    return records, route, tables


st.title("TableTrace")
st.caption("Property values from tables in PDFs, each one tied to the cell it came from.")

papers = {p.paper_id: p for p in load_papers() if p.input_pdf.exists()}
with st.sidebar:
    choices = ["Upload a PDF"] + ([f"Example: {pid}" for pid in papers] if papers else [])
    default = choices.index("Example: billinghurst1992") if "Example: billinghurst1992" in choices else 0
    source = st.selectbox("Document", choices, index=default)
    if source.startswith("Example: "):
        paper = papers[source.removeprefix("Example: ")]
        pdf_path, paper_id = paper.input_pdf, paper.paper_id
        page = st.selectbox("Page (labeled pages)", paper.labeled_pages, index=len(paper.labeled_pages) - 1)
    else:
        upload = st.file_uploader("PDF", type="pdf")
        if upload is None:
            st.info("Upload a PDF, or run `make data` to get the built-in examples.")
            st.stop()
        tmp = Path(tempfile.gettempdir()) / f"tabletrace_{upload.name}"
        tmp.write_bytes(upload.getvalue())
        pdf_path, paper_id = tmp, Path(upload.name).stem
        page = st.number_input("Page", min_value=1, max_value=page_count(tmp), value=1)
    threshold = st.slider("Review threshold", 0.0, 1.0, calibrated_threshold(), 0.01)

records, route, tables = extract(str(pdf_path), int(page), paper_id)
st.write(f"Page {page}: {route['reason']}, so it takes the **{route['route'].replace('_', ' ')}** route. "
         f"{len(tables)} table(s), {len(records)} values.")

df = pd.DataFrame([{
    "material": r["material"], "property": r["property"], "value": round(r["value"], 2), "unit": r["unit"],
    "printed": r["raw_text"], "temperature (C)": r["test_temperature_c"], "table": r["table_index"],
    "row": r["row"], "col": r["col"], "confidence": round(r["confidence"], 2),
    "review": r["confidence"] < threshold} for r in records])

left, right = st.columns([3, 2])
with right:
    if df.empty:
        st.warning("No property values found on this page.")
        picked = None
    else:
        st.metric("Sent to review", f"{int(df['review'].sum())} of {len(df)}")
        event = st.dataframe(df, hide_index=True, on_select="rerun", selection_mode="single-row",
                             width="stretch")
        rows = event.selection.rows if event is not None and hasattr(event, "selection") else []
        picked = rows[0] if rows else None
        st.download_button("Records as JSON lines", "\n".join(json.dumps(r) for r in records),
                           file_name=f"{paper_id}_p{page}.jsonl")
with left:
    img = render_page(Path(pdf_path), int(page), dpi=DPI).convert("RGB")
    draw = ImageDraw.Draw(img)
    k = DPI / 72
    for t in tables:
        x0, y0, x1, y1 = (v * k for v in t["bbox"])
        draw.rectangle((x0, y0, x1, y1), outline=(195, 194, 183), width=2)
    for i, r in enumerate(records):
        if not r["bbox"]:
            continue
        x0, y0, x1, y1 = (v * k for v in r["bbox"])
        color = PICKED if i == picked else (REVIEW if r["confidence"] < threshold else ACCEPT)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4 if i == picked else 2)
    st.image(img, caption="green: accepted · amber: sent to review · blue: the selected row", width="stretch")
