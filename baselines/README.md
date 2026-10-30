# Baselines

Other table extractors run on the same 16 labeled pages as TableTrace, and scored with the same
interpretation, normalization and Hungarian matching. The results are in
[reports/baselines.md](../reports/baselines.md); this page says how each tool was run.

## Protocol

- **Pages.** Every tool sees the 16 labeled pages, cut out of the input PDFs as single-page files
  (`prepare_pages.py`). That's the same input TableTrace gets: the three image-only papers have no
  text layer, the NASA reports and the patent are scans with the OCR layer that came in the PDF,
  and the four born-digital papers keep their text layer, hidden duplicate tables included.
  Models that take images get a render of the same page: 1288 px on the long side for olmOCR
  (what it was trained on) and 1600 px for the others.
- **Settings.** Defaults, as shipped. No tool was tuned on these documents.
- **From output to records.** A tool's tables (cell grids, HTML, markdown or LaTeX) become grids,
  with a spanning cell repeated across the rows and columns it covers, and go through
  `tabletrace.interpret` like TableTrace's own grids. So the header rules, unit handling and
  validation are the same for everyone, and the comparison is about finding tables and reading
  their cells. Cell text is cleaned of markup (`<sub>`, `**bold**`, LaTeX like `\pm` or `_{0.2}`).
- **Captions.** A tool's own caption is used when it returns one (Docling, Marker, GROBID, the
  page-level text of olmOCR, Nougat and Qwen). When a tool gives a table's position but no caption
  (pdfplumber, Camelot), the caption is linked the same way TableTrace links its own. Two labeled
  tables only name their property in the caption.
- **End-to-end records.** The `*_records` runs skip TableTrace's interpretation entirely: the model
  answers with material, property, value, unit and temperature, and only the unit conversion is
  shared.
- **Scoring.** Strict F1 needs the sample name, property and value (within 1%) to match; lenient
  also accepts a sample name with a small misreading. Intervals come from 10,000 bootstrap
  resamples of the documents, not the values, and the differences from TableTrace are paired
  (same resamples).

## How each tool was run

| Tool | Version | Where | Runner | Notes |
|---|---|---|---|---|
| pdfplumber | 0.11.10 | M2 laptop, CPU | `run_pdfplumber.py` | `find_tables()` with the default ruling-line settings, and with `"text"` alignment for both directions |
| Camelot | 2.0.0 | M2 laptop, CPU | `run_camelot.py` | `lattice` and `stream` flavors, defaults |
| Docling | 2.145.0 | M2 laptop, CPU | `run_docling.py` | default `DocumentConverter`: layout model, TableFormer, RapidOCR where there's no text |
| Marker | 1.10.2 | Kaggle CPU (4 cores) | `kaggle/marker_kaggle.py` | defaults, no LLM; Marker 2 needs a separate vision-model server, so the last 1.x release is used |
| GROBID | 0.9.1 | Kaggle CPU (4 cores) | `kaggle/grobid_kaggle.py` | built from the 0.9.1 tag, `processFulltextDocument`, tables from the TEI |
| Nougat | facebook/nougat-base | Kaggle, T4 | `kaggle/vlm_baselines.py` | greedy, float16, `post_process_generation` |
| olmOCR-2 | allenai/olmOCR-2-7B-1025 | Kaggle, 2 x T4 | `kaggle/vlm_baselines.py` | olmOCR's own prompt and 1288 px render; greedy decoding instead of its 0.1 temperature |
| Qwen2.5-VL | Qwen/Qwen2.5-VL-7B-Instruct | Kaggle, 2 x T4 | `kaggle/vlm_baselines.py` | two prompts: transcribe every table as HTML, or list the property values as JSON records |
| Qwen3-VL | Qwen/Qwen3-VL-8B-Instruct | Kaggle, 2 x T4 | `kaggle/vlm_baselines.py` | same two prompts |

The GPU models run in float16 (T4s have no bfloat16), with the vision encoders kept in float32,
greedy decoding, and a stop when a model starts repeating itself. Exact environments for the
local tools are in `requirements/`, and the Kaggle scripts install pinned versions themselves.
The saved outputs of every run are in `raw/`, one file per page, so the scoring can be rerun
without any of the tools:

```bash
python -m tabletrace baselines          # writes reports/baselines.md, .json and the figure
```

To rerun a tool, prepare the pages (`python baselines/prepare_pages.py` after `make data`), then
run its script. The Kaggle scripts read a private dataset made by `kaggle/prepare_upload.py`.

One change to TableTrace's interpretation came out of this. Some tools write a two-line column
header as two header rows (Nougat writes `HfMoNb` above `TaTiZr`), and a transposed table took
its sample name from the last header row only. A name split over two rows is now joined back when
both halves are element formulas. TableTrace's own extractions are byte-identical with and without
the change, in all four experiments.

## What didn't run

- **Marker 1.10** was still running on its Kaggle CPU session after about five hours, so it isn't
  in the results. `kaggle/marker_kaggle.py` and `run_marker.py` are ready to rerun it, ideally on a
  GPU session.
- **Qwen3-VL-8B** ran the table prompt only, in a session of its own (the first attempt ran out of
  GPU memory because Qwen2.5-VL was still loaded). The records prompt wasn't run for it.

## Things to know when reading the numbers

- Several tools read the hidden duplicate tables in four of the MDPI papers too: pdfplumber,
  Camelot and Docling all put two values in each cell on those pages.
- Qwen2.5-VL answered `NO TABLES` on two pages that have one (Yao 2016 page 3 and the 1975 NASA
  table), and every value it did return was correct. Its prompt asked for HTML; it answered in
  markdown, which the scorer reads too.
- With 10 documents the intervals are wide. A difference whose paired interval crosses zero isn't
  a difference this data can show.
- Seconds per page aren't comparable across rows: some ran on a laptop CPU, some on Kaggle GPUs,
  and the GPU models processed four pages at a time.
