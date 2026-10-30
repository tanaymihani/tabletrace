---
license: cc-by-4.0
language:
- en
pretty_name: TableTrace alloy property tables
size_categories:
- n<1K
task_categories:
- image-to-text
- table-question-answering
tags:
- materials-science
- table-extraction
- document-ai
- ocr
- information-extraction
- high-entropy-alloys
configs:
- config_name: default
  data_files:
  - split: test
    path: data/ground_truth.csv
---

# TableTrace alloy property tables

514 mechanical property values (Vickers hardness, yield strength, ultimate strength, elongation)
transcribed from 20 tables in 10 documents: seven open-access journal papers, a US patent
application and two NASA technical reports, two of them scanned typewriter pages. Each value is
tied to its page, table, sample name and test temperature, so it can score any system that turns
PDF tables into records. It's the evaluation set behind
[TableTrace](https://github.com/tanaymihani/tabletrace), where the scorer, the baselines and the
labeling rules live.

## What's here

| Path | What it is |
|---|---|
| `data/ground_truth.csv` | the 514 labels, one row per value |
| `data/papers.csv` | the 10 documents: identifier, license, source URL, SHA-256, which tables were labeled |
| `transcriptions.py` | every labeled table typed in as a small grid; `ground_truth.csv` is generated from it |
| `pages/` | a 200 DPI render of each of the 16 labeled pages |

Columns of `ground_truth.csv`:

| Column | Meaning |
|---|---|
| `label_id` | `paper-Ttable-nnn` |
| `paper_id`, `page`, `table_index` | where the value is printed (page numbers are 1-based in the PDF) |
| `material` | the sample name as printed in the row or column label (citations and footnote marks dropped, subscripts inline: `Nb1.5TaTiZr0.5`) |
| `property` | `vickers_hardness`, `yield_strength`, `ultimate_strength` or `elongation` |
| `value`, `uncertainty`, `unit` | in canonical units: HV, MPa, % |
| `qualifier` | `=`, or `>` for a printed bound (`> 50`) |
| `raw_text`, `raw_unit` | the cell and unit exactly as printed |
| `test_temperature_c` | when the table prints one; blank when it only says room temperature |
| `loading`, `condition` | tension or compression, and the processing state when the table gives it (as-cast, CC, DS, replicate 2) |
| `source` | `text_layer` if the page has a usable text layer, `ocr` if it has to be read from pixels |
| `notes` | anything unusual about the cell (footnote marks, literature values, known errors in the source) |

By property: 183 elongation, 155 ultimate strength, 91 hardness, 85 yield strength. 320 values have a
test temperature and 47 have a printed uncertainty.

## Documents and licenses

| Document | Venue | Identifier | License | Labeled tables (table@page) |
|---|---|---|---|---|
| Chen et al. 2014, *Microstructures and Crackling Noise of AlxNbTiMoV High Entropy Alloys* | Entropy 16(2) 870 | doi:10.3390/e16020870 | CC BY 3.0 | 3@7 |
| Yao et al. 2016, *MoNbTaV Medium-Entropy Alloy* | Entropy 18(5) 189 | doi:10.3390/e18050189 | CC BY 4.0 | 1@3 |
| Tseng et al. 2018, *Effects of Mo, Nb, Ta, Ti, and Zr on Mechanical Properties of Equiatomic Hf-Mo-Nb-Ta-Ti-Zr Alloys* | Entropy 21(1) 15 | doi:10.3390/e21010015 | CC BY 4.0 | 3@7, 4@7, 7@11 |
| Zýka et al. 2019, *Microstructure and Room Temperature Mechanical Properties of Different 3 and 4 Element Medium Entropy Alloys from HfNbTaTiZr System* | Entropy 21(2) 114 | doi:10.3390/e21020114 | CC BY 4.0 | 3@4, 7@14 |
| Haas et al. 2019, *Microstructure and Mechanical Properties of Precipitate Strengthened High Entropy Alloy Al10Co25Cr8Fe15Ni36Ti6 with Additions of Hafnium and Molybdenum* | Entropy 21(2) 169 | doi:10.3390/e21020169 | CC BY 4.0 | 2@6 |
| Rickman et al. 2019, *Materials informatics for the screening of multi-principal elements and high-entropy alloys* | Nature Communications 10 2618 | doi:10.1038/s41467-019-10533-1 | CC BY 4.0 | 2@5 |
| Sanchez et al. 2019, *Design, Microstructure and Mechanical Properties of Cast Medium Entropy Aluminium Alloys* | Scientific Reports 9 6792 | doi:10.1038/s41598-019-43329-w | CC BY 4.0 | 3@9 |
| Yeh 2002, *High-entropy multielement alloys* | US patent application | US 2002/0159914 A1 | public domain (US patent document) | 1@3, 1@4, 4@5, 5@5, 6@5, 6@6, 7@6 |
| Gray 1975, *Tensile and creep properties of titanium-vanadium, titanium-molybdenum, and titanium-niobium alloys* | NASA TM X-3216 | NTRS 19750012451 | US government work, cleared for public use | 1@13 |
| Billinghurst et al. 1992, *Tensile properties of cast titanium alloys: Titanium-6Al-4V ELI and Titanium-5Al-2.5Sn ELI* | NASA TP-3288 | NTRS 19930000884 | US government work, cleared for public use | 2@13, 3@14, 4@14 |

The page images are reproduced from these documents under their licenses. The journal articles
are the authors' work under Creative Commons Attribution; the labels and the transcriptions are
released under CC BY 4.0. Please cite the original articles along with this dataset when you use
the page images.

## How it was made

- **Choice of documents.** The seven journal papers are the ones in LitXAlloy (the alloy part of
  Radical AI's LitXBench, 2026) that report these four properties in tables, so a second,
  independent curation of the same papers exists. The patent and the two NASA reports add the
  other two kinds of documents a materials extraction system has to read, and real scans.
- **Transcription.** Every table was typed from a rendered page image (300 DPI for the scans,
  with zoomed crops for dense or damaged cells) into `transcriptions.py`, before any extraction
  output existed. Canonical units are computed with a small conversion table kept separate from
  the extraction code, so a bug in one can't hide in the other.
- **Rules.** Values marked as calculated, predicted or theoretical are left out, as are summary
  rows (averages), empty cells and other properties. Literature values printed inside a table are
  included and marked in `notes`. Bounds keep their qualifier. Footnoted values keep their number
  and the footnote is explained in `notes`. The full rule table is in
  [labels/README.md](https://github.com/tanaymihani/tabletrace/blob/main/labels/README.md).

## Checks

| Check | Values covered | Result |
|---|---|---|
| LitXAlloy curates the same seven journal papers. Matching is one-to-one on property and value within 1%. | 102 LitXAlloy values | all 102 agree, and all 74 LitXAlloy values whose source cites a table are in these labels |
| The MPEA dataset (Borg et al. 2020) compiles the same seven papers | 124 MPEA values | 113 agree; the other 11 come from figures or text, or are fracture strains rounded to whole percent |
| NASA TM X-3216 prints every UTS twice, in MN/m² and ksi | 97 values | 96 of 97 agree; the other is an error in the report itself (530 MN/m² next to 79 ksi), labeled as printed |
| NASA TP-3288 prints an "Averages" row for each group | 56 values | 13 of 15 group averages reproduce; the other two are errors in the source, confirmed at 450 DPI |
| A second, blind pass over 50 randomly chosen values | 50 values | see "Label noise" below |

## Label noise

A month after the first pass, 50 labels were drawn at random and listed with their sample name,
property, temperature and condition but not their values. Each value was found and read again
from a fresh render of the page before comparing. All 50 match the printed number, so the
transcription error rate is at most 5.8% (one-sided 95% Clopper-Pearson bound). The same person
did both passes, so this checks reading and locating values, not the labeling rules themselves.
The sample and the readings are in
[labels/second_pass](https://github.com/tanaymihani/tabletrace/tree/main/labels/second_pass).

## Intended use and limits

Use it to score table extraction from scientific and technical PDFs: text-layer pages (4
documents, 122 values), born-digital pages rendered to images with the text layer removed (3
documents, 63 values; the original PDFs make an OCR-free control), and real scans (3 documents,
329 values). It's small. One NASA report holds 194 of the 514 values, values from one table are
correlated, and every number derived from it should come with an interval over documents rather
than over values. It covers four mechanical properties only, and every label was typed by one
person, so read the checks above before trusting a single value.

Four of the five MDPI PDFs (all but Chen 2014) hide a shifted duplicate of their tables in the text
layer, inside a clipped form. That's a property of the source files, and a good test for any
text-layer reader; see the TableTrace README.

## Citation

```
@misc{mihani2026tabletrace,
  title  = {TableTrace: alloy property tables from papers, patents and scanned reports},
  author = {Mihani, Tanay},
  year   = {2026},
  url    = {https://github.com/tanaymihani/tabletrace}
}
```
