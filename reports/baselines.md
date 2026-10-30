# Other tools on the same labels

Every system gets the 16 labeled pages that TableTrace reads, and its tables go through
TableTrace's interpretation, normalization and Hungarian matching, so these numbers measure
finding tables and reading their cells. See baselines/README.md for how each tool was run.
Intervals come from resampling the 10 documents with replacement (10,000 times), so a
document counts once however many labels it has. The "records" rows skip TableTrace's
interpretation: the model returns material, property, value and unit itself, and only the
unit conversion is shared.

### Strict F1 [95% bootstrap interval over documents]

| System | Held-out documents | All documents | Text layer | Rendered pages | Real scans |
|---|---|---|---|---|---|
| TableTrace (this repo) | 0.715 [0.50, 0.91] | 0.631 [0.47, 0.89] | 1.000 [1.00, 1.00] | 0.444 [0.00, 0.70] | 0.506 [0.42, 0.83] |
| pdfplumber, ruling lines | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| pdfplumber, word alignment | 0.008 [0.00, 0.02] | 0.004 [0.00, 0.01] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.006 [0.00, 0.03] |
| Camelot lattice | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| Camelot stream | 0.261 [0.00, 0.44] | 0.303 [0.09, 0.56] | 0.612 [0.00, 0.77] | 0.000 [0.00, 0.00] | 0.204 [0.00, 0.50] |
| Docling | 0.452 [0.10, 0.77] | 0.456 [0.26, 0.70] | 0.588 [0.26, 0.77] | 0.190 [0.00, 0.80] | 0.469 [0.19, 0.92] |
| GROBID | 0.432 [0.06, 0.71] | 0.371 [0.10, 0.64] | 0.741 [0.67, 0.88] | 0.000 [0.00, 0.00] | 0.157 [0.00, 0.50] |
| Nougat base | 0.662 [0.33, 0.92] | 0.401 [0.12, 0.74] | 0.492 [0.02, 0.95] | 0.764 [0.00, 1.00] | 0.269 [0.00, 0.74] |
| olmOCR-2 7B | 0.662 [0.49, 0.97] | 0.529 [0.26, 0.85] | 0.943 [0.90, 1.00] | 0.700 [0.00, 1.00] | 0.258 [0.00, 0.48] |
| Qwen2.5-VL 7B, tables | 0.784 [0.49, 0.96] | 0.616 [0.26, 0.94] | 0.870 [0.45, 1.00] | 0.950 [0.80, 1.00] | 0.383 [0.00, 0.96] |
| Qwen2.5-VL 7B, records | 0.288 [0.18, 0.55] | 0.333 [0.23, 0.54] | 0.323 [0.16, 0.61] | 0.740 [0.56, 0.86] | 0.233 [0.09, 0.27] |
| Qwen3-VL 8B, tables | 0.874 [0.51, 1.00] | 0.682 [0.31, 0.98] | 1.000 [1.00, 1.00] | 0.466 [0.00, 1.00] | 0.569 [0.00, 1.00] |

### Lenient F1 [95% bootstrap interval over documents]

| System | Held-out documents | All documents | Text layer | Rendered pages | Real scans |
|---|---|---|---|---|---|
| TableTrace (this repo) | 0.715 [0.50, 0.91] | 0.774 [0.64, 0.90] | 1.000 [1.00, 1.00] | 0.533 [0.33, 0.70] | 0.716 [0.47, 0.83] |
| pdfplumber, ruling lines | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| pdfplumber, word alignment | 0.023 [0.00, 0.06] | 0.011 [0.00, 0.04] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.017 [0.00, 0.08] |
| Camelot lattice | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| Camelot stream | 0.261 [0.00, 0.44] | 0.373 [0.15, 0.56] | 0.612 [0.00, 0.77] | 0.000 [0.00, 0.00] | 0.314 [0.00, 0.50] |
| Docling | 0.466 [0.11, 0.79] | 0.591 [0.31, 0.75] | 0.619 [0.26, 0.77] | 0.238 [0.00, 1.00] | 0.653 [0.19, 0.92] |
| GROBID | 0.432 [0.06, 0.71] | 0.388 [0.10, 0.67] | 0.784 [0.67, 0.88] | 0.000 [0.00, 0.00] | 0.157 [0.00, 0.50] |
| Nougat base | 0.676 [0.35, 0.94] | 0.410 [0.13, 0.75] | 0.525 [0.05, 0.95] | 0.764 [0.00, 1.00] | 0.269 [0.00, 0.74] |
| olmOCR-2 7B | 0.671 [0.50, 0.98] | 0.585 [0.30, 0.96] | 1.000 [1.00, 1.00] | 0.950 [0.80, 1.00] | 0.263 [0.00, 0.48] |
| Qwen2.5-VL 7B, tables | 0.784 [0.49, 0.96] | 0.616 [0.26, 0.94] | 0.870 [0.45, 1.00] | 0.950 [0.80, 1.00] | 0.383 [0.00, 0.96] |
| Qwen2.5-VL 7B, records | 0.288 [0.18, 0.55] | 0.340 [0.24, 0.54] | 0.323 [0.16, 0.61] | 0.740 [0.56, 0.86] | 0.245 [0.09, 0.29] |
| Qwen3-VL 8B, tables | 0.901 [0.58, 1.00] | 0.711 [0.34, 1.00] | 1.000 [1.00, 1.00] | 0.621 [0.12, 1.00] | 0.582 [0.00, 1.00] |

### TableTrace minus each system, strict F1 (paired bootstrap, same resampled documents)

| System | Held-out documents | All documents | Text layer | Rendered pages | Real scans |
|---|---|---|---|---|---|
| pdfplumber, ruling lines | +0.715 [+0.50, +0.91] | +0.631 [+0.47, +0.89] | +1.000 [+1.00, +1.00] | +0.444 [+0.00, +0.70] | +0.506 [+0.42, +0.83] |
| pdfplumber, word alignment | +0.708 [+0.50, +0.90] | +0.627 [+0.47, +0.89] | +1.000 [+1.00, +1.00] | +0.444 [+0.00, +0.70] | +0.500 [+0.42, +0.81] |
| Camelot lattice | +0.715 [+0.50, +0.91] | +0.631 [+0.47, +0.89] | +1.000 [+1.00, +1.00] | +0.444 [+0.00, +0.70] | +0.506 [+0.42, +0.83] |
| Camelot stream | +0.454 [+0.11, +0.79] | +0.328 [+0.18, +0.61] | +0.388 [+0.23, +1.00] | +0.444 [+0.00, +0.70] | +0.302 [-0.03, +0.83] |
| Docling | +0.263 [-0.23, +0.68] | +0.174 [-0.08, +0.54] | +0.412 [+0.23, +0.74] | +0.254 [-0.80, +0.70] | +0.037 [-0.45, +0.64] |
| GROBID | +0.284 [+0.04, +0.70] | +0.259 [+0.12, +0.46] | +0.259 [+0.12, +0.33] | +0.444 [+0.00, +0.70] | +0.349 [-0.03, +0.83] |
| Nougat base | +0.054 [-0.25, +0.48] | +0.229 [-0.08, +0.56] | +0.508 [+0.05, +0.98] | -0.319 [-0.47, +0.00] | +0.237 [-0.27, +0.72] |
| olmOCR-2 7B | +0.053 [-0.21, +0.25] | +0.102 [-0.05, +0.25] | +0.057 [+0.00, +0.10] | -0.256 [-0.47, +0.00] | +0.248 [-0.01, +0.42] |
| Qwen2.5-VL 7B, tables | -0.069 [-0.44, +0.37] | +0.014 [-0.28, +0.29] | +0.130 [+0.00, +0.55] | -0.506 [-1.00, -0.30] | +0.123 [-0.49, +0.59] |
| Qwen2.5-VL 7B, records | +0.427 [+0.00, +0.70] | +0.297 [+0.08, +0.58] | +0.677 [+0.39, +0.84] | -0.296 [-0.86, +0.14] | +0.273 [+0.15, +0.74] |
| Qwen3-VL 8B, tables | -0.159 [-0.44, +0.25] | -0.051 [-0.30, +0.21] | +0.000 [+0.00, +0.00] | -0.021 [-0.67, +0.70] | -0.063 [-0.53, +0.42] |

### What each run returned

| System | Version | Input | Ran on | Seconds per page | Tables | Values | Correct (strict) | Quarantined | Finest location |
|---|---|---|---|---|---|---|---|---|---|
| TableTrace (this repo) | final run | PDF page | M2 CPU | n/a | n/a | 409 | 291 | n/a | cell |
| pdfplumber, ruling lines | 0.11.10 | PDF text layer | M2 CPU | 0.1 | 2 | 0 | 0 | 0 | table |
| pdfplumber, word alignment | 0.11.10 | PDF text layer | M2 CPU | 0.1 | 13 | 43 | 1 | 409 | table |
| Camelot lattice | 2.0.0 | PDF text layer | M2 CPU | 0.4 | 3 | 0 | 0 | 3 | table |
| Camelot stream | 2.0.0 | PDF text layer | M2 CPU | 0.0 | 24 | 259 | 117 | 130 | table |
| Docling | 2.145.0 | PDF page | M2 CPU | 24.1 | 25 | 437 | 217 | 81 | cell |
| GROBID | 0.9.1 | PDF text layer | Kaggle CPU | 2.9 | 16 | 197 | 132 | 63 | table |
| Nougat base | facebook/nougat-base | page image | Kaggle T4 | 6.1 | 21 | 199 | 143 | 61 | page |
| olmOCR-2 7B | allenai/olmOCR-2-7B-1025 | page image | Kaggle T4 x2 | 124.5 | 24 | 299 | 215 | 61 | page |
| Qwen2.5-VL 7B, tables | Qwen/Qwen2.5-VL-7B-Instruct | page image | Kaggle T4 x2 | 46.8 | 18 | 229 | 229 | 1 | page |
| Qwen2.5-VL 7B, records | Qwen/Qwen2.5-VL-7B-Instruct | page image | Kaggle T4 x2 | 236.7 | 0 | 428 | 157 | 28 | page |
| Qwen3-VL 8B, tables | Qwen/Qwen3-VL-8B-Instruct | page image | Kaggle T4 x2 | 169.6 | 25 | 310 | 281 | 10 | page |

## A vision-language model as a third route (olmOCR-2 7B)

TableTrace's confidence score already compares each value with a second reading of the same cell (OCR
on text-layer pages, the PDF's own OCR layer on scans). Here the VLM's reading of the page is added
as a third. Two ways of using it were fixed before looking at any result: *penalize* (a value the VLM
reads differently, or doesn't find, costs what it costs for a second OCR reading) and *boost*
(agreement closes half the gap to 1, nothing else changes). Neither adds a tuned weight.

| Confidence | AUROC, calibration papers | AUROC, held-out papers | Error among accepted at alpha = 10% (held-out) | Accepted (held-out) | Errors caught | Splits where 10% holds | Splits where 5% holds |
|---|---|---|---|---|---|---|---|
| without the VLM (TableTrace as published) | 0.728 | 0.692 | 6.2% | 65 of 155 | 14 of 18 | 47% | 76% |
| penalize | 0.679 | 0.679 | 0.0% | 36 of 155 | 18 of 18 | 65% | 45% |
| boost | 0.726 | 0.755 | 5.5% | 73 of 155 | 14 of 18 | 59% | 57% |

The VLM agreed with 134 of TableTrace's values, disagreed with 47 and had no counterpart for 228. TableTrace was wrong on only 4 of the 47 it disagreed with, so the VLM's disagreement mostly marks its own misreadings, and penalizing it hurts. Its agreement is more useful. Chosen on the calibration papers, the published score without the VLM stays the default (the boost ties it there); the boost does better on the held-out papers, which is a reason to test it on more documents, not to switch on this evidence.
