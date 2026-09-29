# First test run

`results.md` and `metrics.json` in this folder are the evaluation exactly as it came out the first
time I ran the pipeline on the test papers. I froze the configuration before that run (fine-tuned OCR
model v2, region OCR, detection threshold 0.8) and I've left these files unchanged.

That run exposed three bugs. I fixed them afterwards and re-checked everything on the calibration
papers before the final run in `../results.md`:

1. The ruled-header detection treated a table's outer border, and the stems of capital letters in a
   tight header row, as vertical rules. A boxed header with no internal rules collapsed into a single
   cell (NASA TP-3288 Table 2).
2. Table Transformer sometimes merged two data rows into one (`590 800` in a single cell, in the
   patent and in NASA TM X-3216). Rows where the cells hold two lines of numbers now get split.
3. Sparse-text OCR lost some `±` signs and short isolated numbers. Those cells now get re-read on
   their own.

One more change touched a test document but can't affect any score. While building the label review
page I noticed a sentence in the body text ("Table 5 shows the effect of boron") being picked up as a
caption. Captions now can't be followed by a lowercase word. Table numbers aren't used in matching.

Strict F1 on the test papers: 0.499 in this run, 0.715 in the final one.
