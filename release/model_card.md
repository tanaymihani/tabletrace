---
license: apache-2.0
language:
- en
library_name: tesseract
tags:
- ocr
- tesseract
- lstm
- scientific-tables
- materials-science
---

# Tesseract eng_tt: English plus the symbols in materials property tables

A fine-tuned version of Tesseract's `tessdata_best/eng` LSTM model that can output `±`, `σ`, `ε`
and `µ`, which the stock English model doesn't have in its character set. Stock Tesseract reads
"value ± uncertainty" cells as `422+75`, `534404` or `1148`, and σ or ε headers as `oy`, `6,` or `©`.
It was trained for [TableTrace](https://github.com/tanaymihani/tabletrace), a pipeline that
extracts mechanical properties from tables in papers, patents and scanned reports.

## Use

```bash
tesseract table.png out -l eng_tt --tessdata-dir /path/to/this/folder --psm 6
```

It's a drop-in `traineddata` file for Tesseract 5 (LSTM engine).

## Training

- **Base:** `tessdata_best/eng` (float weights), continued with `lstmtraining` for 6,000 iterations
  at learning rate 0.0002.
- **Data:** 10,000 synthetic text lines rendered by
  [`training/make_ocr_data.py`](https://github.com/tanaymihani/tabletrace/blob/main/training/make_ocr_data.py):
  random alloy formulas with real subscripts, values with `±`, property headers with σ and ε, ruled
  table rows and scan noise, plus 30% ordinary English prose. None of it comes from the documents
  the model was evaluated on.
- **Character set:** the base model's set merged with every character in the training text, so
  the dictionary, punctuation and number patterns of the base model are kept.
- **Version history:** a first version trained on table content only read `±` perfectly but forgot
  ordinary English ("Alloy" became `A10Yy`), so this one mixes in prose and keeps the word list. A
  third version with bold and subscripted header lines had lower character error on calibration
  tables but lower extraction F1, so this one was kept.
- The full recipe is [`training/finetune_tesseract.sh`](https://github.com/tanaymihani/tabletrace/blob/main/training/finetune_tesseract.sh)
  (about 35 minutes on a laptop CPU).

## Results

On the table regions of three journal papers rendered to 300 DPI images:

| | Stock `eng` | `eng_tt` |
|---|---|---|
| Character error rate, table regions | 27.9% | 18.8% |
| Word error rate, table regions | 52.0% | 29.6% |
| Character error rate, whole pages | 7.5% | 6.9% |

In the full extraction pipeline it raised strict F1 on the calibration OCR documents from 0.251 to
0.399. On the four held-out OCR documents the effect was mixed: better on two (0.09 to 0.33, and
0.28 to 0.47) and slightly worse on two (0.80 to 0.70, and 0.87 to 0.83).

## Limits

It's tuned for tables in English-language materials papers and reports. It still misreads some
typewritten digits on 1970s scans, drops letters from alloy names on poor-quality pages, and the
evaluation above covers three papers, so treat the error rates as rough.

## License

Apache 2.0, the license of the tessdata_best model it's based on.
