# TableTrace: make data | make run | make eval | make all
PY ?= .venv/bin/python
export HF_HUB_DISABLE_PROGRESS_BARS = 1

.PHONY: all data model run eval test labels litx review ocr-model hf-release clean

all: data model run eval

## download the ten documents (checksummed) and build the image-only PDFs
data:
	$(PY) scripts/download_papers.py
	$(PY) scripts/make_image_only.py

## fetch the fine-tuned Tesseract model if the clone only has its Git LFS pointer (SHA-256 checked)
model:
	$(PY) scripts/fetch_model.py

## run every experiment (stock OCR, fine-tuned OCR, text-layer oracle) into outputs/runs/
run:
	$(PY) -m tabletrace run

## score the runs and write reports/results.md, reports/metrics.json, reports/figures/
eval:
	$(PY) -m tabletrace eval

test:
	$(PY) -m pytest tests

## regenerate labels/ground_truth.csv from the typed transcriptions (and rerun its consistency checks)
labels:
	$(PY) labels/transcriptions.py

## cross-check the labels against LitXAlloy (writes labels/litxalloy_reference.csv, reports/label_qa.md)
litx:
	$(PY) scripts/build_litxalloy_reference.py

## rebuild the fine-tuned Tesseract model from scratch (about 30 minutes on a laptop CPU)
ocr-model:
	$(PY) training/make_ocr_data.py --out data/ocr_training/lines --n 10000 --seed 11
	bash training/finetune_tesseract.sh

clean:
	rm -rf outputs

## build outputs/label_review/index.html: each labeled table next to its labels, for a second pass
review:
	$(PY) scripts/label_review.py

## build the Hugging Face dataset and model folders in outputs/hf (printed commands upload them)
hf-release:
	$(PY) scripts/build_hf_release.py
