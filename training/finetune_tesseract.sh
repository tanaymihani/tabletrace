#!/usr/bin/env bash
# Fine-tune Tesseract's English LSTM on synthetic table-cell lines.
#
# The stock model cannot emit ± σ ε µ at all (they are not in its character set),
# and it has never seen subscripted alloy formulas or ruled table cells. This script
# extends the character set and continues training from tessdata_best/eng, which
# is the float model (the default "fast" model is integerized and cannot be trained).
#
# Usage: bash training/finetune_tesseract.sh            (after make_ocr_data.py)
#        ITERATIONS=6000 bash training/finetune_tesseract.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/data/ocr_training"
LINES="$WORK/lines"
OUT="$ROOT/models/tessdata"
MODEL="${MODEL:-eng_tt}"
ITERATIONS="${ITERATIONS:-6000}"
JOBS="${JOBS:-7}"
mkdir -p "$WORK" "$OUT" "$WORK/ckpt"

echo "== base model (tessdata_best, float weights)"
[ -f "$WORK/eng.traineddata" ] || curl -sSL -o "$WORK/eng.traineddata" \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
combine_tessdata -e "$WORK/eng.traineddata" "$WORK/eng.lstm" > /dev/null
combine_tessdata -u "$WORK/eng.traineddata" "$WORK/eng." > /dev/null
# keep the base model's dictionary, punctuation and number patterns in the new language model
dawg2wordlist "$WORK/eng.lstm-unicharset" "$WORK/eng.lstm-word-dawg" "$WORK/eng.wordlist" > /dev/null 2>&1
dawg2wordlist "$WORK/eng.lstm-unicharset" "$WORK/eng.lstm-punc-dawg" "$WORK/eng.punc" > /dev/null 2>&1
dawg2wordlist "$WORK/eng.lstm-unicharset" "$WORK/eng.lstm-number-dawg" "$WORK/eng.numbers" > /dev/null 2>&1

echo "== script data for the unicharset properties (Latin, Greek, Common)"
mkdir -p "$WORK/langdata"
for f in Latin.unicharset Greek.unicharset Common.unicharset Inherited.unicharset radical-stroke.txt; do
  [ -f "$WORK/langdata/$f" ] || curl -sSL -o "$WORK/langdata/$f" \
    "https://raw.githubusercontent.com/tesseract-ocr/langdata_lstm/main/$f"
done

echo "== line images -> .lstmf ($JOBS jobs)"
find "$LINES" -name '*.png' | sort | \
  xargs -P "$JOBS" -I{} sh -c 'b="${1%.png}"; [ -f "$b.lstmf" ] || tesseract "$1" "$b" --psm 13 lstm.train > /dev/null 2>&1' _ {}
n_lstmf=$(find "$LINES" -name '*.lstmf' | wc -l | tr -d ' ')
echo "   $n_lstmf lstmf files"

echo "== character set: eng + everything in the ground truth (± σ ε µ ° ² ...)"
find "$LINES" -name '*.gt.txt' | sort > "$WORK/gt_files.txt"
xargs unicharset_extractor --output_unicharset "$WORK/tt.unicharset" --norm_mode 1 < "$WORK/gt_files.txt" > /dev/null 2>&1
merge_unicharsets "$WORK/eng.lstm-unicharset" "$WORK/tt.unicharset" "$WORK/merged.unicharset" > /dev/null 2>&1
mkdir -p "$WORK/lang/$MODEL"  # combine_lang_model does not create its output directory
combine_lang_model --input_unicharset "$WORK/merged.unicharset" --script_dir "$WORK/langdata" \
  --words "$WORK/eng.wordlist" --puncs "$WORK/eng.punc" --numbers "$WORK/eng.numbers" \
  --output_dir "$WORK/lang" --lang "$MODEL" > "$WORK/combine_lang_model.log" 2>&1
[ -f "$WORK/lang/$MODEL/$MODEL.traineddata" ] || { echo "combine_lang_model failed, see $WORK/combine_lang_model.log"; exit 1; }
echo "   characters: $(head -1 "$WORK/merged.unicharset") (eng had $(head -1 "$WORK/eng.lstm-unicharset"))"

echo "== train / eval lists (90/10, fixed order)"
find "$LINES" -name '*.lstmf' | sort > "$WORK/all.list"
awk 'NR % 10 != 0' "$WORK/all.list" > "$WORK/train.list"
awk 'NR % 10 == 0' "$WORK/all.list" > "$WORK/eval.list"

echo "== lstmtraining: $ITERATIONS iterations"
lstmtraining \
  --continue_from "$WORK/eng.lstm" \
  --old_traineddata "$WORK/eng.traineddata" \
  --traineddata "$WORK/lang/$MODEL/$MODEL.traineddata" \
  --model_output "$WORK/ckpt/$MODEL" \
  --train_listfile "$WORK/train.list" \
  --eval_listfile "$WORK/eval.list" \
  --learning_rate 0.0002 \
  --max_iterations "$ITERATIONS" \
  --target_error_rate 0.005 2>&1 | tee "$WORK/training.log" | grep -E "At iteration|Finished|best|Eval" || true

echo "== package the model"
lstmtraining --stop_training \
  --continue_from "$WORK/ckpt/${MODEL}_checkpoint" \
  --traineddata "$WORK/lang/$MODEL/$MODEL.traineddata" \
  --model_output "$OUT/$MODEL.traineddata" > /dev/null
echo "wrote $OUT/$MODEL.traineddata ($(du -h "$OUT/$MODEL.traineddata" | cut -f1))"
