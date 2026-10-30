"""GPU baselines, run on Kaggle's free T4 x2 machines: Nougat, olmOCR-2 and Qwen VL models.

Reads the page images from the private dataset made by prepare_upload.py and writes one JSON
file per page and tool to /kaggle/working/raw/<tool>/, in the format of baselines/common.py.
The text of each answer is stored as is, and all parsing happens later in the main repo.

Which models run is set by the MODELS line below, so the same script can be pushed as two
kernels that run at the same time. Everything uses greedy decoding in float16 (T4s have no
bfloat16), with the vision encoders in float32, where float16 can overflow.
"""

import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

MODELS = os.environ.get("TT_MODELS", "nougat,olmocr2").split(",")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "accelerate>=1.1",
                "python-Levenshtein", "nltk"], check=True)  # the last two: Nougat's post-processing

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList  # noqa: E402

DATA = os.path.dirname(glob.glob("/kaggle/input/**/pages.json", recursive=True)[0])
OUT = "/kaggle/working/raw"
PAGES = json.load(open(os.path.join(DATA, "pages.json")))
START = time.time()
BATCH = 4

TABLE_PROMPT = (
    "Transcribe every table on this page.\n\n"
    "For each table, first write its caption on its own line exactly as printed (for example "
    "\"Table 3. Mechanical properties of ...\"), then the table as one HTML <table>. Use <th> for "
    "header cells, and colspan or rowspan for merged cells. Copy each cell exactly as printed, "
    "including ±, units, footnote marks and words such as NA. Write subscripts inline "
    "(Nb1.5TaTiZr0.5). Don't add, compute, convert or summarize anything. If there is no table "
    "on the page, answer NO TABLES."
)
RECORDS_PROMPT = (
    "List every mechanical property value printed in the tables on this page, for these four "
    "properties only: Vickers hardness, yield strength (including 0.2% proof stress), ultimate "
    "tensile or compressive strength, and elongation or fracture strain.\n\n"
    "Answer with a JSON array and nothing else, one object per value, with these keys:\n"
    "- \"material\": the sample name exactly as printed in its row or column\n"
    "- \"property\": one of \"vickers_hardness\", \"yield_strength\", \"ultimate_strength\", \"elongation\"\n"
    "- \"value\": the number as printed, without its ± uncertainty\n"
    "- \"unit\": the unit as printed, for example \"HV\", \"MPa\", \"GPa\", \"ksi\" or \"%\"\n"
    "- \"test_temperature\": the test temperature with its unit as printed (for example \"-423 F\" "
    "or \"650 C\"), or null if the table doesn't give one\n"
    "- \"table\": the printed table number, or null\n\n"
    "Leave out values marked as calculated, predicted or theoretical, and averages. If there are "
    "none, answer []."
)
# olmocr/prompts/prompts.py, build_no_anchoring_v4_yaml_prompt (olmocr 0.4.27)
OLMOCR_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


def save(tool, version, page, seconds, text, settings):
    d = os.path.join(OUT, tool)
    os.makedirs(d, exist_ok=True)
    doc = {"tool": tool, "version": version, "paper_id": page["paper_id"], "page": page["page"],
           "seconds": round(seconds, 2), "settings": settings, "tables": [], "text": text}
    with open(os.path.join(d, f"{page['paper_id']}_p{page['page']}.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


def done(tool, page):
    return os.path.exists(os.path.join(OUT, tool, f"{page['paper_id']}_p{page['page']}.json"))


class StopOnLoop(StoppingCriteria):
    """Stop a sequence that has started repeating the same block of text."""

    def __init__(self, tokenizer, prompt_len):
        self.tok, self.prompt_len = tokenizer, prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        stop = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if input_ids.shape[1] % 64:
            return stop
        for i in range(input_ids.shape[0]):
            tail = self.tok.decode(input_ids[i, max(self.prompt_len, input_ids.shape[1] - 600):], skip_special_tokens=True)
            if re.search(r"(.{20,300}?)\1{5,}$", tail, re.S):
                stop[i] = True
        return stop


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def drop_hf_cache(repo):
    path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}")
    shutil.rmtree(path, ignore_errors=True)


def run_nougat():
    from transformers import NougatProcessor, VisionEncoderDecoderModel

    repo = "facebook/nougat-base"
    processor = NougatProcessor.from_pretrained(repo)
    model = VisionEncoderDecoderModel.from_pretrained(repo, torch_dtype=torch.float16).to("cuda:0").eval()
    todo = [p for p in PAGES if not done("nougat", p)]
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        images = [Image.open(os.path.join(DATA, p["img_page"])).convert("RGB") for p in batch]
        t0 = time.time()
        pixel_values = processor(images, return_tensors="pt").pixel_values.to("cuda:0", torch.float16)
        with torch.inference_mode():
            out = model.generate(pixel_values, min_length=1, max_new_tokens=3584, num_beams=1, do_sample=False,
                                 bad_words_ids=[[processor.tokenizer.unk_token_id]])
        texts = processor.batch_decode(out, skip_special_tokens=True)
        secs = (time.time() - t0) / len(batch)
        for p, t in zip(batch, texts):
            save("nougat", repo, p, secs, processor.post_process_generation(t, fix_markdown=False),
                 {"image": "1600 px longest side, resized by the processor", "max_new_tokens": 3584})
        log(f"nougat {[p['paper_id'] + '_p' + str(p['page']) for p in batch]} {secs:.1f}s/page")
    free(model, processor)
    drop_hf_cache(repo)


def load_qwen(repo, processor_repo=None):
    from transformers import AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(processor_repo or repo)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        repo, torch_dtype=torch.float16, device_map="auto", max_memory={0: "12GiB", 1: "14GiB"}).eval()
    visual = getattr(model, "visual", None) or getattr(model.model, "visual", None)
    if visual is not None:
        visual.to(torch.float32)
    return model, processor


def run_chat(tool, repo, prompt, image_key, max_new_tokens, processor_repo=None, text_first=True, model=None,
             processor=None, settings=None):
    own = model is None
    if own:
        model, processor = load_qwen(repo, processor_repo)
    todo = [p for p in PAGES if not done(tool, p)]
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        images = [Image.open(os.path.join(DATA, p[image_key])).convert("RGB") for p in batch]
        content = [{"type": "text", "text": prompt}, {"type": "image"}] if text_first else \
            [{"type": "image"}, {"type": "text", "text": prompt}]
        chat = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True)
        t0 = time.time()
        try:
            inputs = processor(text=[chat] * len(batch), images=images, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            n = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     stopping_criteria=StoppingCriteriaList([StopOnLoop(processor.tokenizer, n)]))
            texts = processor.batch_decode(out[:, n:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            log(f"{tool}: out of memory on a batch of {len(batch)}, one page at a time")
            free()
            texts = []
            for img in images:
                inputs = processor(text=[chat], images=[img], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                n = inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         stopping_criteria=StoppingCriteriaList([StopOnLoop(processor.tokenizer, n)]))
                texts.append(processor.batch_decode(out[:, n:], skip_special_tokens=True)[0])
                free()
        secs = (time.time() - t0) / len(batch)
        for p, t in zip(batch, texts):
            save(tool, repo, p, secs, t, {"prompt": prompt, "image": image_key, "max_new_tokens": max_new_tokens,
                                          "decoding": "greedy, float16, T4 x2", **(settings or {})})
        log(f"{tool} {[p['paper_id'] + '_p' + str(p['page']) for p in batch]} {secs:.1f}s/page; "
            f"first answer starts: {texts[0][:160]!r}")
    return model, processor


def main():
    log(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
        f"gpus {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, models {MODELS}")
    for name in MODELS:
        try:
            if name == "nougat":
                run_nougat()
            elif name == "olmocr2":
                repo = "allenai/olmOCR-2-7B-1025"
                m, pr = run_chat("olmocr2", repo, OLMOCR_PROMPT, "img_olmocr", 6000,
                                 processor_repo="Qwen/Qwen2.5-VL-7B-Instruct",
                                 settings={"note": "olmOCR's own prompt and 1288 px render, greedy instead of temperature 0.1"})
                free(m, pr)
                drop_hf_cache(repo)
            elif name in ("qwen25vl", "qwen3vl"):
                repo = {"qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct", "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct"}[name]
                m, pr = run_chat(f"{name}_tables", repo, TABLE_PROMPT, "img_page", 8192)
                if time.time() - START < 8 * 3600:
                    run_chat(f"{name}_records", repo, RECORDS_PROMPT, "img_page", 8192, model=m, processor=pr)
                free(m, pr)
                drop_hf_cache(repo)
        except Exception:
            log(f"{name} failed:\n{traceback.format_exc()}")
            free()
    log("finished")


if __name__ == "__main__":
    main()
