"""GROBID 0.9.1 on a Kaggle CPU session: build it, start the service, send it every labeled page.

Each page goes to /api/processFulltextDocument with table coordinates on. The tables in the TEI
answer (<figure type="table">) are saved as rows, with a cell's cols attribute repeated across
the columns it spans, plus the caption (head and figDesc) and the box. The full TEI is kept too.
Writes /kaggle/working/raw/grobid/<paper_id>_p<page>.json.
"""

import glob
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET

import requests

DATA = os.path.dirname(glob.glob("/kaggle/input/**/pages.json", recursive=True)[0])
OUT = "/kaggle/working/raw/grobid"
PAGES = json.load(open(os.path.join(DATA, "pages.json")))
SRC = "/tmp/grobid"
TEI = "{http://www.tei-c.org/ns/1.0}"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def text_of(el):
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def tables_from_tei(tei):
    root = ET.fromstring(tei)
    out = []
    for fig in root.iter(f"{TEI}figure"):
        if fig.get("type") != "table":
            continue
        rows = []
        for row in fig.iter(f"{TEI}row"):
            cells = []
            for cell in row.findall(f"{TEI}cell"):
                span = int(cell.get("cols", "1") or 1)
                cells.extend([text_of(cell)] * max(1, min(span, 50)))
            rows.append(cells)
        caption = " ".join(x for x in (text_of(fig.find(f"{TEI}head")), text_of(fig.find(f"{TEI}figDesc"))) if x)
        bbox = None
        coords = fig.get("coords")
        if coords:
            first = coords.split(";")[0].split(",")
            if len(first) == 5:
                _, x, y, w, h = (float(v) for v in first)
                bbox = [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]
        out.append({"rows": rows, "caption": caption or None, "bbox": bbox})
    return out


def main():
    sh("java -version 2>&1 | head -1 || true")
    if subprocess.run("java -version 2>&1 | grep -q '\"21\\|version \"2[1-9]'", shell=True).returncode:
        sh("apt-get update -qq && apt-get install -y -qq openjdk-21-jdk-headless > /dev/null")
    sh(f"git clone -q --depth 1 --branch 0.9.1 https://github.com/kermitt2/grobid.git {SRC}")
    sh(f"cd {SRC} && ./gradlew clean install -x test --no-daemon -q")
    service = subprocess.Popen(f"cd {SRC} && ./gradlew run --no-daemon -q", shell=True)
    for _ in range(120):
        try:
            if requests.get("http://localhost:8070/api/isalive", timeout=5).text.strip() == "true":
                break
        except requests.RequestException:
            pass
        time.sleep(5)
    else:
        raise SystemExit("GROBID service did not start")
    os.makedirs(OUT, exist_ok=True)
    for p in PAGES:
        pdf = os.path.join(DATA, f"{p['paper_id']}_p{p['page']}.pdf")
        t0 = time.time()
        with open(pdf, "rb") as f:
            r = requests.post("http://localhost:8070/api/processFulltextDocument", files={"input": f},
                              data={"teiCoordinates": "figure", "consolidateHeader": "0",
                                    "consolidateCitations": "0"}, timeout=600)
        seconds = time.time() - t0
        tei = r.text if r.ok else ""
        tables = tables_from_tei(tei) if tei else []
        doc = {"tool": "grobid", "version": "0.9.1", "paper_id": p["paper_id"], "page": p["page"],
               "seconds": round(seconds, 2), "settings": {"endpoint": "processFulltextDocument", "status": r.status_code,
                                                          "device": "Kaggle CPU, 4 cores"},
               "tables": tables, "text": None, "tei": tei}
        with open(os.path.join(OUT, f"{p['paper_id']}_p{p['page']}.json"), "w") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"grobid {p['paper_id']} p{p['page']}: HTTP {r.status_code}, {len(tables)} tables, {seconds:.1f}s", flush=True)
    service.terminate()


if __name__ == "__main__":
    main()
