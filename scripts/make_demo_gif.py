"""Record `python -m tabletrace extract` on one scanned page as a short GIF.

The command is run for real and its output is drawn line by line, so the GIF shows exactly
what the CLI prints. Writes reports/figures/demo.gif.

Usage: python scripts/make_demo_gif.py [--paper billinghurst1992 --page 14]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BG, BAR, INK, INK_2, MUTED = "#fcfcfb", "#ecebe6", "#0b0b0b", "#3d3c39", "#898781"
REVIEW, ACCEPT = "#9a5b00", "#0f7f58"  # status colors, always next to the word itself
MONO = ["/System/Library/Fonts/Menlo.ttc", "/Library/Fonts/Courier New.ttf", "DejaVuSansMono.ttf"]


def mono(size: int) -> ImageFont.FreeTypeFont:
    for path in MONO:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="billinghurst1992")
    ap.add_argument("--page", type=int, default=14)
    args = ap.parse_args()

    command = f"python -m tabletrace extract {args.paper} --page {args.page}"
    env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    out = subprocess.run([sys.executable, "-m", "tabletrace", "extract", args.paper, "--page", str(args.page)],
                         capture_output=True, text=True, cwd=ROOT, env=env, check=True).stdout.rstrip().splitlines()

    font = mono(15)
    char_w = font.getlength("M")
    width = int(max(len(line) for line in out + ["$ " + command]) * char_w) + 56
    lh, top = 23, 52
    height = top + lh * (len(out) + 2) + 18

    def frame(typed: str, shown: int, cursor: bool) -> Image.Image:
        img = Image.new("RGB", (width, height), BG)
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, width, 30), fill=BAR)
        for i, c in enumerate(("#e0443e", "#dea123", "#1aab29")):
            d.ellipse((14 + 20 * i, 9, 26 + 20 * i, 21), fill=c)
        d.text((28, top), "$", font=font, fill=MUTED)
        d.text((28 + 2 * char_w, top), typed + ("_" if cursor else ""), font=font, fill=INK)
        for i, line in enumerate(out[:shown]):
            y = top + lh * (i + 1)
            x = 28
            for word, color in ((" REVIEW", REVIEW), (" accept", ACCEPT)):
                if line.endswith(word):
                    d.text((x, y), line[: -len(word)], font=font, fill=INK_2)
                    d.text((x + char_w * (len(line) - len(word)), y), word, font=font, fill=color)
                    break
            else:
                d.text((x, y), line, font=font, fill=INK if i < 3 or i == len(out) - 1 else INK_2)
        return img

    frames, durations = [], []
    for n in range(0, len(command) + 1, 2):
        frames.append(frame(command[:n], 0, True))
        durations.append(60)
    frames.append(frame(command, 0, False))
    durations.append(900)  # the pipeline runs
    for k in range(1, len(out) + 1):
        frames.append(frame(command, k, False))
        durations.append(650 if k <= 3 else 380)
    durations[-1] = 4500
    target = ROOT / "reports" / "figures" / "demo.gif"
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {target.relative_to(ROOT)}: {len(frames)} frames, {sum(durations) / 1000:.1f} s, {width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
