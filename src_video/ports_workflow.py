"""Render a self-contained, illustrative PORTS routing walkthrough."""

from __future__ import annotations

import argparse
import io
import math
import os
import shutil
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG, PANEL, LINE = "#0c1425", "#141f35", "#2c3d59"
WHITE, MUTED, BLUE, VIOLET, GREEN = "#f2f6ff", "#9fb0cc", "#7dbfff", "#c2a4ff", "#63edb6"
MODELS = ("A", "B")
SWITCH = 25  # hundredths of an illustrative cost unit
PRIMITIVES = ("Open app", "New recipe", "Fill title", "Fill ingredients", "Fill directions", "Save")
PHASES = ("Task", "Initial plan", "Primitives", "Skill library", "DAG + costs", "Best route")
DURATIONS = (5, 6, 6, 8, 12, 8)


@dataclass(frozen=True)
class Edge:
    start: int
    end: int
    name: str
    level: int
    costs: tuple[int, int]


BASE = tuple(Edge(i, i + 1, name, 0, pair) for i, (name, pair) in enumerate(zip(
    PRIMITIVES, ((85, 135), (95, 135), (90, 65), (105, 75), (110, 80), (75, 110)))))
SKILLS = (
    Edge(2, 3, "Name recipe", 1, (75, 55)),
    Edge(0, 2, "Open editor", 2, (110, 180)),
    Edge(2, 5, "Populate recipe", 2, (220, 120)),
    Edge(0, 6, "Create recipe", 3, (420, 450)),
)
EDGES = BASE + SKILLS


def solve(edges=EDGES):
    """Shortest path by (primitive boundary, last model), with backpointers."""
    table = [{None: (0, ())}] + [{} for _ in PRIMITIVES]
    counts = [1] + [0] * len(PRIMITIVES)
    for start, states in enumerate(table[:-1]):
        for edge in (item for item in edges if item.start == start):
            counts[edge.end] += counts[start]
            for previous, (cost, path) in states.items():
                for model, edge_cost in zip(MODELS, edge.costs):
                    total = cost + edge_cost + (SWITCH if previous not in (None, model) else 0)
                    target = table[edge.end]
                    if model not in target or total < target[model][0]:
                        target[model] = total, (*path, (edge, model))
    best_cost, best_path = min(table[-1].values(), key=lambda item: item[0])
    return table, counts[-1], best_cost, best_path


TABLE, COUNT, COST, PATH = solve()


@lru_cache(maxsize=None)
def font(size, bold=False):
    names = ("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")
    for name in names:
        for candidate in (name, str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name)):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


def ease(value):
    x = max(0.0, min(1.0, value))
    return x * x * (3 - 2 * x)


def text(draw, xy, label, size=20, color=WHITE, bold=False, anchor="lt"):
    draw.text(xy, label, font=font(size, bold), fill=color, anchor=anchor)


def card(draw, box, outline=LINE, fill=PANEL, radius=16):
    draw.rounded_rectangle(box, radius, fill, outline, width=2)


def arrow(draw, points, color, width=3, progress=1):
    if progress <= 0:
        return
    point = progress * (len(points) - 1)
    i = min(int(point), len(points) - 2)
    a, b = points[i], points[i + 1]
    f = point - i
    end = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
    draw.line([*points[:i + 1], end], color, width=width)
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    draw.polygon([end, *((end[0] - 11 * math.cos(angle + side * .48),
                         end[1] - 11 * math.sin(angle + side * .48)) for side in (-1, 1))], fill=color)


def shell(draw, phase, heading, caption, elapsed):
    text(draw, (60, 37), "PORTS", 26, GREEN, True)
    text(draw, (160, 41), "ROUTING WALKTHROUGH", 16, MUTED, True)
    text(draw, (1220, 41), "ILLUSTRATIVE EXAMPLE", 15, MUTED, True, "rt")
    for i, label in enumerate(PHASES):
        x = 60 + 195 * i
        active = i == phase
        draw.line((x, 95, x + 178, 95), GREEN if active else LINE, width=4)
        text(draw, (x, 108), f"{i + 1:02d}  {label}", 16, GREEN if active else MUTED, active)
    text(draw, (60, 155), heading, 41, WHITE, True)
    draw.line((60, 631, 1220, 631), LINE, width=1)
    text(draw, (60, 653), caption, 21, MUTED)
    draw.line((0, 716, W * elapsed / sum(DURATIONS), 716), GREEN, width=4)


def primitive_strip(draw, progress, y=290):
    for i, name in enumerate(PRIMITIVES):
        x = 60 + i * 195
        reveal = ease(progress * 7 - i)
        if not reveal:
            continue
        yy = y + round(15 * (1 - reveal))
        card(draw, (x, yy, x + 178, yy + 112), BLUE)
        text(draw, (x + 14, yy + 19), f"P{i + 1}", 18, BLUE, True)
        text(draw, (x + 89, yy + 69), name, 18, WHITE, True, "mm")
        if i:
            arrow(draw, [(x - 15, yy + 55), (x - 4, yy + 55)], BLUE, 2, reveal)


def graph(draw, t, selected=False):
    xs = [110 + 176 * i for i in range(7)]
    y = 457 if not selected else 403
    arches = {SKILLS[1]: 62, SKILLS[2]: 98, SKILLS[3]: 180}
    ready = min(6, int(max(0, t - 4.3) * 1.1)) if not selected else 6
    active = set(edge for edge, _ in PATH) if selected else set()
    for i, edge in enumerate(EDGES):
        progress = (ease((t - i * .52) / .7) if i < 6 else ease((t - 1.2 - (i - 6) * .7) / .8))
        if selected:
            progress = 1 if edge not in active else ease(t * 1.1 - list(e for e, _ in PATH).index(edge) * .45)
        if not progress:
            continue
        x0, x1 = xs[edge.start] + 17, xs[edge.end] - 17
        height = arches.get(edge, -43 if edge.level else 0)
        points = [(x0 + (x1 - x0) * a / 48, y - height * math.sin(math.pi * a / 48))
                  for a in range(49)]
        color = (GREEN if edge in active else LINE) if selected else (VIOLET if edge.level else BLUE)
        arrow(draw, points, color, 5 if edge in active else 2, progress)
        if progress > .92 and (not selected or edge in active):
            mx = (x0 + x1) / 2
            if edge.level:
                label = f"L{edge.level}  {edge.name}"
                text(draw, (mx, y - height - (29 if height > 0 else -9)), label, 17, color, True, "mt")
            elif selected:
                text(draw, (mx, y - 34), edge.name, 17, color, True, "mt")
            if not selected:
                offset = 9 if height > 0 else (32 if height < 0 else -24)
                text(draw, (mx, y - height + offset), f"{edge.costs[0]/100:.2f} / {edge.costs[1]/100:.2f}",
                     15, MUTED, anchor="mt")
    for i, x in enumerate(xs):
        color = GREEN if (selected and i in {0, *(edge.end for edge, _ in PATH)}) else (
            BLUE if not selected and i <= ready else MUTED)
        draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill=BG, outline=color, width=2)
        text(draw, (x, y), str(i), 16, color, True, "mm")
        if not selected and i <= ready:
            values = "0.00" if not i else " / ".join(f"{TABLE[i][m][0]/100:.2f}" for m in MODELS)
            text(draw, (x, 580), values, 16, GREEN, True, "mt")
    if not selected:
        text(draw, (60, 224), "Edge costs: Model A / Model B", 18, MUTED)
        text(draw, (1220, 224), "Model switch: +0.25", 18, MUTED, anchor="rt")
        text(draw, (60, 553), "BEST CUMULATIVE COST PER BOUNDARY  /  LAST MODEL A OR B", 14, MUTED, True)


def frame(elapsed):
    phase, t = 0, elapsed
    while phase < len(DURATIONS) - 1 and t >= DURATIONS[phase]:
        t -= DURATIONS[phase]
        phase += 1
    image = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(image)
    if phase == 0:
        shell(d, phase, "A task arrives", "AndroidWorld  /  One Broccoli recipe  /  Example values", elapsed)
        card(d, (60, 236, 778, 571), BLUE)
        text(d, (91, 270), "RecipeAddSingleRecipe", 31, BLUE, True)
        text(d, (91, 329), "Add Lemon Herb Pasta to Broccoli.", 28, WHITE, True)
        for i, line in enumerate(("Title: Lemon Herb Pasta", "Ingredients: pasta, lemon, parsley",
                                  "Directions: cook pasta, then toss with lemon and herbs.")):
            text(d, (91, 391 + i * 46), line, 20, MUTED)
        p = ease((t - .6) / 1.4)
        arrow(d, [(795, 403), (906, 403)], GREEN, 4, p)
        if t > 1.4:
            card(d, (924, 340, 1220, 468), GREEN)
            text(d, (1072, 387), "Compact planner", 23, WHITE, True, "mm")
            text(d, (1072, 423), "Task in, draft route out", 17, MUTED, anchor="mm")
    elif phase == 1:
        shell(d, phase, "The small model drafts a route", "The draft describes what to do, before PORTS chooses how to execute it.", elapsed)
        for i, (title, detail) in enumerate((("Open the recipe editor", "Launch Broccoli and start a new recipe."),
                                             ("Enter the recipe", "Fill the title, ingredients, and directions."),
                                             ("Save it", "Commit the recipe to the library."))):
            p = ease((t - i * .8) / .6)
            if p:
                y = 229 + i * 118 + round(15 * (1 - p))
                card(d, (60, y, 1220, y + 95))
                text(d, (89, y + 34), f"{i + 1:02d}", 28, GREEN, True)
                text(d, (158, y + 17), title, 26, WHITE, True)
                text(d, (158, y + 52), detail, 19, MUTED)
    elif phase == 2:
        shell(d, phase, "Decompose the draft into primitives", "The ordered primitives remain available as a complete fallback route.", elapsed)
        primitive_strip(d, min(1, t / 4.2))
        if t > 2.1:
            for left, right, label in ((60, 433, "OPEN EDITOR"), (450, 1018, "ENTER RECIPE"), (1035, 1213, "COMMIT")):
                d.line((left, 447, right, 447), LINE, width=2)
                text(d, ((left + right) / 2, 469), label, 17, MUTED, True, "mt")
    elif phase == 3:
        shell(d, phase, "Match spans in the skill library", "A compatible skill can replace a contiguous span without changing the task goal.", elapsed)
        for i, (level, title, span) in enumerate((
                ("L3", "Task workflow", "Create recipe   |   P1-P6"),
                ("L2", "App-specific subroutines", "Open editor   |   P1-P2      Populate recipe   |   P3-P5"),
                ("L1", "Operation skill", "Name recipe   |   P3"))):
            p = ease((t - i * .95) / .7)
            if p:
                y = 225 + i * 115
                x = 60 + round(40 * (1 - p))
                card(d, (x, y, 1220, y + 92), VIOLET)
                text(d, (x + 23, y + 33), level, 28, VIOLET, True)
                text(d, (x + 99, y + 15), title, 24, WHITE, True)
                text(d, (x + 99, y + 53), span, 20, MUTED)
                if i and t > i + .45:
                    arrow(d, [(105, y - 22), (105, y - 5)], VIOLET, 2)
    elif phase == 4:
        shell(d, phase, "Score candidate routes on a DAG",
              f"{COUNT} structural routes  /  Dynamic programming retains the best cost for each (boundary, last model).", elapsed)
        graph(d, t)
    else:
        shell(d, phase, "Backtrack the minimum-cost route",
              "This is the selected execution plan, not a claim that the task has already succeeded.", elapsed)
        graph(d, t, selected=True)
        for i, (edge, model) in enumerate(PATH):
            if t < .4 + .45 * i:
                continue
            x = 60 + 273 * i
            card(d, (x, 463, x + 251, 588), GREEN)
            text(d, (x + 15, 484), f"{i+1:02d}   MODEL {model}", 16, GREEN, True)
            text(d, (x + 15, 519), edge.name, 22, WHITE, True)
            text(d, (x + 15, 554), f"Edge cost  {edge.costs[MODELS.index(model)]/100:.2f}", 17, MUTED)
        if t > 2:
            text(d, (1220, 478), f"{COST/100:.2f}", 50, GREEN, True, "rt")
            text(d, (1220, 537), "TOTAL COST", 15, MUTED, True, "rt")
            switch_count = sum(a[1] != b[1] for a, b in zip(PATH, PATH[1:]))
            text(d, (1220, 564), f"Includes {switch_count} switches (+{switch_count * SWITCH / 100:.2f})",
                 16, MUTED, anchor="rt")
    return image


def write_avi(target, fps, images, count):
    """Dependency-light Motion JPEG AVI fallback (one video stream, no audio)."""
    def chunk(tag, payload):
        return tag + struct.pack("<I", len(payload)) + payload + b"\0" * (len(payload) % 2)

    def listing(tag, payload):
        return chunk(b"LIST", tag + payload)

    main = struct.pack("<14I", round(1_000_000 / fps), 0, 0, 16, count, 0, 1, 0, W, H, 0, 0, 0, 0)
    stream = struct.pack("<4s4sIHH8I4h", b"vids", b"MJPG", 0, 0, 0, 0, 1, fps, 0, count,
                         0, 0xFFFFFFFF, 0, 0, 0, W, H)
    bitmap = struct.pack("<IiiHH4sIiiII", 40, W, H, 1, 24, b"MJPG", W * H * 3, 0, 0, 0, 0)
    with target.open("wb") as out:
        out.write(b"RIFF\0\0\0\0AVI ")
        out.write(listing(b"hdrl", chunk(b"avih", main) +
                          listing(b"strl", chunk(b"strh", stream) + chunk(b"strf", bitmap))))
        start = out.tell()
        out.write(b"LIST\0\0\0\0movi")
        index = bytearray()
        for image in images:
            with io.BytesIO() as buffer:
                image.save(buffer, "JPEG", quality=86)
                data = buffer.getvalue()
            index.extend(struct.pack("<4sIII", b"00dc", 16, out.tell() - start - 8, len(data)))
            out.write(chunk(b"00dc", data))
        end = out.tell()
        out.write(chunk(b"idx1", index))
        size = out.tell()
        out.seek(start + 4)
        out.write(struct.pack("<I", end - start - 8))
        out.seek(4)
        out.write(struct.pack("<I", size - 8))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=24, choices=range(12, 61))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--preview", type=Path, help="Write a montage of the six scenes instead of encoding")
    args = parser.parse_args(argv)
    if args.preview:
        preview = Image.new("RGB", (W * 3 // 2, H))
        for i, elapsed in enumerate((3, 9, 15, 23, 36, 43)):
            preview.paste(frame(elapsed).resize((W // 2, H // 2)), ((i % 3) * W // 2, (i // 3) * H // 2))
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        preview.save(args.preview)
        return
    executable = shutil.which(args.ffmpeg)
    target = args.output or Path("deliverables") / ("ports-recipe-workflow.mp4" if executable else "ports-recipe-workflow.avi")
    if target.suffix.lower() not in {".mp4", ".avi"}:
        parser.error("--output must end in .mp4 or .avi")
    if target.suffix.lower() == ".mp4" and not executable:
        parser.error("MP4 requires FFmpeg with libx264; choose .avi for the Pillow-only fallback")
    target.parent.mkdir(parents=True, exist_ok=True)
    count = sum(DURATIONS) * args.fps
    images = (frame(i / args.fps) for i in range(count))
    temporary = target.with_name(f".{target.stem}.encoding{target.suffix}")
    try:
        if target.suffix.lower() == ".mp4":
            from .render import _encode

            _encode(executable, temporary, (W, H), args.fps, images)
        else:
            write_avi(temporary, args.fps, images, count)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(target.resolve())


if __name__ == "__main__":
    main()
