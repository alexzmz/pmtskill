"""Render two annotated views of one episode from a shared screenshot timeline."""

from __future__ import annotations

import bisect
import hashlib
import math
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path


def _time(row):
    value = float(row["t"])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid relative timestamp: {value!r}")
    return value


def _timeline(frames, steps, duration, fps, pixel_hash, *, max_step_delta=1.0):
    """Reserve one CFR tick per exact, time-local step observation.

    Repeated/static screens are disambiguated by timestamp, and a trajectory
    frame tagged with that step is preferred to a coincident sampled frame.
    No annotation is drawn over a merely nearby, different screenshot.
    """
    frame_times = [_time(frame) for frame in frames]
    matches = []
    for step in steps:
        t = _time(step)
        number = int(step["step"])
        step_hash = pixel_hash(step["path"])
        lo = bisect.bisect_left(frame_times, t - max_step_delta)
        hi = bisect.bisect_right(frame_times, t + max_step_delta)
        candidates = sorted(range(lo, hi), key=lambda i: (
            frames[i].get("source") != "trajectory" or frames[i].get("step") != number,
            abs(frame_times[i] - t), i,
        ))
        source = next((i for i in candidates if pixel_hash(frames[i]["path"]) == step_hash), None)
        matches.append({"step": number, "step_t": t, "status": "exact" if source is not None else "unmatched",
                        "source_t": frame_times[source] if source is not None else None,
                        "source": frames[source].get("source") if source is not None else None,
                        "source_path": str(frames[source]["path"]) if source is not None else None,
                        "_source": source})

    exact = [match for match in matches if match["_source"] is not None]
    count = max(math.ceil(duration * fps) + 1, len(exact) + 2 if exact else 1)
    # Include the initial and final recorded screens, and preserve step order
    # even if multiple trajectory observations fall between two output ticks.
    timeline = [max(0, bisect.bisect_right(frame_times, min(i / fps, duration)) - 1) for i in range(count)]
    timeline[0] = 0
    timeline[-1] = len(frames) - 1
    marks = {}
    previous = 0
    for index, match in enumerate(exact):
        source = match["_source"]
        preferred = round(frame_times[source] * fps)
        tick = min(max(preferred, previous + 1), count - 1 - (len(exact) - index))
        timeline[tick] = source
        marks[tick] = match["step"]
        match["video_t"] = tick / fps
        previous = tick

    # Keep the label readable through a short run of *identical* pixels only.
    # Never overlay a label over a changed screen or another step's own tick.
    for tick, number in list(marks.items()):
        digest = pixel_hash(frames[timeline[tick]]["path"])
        for nearby in range(tick + 1, min(count, tick + math.ceil(fps * 0.55))):
            if nearby in marks or pixel_hash(frames[timeline[nearby]]["path"]) != digest:
                break
            marks[nearby] = number
    for match in matches:
        del match["_source"]
    return timeline, marks, matches


def _same_screen(first, second):
    """Permit minor transient differences, but suppress boxes across UI changes."""
    from PIL import ImageChops

    if abs(first.width / first.height - second.width / second.height) > 0.02:
        return False
    difference = ImageChops.difference(first.resize((64, 96)), second.resize((64, 96)))
    channels = difference.split()
    changed = ImageChops.lighter(channels[0], ImageChops.lighter(channels[1], channels[2]))
    hist = changed.histogram()
    return sum(hist[24:]) <= 0.035 * 64 * 96


def _choose_controls(samples, sample_times, frame, screen_matches, max_age, *, display_t=None):
    t = _time(frame)
    index = bisect.bisect_right(sample_times, t + 1e-6) - 1
    if index < 0:
        return None
    sample = samples[index]
    freshness_t = max(t, display_t) if display_t is not None else t
    # Without a source image there is no evidence that the old controls still
    # belong to this screen, so even unlimited mode has a short expiry.
    age_limit = max_age if max_age is not None else (math.inf if sample.get("path") else 2.0)
    if freshness_t - sample_times[index] > age_limit:
        return None
    if sample.get("path") and sample["path"] != frame["path"]:
        try:
            if not screen_matches(sample["path"], frame["path"]):
                return None
        except OSError:
            return None
    return sample


def _font(size):
    from PIL import ImageFont

    for family in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(family, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label(draw, at, text, font, *, fill, background):
    try:
        extent = draw.textbbox((0, 0), text, font=font)
    except UnicodeEncodeError:
        text = text.encode("ascii", errors="replace").decode("ascii")
        extent = draw.textbbox((0, 0), text, font=font)
    x, y = at
    width, height = extent[2] - extent[0], extent[3] - extent[1]
    draw.rectangle((x, y, x + width + 10, y + height + 8), fill=background)
    draw.text((x + 5 - extent[0], y + 4 - extent[1]), text, font=font, fill=fill)


def _draw_step(image, number, font):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    inset = max(3, image.width // 130)
    color = (255, 209, 43)
    draw.rectangle((inset, inset, image.width - inset - 1, image.height - inset - 1),
                   outline=color, width=inset)
    _label(draw, (2 * inset, 2 * inset), f"Step {number}", font, fill=(0, 0, 0), background=color)


def _draw_controls(image, sample, font):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size
    color = (30, 240, 214)
    if sample is None:
        _label(draw, (5, 5), "Waiting for a fresh UI sample", font, fill=(255, 255, 255), background=(20, 25, 31))
        return 0
    src_width, src_height = sample.get("size") or image.size
    if src_width <= 0 or src_height <= 0 or abs(src_width / src_height - width / height) > 0.02:
        return 0
    sx, sy = width / src_width, height / src_height
    thickness = max(2, width // 320)
    drawn = 0
    for index, box in enumerate(sample.get("boxes", []), 1):
        x0, y0, x1, y1 = box["bounds"]
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
            continue
        left, right = (max(0, min(width - 1, round(x * sx))) for x in (x0, x1))
        top, bottom = (max(0, min(height - 1, round(y * sy))) for y in (y0, y1))
        if right <= left or bottom <= top:
            continue
        draw.rectangle((left, top, right, bottom), outline=color, width=thickness)
        raw_label = box.get("label")
        label = " ".join(str(index if raw_label is None else raw_label).split())[:40] or str(index)
        _label(draw, (left, top), label, font, fill=(0, 0, 0), background=color)
        drawn += 1
    return drawn


def _fit(image, size):
    from PIL import Image, ImageOps

    if image.size == size:
        return image
    # Preserve a one-pixel odd-size pad as a pad, not a rescale.
    if 0 <= size[0] - image.width <= 1 and 0 <= size[1] - image.height <= 1:
        canvas = Image.new("RGB", size)
        canvas.paste(image)
        return canvas
    return ImageOps.pad(image, size, color=(0, 0, 0))


def _encode(ffmpeg, target, size, fps, images):
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{size[0]}x{size[1]}", "-framerate", f"{fps:.10g}", "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(target)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for image in images:
                process.stdin.write(_fit(image, size).tobytes())
            process.stdin.close()
            if process.wait(timeout=120) != 0:
                raise BrokenPipeError("ffmpeg exited unsuccessfully")
        except BrokenPipeError as exc:
            errors.seek(0)
            message = errors.read().decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"ffmpeg failed while encoding {target.name}: {message or exc}") from exc
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass


def render_episode(output_dir, *, frames, steps, controls, duration, fps=8, ffmpeg="ffmpeg", max_control_age=None):
    """Create steps.mp4 and controls.mp4 from episode-relative timestamps.

    ``frames`` and ``steps`` contain ``t`` (seconds) and PNG ``path``;
    trajectory steps also have integer ``step``. ``controls`` contain ``t``
    and ``boxes`` (``bounds=[left, top, right, bottom]`` and optional ``label``),
    optionally ``size=[width, height]`` and ``path`` to the sampled screen.
    Paths may be absolute or relative to ``output_dir``. No unmatched screen
    is substituted for a trajectory observation. If ``max_control_age`` is
    ``None``, controls remain visible as long as their sampled screen still
    resembles the current screen; observations without an image expire in 2 s.
    """
    from PIL import Image

    output_dir = Path(output_dir).resolve()
    fps, duration = map(float, (fps, duration))
    max_control_age = float(max_control_age) if max_control_age is not None else None
    if not math.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError("fps must be positive and at most 120")
    if not math.isfinite(duration) or duration < 0 or (
        max_control_age is not None and (not math.isfinite(max_control_age) or max_control_age < 0)
    ):
        raise ValueError("duration and max_control_age must be finite and non-negative")
    frames, steps, controls = (sorted(rows, key=_time) for rows in (frames, steps, controls))
    if not frames:
        raise ValueError("No episode screenshots were recorded; cannot render video")
    executable = shutil.which(str(ffmpeg))
    if executable is None:
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg!r}; install ffmpeg with libx264 support or set its path")
    duration = max(duration, _time(frames[-1]))

    @lru_cache(maxsize=6)
    def load_image(path):
        with Image.open(output_dir / Path(path)) as image:
            return image.convert("RGB")

    @lru_cache(maxsize=None)
    def pixel_hash(path):
        image = load_image(path)
        return image.size, hashlib.sha256(image.tobytes()).digest()

    @lru_cache(maxsize=256)
    def screen_matches(first, second):
        return _same_screen(load_image(first), load_image(second))

    width, height = load_image(frames[0]["path"]).size
    size = (width + width % 2, height + height % 2)
    step_font, control_font = _font(max(18, width // 21)), _font(max(11, width // 40))
    timeline, marks, matches = _timeline(frames, steps, duration, fps, pixel_hash)
    sample_times = [_time(sample) for sample in controls]
    control_frames = 0

    def images(which):
        nonlocal control_frames
        for index, source in enumerate(timeline):
            frame = frames[source]
            image = load_image(frame["path"])
            if which == "steps" and index in marks:
                image = image.copy()
                _draw_step(image, marks[index], step_font)
            if which == "controls":
                sample = _choose_controls(controls, sample_times, frame, screen_matches, max_control_age,
                                          display_t=min(index / fps, duration))
                image = image.copy()
                control_frames += bool(_draw_controls(image, sample, control_font))
            yield image

    step_video, controls_video = output_dir / "steps.mp4", output_dir / "controls.mp4"
    # Publish neither replacement until both encode operations have succeeded.
    with tempfile.TemporaryDirectory(prefix="encode-", dir=output_dir) as temporary:
        for kind in ("steps", "controls"):
            _encode(executable, Path(temporary) / f"{kind}.mp4", size, fps, images(kind))
        (Path(temporary) / "steps.mp4").replace(step_video)
        (Path(temporary) / "controls.mp4").replace(controls_video)
    return {"step_video": str(step_video), "controls_video": str(controls_video),
            "fps": fps, "output_frames": len(timeline), "duration_s": len(timeline) / fps,
            "matched_steps": sum(match["status"] == "exact" for match in matches),
            "unmatched_steps": sum(match["status"] == "unmatched" for match in matches),
            "control_overlay_frames": control_frames, "step_matches": matches}
