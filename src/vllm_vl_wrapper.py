"""Multimodal vLLM adapter for Android World's M3A agent.

The text-only wrapper in :mod:`vllm_wrapper` deliberately ignores images.
This module instead renders each model's Hugging Face chat template and sends
the screenshots through vLLM's ``multi_modal_data`` input.  It also translates
MobileRL's native ``do(action=...)`` syntax into the JSON action protocol used
by Android World's stock M3A agent.
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from typing import Any, Optional, Sequence


M3A_SYSTEM_PROMPT = """You are an Android GUI agent. Use the screenshots and
the UI-element information in the user prompt to choose the next action. The
first image is the raw screen and the second is the same screen annotated with
element indexes. Follow the exact Reason/Action JSON response format requested
by the user prompt. Do not wrap the response in Markdown fences."""

SUMMARY_SYSTEM_PROMPT = """You summarize one Android GUI interaction. Compare
the before and after screenshots and return only the concise single-line
summary requested by the user prompt."""

MOBILERL_SYSTEM_PROMPT = """You are a mobile GUI agent. The first image is the
raw current screenshot and the second is an annotated view of the same screen.
The user prompt also contains bbox_1000 coordinates derived from Android XML.
Choose exactly one next action. Coordinates are relative to a 0..1000 screen.

Respond in the native MobileRL format:
<think>brief reasoning</think>
<answer>do(action="Tap", element=[x1,y1,x2,y2])</answer>

Allowed actions are:
- do(action="Tap", element=[x1,y1,x2,y2])
- do(action="Long Press", element=[x1,y1,x2,y2])
- do(action="Type", text="text", element=[optional bbox])
- do(action="Swipe", direction="up|down|left|right", element=[optional bbox])
- do(action="Launch", app="app name")
- do(action="Back"), do(action="Home"), or do(action="Wait")
- do(action="Answer", text="answer") for a question
- finish(message="optional result") only after the goal is complete

For an action-selection request, use this native format even if the embedded
AndroidWorld user prompt mentions JSON. The host translates it safely."""

_ACTION_PROMPT_MARKERS = (
    "Now output an action",
    "Your Answer:",
)
_SUMMARY_PROMPT_MARKERS = (
    "Summary of this step:",
    "summerize the latest step",
)
_JSON_ACTION_TYPES = {
    "answer",
    "click",
    "double_tap",
    "input_text",
    "keyboard_enter",
    "long_press",
    "navigate_back",
    "navigate_home",
    "open_app",
    "scroll",
    "status",
    "swipe",
    "wait",
}


def resolve_prompt_profile(model_path: str, requested: str) -> str:
    """Resolve ``auto`` without loading model metadata."""
    if requested != "auto":
        return requested
    return "mobilerl" if "mobilerl" in model_path.lower() else "m3a"


def _is_action_prompt(prompt: str) -> bool:
    if any(marker in prompt for marker in _SUMMARY_PROMPT_MARKERS):
        return False
    return any(marker in prompt for marker in _ACTION_PROMPT_MARKERS)


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    fenced = re.fullmatch(
        r"```(?:json|python|text)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced.group(1).strip() if fenced else cleaned


def _extract_balanced_call(text: str) -> str | None:
    """Return the first native action call while respecting quoted strings."""
    match = re.search(r"\b(?:do|finish)\s*\(", text, flags=re.IGNORECASE)
    if not match:
        return None
    start = match.start()
    opening = text.find("(", start)
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening, len(text)):
        character = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in ("'", '"'):
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _literal_call_arguments(
    expression: str,
) -> tuple[str, dict[str, Any]] | None:
    """Parse MobileRL action arguments without executing model output."""
    try:
        node = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not isinstance(node, ast.Call) or node.args:
        return None
    if not isinstance(node.func, ast.Name):
        return None
    function_name = node.func.id.lower()
    if function_name not in {"do", "finish"}:
        return None
    values: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            return None
        try:
            values[keyword.arg.lower()] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError):
            return None
    return function_name, values


def _canonical_action_name(value: Any) -> str:
    return re.sub(r"[\s_-]+", "", str(value or "")).lower()


def _bbox_center(
    element: Any,
    screen_size: tuple[int, int] | None,
    coordinate_mode: str,
) -> tuple[int, int] | None:
    if not isinstance(element, (list, tuple)) or len(element) not in (2, 4):
        return None
    try:
        coordinates = [float(value) for value in element]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coordinates):
        return None
    if len(coordinates) == 2:
        x, y = coordinates
    else:
        x = (coordinates[0] + coordinates[2]) / 2
        y = (coordinates[1] + coordinates[3]) / 2

    width, height = screen_size or (1000, 1000)
    effective_mode = coordinate_mode
    if effective_mode == "auto":
        in_relative_range = all(0 <= value <= 1000 for value in coordinates)
        effective_mode = (
            "normalized_1000"
            if in_relative_range and (width > 1000 or height > 1000)
            else "pixels"
        )
    if effective_mode == "normalized_1000":
        x = x * width / 1000
        y = y * height / 1000
    x = min(max(round(x), 0), max(width - 1, 0))
    y = min(max(round(y), 0), max(height - 1, 0))
    return x, y


def _mobile_call_to_json_action(
    function_name: str,
    values: dict[str, Any],
    *,
    screen_size: tuple[int, int] | None,
    coordinate_mode: str,
) -> dict[str, Any] | None:
    if function_name == "finish":
        message = values.get("message")
        if message:
            # AndroidWorld answer actions populate interaction_cache. This is
            # required for query tasks; state-based tasks are independently
            # stopped by their success evaluator after the same step.
            return {"action_type": "answer", "text": str(message)}
        return {"action_type": "status", "goal_status": "complete"}

    name = _canonical_action_name(values.get("action"))
    element = values.get("element", values.get("bbox"))
    point = _bbox_center(element, screen_size, coordinate_mode)

    if name in {"tap", "click"}:
        return (
            {"action_type": "click", "x": point[0], "y": point[1]}
            if point
            else None
        )
    if name in {"longpress", "pressandhold"}:
        return (
            {"action_type": "long_press", "x": point[0], "y": point[1]}
            if point
            else None
        )
    if name in {"type", "input", "inputtext"}:
        text = values.get(
            "text",
            values.get("content", values.get("argument")),
        )
        if text is None:
            return None
        action: dict[str, Any] = {
            "action_type": "input_text",
            "text": str(text),
        }
        if point:
            action.update({"x": point[0], "y": point[1]})
        return action
    if name in {"swipe", "scroll"}:
        direction = str(
            values.get("direction") or values.get("argument") or ""
        ).strip().lower()
        if direction not in {"up", "down", "left", "right"}:
            return None
        # MobileRL describes the finger gesture direction, matching AW swipe
        # (and deliberately not AW's content-oriented scroll action).
        return {"action_type": "swipe", "direction": direction}
    if name in {"launch", "open", "openapp"}:
        app_name = (
            values.get("app")
            or values.get("app_name")
            or values.get("text")
        )
        return (
            {"action_type": "open_app", "app_name": str(app_name)}
            if app_name
            else None
        )
    if name in {"back", "navigateback", "pressback"}:
        return {"action_type": "navigate_back"}
    if name in {"home", "navigatehome", "presshome"}:
        return {"action_type": "navigate_home"}
    if name == "wait":
        return {"action_type": "wait"}
    if name in {"finish", "complete", "done"}:
        return {"action_type": "status", "goal_status": "complete"}
    if name in {"infeasible", "impossible"}:
        return {"action_type": "status", "goal_status": "infeasible"}
    if name == "answer":
        answer = values.get("text", values.get("content"))
        return (
            {"action_type": "answer", "text": str(answer)}
            if answer is not None
            else None
        )
    if name in {"enter", "keyboardenter", "pressenter"}:
        return {"action_type": "keyboard_enter"}
    return None


def _extract_reason(text: str) -> str:
    think = re.search(r"<think>(.*?)</think>", text, re.I | re.S)
    if think and think.group(1).strip():
        return " ".join(think.group(1).split())
    reason = re.search(
        r"\bReason\s*:\s*(.*?)(?:\bAction\s*:|$)",
        text,
        re.I | re.S,
    )
    if reason and reason.group(1).strip():
        return " ".join(reason.group(1).split())
    return "Converted from the model's native mobile action format."


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _is_m3a_action_output(text: str) -> bool:
    reason = re.search(
        r"\bReason\s*:\s*(.*?)\s*\bAction\s*:",
        text,
        re.I | re.S,
    )
    if not reason or not reason.group(1).strip():
        return False
    action_section = text[reason.end() :]
    action = _extract_json_object(action_section)
    return bool(
        action
        and action.get("action_type") in _JSON_ACTION_TYPES
    )


def normalize_action_output(
    text: str,
    *,
    screen_size: tuple[int, int] | None,
    coordinate_mode: str,
) -> tuple[str, bool]:
    """Return an M3A-compatible response and whether it was converted."""
    cleaned = _strip_fences(text)
    has_reason = re.search(r"\bReason\s*:", cleaned, re.I)
    has_action = re.search(r"\bAction\s*:", cleaned, re.I)
    if has_reason and has_action:
        return cleaned, cleaned != text.strip()

    json_value = _extract_json_object(cleaned)
    if json_value is not None:
        candidate = json_value.get("action", json_value)
        if isinstance(candidate, dict):
            action_type = candidate.get("action_type")
            if action_type in _JSON_ACTION_TYPES:
                normalized = json.dumps(candidate, ensure_ascii=False)
                return f"Reason: {_extract_reason(cleaned)}\nAction: {normalized}", True

    call = _extract_balanced_call(cleaned)
    parsed_call = _literal_call_arguments(call) if call else None
    if parsed_call is not None:
        function_name, values = parsed_call
        action = _mobile_call_to_json_action(
            function_name,
            values,
            screen_size=screen_size,
            coordinate_mode=coordinate_mode,
        )
        if action is not None:
            normalized = json.dumps(action, ensure_ascii=False)
            return f"Reason: {_extract_reason(cleaned)}\nAction: {normalized}", True

    return cleaned, cleaned != text.strip()


class VLLMMultimodalWrapper:
    """Use a local vLLM vision-language model through Android World's API."""

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        dtype: str = "auto",
        max_num_seqs: int = 1,
        enforce_eager: bool = False,
        sampling_seed: int | None = 42,
        max_images_per_prompt: int = 2,
        image_max_pixels: int = 500_000,
        prompt_profile: str = "auto",
        bbox_coordinate_mode: str = "auto",
        system_prompt: str | None = None,
        normalize_actions: bool = True,
        raise_on_error: bool = False,
        _llm: Any = None,
        _processor: Any = None,
        _sampling_params: Any = None,
    ) -> None:
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_images_per_prompt = max_images_per_prompt
        self.image_max_pixels = image_max_pixels
        self.prompt_profile = resolve_prompt_profile(model_path, prompt_profile)
        self.bbox_coordinate_mode_requested = bbox_coordinate_mode
        self.bbox_coordinate_mode = (
            "normalized_1000"
            if bbox_coordinate_mode == "auto"
            and self.prompt_profile == "mobilerl"
            else bbox_coordinate_mode
        )
        self.system_prompt = system_prompt
        self.normalize_actions = normalize_actions
        self.raise_on_error = raise_on_error

        if _llm is None or _sampling_params is None:
            from vllm import LLM, SamplingParams

            if _llm is None:
                llm_kwargs: dict[str, Any] = {
                    "model": model_path,
                    "tensor_parallel_size": tensor_parallel_size,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "trust_remote_code": True,
                    "dtype": dtype,
                    "max_num_seqs": max_num_seqs,
                    "enforce_eager": enforce_eager,
                    "limit_mm_per_prompt": {
                        "image": max_images_per_prompt,
                    },
                }
                if max_model_len is not None:
                    llm_kwargs["max_model_len"] = max_model_len
                _llm = LLM(**llm_kwargs)
            if _sampling_params is None:
                sampling_kwargs: dict[str, Any] = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                if sampling_seed is not None:
                    sampling_kwargs["seed"] = sampling_seed
                _sampling_params = SamplingParams(**sampling_kwargs)
        if _processor is None:
            from transformers import AutoProcessor

            _processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

        self.llm = _llm
        self.processor = _processor
        self.sampling_params = _sampling_params

        self._request_count = 0
        self._action_request_count = 0
        self._summary_request_count = 0
        self._error_count = 0
        self._prompt_tokens = 0
        self._generated_tokens = 0
        self._total_latency_s = 0.0
        self._image_count = 0
        self._resized_image_count = 0
        self._converted_action_count = 0
        self._native_action_count = 0
        self._unparsed_action_count = 0
        self._last_errors: list[str] = []

    def _to_pil(self, image: Any) -> Any:
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        try:
            return Image.fromarray(image).convert("RGB")
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "AndroidWorld screenshot must be a PIL image or an array "
                "compatible with PIL.Image.fromarray()."
            ) from exc

    def _prepare_images(
        self,
        images: Sequence[Any],
    ) -> tuple[list[Any], tuple[int, int] | None]:
        from PIL import Image

        selected = list(images[: self.max_images_per_prompt])
        prepared: list[Any] = []
        original_screen_size: tuple[int, int] | None = None
        for source in selected:
            image = self._to_pil(source)
            if original_screen_size is None:
                original_screen_size = image.size
            exceeds_pixel_limit = (
                self.image_max_pixels
                and image.width * image.height > self.image_max_pixels
            )
            if exceeds_pixel_limit:
                scale = math.sqrt(
                    self.image_max_pixels / (image.width * image.height)
                )
                target = (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                )
                image = image.resize(target, Image.Resampling.LANCZOS)
                self._resized_image_count += 1
            prepared.append(image)
        return prepared, original_screen_size

    def _select_system_prompt(self, action_prompt: bool) -> str:
        if self.system_prompt:
            return self.system_prompt
        if not action_prompt:
            return SUMMARY_SYSTEM_PROMPT
        if self.prompt_profile == "mobilerl":
            return MOBILERL_SYSTEM_PROMPT
        return M3A_SYSTEM_PROMPT

    @staticmethod
    def _image_content(image: Any, include_value: bool) -> dict[str, Any]:
        content: dict[str, Any] = {"type": "image"}
        if include_value:
            content["image"] = image
        return content

    def _render_chat_prompt(
        self,
        text_prompt: str,
        images: Sequence[Any],
        system_prompt: str,
    ) -> str:
        """Render Qwen/GLM-style templates with compatibility fallbacks."""
        errors: list[Exception] = []
        template_owner = self.processor
        apply_template = getattr(template_owner, "apply_chat_template", None)
        if apply_template is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            apply_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_template is None:
            raise AttributeError(
                "The model processor/tokenizer has no apply_chat_template()."
            )

        for include_image_value in (False, True):
            image_parts = [
                self._image_content(image, include_image_value)
                for image in images
            ]
            user_content = image_parts + [
                {"type": "text", "text": text_prompt}
            ]
            candidates = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                [
                    {
                        "role": "user",
                        "content": image_parts
                        + [
                            {
                                "type": "text",
                                "text": f"{system_prompt}\n\n{text_prompt}",
                            }
                        ],
                    }
                ],
            ]
            for messages in candidates:
                try:
                    rendered = apply_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if not isinstance(rendered, str):
                        raise TypeError(
                            "apply_chat_template(tokenize=False) did not "
                            "return text."
                        )
                    return rendered
                except (KeyError, TypeError, ValueError, AttributeError) as exc:
                    errors.append(exc)
        details = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
        raise RuntimeError(f"Unable to render the VL chat template: {details}")

    def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
        self,
        text_prompt: str,
        images: Optional[list[Any]] = None,
    ) -> tuple[str, Optional[bool], Any]:
        """Generate with real screenshots and return AndroidWorld's tuple."""
        started_at = time.perf_counter()
        self._request_count += 1
        action_prompt = _is_action_prompt(text_prompt)
        if action_prompt:
            self._action_request_count += 1
        else:
            self._summary_request_count += 1
        try:
            prepared_images, screen_size = self._prepare_images(images or [])
            self._image_count += len(prepared_images)
            system_prompt = self._select_system_prompt(action_prompt)
            rendered_prompt = self._render_chat_prompt(
                text_prompt,
                prepared_images,
                system_prompt,
            )
            request: dict[str, Any] = {"prompt": rendered_prompt}
            if prepared_images:
                request["multi_modal_data"] = {"image": prepared_images}
            outputs = self.llm.generate([request], self.sampling_params)
            request_output = outputs[0]
            candidate = request_output.outputs[0]
            raw_text = str(candidate.text).strip()

            prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
            generated_token_ids = getattr(candidate, "token_ids", None)
            if prompt_token_ids is not None:
                self._prompt_tokens += len(prompt_token_ids)
            if generated_token_ids is not None:
                self._generated_tokens += len(generated_token_ids)

            generated_text = raw_text
            converted = False
            action_format = "summary"
            if action_prompt and self.normalize_actions:
                generated_text, converted = normalize_action_output(
                    raw_text,
                    screen_size=screen_size,
                    coordinate_mode=self.bbox_coordinate_mode,
                )
                if converted:
                    self._converted_action_count += 1
                    action_format = "normalized"
                elif _is_m3a_action_output(generated_text):
                    self._native_action_count += 1
                    action_format = "m3a"
                else:
                    self._unparsed_action_count += 1
                    action_format = "unparsed"
            elif action_prompt:
                action_format = "normalization-disabled"
            raw_response = {
                "raw_text": raw_text,
                "normalized_text": generated_text,
                "action_normalized": converted,
                "action_format": action_format,
                "prompt_profile": self.prompt_profile,
                "image_count": len(prepared_images),
                "screen_size": list(screen_size) if screen_size else None,
            }
            return generated_text, None, raw_response
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._error_count += 1
            error_text = f"{type(exc).__name__}: {exc}"
            self._last_errors.append(error_text)
            self._last_errors = self._last_errors[-10:]
            if self.raise_on_error:
                raise RuntimeError(
                    f"vLLM multimodal inference failed: {error_text}"
                ) from exc
            return (
                f"vLLM multimodal inference failed: {error_text}",
                False,
                {"error": error_text},
            )
        finally:
            self._total_latency_s += time.perf_counter() - started_at

    def get_stats(self) -> dict[str, Any]:
        average_latency = (
            self._total_latency_s / self._request_count
            if self._request_count
            else 0.0
        )
        return {
            "backend": "vllm-in-process-multimodal",
            "model_path": self.model_path,
            "prompt_profile": self.prompt_profile,
            "bbox_coordinate_mode_requested": (
                self.bbox_coordinate_mode_requested
            ),
            "bbox_coordinate_mode": self.bbox_coordinate_mode,
            "request_count": self._request_count,
            "action_request_count": self._action_request_count,
            "summary_request_count": self._summary_request_count,
            "error_count": self._error_count,
            "prompt_tokens": self._prompt_tokens,
            "generated_tokens": self._generated_tokens,
            "image_count": self._image_count,
            "resized_image_count": self._resized_image_count,
            "converted_action_count": self._converted_action_count,
            "native_action_count": self._native_action_count,
            "unparsed_action_count": self._unparsed_action_count,
            "total_latency_s": round(self._total_latency_s, 3),
            "mean_latency_s": round(average_latency, 3),
            "last_errors": list(self._last_errors),
        }
