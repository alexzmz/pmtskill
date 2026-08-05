from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import task_runner_vl as runner  # noqa: E402
from vllm_vl_wrapper import (  # noqa: E402
    VLLMMultimodalWrapper,
    normalize_action_output,
    resolve_prompt_profile,
)


class _FakeProcessor:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "<rendered-vl-prompt>"


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.inputs = None
        self.sampling_params = None

    def generate(self, inputs, sampling_params):
        self.inputs = inputs
        self.sampling_params = sampling_params
        candidate = SimpleNamespace(text=self.text, token_ids=[1, 2, 3])
        output = SimpleNamespace(outputs=[candidate], prompt_token_ids=[4, 5])
        return [output]


class ActionNormalizationTest(unittest.TestCase):
    def test_auto_profile_detects_mobilerl_only(self):
        self.assertEqual(
            resolve_prompt_profile("/models/MobileRL-9B", "auto"),
            "mobilerl",
        )
        self.assertEqual(resolve_prompt_profile("/models/Qwen2.5-VL", "auto"), "m3a")
        self.assertEqual(resolve_prompt_profile("anything", "mobilerl"), "mobilerl")

    def test_mobile_tap_bbox_is_scaled_to_screen(self):
        response = (
            "<think>Tap the visible button.</think>\n"
            '<answer>do(action="Tap", element=[100,200,300,400])</answer>'
        )
        normalized, converted = normalize_action_output(
            response,
            screen_size=(1080, 2400),
            coordinate_mode="normalized_1000",
        )
        self.assertTrue(converted)
        self.assertIn('"action_type": "click"', normalized)
        self.assertIn('"x": 216', normalized)
        self.assertIn('"y": 720', normalized)
        self.assertIn("Reason: Tap the visible button.", normalized)

    def test_mobile_type_preserves_escaped_text_and_target(self):
        response = (
            '<answer>do(action="Type", text="Ada \\\"Lovelace\\\"", '
            "element=[10,20,30,40])</answer>"
        )
        normalized, converted = normalize_action_output(
            response,
            screen_size=(1000, 1000),
            coordinate_mode="pixels",
        )
        self.assertTrue(converted)
        self.assertIn('"action_type": "input_text"', normalized)
        self.assertIn('Ada \\\"Lovelace\\\"', normalized)
        self.assertIn('"x": 20', normalized)
        self.assertIn('"y": 30', normalized)

    def test_json_only_action_is_wrapped_for_m3a(self):
        normalized, converted = normalize_action_output(
            '{"action_type":"navigate_back"}',
            screen_size=None,
            coordinate_mode="auto",
        )
        self.assertTrue(converted)
        self.assertTrue(normalized.startswith("Reason:"))
        self.assertIn("Action:", normalized)

    def test_valid_reason_action_is_not_rewritten(self):
        original = (
            'Reason: done\nAction: '
            '{"action_type":"status","goal_status":"complete"}'
        )
        normalized, converted = normalize_action_output(
            original,
            screen_size=None,
            coordinate_mode="auto",
        )
        self.assertFalse(converted)
        self.assertEqual(normalized, original)

    def test_model_output_is_never_executed_as_python(self):
        malicious = '<answer>do(action=__import__("os").system("bad"))</answer>'
        normalized, converted = normalize_action_output(
            malicious,
            screen_size=(1080, 2400),
            coordinate_mode="auto",
        )
        self.assertFalse(converted)
        self.assertEqual(normalized, malicious)

    def test_finish_message_becomes_answer_for_query_tasks(self):
        normalized, converted = normalize_action_output(
            'finish(message="The answer is 7")',
            screen_size=(1080, 2400),
            coordinate_mode="auto",
        )
        self.assertTrue(converted)
        self.assertIn('"action_type": "answer"', normalized)
        self.assertIn('"text": "The answer is 7"', normalized)


class MultimodalWrapperTest(unittest.TestCase):
    def test_predict_mm_sends_both_images_and_normalizes_action(self):
        model_text = (
            "<think>The button is centered.</think>"
            '<answer>do(action="Tap", element=[400,400,600,600])</answer>'
        )
        fake_llm = _FakeLLM(model_text)
        processor = _FakeProcessor()
        sampling_params = object()
        wrapper = VLLMMultimodalWrapper(
            "/models/mobilerl-9b",
            image_max_pixels=0,
            prompt_profile="auto",
            bbox_coordinate_mode="normalized_1000",
            _llm=fake_llm,
            _processor=processor,
            _sampling_params=sampling_params,
        )
        images = [
            Image.new("RGB", (1080, 2400), "white"),
            Image.new("RGB", (1080, 2400), "black"),
        ]
        prompt = "Now output an action.\nYour Answer:"
        text, is_safe, raw = wrapper.predict_mm(prompt, images)

        self.assertIsNone(is_safe)
        self.assertIn('"action_type": "click"', text)
        self.assertIn('"x": 540', text)
        self.assertIn('"y": 1200', text)
        self.assertTrue(raw["action_normalized"])
        request = fake_llm.inputs[0]
        self.assertEqual(request["prompt"], "<rendered-vl-prompt>")
        self.assertEqual(len(request["multi_modal_data"]["image"]), 2)
        self.assertIs(fake_llm.sampling_params, sampling_params)

        user_parts = processor.messages[-1]["content"]
        self.assertEqual(
            [part["type"] for part in user_parts],
            ["image", "image", "text"],
        )
        stats = wrapper.get_stats()
        self.assertEqual(stats["image_count"], 2)
        self.assertEqual(stats["converted_action_count"], 1)
        self.assertEqual(stats["error_count"], 0)

    def test_summary_request_is_not_action_normalized(self):
        fake_llm = _FakeLLM("The tap opened Settings.")
        wrapper = VLLMMultimodalWrapper(
            "/models/mobilerl-9b",
            image_max_pixels=0,
            _llm=fake_llm,
            _processor=_FakeProcessor(),
            _sampling_params=object(),
        )
        text, _, raw = wrapper.predict_mm(
            "Summary of this step:",
            [Image.new("RGB", (20, 20), "white")],
        )
        self.assertEqual(text, "The tap opened Settings.")
        self.assertFalse(raw["action_normalized"])
        self.assertEqual(raw["action_format"], "summary")
        self.assertEqual(wrapper.get_stats()["summary_request_count"], 1)

    def test_native_m3a_action_is_counted_as_valid_not_unparsed(self):
        fake_llm = _FakeLLM(
            'Reason: go back\nAction: {"action_type":"navigate_back"}'
        )
        wrapper = VLLMMultimodalWrapper(
            "/models/qwen2.5-vl",
            image_max_pixels=0,
            _llm=fake_llm,
            _processor=_FakeProcessor(),
            _sampling_params=object(),
        )
        _, _, raw = wrapper.predict_mm(
            "Now output an action.\nYour Answer:",
            [Image.new("RGB", (20, 20), "white")],
        )
        self.assertEqual(raw["action_format"], "m3a")
        stats = wrapper.get_stats()
        self.assertEqual(stats["native_action_count"], 1)
        self.assertEqual(stats["unparsed_action_count"], 0)


class RunnerTest(unittest.TestCase):
    def test_parser_exposes_vl_defaults(self):
        args = runner.build_parser().parse_args([])
        self.assertEqual(args.prompt_profile, "auto")
        self.assertEqual(args.max_images_per_prompt, 2)
        self.assertEqual(args.image_max_pixels, 500_000)
        self.assertEqual(args.max_tokens, 1024)
        self.assertTrue(args.normalize_actions)
        self.assertTrue(args.enforce_eager)
        self.assertTrue(args.include_ui_bboxes)

    def test_bbox_prompt_adapter_adds_normalized_xml_bounds(self):
        fake_module = SimpleNamespace(
            m3a_utils=SimpleNamespace(
                validate_ui_element=lambda _element, _size: True
            )
        )
        runner._install_bbox_prompt_adapter(fake_module)
        element = SimpleNamespace(
            text="Save",
            content_description=None,
            hint_text=None,
            tooltip=None,
            class_name="android.widget.Button",
            package_name="example",
            resource_name="save",
            resource_id=None,
            is_clickable=True,
            is_long_clickable=False,
            is_editable=False,
            is_scrollable=False,
            is_focusable=True,
            is_focused=False,
            is_selected=False,
            is_checked=False,
            is_enabled=True,
            is_visible=True,
            bbox_pixels=SimpleNamespace(
                x_min=108,
                y_min=240,
                x_max=324,
                y_max=480,
            ),
            bbox=SimpleNamespace(
                x_min=0.1,
                y_min=0.1,
                x_max=0.3,
                y_max=0.2,
            ),
        )
        description = fake_module._generate_ui_elements_description_list(
            [element],
            (1080, 2400),
        )
        self.assertIn('"bbox_1000": [100, 100, 300, 200]', description)
        self.assertIn('"bbox_pixels": [108, 240, 324, 480]', description)
        self.assertIn('"text": "Save"', description)

    def test_main_reuses_detailed_reporting_with_m3a(self):
        with mock.patch.object(
            runner.task_runner_detail,
            "run_evaluation",
            return_value=0,
        ) as run_evaluation:
            result = runner.main(
                [
                    "--model_path",
                    "/models/mobilerl-9b",
                    "--tasks",
                    "ContactsAddContact",
                ]
            )

        self.assertEqual(result, 0)
        kwargs = run_evaluation.call_args.kwargs
        self.assertEqual(kwargs["condition"], "vl_baseline")
        self.assertEqual(kwargs["model_info"]["backend"], "vllm-multimodal")
        self.assertEqual(kwargs["model_info"]["prompt_profile"], "mobilerl")
        self.assertEqual(kwargs["agent_name"], "m3a_vllm_vl_mobilerl")
        self.assertTrue(callable(kwargs["agent_factory"]))

    def test_vl_validation_rejects_invalid_image_limit(self):
        args = runner.build_parser().parse_args(["--max_images_per_prompt", "0"])
        with self.assertRaisesRegex(ValueError, "at least 1"):
            runner.validate_vl_args(args)


if __name__ == "__main__":
    unittest.main()
