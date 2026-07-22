import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "generate_trafficcomposer_ir.py"
SPEC = importlib.util.spec_from_file_location("generate_trafficcomposer_ir", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrafficComposerGenerationTests(unittest.TestCase):
    def test_chat_endpoint_is_converted_to_sdk_base_url(self):
        self.assertEqual(
            MODULE.api_base_url("https://example.test/v1/chat/completions"),
            "https://example.test/v1",
        )
        self.assertEqual(MODULE.api_base_url("https://example.test/v1/"), "https://example.test/v1")

    def test_generate_one_uses_trafficcomposer_prompt_and_postprocess(self):
        raw = "prefix\n<YAML>\nenvironment:\n  weather: clear\nparticipant: {}\n</YAML>"
        create = unittest.mock.Mock(return_value=types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=raw))]
        ))
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        ))
        returned_raw, processed = MODULE.generate_one("a road scene", client)
        self.assertEqual(returned_raw, raw)
        self.assertIn("environment:", processed)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], MODULE.OPENAI_MODEL)
        self.assertGreaterEqual(len(kwargs["messages"]), 3)

    def test_description_files_are_sorted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "b.txt").write_text("b", encoding="utf-8")
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "skip.json").write_text("{}", encoding="utf-8")
            self.assertEqual([path.name for path in MODULE.description_files(root)], ["a.txt", "b.txt"])

    def test_generate_one_accepts_multimodal_messages(self):
        raw = "<YAML>\nparticipant: {}\n</YAML>"
        create = unittest.mock.Mock(return_value=types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=raw))]
        ))
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        ))
        messages = [{"role": "user", "content": [{"type": "text", "text": "scene"}]}]
        MODULE.generate_one("ignored", client, messages=messages)
        self.assertIs(create.call_args.kwargs["messages"], messages)

    def test_compact_multimodal_keeps_only_target_image(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,x"}}
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "demo"}, image]},
            {"role": "assistant", "content": "gold"},
            {"role": "user", "content": [{"type": "text", "text": "target"}, image]},
        ]
        compact = MODULE.compact_multimodal_messages(messages)
        self.assertEqual(len(compact[0]["content"]), 1)
        self.assertEqual(len(compact[2]["content"]), 2)


if __name__ == "__main__":
    unittest.main()
