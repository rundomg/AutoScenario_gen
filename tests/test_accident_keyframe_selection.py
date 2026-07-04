import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("requests", types.SimpleNamespace())
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))

from agents.accident_keyframe_selector import AccidentKeyframeSelector
from tools.accident_keyframe_sampler import sample_video_frames


class FakeCapture:
    def __init__(self, fps=10.0, total_frames=30):
        self._fps = fps
        self._total = total_frames
        self.pos = 0
        self.read_indices = []

    def isOpened(self):
        return True

    def get(self, prop_id):
        if prop_id == 5:
            return self._fps
        if prop_id == 7:
            return self._total
        return 0

    def set(self, prop_id, value):
        if prop_id == 1:
            self.pos = int(value)

    def read(self):
        self.read_indices.append(self.pos)
        return True, f"frame@{self.pos}"

    def release(self):
        pass


class AccidentKeyframeSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.video_path = Path(self.tmp.name) / "accid.mp4"
        self.video_path.write_bytes(b"\x00")

    def _imwrite(self, path, frame):
        Path(path).write_text(str(frame))
        return True

    def test_samples_two_frames_per_second(self):
        capture = FakeCapture(fps=10.0, total_frames=30)
        manifest = sample_video_frames(
            str(self.video_path),
            str(Path(self.tmp.name) / "frames"),
            scene_id="accid",
            samples_per_second=2.0,
            capture_factory=lambda _p: capture,
            imwrite=self._imwrite,
        )

        self.assertEqual(capture.read_indices, [0, 5, 10, 15, 20, 25])
        self.assertEqual([f["timestamp_s"] for f in manifest["frames"]], [0, 0.5, 1, 1.5, 2, 2.5])
        self.assertTrue((Path(self.tmp.name) / "frames" / "frames_manifest.json").exists())

    def test_selector_accepts_valid_json(self):
        selector = AccidentKeyframeSelector()
        payload = {
            "collision_second": 1.5,
        }
        output = Path(self.tmp.name) / "selection.json"
        output.write_text(json.dumps(payload), encoding="utf-8")

        parsed, error = selector.extract_decision_data(str(output))
        self.assertIsNone(error)
        self.assertEqual(parsed["collision_second"], 1.5)

    def test_selector_rejects_extra_keys(self):
        selector = AccidentKeyframeSelector()
        payload = {
            "collision_second": 1.5,
            "collision_confidence": "medium",
        }
        output = Path(self.tmp.name) / "selection_extra.json"
        output.write_text(json.dumps(payload), encoding="utf-8")

        parsed, error = selector.extract_decision_data(str(output))
        self.assertIsNone(parsed)
        self.assertIn("Unexpected keys", error)


if __name__ == "__main__":
    unittest.main()
