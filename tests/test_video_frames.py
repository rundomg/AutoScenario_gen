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

from tools.video_frames import extract_anchor_frames


class FakeCapture:
    """Minimal cv2.VideoCapture stand-in driven by (fps, total_frames)."""

    CAP_PROPS = {5: "fps", 7: "total_frames", 1: "pos"}

    def __init__(self, fps=30.0, total_frames=300):
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


class ExtractAnchorFramesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # extract_anchor_frames checks os.path.exists(video_path).
        self.video_path = Path(self.tmp.name) / "video.mp4"
        self.video_path.write_bytes(b"\x00")
        self.written = []

    def _imwrite(self, path, frame):
        Path(path).write_text(str(frame))
        self.written.append(path)
        return True

    def test_extracts_start_and_end_with_correct_indices(self):
        cap = FakeCapture(fps=30.0, total_frames=300)
        out_dir = Path(self.tmp.name) / "frames"
        manifest = extract_anchor_frames(
            str(self.video_path),
            start_s=4.0,
            end_s=7.0,
            output_dir=str(out_dir),
            scene_id="s0000_c0",
            capture_factory=lambda _p: cap,
            imwrite=self._imwrite,
        )
        # 4s and 7s at 30 fps -> frame 120 and 210.
        self.assertEqual(cap.read_indices, [120, 210])
        self.assertEqual(manifest["fps"], 30.0)
        self.assertTrue(manifest["start_frame_path"].endswith("s0000_c0_start_frame.jpg"))
        self.assertTrue(manifest["end_anchor_frame_path"].endswith("s0000_c0_end_anchor_frame.jpg"))
        self.assertTrue((out_dir / "frames.json").exists())
        saved = json.loads((out_dir / "frames.json").read_text())
        self.assertEqual(saved["frames"][0]["frame_index"], 120)
        self.assertTrue(saved["frames"][0]["is_start"])
        self.assertTrue(saved["frames"][-1]["is_end_anchor"])

    def test_context_frames_sampled_between_anchors(self):
        cap = FakeCapture(fps=10.0, total_frames=200)
        out_dir = Path(self.tmp.name) / "frames2"
        manifest = extract_anchor_frames(
            str(self.video_path),
            start_s=2.0,
            end_s=5.0,
            output_dir=str(out_dir),
            scene_id="s0",
            context_sample_rate_s=1.0,
            capture_factory=lambda _p: cap,
            imwrite=self._imwrite,
        )
        # start(2s=20), ctx(3s=30), ctx(4s=40), end(5s=50).
        self.assertEqual(cap.read_indices, [20, 30, 40, 50])
        context = [f for f in manifest["frames"] if not f["is_start"] and not f["is_end_anchor"]]
        self.assertEqual(len(context), 2)

    def test_inverted_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            extract_anchor_frames(
                str(self.video_path), 5.0, 5.0, self.tmp.name, "s0",
                capture_factory=lambda _p: FakeCapture(), imwrite=self._imwrite,
            )

    def test_end_beyond_duration_rejected(self):
        cap = FakeCapture(fps=30.0, total_frames=300)  # 10 s long
        with self.assertRaises(ValueError):
            extract_anchor_frames(
                str(self.video_path), 1.0, 99.0, self.tmp.name, "s0",
                capture_factory=lambda _p: cap, imwrite=self._imwrite,
            )


if __name__ == "__main__":
    unittest.main()
