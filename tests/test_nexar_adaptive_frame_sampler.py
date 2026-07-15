import tempfile
import unittest
from pathlib import Path

from tools.nexar_adaptive_frame_sampler import (
    desired_frame_count,
    plan_adaptive_frames,
    sample_video,
)


class _FakeCapture:
    def __init__(self, _path, fps=30.0, total_frames=900):
        self.fps = fps
        self.total_frames = total_frames
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == 5 else self.total_frames if prop == 7 else 0

    def set(self, prop, value):
        if prop == 1:
            self.position = int(value)
        return True

    def read(self):
        return True, {"frame_index": self.position}

    def release(self):
        self.released = True


class NexarAdaptiveFrameSamplerTests(unittest.TestCase):
    def test_budget_is_bounded(self):
        self.assertEqual(desired_frame_count(0.03), 6)
        self.assertEqual(desired_frame_count(1.0), 7)
        self.assertEqual(desired_frame_count(10.0), 12)

    def test_plan_gets_denser_toward_event(self):
        plan = plan_adaptive_frames(
            time_of_alert_s=10.0,
            time_of_event_s=13.0,
            fps=30.0,
            total_frames=1200,
        )
        indices = [item["frame_index"] for item in plan["frames"]]
        gaps = [right - left for left, right in zip(indices, indices[1:])]

        self.assertEqual(indices[0], 300)
        self.assertEqual(indices[-1], 390)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertGreater(gaps[0], gaps[-1])
        self.assertLessEqual(len(indices), 12)

    def test_very_short_window_uses_only_unique_available_frames(self):
        plan = plan_adaptive_frames(
            time_of_alert_s=10.0,
            time_of_event_s=10.033,
            fps=30.0,
            total_frames=1200,
        )

        self.assertTrue(plan["limited_by_available_frames"])
        self.assertEqual(plan["actual_frame_count"], 2)
        self.assertEqual(
            len({item["frame_index"] for item in plan["frames"]}),
            2,
        )

    def test_sample_video_writes_manifest_and_planned_images(self):
        written = []

        def fake_writer(path, frame, _params):
            written.append((Path(path).name, frame["frame_index"]))
            Path(path).write_bytes(b"jpg")
            return True

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "00000.mp4"
            video.write_bytes(b"video")
            output = Path(tmp) / "out"
            manifest = sample_video(
                video,
                {"time_of_alert": "10.0", "time_of_event": "11.0"},
                output,
                capture_factory=_FakeCapture,
                image_writer=fake_writer,
            )

            self.assertEqual(len(written), 7)
            self.assertEqual(manifest["actual_frame_count"], 7)
            self.assertTrue((output / "frames_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
