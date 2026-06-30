#!/usr/bin/env python
"""Run a generated AutoScenario CARLA script and record it to a video file.

The generated scenario scripts (e.g.
``results/.../accidents_4/s0000_c0_r000.py``) load a CARLA world, spawn the
ego plus the risk actors, switch the world into synchronous fixed-step mode and
drive it with ``world.tick()`` for the whole scenario duration.

In CARLA synchronous mode only the client that actually ticks the world reliably
receives sensor data, so a separate "listener" process gets zero camera frames.
This recorder therefore runs the scenario **in this same process** (via ``exec``
so the script is left untouched) and wraps ``carla.World.tick`` so that, after
every tick, it attaches an RGB camera to the ego (once spawned) and pulls the
matching frame. The frames are streamed into an MP4 file.

Usage
-----
    python tools/record_scenario.py path/to/s0000_c0_r000.py
    python tools/record_scenario.py path/to/s0000_c0_r000.py --out run.mp4 --view chase

The output video defaults to ``<scenario>.mp4`` next to the scenario script.
"""

import argparse
import os
import queue
import sys

import numpy as np

import carla


# ---------------------------------------------------------------------------
# Camera placement presets (relative to the ego vehicle).
# ---------------------------------------------------------------------------
def _camera_transform(view):
    if view == "front":
        return carla.Transform(
            carla.Location(x=1.2, y=0.0, z=1.6),
            carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
        )
    if view == "bev":
        return carla.Transform(
            carla.Location(x=0.0, y=0.0, z=35.0),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )
    # Default: third-person chase camera behind and above the ego.
    return carla.Transform(
        carla.Location(x=-6.5, y=0.0, z=3.5),
        carla.Rotation(pitch=-14.0, yaw=0.0, roll=0.0),
    )


def _find_ego(world):
    """Return the hero/ego vehicle if it is already spawned, else None."""
    try:
        vehicles = world.get_actors().filter("vehicle.*")
    except Exception:
        return None
    for vehicle in vehicles:
        try:
            if vehicle.attributes.get("role_name") == "hero":
                return vehicle
        except Exception:
            continue
    return None


def _image_to_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    return array[:, :, :3][:, :, ::-1].copy()  # -> RGB


class _VideoWriter:
    """Thin wrapper that prefers imageio(ffmpeg) and falls back to OpenCV."""

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self._backend = None
        self._writer = None
        try:
            import imageio.v2 as imageio  # noqa: WPS433 (runtime optional dep)

            # Force the FFMPEG plugin explicitly: without imageio-ffmpeg
            # installed imageio silently picks a wrong plugin (e.g. tifffile)
            # that only fails later when a frame is appended.
            self._writer = imageio.get_writer(
                path,
                format="FFMPEG",
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
            self._backend = "imageio"
            return
        except Exception:
            self._writer = None
        try:
            import cv2  # noqa: WPS433

            self._cv2 = cv2
            self._backend = "cv2"
        except Exception as exc:
            raise RuntimeError(
                "Neither imageio[ffmpeg] nor opencv-python is available to "
                "encode the video. Install one, e.g. `pip install imageio "
                "imageio-ffmpeg`."
            ) from exc

    def append(self, rgb_frame):
        if self._backend == "imageio":
            self._writer.append_data(rgb_frame)
            return
        cv2 = self._cv2
        if self._writer is None:
            height, width = rgb_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self.path, fourcc, self.fps, (width, height)
            )
        self._writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self._writer is None:
            return
        try:
            if self._backend == "imageio":
                self._writer.close()
            else:
                self._writer.release()
        except Exception:
            pass


class _Recorder:
    """Attaches a camera to the ego and grabs one frame per world tick."""

    def __init__(self, out_path, args, fps):
        self.out_path = out_path
        self.args = args
        self.fps = fps
        self.camera = None
        self.frame_queue = queue.Queue()
        self.writer = _VideoWriter(out_path, fps)
        self.frames_written = 0
        self._spawned_this_tick = False

    def _attach_camera(self, world):
        ego = _find_ego(world)
        if ego is None:
            return
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.args.width))
        camera_bp.set_attribute("image_size_y", str(self.args.height))
        camera_bp.set_attribute("fov", str(self.args.fov))
        camera_bp.set_attribute("sensor_tick", "0.0")  # one frame per world tick
        self.camera = world.spawn_actor(
            camera_bp, _camera_transform(self.args.view), attach_to=ego
        )
        self.camera.listen(self.frame_queue.put)
        self._spawned_this_tick = True
        print(f"[record] ego found (id={ego.id}); {self.args.view} camera attached.")

    def on_tick(self, world):
        # Lazily attach the camera the first tick the ego exists.
        if self.camera is None:
            self._attach_camera(world)
            return
        # The camera spawned during a tick produces its first frame only on the
        # *next* tick, so skip the grab on the spawn tick to avoid a stall.
        if self._spawned_this_tick:
            self._spawned_this_tick = False
            return
        try:
            image = self.frame_queue.get(timeout=2.0)
        except queue.Empty:
            return
        self.writer.append(_image_to_rgb(image))
        self.frames_written += 1

    def close(self):
        # Always tear the camera down first so a writer error can't leak the
        # sensor into the live simulation.
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            try:
                self.camera.destroy()
            except Exception:
                pass
            self.camera = None
        # Drain any frames still queued after the scenario loop ended.
        while True:
            try:
                image = self.frame_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self.writer.append(_image_to_rgb(image))
                self.frames_written += 1
            except Exception:
                break
        self.writer.close()


def record(args):
    out_path = os.path.abspath(
        args.out or os.path.splitext(os.path.abspath(args.scenario))[0] + ".mp4"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    tick_dt = float(os.environ.get("AUTOSCENARIO_RISK_TICK_DT", "0.05"))
    fps = max(1, int(round(1.0 / max(tick_dt, 1e-3))))

    recorder = _Recorder(out_path, args, fps)

    # Wrap carla.World.tick so every tick the running scenario performs also
    # advances our recording. Patching at the class level catches the world
    # object the scenario creates internally without needing a reference to it.
    original_tick = carla.World.tick

    def patched_tick(self, *call_args, **call_kwargs):
        result = original_tick(self, *call_args, **call_kwargs)
        try:
            recorder.on_tick(self)
        except Exception as exc:  # never let recording break the scenario
            print(f"[record] frame capture error: {exc}")
        return result

    carla.World.tick = patched_tick

    scenario_path = os.path.abspath(args.scenario)
    print(f"[record] running scenario in-process: {scenario_path}")
    with open(scenario_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    code = compile(source, scenario_path, "exec")
    namespace = {"__name__": "__main__", "__file__": scenario_path}

    try:
        exec(code, namespace)  # noqa: S102 (running a generated scenario by design)
    finally:
        carla.World.tick = original_tick
        recorder.close()

    duration = recorder.frames_written / float(fps)
    print(
        f"[record] wrote {recorder.frames_written} frames "
        f"({duration:.1f}s @ {fps}fps) -> {out_path}"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run a generated CARLA scenario script and record it to MP4."
    )
    parser.add_argument("scenario", help="Path to the generated scenario .py script.")
    parser.add_argument("--out", help="Output video path (default: <scenario>.mp4).")
    parser.add_argument(
        "--view",
        choices=["chase", "front", "bev"],
        default="chase",
        help="Camera viewpoint relative to the ego (default: chase).",
    )
    parser.add_argument("--width", type=int, default=1280, help="Frame width.")
    parser.add_argument("--height", type=int, default=720, help="Frame height.")
    parser.add_argument("--fov", type=float, default=90.0, help="Camera field of view.")
    args = parser.parse_args()

    if not os.path.exists(args.scenario):
        parser.error(f"scenario script not found: {args.scenario}")
    sys.exit(record(args))


if __name__ == "__main__":
    main()
