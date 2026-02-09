"""
Passive Recording - Capture real mouse movements during normal computer use.

Records cursor positions and clicks in the background. On each click, segments
the trajectory from the previous action, applies Savitzky-Golay smoothing to
reduce noise, and saves in TrajectoryData format for TRM training.

Usage:
    uv run python -m anorha_control.exploration.passive_recorder
    uv run python -m anorha_control.exploration.passive_recorder --screenshots --output-dir data/trajectories
"""
import argparse
import json
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    from pynput import mouse
except ImportError:
    mouse = None

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None


@dataclass
class TrajectoryData:
    """Trajectory data for TRM training. Matches smart_data_gatherer format."""
    task_id: str
    target: Dict[str, float]
    trajectory: List[Dict[str, Any]]
    success: bool = True
    screen_size: List[int] = None
    task_type: str = "click"
    target_label: str = ""
    task_category: str = ""
    source: str = "human"
    timestamp: str = ""
    normalized: bool = True
    screenshot_path: str = ""

    def __post_init__(self):
        if self.screen_size is None:
            self.screen_size = [1920, 1080]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _smooth_trajectory(
    points: List[Tuple[float, float, float]],
    window_length: int = 31,
    polyorder: int = 4,
) -> List[Dict[str, Any]]:
    """
    Apply Savitzky-Golay smoothing and compute velocity.
    points: list of (t_ms, x, y)
    Returns list of {t, x, y, vx, vy, click}.
    """
    if len(points) < 3:
        return []
    if savgol_filter is None:
        # Fallback: no smoothing, compute velocity from raw
        return _trajectory_with_velocity_raw(points)

    t_arr = [p[0] for p in points]
    x_arr = [p[1] for p in points]
    y_arr = [p[2] for p in points]

    n = len(points)
    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl < 3:
        return _trajectory_with_velocity_raw(points)
    poly = min(polyorder, wl - 1)

    try:
        x_smooth = savgol_filter(x_arr, wl, poly, mode="nearest")
        y_smooth = savgol_filter(y_arr, wl, poly, mode="nearest")
    except Exception:
        return _trajectory_with_velocity_raw(points)

    result = []
    for i in range(n):
        t = t_arr[i]
        x = float(x_smooth[i])
        y = float(y_smooth[i])
        if i < n - 1:
            dt = (t_arr[i + 1] - t) / 1000.0
            vx = (x_smooth[i + 1] - x_smooth[i]) / dt if dt > 0 else 0
            vy = (y_smooth[i + 1] - y_smooth[i]) / dt if dt > 0 else 0
        else:
            vx = result[-1]["vx"] if result else 0
            vy = result[-1]["vy"] if result else 0
        result.append({
            "t": t,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "click": i == n - 1,
        })
    return result


def _trajectory_with_velocity_raw(points: List[Tuple[float, float, float]]) -> List[Dict[str, Any]]:
    """Compute trajectory with velocity from raw points (no smoothing)."""
    result = []
    for i in range(len(points)):
        t, x, y = points[i]
        if i < len(points) - 1:
            dt = (points[i + 1][0] - t) / 1000.0
            vx = (points[i + 1][1] - x) / dt if dt > 0 else 0
            vy = (points[i + 1][2] - y) / dt if dt > 0 else 0
        else:
            vx = result[-1]["vx"] if result else 0
            vy = result[-1]["vy"] if result else 0
        result.append({
            "t": t,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "click": i == len(points) - 1,
        })
    return result


def _count_direction_reversals(points: List[Tuple[float, float, float]]) -> int:
    """Count significant direction reversals (zigzag)."""
    if len(points) < 4:
        return 0
    count = 0
    for i in range(2, len(points)):
        dx1 = points[i - 1][1] - points[i - 2][1]
        dy1 = points[i - 1][2] - points[i - 2][2]
        dx2 = points[i][1] - points[i - 1][1]
        dy2 = points[i][2] - points[i - 1][2]
        dot = dx1 * dx2 + dy1 * dy2
        if dot < -100:  # Significant reversal
            count += 1
    return count


def _normalize_trajectory(
    trajectory: List[Dict[str, Any]],
    target: Tuple[float, float],
    screen_w: int,
    screen_h: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Normalize trajectory and target to 0-1 range."""
    target_norm = {"x": target[0] / screen_w, "y": target[1] / screen_h}
    traj_norm = []
    for p in trajectory:
        traj_norm.append({
            "t": p["t"],
            "x": p["x"] / screen_w,
            "y": p["y"] / screen_h,
            "vx": p.get("vx", 0) / screen_w,
            "vy": p.get("vy", 0) / screen_h,
            "click": p.get("click", False),
        })
    return traj_norm, target_norm


class PassiveRecorder:
    """Records mouse movements and clicks, exports as TrajectoryData."""

    def __init__(
        self,
        output_dir: Path,
        min_interval_ms: int = 16,
        buffer_seconds: float = 5.0,
        max_trajectory_seconds: float = 2.0,
        min_points: int = 3,
        max_direction_reversals: int = 10,
        save_screenshots: bool = False,
        smooth: bool = True,
        smooth_window: int = 31,
        smooth_polyorder: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_ms = min_interval_ms
        self.buffer_seconds = buffer_seconds
        self.max_trajectory_seconds = max_trajectory_seconds
        self.min_points = min_points
        self.max_direction_reversals = max_direction_reversals
        self.save_screenshots = save_screenshots
        self.smooth = smooth
        self.smooth_window = smooth_window
        self.smooth_polyorder = smooth_polyorder

        self._buffer: deque = deque(maxlen=int(buffer_seconds * 1000 / min_interval_ms) + 100)
        self._last_click_time: Optional[float] = None
        self._last_move_time: float = 0
        self._trajectories: List[TrajectoryData] = []
        self._lock = threading.Lock()
        self._running = False
        self._listener: Optional[mouse.Listener] = None

        if pyautogui:
            self._screen_w, self._screen_h = pyautogui.size()
        else:
            self._screen_w, self._screen_h = 1920, 1080

        if save_screenshots:
            self._screenshots_dir = self.output_dir / "screenshots"
            self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._screenshots_dir = None

    def _on_move(self, x: float, y: float):
        if not self._running:
            return
        now = time.time()
        if (now - self._last_move_time) * 1000 < self.min_interval_ms:
            return
        self._last_move_time = now
        with self._lock:
            t_ms = int((now - self._start_time) * 1000) if hasattr(self, "_start_time") else 0
            self._buffer.append((now, x, y))

    def _on_click(self, x: float, y: float, button, pressed):
        if not pressed or not self._running:
            return
        now = time.time()
        with self._lock:
            self._process_click(now, x, y)

    def _process_click(self, click_time: float, click_x: float, click_y: float):
        """Extract trajectory from buffer and save."""
        if not self._buffer:
            return

        # Segment: from last_click (or max_trajectory_seconds ago) to now
        cutoff = click_time - self.max_trajectory_seconds
        if self._last_click_time is not None:
            cutoff = max(cutoff, self._last_click_time)

        points: List[Tuple[float, float, float]] = []
        first_t = None
        for t, px, py in self._buffer:
            if t >= cutoff and t <= click_time:
                if first_t is None:
                    first_t = t
                t_ms = int((t - first_t) * 1000)
                points.append((t_ms, px, py))

        self._last_click_time = click_time

        if len(points) < self.min_points:
            return

        # Quality filter: direction reversals
        if _count_direction_reversals(points) > self.max_direction_reversals:
            return

        # Smooth
        if self.smooth and savgol_filter:
            trajectory = _smooth_trajectory(
                points,
                window_length=self.smooth_window,
                polyorder=self.smooth_polyorder,
            )
        else:
            trajectory = _trajectory_with_velocity_raw(points)

        if len(trajectory) < self.min_points:
            return

        # Ensure last point is at click
        trajectory[-1]["x"] = click_x
        trajectory[-1]["y"] = click_y
        trajectory[-1]["click"] = True

        # Normalize
        target = (click_x, click_y)
        traj_norm, target_norm = _normalize_trajectory(
            trajectory, target, self._screen_w, self._screen_h
        )

        # Screenshot (optional)
        screenshot_path = ""
        if self.save_screenshots and self._screenshots_dir:
            try:
                from ..utils.screen import ScreenCapture
                screen = ScreenCapture()
                img = screen.capture()
                task_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self._trajectories):04d}"
                img_path = self._screenshots_dir / f"{task_id}.jpg"
                img.resize((384, 384), Image.Resampling.LANCZOS).save(img_path, quality=85)
                screenshot_path = str(img_path.relative_to(self.output_dir))
            except Exception:
                pass

        traj_data = TrajectoryData(
            task_id=f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self._trajectories):04d}",
            target=target_norm,
            trajectory=traj_norm,
            success=True,
            screen_size=[self._screen_w, self._screen_h],
            task_type="click",
            target_label="click" if screenshot_path else "",
            source="human",
            timestamp=datetime.now().isoformat(),
            normalized=True,
            screenshot_path=screenshot_path,
        )
        self._trajectories.append(traj_data)

    def _save_batch(self):
        with self._lock:
            if not self._trajectories:
                return
            batch = list(self._trajectories)
            self._trajectories.clear()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = self.output_dir / f"passive_trajectories_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump([t.to_dict() for t in batch], f, indent=2)
        print(f"   Saved {len(batch)} trajectories to {out_file.name}")

    def start(self):
        if not mouse:
            print("Error: pynput not installed. Run: uv pip install pynput")
            sys.exit(1)
        self._running = True
        self._start_time = time.time()
        self._listener = mouse.Listener(on_move=self._on_move, on_click=self._on_click)
        self._listener.start()
        print("Passive recorder started. Move mouse and click normally. Ctrl+C to stop.")

    def stop(self):
        self._running = False
        if self._listener:
            self._listener.stop()
            self._listener = None
        self._save_batch()


def main():
    parser = argparse.ArgumentParser(description="Passive mouse recording for TRM training")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/trajectories",
        help="Output directory for trajectory JSON files",
    )
    parser.add_argument(
        "--min-interval-ms",
        type=int,
        default=16,
        help="Minimum interval between position samples (ms). ~16 = 60Hz",
    )
    parser.add_argument(
        "--screenshots",
        action="store_true",
        help="Capture screenshot at each click (for Vision TRM)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable Savitzky-Golay smoothing",
    )
    args = parser.parse_args()

    recorder = PassiveRecorder(
        output_dir=Path(args.output_dir),
        min_interval_ms=args.min_interval_ms,
        save_screenshots=args.screenshots,
        smooth=not args.no_smooth,
    )

    try:
        recorder.start()
        while True:
            time.sleep(5)
            if recorder._trajectories:
                recorder._save_batch()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recorder.stop()
        print("Done.")


if __name__ == "__main__":
    main()
