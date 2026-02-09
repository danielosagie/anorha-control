"""
Smart Data Gatherer - VLM-guided exploration for TRM training data.

This replaces the old random exploration with intelligent, VLM-guided actions.
Outputs trajectory data in the format required for TRM training.

Usage:
    uv run python -m anorha_control.exploration.smart_data_gatherer

Features:
- VLM-guided action selection (not random!)
- Records mouse trajectories in TRM training format
- Progress tracking with targets
- Single-command headless data gathering
- Automatic data classification for training quality
"""
import asyncio
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import os

from PIL import Image

# Local imports
from ..models.vlm_subsystems import VLMSubsystems, GroundingResult
from .grounding_harness import GroundingHarness, GroundingHarnessResult, _get_synonyms
from ..models.local_llm import LocalLLM, TaskPlanner
from ..utils.overlay import get_indicator
from .task_curriculum import TaskCurriculum, Difficulty
from .execution_backend import BrowserBackend, DesktopBackend, ExecutionBackend

try:
    import torch
    import warnings
    warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning, module="torch.utils.data")
except ImportError:
    torch = None


@dataclass
class TrajectoryPoint:
    """A single point in a mouse trajectory."""
    t: int  # Timestamp in ms
    x: int
    y: int
    vx: float = 0.0  # Velocity x
    vy: float = 0.0  # Velocity y
    click: bool = False


@dataclass
class TrajectoryData:
    """Complete trajectory data for TRM training. Uses normalized coords (0-1) for scale across screen sizes."""
    task_id: str
    target: Dict[str, float]  # {x, y} in 0-1 normalized
    trajectory: List[Dict[str, Any]] = field(default_factory=list)  # x, y, vx, vy in 0-1
    success: bool = False
    screen_size: List[int] = field(default_factory=lambda: [1920, 1080])
    task_type: str = ""  # click, type, scroll
    target_label: str = ""  # e.g. "Submit", "Username" - for Vision TRM training
    task_category: str = ""  # TaskCategory.value for task embedding (forms, precision, etc.)
    source: str = "vlm"  # vlm, human, random
    timestamp: str = ""
    normalized: bool = True  # coords are 0-1; screen_size used for denormalization when executing
    screenshot_path: str = ""  # Optional: path to saved screenshot for Vision TRM training
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GatheringProgress:
    """Tracks data gathering progress."""
    total_trajectories: int = 0
    successful_trajectories: int = 0
    failed_trajectories: int = 0
    episodes_completed: int = 0
    start_time: float = field(default_factory=time.time)
    
    # Rolling window for ETA (last N episodes)
    _recent_episodes: List[Tuple[float, int]] = field(default_factory=list, repr=False)  # (end_time, successful_count)
    ROLLING_WINDOW: int = 25  # Larger window = smoother ETA
    MIN_SPAN_MINUTES: float = 15.0  # Use overall rate until we have enough recent data
    
    # Training targets
    target_trajectories: int = 5000
    target_success_rate: float = 0.80
    
    def record_episode(self, end_time: float, successful_count: int):
        """Record episode for rolling ETA. Keeps last N episodes."""
        self._recent_episodes.append((end_time, successful_count))
        if len(self._recent_episodes) > self.ROLLING_WINDOW:
            self._recent_episodes.pop(0)
    
    @property
    def success_rate(self) -> float:
        if self.total_trajectories == 0:
            return 0.0
        return self.successful_trajectories / self.total_trajectories
    
    @property
    def progress_percent(self) -> float:
        return min(100, (self.successful_trajectories / self.target_trajectories) * 100)
    
    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600
    
    def _overall_speed(self) -> float:
        """Speed based on full run (used when rolling window is insufficient)."""
        if self.elapsed_hours <= 0:
            return 0.0
        return self.successful_trajectories / self.elapsed_hours
    
    @property
    def trajectories_per_hour(self) -> float:
        """Speed: rolling window when span >= MIN_SPAN_MINUTES, else overall rate."""
        return self._trajectories_per_hour_impl()[0]

    def _trajectories_per_hour_impl(self) -> Tuple[float, str]:
        """Returns (speed, source_label) for display."""
        overall = self._overall_speed()
        if len(self._recent_episodes) < 2:
            return overall, "overall"
        first_time = self._recent_episodes[0][0]
        last_time = self._recent_episodes[-1][0]
        span_minutes = (last_time - first_time) / 60
        if span_minutes < self.MIN_SPAN_MINUTES:
            return overall, "overall"
        span_hours = span_minutes / 60
        successful_in_window = sum(c for _, c in self._recent_episodes)
        rolling_speed = successful_in_window / span_hours
        if rolling_speed <= 0:
            return overall, "overall"
        n = len(self._recent_episodes)
        return rolling_speed, f"last {n} ep"
    
    @property
    def eta_hours(self) -> float:
        """ETA to hit target based on speed. Uses overall rate when rolling is sparse."""
        speed = self.trajectories_per_hour
        if speed <= 0:
            return float('inf')
        remaining = self.target_trajectories - self.successful_trajectories
        if remaining <= 0:
            return 0.0
        return remaining / speed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trajectories": self.total_trajectories,
            "successful_trajectories": self.successful_trajectories,
            "failed_trajectories": self.failed_trajectories,
            "success_rate": f"{self.success_rate:.2%}",
            "progress_percent": f"{self.progress_percent:.1f}%",
            "elapsed_hours": f"{self.elapsed_hours:.1f}h",
            "trajectories_per_hour": f"{self.trajectories_per_hour:.0f}/h",
            "eta_hours": f"{self.eta_hours:.1f}h" if self.eta_hours < 1000 else "calculating..."
        }


@dataclass
class GathererConfig:
    """Configuration for smart data gathering."""
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 800
    
    # VLM settings - moondream most reliable for JSON; llava often returns garbage
    vlm_model: str = "moondream"  # Stable. Alt: llava-phi3, Me7war/Astria
    vlm_backend: str = "ollama"
    vlm_url: str = "http://localhost:11434"
    vlm_timeout: float = 600.0  # CPU inference can take 10+ min for 5k tokens
    use_gpu: bool = True  # Enable GPU for OCR (and VLM via Ollama when available)
    # Resize images before sending to VLM (reduces tokens, faster inference)
    # (768, 480) = ~3x faster. None = full resolution.
    vlm_image_max_size: Optional[Tuple[int, int]] = (768, 480)
    
    # Data settings
    data_dir: Path = Path("data/trajectories")
    save_every: int = 10  # Save after N successful trajectories
    save_screenshots: bool = False  # Save screenshots for Vision TRM training (crap-top path)
    target_trajectories: int = 100000  # Target: 100k for good TRM training
    
    # Episode settings
    max_episode_steps: int = 15
    grounding_timeout: float = 30.0  # Cap grounding (DOM+OCR+VLM) at 30s to fail fast
    step_timeout: float = 90.0  # Per-step safety cap (grounding + execution)
    grounding_refinement_max: int = 2  # Retry grounding with alternative phrasings when no visible change
    grounding_model: str = "vlm"  # "vlm" | "uground" | "vision_trm" | "anorha_trm" (unified)
    uground_4bit: bool = False  # Use 4-bit quant for UGround (laptop, ~2GB VRAM)
    vision_trm_checkpoint: str = "checkpoints/vision_trm_best.pt"
    anorha_trm_checkpoint: str = "checkpoints/anorha_trm_best.pt"
    action_delay: float = 0.2  # Fallback when action type unknown
    # Action-specific delays: type needs longer for phash to detect input change
    action_delay_type: float = 0.4
    action_delay_click: float = 0.3
    action_delay_press_enter: float = 0.5
    # Success detection: phash threshold (sim < X = change detected). Higher = more lenient.
    phash_success_threshold: float = 0.99  # 0.98 was too strict; subtle changes (typing, dropdowns) missed
    # Extra sample times for slow sites (form submit, navigation). Added to base sample_times.
    phash_extra_wait_times: List[float] = field(default_factory=lambda: [1.0, 1.5])  # s
    # Longer waits for submit/login/calculate clicks (navigation can take 2–3s)
    phash_extra_wait_times_submit: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])  # s
    use_vlm_verification: bool = False  # VLM confirm success (slow, ~1 call per trajectory)
    replan_on_failure: bool = True  # Re-plan when step fails (no visible change) so VLM can adapt
    max_failure_replans: int = 2  # Max re-plans per episode due to failure (avoid thrashing)
    max_difficulty: Difficulty = Difficulty.MEDIUM
    execution_backend: str = "browser"  # "browser" or "desktop" for full computer use
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)


class SmartDataGatherer:
    """
    VLM-guided data gatherer for TRM training.
    
    Unlike random exploration (17% success), this uses VLM to:
    1. Plan intelligent actions
    2. Ground elements precisely  
    3. Verify action success
    4. Record high-quality trajectories
    
    Target: 5000+ successful trajectories for TRM training
    """
    
    def __init__(self, config: GathererConfig = None):
        self.config = config or GathererConfig()
        
        # Progress tracking
        self.progress = GatheringProgress(target_trajectories=self.config.target_trajectories)
        self.trajectories: List[TrajectoryData] = []
        
        # VLM subsystems
        self.vlm = VLMSubsystems(
            model=self.config.vlm_model,
            base_url=self.config.vlm_url,
            backend_type=self.config.vlm_backend,
            use_ocr_gpu=self.config.use_gpu,
            timeout=self.config.vlm_timeout,
        )
        
        # Check VLM availability
        self._check_vlm_connection()
        
        # Task curriculum
        self.curriculum = TaskCurriculum(max_difficulty=self.config.max_difficulty)
        
        # Execution backend (browser or desktop)
        if self.config.execution_backend == "desktop":
            self.backend: ExecutionBackend = DesktopBackend()
        else:
            self.backend = BrowserBackend(
                viewport_width=self.config.viewport_width,
                viewport_height=self.config.viewport_height,
                headless=self.config.headless,
            )
        
        # Control
        self._running = False
        self._paused = False
        self._killed = False
        
        # Overlay for hotkeys - configured for viewport size ONLY if visible
        # If headless, we want it on the whole screen so user sees it
        overlay_w = None if self.config.headless else self.config.viewport_width
        overlay_h = None if self.config.headless else self.config.viewport_height
        
        self.indicator = get_indicator(
            on_kill=self._on_kill, 
            on_pause=self._on_pause,
            width=overlay_w,
            height=overlay_h
        )
        
        # Robust grounding harness (multi-strategy for 97% accuracy)
        uground_backend = None
        vision_trm_backend = None
        anorha_trm_backend = None
        if self.config.grounding_model == "anorha_trm":
            try:
                from ..training.unified_trm import AnorhaTRMBackend
                ckpt = getattr(self.config, "anorha_trm_checkpoint", "checkpoints/anorha_trm_best.pt")
                if Path(ckpt).exists():
                    anorha_trm_backend = AnorhaTRMBackend(ckpt)
                    print("[DataGatherer] Anorha TRM grounding enabled (unified, crap-top)")
                else:
                    print(f"[DataGatherer] Anorha TRM checkpoint not found: {ckpt}, using VLM")
            except Exception as e:
                print(f"[DataGatherer] Anorha TRM init failed: {e}, using VLM")
        elif self.config.grounding_model == "vision_trm":
            try:
                from ..training.vision_trm_training import VisionTRMBackend
                ckpt = getattr(self.config, "vision_trm_checkpoint", "checkpoints/vision_trm_best.pt")
                if Path(ckpt).exists():
                    vision_trm_backend = VisionTRMBackend(ckpt)
                    print("[DataGatherer] Vision TRM grounding enabled (crap-top, fully local)")
                else:
                    print(f"[DataGatherer] Vision TRM checkpoint not found: {ckpt}, using VLM")
            except Exception as e:
                print(f"[DataGatherer] Vision TRM init failed: {e}, using VLM")
        elif self.config.grounding_model == "uground":
            try:
                from ..models.uground_backend import UGroundBackend, get_uground_available
                if get_uground_available():
                    import torch
                    ug_device = "cuda:0" if (self.config.use_gpu and torch.cuda.is_available()) else "auto"
                    uground_backend = UGroundBackend(
                        load_in_4bit=getattr(self.config, "uground_4bit", False),
                        device=ug_device,
                    )
                    print(f"[DataGatherer] UGround grounding enabled (GUI-specialized) [device={ug_device}]")
                    print("[DataGatherer] Pre-loading UGround (~30s)...")
                    uground_backend.preload()
                    print("[DataGatherer] UGround ready.")
                else:
                    print("[DataGatherer] UGround requested but transformers not available, using VLM")
            except Exception as e:
                print(f"[DataGatherer] UGround init failed: {e}, using VLM")
        self.grounding_harness = GroundingHarness(
            self.vlm,
            uground_backend=uground_backend,
            vision_trm_backend=vision_trm_backend,
            anorha_trm_backend=anorha_trm_backend,
        )
        
        # Load existing progress
        self._load_progress()
        
        if self.config.use_gpu:
            import torch
            if torch.cuda.is_available():
                print(f"[DataGatherer] 🚀 GPU acceleration enabled (CUDA: {torch.cuda.get_device_name(0)})")
            else:
                print("[DataGatherer] GPU requested but CUDA unavailable; using CPU (check PyTorch CUDA install)")
    
    def _port_from_url(self, url: str) -> int:
        """Extract port from URL, default 8080."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.port or 8080
    
    def _resize_for_vlm(
        self, img: Image.Image
    ) -> Tuple[Image.Image, float, float]:
        """
        Resize image for VLM to reduce tokens when vision encoder is on CPU.
        Returns (resized_img, scale_x, scale_y) for converting coords back.
        """
        max_size = self.config.vlm_image_max_size
        if not max_size:
            return img, 1.0, 1.0
        w, h = img.size
        mx, my = max_size
        if w <= mx and h <= my:
            return img, 1.0, 1.0
        ratio = min(mx / w, my / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        scale_x = w / new_w
        scale_y = h / new_h
        return resized, scale_x, scale_y
    
    def _check_vlm_connection(self):
        """Check if VLM backend is available and show helpful message if not."""
        import requests
        
        backend = self.config.vlm_backend
        url = self.config.vlm_url
        
        if backend == "llamacpp":
            try:
                r = requests.get(f"{url}/health", timeout=2)
                if r.status_code == 200:
                    print(f"[VLM] ✅ llama.cpp server running at {url}")
                    if self.config.use_gpu:
                        port = self._port_from_url(url)
                        print(f"   💡 If VLM is slow (~30s/image), restart with GPU:")
                        print(f"      uv run python -m anorha_control.model_server start {self.config.vlm_model} --port {port}")
                    return True
            except:
                pass
            
            print(f"[VLM] ⚠️ llama.cpp server not responding at {url}")
            port = self._port_from_url(url)
            print(f"      Start with GPU: uv run python -m anorha_control.model_server start {self.config.vlm_model} --port {port}")
            print(f"      Or use --start-server with gather to auto-start")
            print(f"      Or use Ollama: remove --llamacpp flag")
            
        else:  # ollama
            try:
                r = requests.get(f"{url}/api/tags", timeout=2)
                if r.status_code == 200:
                    print(f"[VLM] ✅ Ollama running at {url}")
                    models = [m.get("name", "") for m in r.json().get("models", [])]
                    if not any("llava" in m.lower() or "astria" in m.lower() for m in models):
                        print(f"[VLM] 💡 Pull a VLM: ollama pull llava")
                    return True
            except:
                pass
            
            print(f"[VLM] ⚠️ Ollama not responding at {url}")
            print(f"      Start: ollama serve")
            print(f"      Pull: ollama pull llava  (or Me7war/Astria)")
        
        return False

    
    def _on_kill(self):
        print("\n🛑 Kill switch triggered!")
        self._killed = True
        self._running = False
    
    def _on_pause(self, paused: bool):
        self._paused = paused
        print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
    
    def _load_progress(self):
        """Load existing trajectories and progress."""
        progress_file = self.config.data_dir / "progress.json"
        if progress_file.exists():
            with open(progress_file) as f:
                data = json.load(f)
                self.progress.successful_trajectories = data.get("successful_trajectories", 0)
                self.progress.failed_trajectories = data.get("failed_trajectories", 0)
                self.progress.total_trajectories = data.get("total_trajectories", 0)
                self.progress.episodes_completed = data.get("episodes_completed", 0)
                print(f"[DataGatherer] Loaded progress: {self.progress.successful_trajectories} successful trajectories")
    
    def _save_progress(self):
        """Save current progress."""
        progress_file = self.config.data_dir / "progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                "successful_trajectories": self.progress.successful_trajectories,
                "failed_trajectories": self.progress.failed_trajectories,
                "total_trajectories": self.progress.total_trajectories,
                "episodes_completed": self.progress.episodes_completed,
                "target_trajectories": self.progress.target_trajectories,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
    
    def _save_trajectories(self, batch: List[TrajectoryData]):
        """Save a batch of trajectories."""
        if not batch:
            return
        
        # Save to timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = self.config.data_dir / f"trajectories_{timestamp}.json"
        
        with open(data_file, 'w') as f:
            json.dump([t.to_dict() for t in batch], f, indent=2)
        
        print(f"   💾 Saved {len(batch)} trajectories to {data_file.name}")
        self._save_progress()
    
    async def _init_backend(self):
        """Initialize execution backend (browser or desktop)."""
        if isinstance(self.backend, BrowserBackend):
            await self.backend.init()
    
    async def _close_backend(self):
        """Close backend (browser)."""
        if isinstance(self.backend, BrowserBackend):
            await self.backend.close()
    
    async def _ensure_backend(self) -> bool:
        """Ensure backend is ready. Returns True if ready."""
        return await self.backend.ensure_ready()
    
    async def _screenshot(self) -> Image.Image:
        """Capture screenshot."""
        return await self.backend.screenshot()
    
    def _is_page_crash(self, e: Exception) -> bool:
        """Check if error indicates page crash (needs browser restart)."""
        msg = str(e).lower()
        return "crashed" in msg or "page closed" in msg
    
    async def _navigate(self, url: str) -> bool:
        """Navigate to URL or desktop context."""
        for attempt in range(3):
            try:
                ok = await self.backend.navigate(url)
                if ok:
                    await asyncio.sleep(1)
                return ok
            except Exception as e:
                print(f"   ⚠️ Navigation failed (attempt {attempt + 1}/3): {str(e)[:80]}...")
                if self._is_page_crash(e) and isinstance(self.backend, BrowserBackend):
                    print("   🔧 Browser crashed - restarting...")
                    await self._close_backend_safe()
                    await asyncio.sleep(3)
                    await self._init_backend()
                elif attempt < 2:
                    await asyncio.sleep(2)
        return False
    
    async def _record_trajectory(
        self, 
        start: Tuple[int, int], 
        end: Tuple[int, int],
        duration_ms: int = 500
    ) -> List[TrajectoryPoint]:
        """
        Record a mouse trajectory from start to end.
        Simulates smooth movement and records points.
        """
        trajectory = []
        steps = max(5, duration_ms // 50)  # ~50ms per point
        
        for i in range(steps + 1):
            t = i * (duration_ms // steps)
            progress = i / steps
            
            # Smooth easing (ease-out-cubic)
            eased = 1 - ((1 - progress) ** 3)
            
            x = int(start[0] + (end[0] - start[0]) * eased)
            y = int(start[1] + (end[1] - start[1]) * eased)
            
            # Calculate velocity
            if i > 0:
                prev = trajectory[-1]
                dt = (t - prev["t"]) / 1000  # seconds
                vx = (x - prev["x"]) / dt if dt > 0 else 0
                vy = (y - prev["y"]) / dt if dt > 0 else 0
            else:
                vx, vy = 0, 0
            
            trajectory.append({
                "t": t,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "click": (i == steps)  # Click at end
            })
        
        return trajectory
    
    async def _execute_and_record(
        self,
        target_x: int,
        target_y: int,
        action_type: str = "click",
        text: str = None,
        target_label: str = "",
        value_label: str = "",
        task_objective: str = "",
        task_category: str = "",
    ) -> Optional[TrajectoryData]:
        """
        Execute an action and record the trajectory.
        """
        # Get current mouse position (center of viewport as starting point)
        vw, vh = self.backend.viewport_size()
        start_x = random.randint(100, vw - 100)
        start_y = random.randint(100, vh - 100)
        
        # Record trajectory (snappy: 150-350ms like human)
        trajectory = await self._record_trajectory(
            (start_x, start_y),
            (target_x, target_y),
            duration_ms=random.randint(150, 350)
        )
        
        # Execute the actual action
        before_screenshot = await self._screenshot()
        
        if action_type == "click":
            await self.backend.click(target_x, target_y)
        elif action_type == "press_enter":
            await self.backend.click(target_x, target_y)
            await asyncio.sleep(0.05)
            await self.backend.press_key("Enter")
        elif action_type == "type" and text:
            await self.backend.click(target_x, target_y)
            await asyncio.sleep(0.05)
            await self.backend.type_text(text)
        
        # Multi-frame sampling: base + extra for slow sites (form submit, navigation)
        delay = getattr(self.config, f"action_delay_{action_type}", None) or self.config.action_delay
        sample_times = [delay, delay + 0.3, delay + 0.5]
        tl = (target_label or "").lower()
        is_submit_click = (
            action_type == "click"
            and any(k in tl for k in ("submit", "login", "calculate", "search submit", "go", "sign in"))
        )
        extra = (
            getattr(self.config, "phash_extra_wait_times_submit", None) or []
            if is_submit_click
            else getattr(self.config, "phash_extra_wait_times", None) or []
        )
        sample_times.extend(delay + t for t in extra)
        success = False
        last_t = 0.0
        after_screenshot = before_screenshot  # fallback
        for t in sample_times:
            await asyncio.sleep(t - last_t)
            last_t = t
            frame = await self._screenshot()
            after_screenshot = frame
            if self._check_action_success(
                before_screenshot, frame, center=(target_x, target_y), action_type=action_type
            ):
                success = True
                break
        
        # Optional VLM verification (overrides phash result)
        if self.config.use_vlm_verification:
            action_desc = (
                f"typed '{value_label}' into {target_label}" if action_type == "type" and value_label
                else f"clicked {target_label}" if target_label
                else f"{action_type} at ({target_x}, {target_y})"
            )
            result = self.vlm.verify(action_desc, before_screenshot, after_screenshot, task=task_objective)
            success = result.success
            if result.reason:
                print(f"      [VLM verify] {result.reason[:80]}")
        
        # Store in normalized format (0-1) for scale across screen sizes
        w, h = self.backend.viewport_size()
        target_norm = {"x": target_x / w, "y": target_y / h}
        traj_norm = [
            {
                "t": p["t"],
                "x": p["x"] / w,
                "y": p["y"] / h,
                "vx": p.get("vx", 0) / w,
                "vy": p.get("vy", 0) / h,
                "click": p.get("click", False),
            }
            for p in trajectory
        ]
        task_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}"
        screenshot_path = ""
        if self.config.save_screenshots and success and target_label:
            screenshots_dir = self.config.data_dir / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            img_path = screenshots_dir / f"{task_id}.jpg"
            try:
                before_screenshot.resize((384, 384), Image.Resampling.LANCZOS).save(img_path, quality=85)
                screenshot_path = str(img_path.relative_to(self.config.data_dir))
            except Exception:
                pass
        traj_data = TrajectoryData(
            task_id=task_id,
            target=target_norm,
            trajectory=traj_norm,
            success=success,
            screen_size=[w, h],
            task_type=action_type,
            target_label=target_label,
            task_category=task_category,
            source="vlm",
            timestamp=datetime.now().isoformat(),
            normalized=True,
            screenshot_path=screenshot_path,
        )
        
        return traj_data
    
    def _crop_roi(self, img: Image.Image, cx: int, cy: int, size: int = 400) -> Image.Image:
        """Crop 400x400 region centered on (cx, cy), clamped to image bounds."""
        w, h = img.size
        if w == 0 or h == 0:
            return img
        half = size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        if x2 <= x1 or y2 <= y1:
            return img  # Invalid crop, use full image
        return img.crop((x1, y1, x2, y2))
    
    def _check_action_success(
        self, before: Image.Image, after: Image.Image, center: Tuple[int, int] = None,
        action_type: str = "click"
    ) -> bool:
        """
        Check if action caused meaningful change. Uses perceptual hash.
        For click: use ROI only (avoids false positives from global animations).
        For type/press_enter: check full image (dropdown, results) OR ROI (input change).
        """
        from ..utils.hashing import phash_image, hash_similarity
        threshold = getattr(self.config, "phash_success_threshold", 0.99)
        # Slightly more lenient for type ROI (password dots, small inputs)
        type_threshold = min(threshold + 0.005, 0.999)

        def _similarity(b: Image.Image, a: Image.Image) -> float:
            return hash_similarity(phash_image(b), phash_image(a))

        if action_type == "click" and center:
            # Click: ROI only to avoid global-animation false positives
            b_roi = self._crop_roi(before, center[0], center[1])
            a_roi = self._crop_roi(after, center[0], center[1])
            return _similarity(b_roi, a_roi) < threshold

        # Type/press_enter: full image (dropdown, results) OR ROI (input-only change)
        if _similarity(before, after) < threshold:
            return True
        if center:
            b_roi = self._crop_roi(before, center[0], center[1])
            a_roi = self._crop_roi(after, center[0], center[1])
            if _similarity(b_roi, a_roi) < type_threshold:
                return True
        return False
    
    def _ocr_fallback_steps(self, task, vlm_img: Image.Image) -> List[Dict[str, Any]]:
        """
        When VLM fails, use OCR to find task-relevant text and create click steps.
        Extracts keywords from task + sample_data and tries find_text for each.
        """
        # Build search terms: task words + sample_data values (e.g. "Cappuccino", "Login")
        words = []
        for w in task.objective.lower().replace(",", " ").replace(".", " ").split():
            if len(w) > 2 and w.isalnum():
                words.append(w)
        if task.sample_data:
            for v in task.sample_data.values():
                if isinstance(v, str) and len(v) > 2:
                    words.append(v)
                elif isinstance(v, dict):
                    for sv in v.values():
                        if isinstance(sv, str) and len(sv) > 2:
                            words.append(sv)
        
        # Try OCR for each term (prioritize longer/more specific)
        seen = set()
        for term in sorted(words, key=len, reverse=True):
            if term.lower() in seen:
                continue
            seen.add(term.lower())
            coords = self.vlm.find_text(term, vlm_img)
            if coords:
                return [{"action": "click", "target": term, "value": ""}]
        
        return []
    
    def _get_page(self):
        """Get Playwright page if using BrowserBackend."""
        return getattr(self.backend, "page", None)
    
    async def _get_playwright_elements(self) -> List[Dict[str, Any]]:
        """
        Get clickable elements from DOM (buttons, links, inputs) with bbox.
        Used by grounding harness for fast text match when VLM/OCR would fail.
        """
        page = self._get_page()
        if not page:
            return []
        elements = []
        max_elements = 30  # Limit DOM queries for speed
        try:
            selectors = [
                "button, input[type=submit], input[type=button], [role=button]",
                "a[href]",
                "input:not([type=hidden]):not([type=submit]):not([type=button]), textarea",
            ]
            for sel in selectors:
                if len(elements) >= max_elements:
                    break
                els = await page.query_selector_all(sel)
                for el in els[:max_elements - len(elements)]:
                    if not el or self._killed:
                        break
                    try:
                        await el.scroll_into_view_if_needed(timeout=3000)
                        box = await el.bounding_box()
                        if not box:
                            continue
                        text = (await el.inner_text() or "").strip()
                        placeholder = await el.get_attribute("placeholder") or ""
                        if not text and placeholder:
                            text = placeholder
                        if not text and "input" in sel:
                            text = await el.get_attribute("name") or await el.get_attribute("aria-label") or ""
                        elements.append({
                            "text": text.strip(),
                            "x": int(box["x"]),
                            "y": int(box["y"]),
                            "width": int(box["width"]),
                            "height": int(box["height"]),
                        })
                    except Exception:
                        continue
        except Exception:
            pass
        return elements

    async def _close_backend_safe(self):
        """Close backend, swallowing errors on interrupt."""
        try:
            await self._close_backend()
        except Exception:
            pass
    
    async def _playwright_first_plan(self, task) -> List[Dict[str, Any]]:
        """
        Playwright-first: DOM-based planning. Fast, no VLM.
        Returns steps with optional "bbox" (x,y,w,h) for direct execution.
        """
        page = self._get_page()
        if not page:
            return []
        
        steps = []
        obj_lower = task.objective.lower()
        sample_data = task.sample_data or {}
        site = (task.site or "").lower()
        
        # 1. Calculator: BasicCalculator has First number, Second number, Operation, Calculate
        if ("solve" in obj_lower or "calculator" in obj_lower) and "testsheepnz" in site:
            steps = await self._playwright_calculator_steps(task)
            if steps:
                return steps
        
        # 2. GitHub search: input[name="q"], press Enter (no submit button)
        if "github.com" in site and ("search" in obj_lower or "query" in sample_data):
            steps = await self._playwright_github_search_steps(task)
            if steps:
                return steps
        
        # 3. Wikipedia: extract query from "article about 'X'", type in search, Enter, click first result
        if "wikipedia" in site and ("article" in obj_lower or "find" in obj_lower):
            steps = await self._playwright_wikipedia_search_steps(task)
            if steps:
                return steps
        
        # 4. The Internet - Form Authentication: navigate to /login, then fill username/password
        if "herokuapp" in site and ("form authentication" in obj_lower or "login" in obj_lower) and sample_data:
            steps = await self._playwright_the_internet_form_auth(task)
            if steps:
                return steps
        
        # 5. UI Playground textinput: type in input, then click the button (button text updates)
        if "textinput" in site and "button" in obj_lower and "click" in obj_lower:
            steps = await self._playwright_textinput_steps(task)
            if steps:
                return steps
        
        # 6. Generic search task
        if "search" in obj_lower or "find" in obj_lower or "query" in sample_data:
            query = sample_data.get("query") or sample_data.get("q") or ""
            if not query and "search" in obj_lower:
                if " for " in task.objective:
                    parts = task.objective.split(" for ", 1)
                    if len(parts) > 1:
                        query = parts[1].split(",")[0].split(" and ")[0].strip().strip("'\"").strip()
                if not query:
                    query = "book" if "book" in obj_lower else "search"
            if query:
                search_el = await page.query_selector(
                    'input[name="q"], input[type="search"], input[name*="search"], '
                    'input[placeholder*="search"], input[placeholder*="Search"], '
                    'input[aria-label*="search"], input[id*="search"]'
                )
                if search_el:
                    try:
                        step1 = {"action": "type", "target": "search", "value": query}
                        try:
                            await search_el.scroll_into_view_if_needed()
                            box = await search_el.bounding_box()
                            if box:
                                step1["x"] = int(box["x"] + box["width"] / 2)
                                step1["y"] = int(box["y"] + box["height"] / 2)
                        except Exception:
                            pass
                        steps.append(step1)
                        submit = await page.query_selector('button[type="submit"], input[type="submit"], [aria-label*="search" i]')
                        if submit:
                            step2 = {"action": "click", "target": "search submit", "value": ""}
                            try:
                                await submit.scroll_into_view_if_needed()
                                box = await submit.bounding_box()
                                if box:
                                    step2["x"] = int(box["x"] + box["width"] / 2)
                                    step2["y"] = int(box["y"] + box["height"] / 2)
                            except Exception:
                                pass
                            steps.append(step2)
                        else:
                            step_enter = {"action": "press_enter", "target": "search", "value": ""}
                            if "x" in step1 and "y" in step1:
                                step_enter["x"] = step1["x"]
                                step_enter["y"] = step1["y"]
                            steps.append(step_enter)
                        print(f"   📋 Playwright search: type '{query[:20]}...' in search box")
                    except Exception:
                        pass
        
        if not steps and sample_data:
            steps = await self._playwright_form_fallback_steps(task)
            if steps:
                pass
        
        if not steps and ("type" in obj_lower or "input" in obj_lower or "number" in obj_lower):
            steps = await self._generic_input_fallback(task)
        
        return steps
    
    async def _playwright_calculator_steps(self, task) -> List[Dict[str, Any]]:
        """BasicCalculator: parse expression (e.g. 42 + 13) into type/click steps."""
        page = self._get_page()
        if not page:
            return []
        sample_data = task.sample_data or {}
        expr = sample_data.get("expression", "")
        if not expr:
            return []
        # Parse simple "A op B" (e.g. 42 + 13, 25 * 4). Strip parens for "(25 * 4) - 17" -> take first op
        import re
        expr_clean = re.sub(r"[()]", " ", expr).strip()
        op_map = {"+": "Add", "-": "Subtract", "*": "Multiply", "/": "Divide"}
        m = re.search(r"(\d+)\s*([+\-*/])\s*(\d+)", expr_clean)
        if not m:
            return []
        first, op_char, second = m.group(1), m.group(2), m.group(3)
        op_label = op_map.get(op_char, "Add")
        steps = []
        labels = []
        try:
            labels = await page.query_selector_all("label")
            # First number: by label or nth input
            first_el = await page.query_selector(
                'input[name*="first"], input[id*="first"], input[placeholder*="first"]'
            )
            if not first_el:
                for lb in labels:
                    lt = (await lb.inner_text() or "").lower()
                    if "first" in lt and "number" in lt:
                        fid = await lb.get_attribute("for")
                        if fid:
                            first_el = await page.query_selector(f'#{fid}')
                        break
            if not first_el:
                inputs = await page.query_selector_all("input:not([type=hidden]):not([type=submit])")
                if len(inputs) >= 1:
                    first_el = inputs[0]
            if first_el:
                await first_el.scroll_into_view_if_needed()
                box = await first_el.bounding_box()
                if box:
                    steps.append({
                        "action": "type", "target": "First number", "value": first,
                        "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                    })
            # Operation: click Add/Subtract/etc
            op_loc = page.locator(f'text={op_label}')
            if await op_loc.count() > 0:
                first_op = op_loc.first
                await first_op.scroll_into_view_if_needed()
                box = await first_op.bounding_box()
                if box:
                    steps.append({
                        "action": "click", "target": op_label, "value": "",
                        "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                    })
            # Second number input
            second_el = await page.query_selector(
                'input[name*="second"], input[id*="second"]'
            )
            if not second_el:
                for lb in labels:
                    lt = (await lb.inner_text() or "").lower()
                    if "second" in lt and "number" in lt:
                        fid = await lb.get_attribute("for")
                        if fid:
                            second_el = await page.query_selector(f'#{fid}')
                        break
            if not second_el:
                inputs = await page.query_selector_all("input:not([type=hidden]):not([type=submit])")
                if len(inputs) >= 2:
                    second_el = inputs[1]
            if second_el:
                await second_el.scroll_into_view_if_needed()
                box = await second_el.bounding_box()
                if box:
                    steps.append({
                        "action": "type", "target": "Second number", "value": second,
                        "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                    })
            # Calculate button
            calc_el = await page.query_selector(
                'button:has-text("Calculate"), input[value="Calculate"], [role="button"]:has-text("Calculate")'
            )
            if calc_el:
                await calc_el.scroll_into_view_if_needed()
                box = await calc_el.bounding_box()
                if box:
                    steps.append({
                        "action": "click", "target": "Calculate", "value": "",
                        "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                    })
            if steps:
                print(f"   📋 Playwright calculator: {expr} → {len(steps)} steps")
        except Exception as e:
            print(f"   ⚠️ Calculator steps failed: {e}")
        return steps
    
    async def _try_playwright_github_first_repo(self, target: str) -> Optional[Tuple[int, int]]:
        """If on GitHub search results and target is 'first repository', get first repo link bbox."""
        if "first repository" not in (target or "").lower():
            return None
        page = self._get_page()
        if not page:
            return None
        try:
            url = page.url or ""
            if "github.com" not in url or "search" not in url:
                return None
            await asyncio.sleep(1)  # Brief wait for results
            for sel in [
                'a[data-hovercard-type="repository"]',
                'a[href^="/"][href*="/"].Link',
                'div[data-testid="results-list"] a[href*="/"]',
                'a[href^="/"][href*="/"]',
            ]:
                el = await page.query_selector(sel)
                if el:
                    await el.scroll_into_view_if_needed()
                    box = await el.bounding_box()
                    if box:
                        return (int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2))
        except Exception:
            pass
        return None
    
    async def _playwright_github_search_steps(self, task) -> List[Dict[str, Any]]:
        """GitHub search: input[name="q"], type query, press Enter (no submit button)."""
        page = self._get_page()
        if not page:
            return []
        sample_data = task.sample_data or {}
        obj_lower = (task.objective or "").lower()
        query = sample_data.get("query") or sample_data.get("q") or ""
        if not query and "search" in obj_lower:
            if " for " in task.objective:
                parts = task.objective.split(" for ", 1)
                if len(parts) > 1:
                    query = parts[1].split(",")[0].split(" and ")[0].strip().strip("'\"").strip()
        if not query:
            return []
        steps = []
        try:
            # GitHub uses input[name="q"] for main search
            search_el = await page.query_selector('input[name="q"]')
            if not search_el:
                search_el = await page.query_selector(
                    'input[placeholder*="Search"], input[aria-label*="Search"]'
                )
            if search_el:
                await search_el.scroll_into_view_if_needed()
                box = await search_el.bounding_box()
                if box:
                    step1 = {
                        "action": "type", "target": "search", "value": query,
                        "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                    }
                    steps.append(step1)
                    step2 = {
                        "action": "press_enter", "target": "search", "value": "",
                        "x": step1["x"], "y": step1["y"]
                    }
                    steps.append(step2)
                    if "view" in obj_lower or "top result" in obj_lower:
                        steps.append({"action": "click", "target": "first repository", "value": ""})
                    print(f"   📋 Playwright GitHub search: type '{query[:30]}...' + Enter")
        except Exception as e:
            print(f"   ⚠️ GitHub search failed: {e}")
        return steps
    
    async def _playwright_wikipedia_search_steps(self, task) -> List[Dict[str, Any]]:
        """Wikipedia: extract query from objective, type in search, Enter, click first result."""
        page = self._get_page()
        if not page:
            return []
        obj = task.objective or ""
        obj_lower = obj.lower()
        if "hyperlinks only" in obj_lower or "no search" in obj_lower:
            return []  # Use links only, not search
        query = ""
        if "article about" in obj_lower:
            after = obj_lower.split("article about", 1)[-1].strip()
            query = after.split(" by ")[0].split(" using ")[0].strip().strip("'\"").strip()
        if not query:
            return []
        steps = []
        try:
            search_el = await page.query_selector(
                '#searchInput, input[name="search"], input[placeholder*="Search"]'
            )
            if search_el:
                await search_el.scroll_into_view_if_needed()
                box = await search_el.bounding_box()
                if box:
                    step1 = {
                        "action": "type", "target": "Search Wikipedia", "value": query,
                        "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                    }
                    steps.append(step1)
                    step2 = {
                        "action": "press_enter", "target": "Search Wikipedia", "value": "",
                        "x": step1["x"], "y": step1["y"]
                    }
                    steps.append(step2)
                    steps.append({"action": "click", "target": query, "value": ""})
                    print(f"   📋 Playwright Wikipedia search: '{query}' + Enter")
        except Exception as e:
            print(f"   ⚠️ Wikipedia search failed: {e}")
        return steps
    
    async def _playwright_form_fallback_steps(self, task) -> List[Dict[str, Any]]:
        """
        Use Playwright to find form fields and create type steps from sample_data.
        Works when VLM fails but we have labeled inputs (name, email, address, etc.).
        """
        page = self._get_page()
        if not page or not task.sample_data:
            return []
        
        # Map sample_data keys to common field labels (partial match)
        key_to_labels = {
            "name": ["full name", "name", "username", "user name"],
            "email": ["email", "e-mail"],
            "password": ["password", "pass"],
            "address": ["address", "current address", "permanent address", "street"],
            "phone": ["phone", "mobile", "tel"],
        }
        
        steps = []
        used_keys = set()
        
        try:
            inputs = await page.query_selector_all("input:not([type=hidden]):not([type=submit]):not([type=button]), textarea")
            for el in inputs:
                if not el:
                    continue
                placeholder = await el.get_attribute("placeholder") or ""
                name = await el.get_attribute("name") or ""
                id_attr = await el.get_attribute("id") or ""
                
                # Try <label for="id"> if we have id
                label_text = ""
                if id_attr:
                    label_el = await page.query_selector(f'label[for="{id_attr}"]')
                    if label_el:
                        label_text = (await label_el.inner_text() or "").strip()
                
                # Build label from: explicit label, placeholder, name, or id
                label = (label_text or placeholder or name or id_attr).lower().strip()
                if not label:
                    continue
                
                # Find matching sample_data key
                for s_key, s_val in task.sample_data.items():
                    if not isinstance(s_val, str):
                        continue
                    k = s_key.lower()
                    # Allow address to fill multiple fields (current + permanent)
                    if k in used_keys and k != "address":
                        continue
                    labels = key_to_labels.get(k, [k])
                    if any(l in label or label in l for l in labels) or k in label or label in k:
                        display_label = (label_text or placeholder or name or id_attr).strip()
                        step = {"action": "type", "target": display_label or s_key, "value": s_val}
                        # Add bbox so we skip VLM grounding
                        try:
                            await el.scroll_into_view_if_needed()
                            box = await el.bounding_box()
                            if box:
                                step["x"] = int(box["x"] + box["width"] / 2)
                                step["y"] = int(box["y"] + box["height"] / 2)
                        except Exception:
                            pass
                        steps.append(step)
                        used_keys.add(s_key)
                        break
            
            if steps:
                submit_btn = await page.query_selector("button[type=submit], input[type=submit]")
                submit_text = await submit_btn.inner_text() if submit_btn else None
                step = {"action": "click", "target": (submit_text or "Submit").strip(), "value": ""}
                try:
                    if submit_btn:
                        await submit_btn.scroll_into_view_if_needed()
                        box = await submit_btn.bounding_box()
                        if box:
                            step["x"] = int(box["x"] + box["width"] / 2)
                            step["y"] = int(box["y"] + box["height"] / 2)
                except Exception:
                    pass
                steps.append(step)
                print(f"   📋 Playwright form fallback: {len(steps)} steps from DOM")
        
        except Exception as e:
            print(f"   ⚠️ Playwright form fallback failed: {e}")
        
        return steps
    
    async def _playwright_the_internet_form_auth(self, task) -> List[Dict[str, Any]]:
        """The Internet: click Form Authentication link, then fill username/password."""
        page = self._get_page()
        if not page or not task.sample_data:
            return []
        username = task.sample_data.get("username", "")
        password = task.sample_data.get("password", "")
        if not username or not password:
            return []
        steps = []
        try:
            url = (page.url or "").lower()
            # If on homepage, navigate to /login first (click Form Authentication link)
            if "/login" not in url:
                link = await page.query_selector('a[href="/login"]')
                if not link:
                    link = await page.get_by_text("Form Authentication", exact=False).first
                if link:
                    await link.scroll_into_view_if_needed()
                    box = await link.bounding_box()
                    if box:
                        steps.append({
                            "action": "click", "target": "Form Authentication", "value": "",
                            "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                        })
            if not steps:
                # Already on /login, find form directly
                pass
            # Add username, password, login steps (need to run after navigation)
            for field, label, val in [
                ("username", "Username", username),
                ("password", "Password", password),
            ]:
                el = await page.query_selector(f'input[name="{field}"], input[id="{field}"]')
                if el:
                    await el.scroll_into_view_if_needed()
                    box = await el.bounding_box()
                    if box:
                        steps.append({
                            "action": "type", "target": label, "value": val,
                            "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                        })
            login_btn = await page.query_selector('button[type="submit"], input[type="submit"]')
            if login_btn:
                await login_btn.scroll_into_view_if_needed()
                box = await login_btn.bounding_box()
                if box:
                    steps.append({
                        "action": "click", "target": "Login", "value": "",
                        "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                    })
            if steps:
                # If we only have form steps (no nav), we're on /login - good
                # If we have nav + form, we need to run nav first then re-query for form
                # For simplicity: return nav step first; form steps will be re-queried on next plan
                # Actually the problem: after click Form Authentication, the page changes. The form
                # steps we got are from the CURRENT page (homepage). So we can't get form coords now.
                # We need to either: 1) only return nav step, then replan for form, or 2) navigate via
                # page.goto and then get form coords. Let me use page.goto for reliability.
                if "/login" not in url:
                    base = task.site.rstrip("/")
                    try:
                        await page.goto(f"{base}/login", wait_until="domcontentloaded")
                        await asyncio.sleep(0.5)
                    except Exception:
                        pass
                    steps = []
                    for field, label, val in [
                        ("username", "Username", username),
                        ("password", "Password", password),
                    ]:
                        el = await page.query_selector(f'input[name="{field}"], input[id="{field}"]')
                        if el:
                            await el.scroll_into_view_if_needed()
                            box = await el.bounding_box()
                            if box:
                                steps.append({
                                    "action": "type", "target": label, "value": val,
                                    "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                                })
                    login_btn = await page.query_selector('button[type="submit"], input[type="submit"]')
                    if login_btn:
                        await login_btn.scroll_into_view_if_needed()
                        box = await login_btn.bounding_box()
                        if box:
                            steps.append({
                                "action": "click", "target": "Login", "value": "",
                                "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                            })
                if steps:
                    print(f"   📋 Playwright the-internet form auth: {len(steps)} steps")
        except Exception as e:
            print(f"   ⚠️ The-internet form auth failed: {e}")
        return steps
    
    async def _playwright_textinput_steps(self, task) -> List[Dict[str, Any]]:
        """UI Playground textinput: type in input, then click the button (text updates)."""
        page = self._get_page()
        if not page:
            return []
        steps = []
        try:
            inputs = await page.query_selector_all("input:not([type=hidden]):not([type=submit])")
            if not inputs:
                return []
            el = inputs[0]
            value = "test"
            await el.scroll_into_view_if_needed()
            box = await el.bounding_box()
            if box:
                steps.append({
                    "action": "type", "target": "input", "value": value,
                    "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                })
            # Button that updates its text when we type (often next to input)
            btn = await page.query_selector("button, input[type=button]")
            if btn:
                await btn.scroll_into_view_if_needed()
                box = await btn.bounding_box()
                if box:
                    steps.append({
                        "action": "click", "target": "Button", "value": "",
                        "x": int(box["x"] + box["width"] / 2), "y": int(box["y"] + box["height"] / 2)
                    })
            if steps:
                print(f"   📋 Playwright textinput: type + click button ({len(steps)} steps)")
        except Exception as e:
            print(f"   ⚠️ Playwright textinput failed: {e}")
        return steps
    
    async def _generic_input_fallback(self, task) -> List[Dict[str, Any]]:
        """
        Fallback when task is "type into input" but we have no sample_data.
        Finds first visible input and creates a type step (e.g. "123" for numbers).
        """
        page = self._get_page()
        if not page:
            return []
        obj_lower = task.objective.lower()
        try:
            inputs = await page.query_selector_all(
                "input:not([type=hidden]):not([type=submit]):not([type=button]), textarea"
            )
            if not inputs:
                return []
            el = inputs[0]
            value = "123" if "number" in obj_lower else "test"
            step = {"action": "type", "target": "input", "value": value}
            try:
                await el.scroll_into_view_if_needed()
                box = await el.bounding_box()
                if box:
                    step["x"] = int(box["x"] + box["width"] / 2)
                    step["y"] = int(box["y"] + box["height"] / 2)
            except Exception:
                pass
            steps = [step]
            print(f"   📋 Playwright generic input: type '{value}' in first input")
            return steps
        except Exception:
            return []
    
    def _is_browser_error(self, e: Exception) -> bool:
        """Check if error indicates browser crash (needs restart)."""
        msg = str(e).lower()
        return any(x in msg for x in [
            "target closed", "context closed", "connection closed",
            "execution context", "page closed", "browser closed",
            "page crashed", "protocol error", "session closed"
        ])
    
    async def _run_episode(self) -> List[TrajectoryData]:
        """
        Run one VLM-guided exploration episode.
        Returns list of trajectory data collected. Re-raises browser errors.
        """
        try:
            return await self._run_episode_impl()
        except Exception as e:
            if self._is_browser_error(e):
                raise  # Let main loop restart browser
            print(f"   🔧 Episode error (recovering): {type(e).__name__}: {e}")
            self.progress.episodes_completed += 1
            self._save_progress()
            return []
    
    def _fmt_duration(self, sec: float) -> str:
        """Format duration: ms if < 1s, seconds if < 60s, else minutes."""
        if sec < 1:
            return f"({int(sec * 1000)}ms)"
        if sec < 60:
            return f"({sec:.1f}s)"
        m = int(sec // 60)
        s = sec % 60
        return f"({m}m {s:.1f}s)"
    
    def _build_state_context(
        self, last_step_desc: str, screenshot: Image.Image, task_objective: str
    ) -> str:
        """Build context for re-planning. Skip OCR (saves 2-10s)—VLM has screenshot."""
        if not last_step_desc:
            return ""
        return f"Done: {last_step_desc}. Next: {task_objective}"
    
    def _task_has_more_to_do(
        self, task, steps_done: int, last_action: str = ""
    ) -> bool:
        """Heuristic: task implies more work after these steps."""
        obj = (task.objective or "").lower()
        last = (last_action or "").lower()
        # Don't re-plan after we just submitted—form is done
        if "submit" in last or "login" in last:
            return False
        # Don't re-plan if we've done 3+ steps on a simple form task
        if steps_done >= 3 and ("form" in obj or "submit" in obj or "fill" in obj):
            return False
        more_phrases = [
            "and view", "and click", "add to cart", "proceed to", "find the answer",
            "view the top", "open the file", "view their",
            "and go", "and navigate", "go to", "and open", "and find", "and search",
            "file explorer", "downloads folder", "downloads", "navigate to",
        ]
        if any(p in obj for p in more_phrases):
            return True
        if " and " in obj and steps_done < 2:
            return True
        if steps_done < 2 and len(obj) > 50:
            return True
        return False
    
    async def _run_episode_impl(self) -> List[TrajectoryData]:
        """Inner episode logic - may raise on browser/VLM errors. Iterative re-planning."""
        episode_trajectories = []
        task = self.curriculum.sample_task(backend=self.config.execution_backend)
        
        print(f"\n📍 Episode {self.progress.episodes_completed + 1}")
        print(f"   Site: {task.site}")
        print(f"   Task: {task.objective}")
        
        ep_start = time.time()
        
        # Navigate (retry once on failure)
        if not await self._navigate(task.site):
            return []
        
        sample_data = task.sample_data or {}
        steps_done = 0
        last_success = False
        last_step_desc = ""  # "click 'Submit'" - passed to next plan for context
        last_identical_key = None
        last_identical_count = 0
        failure_replan_count = 0

        while steps_done < self.config.max_episode_steps and not self._killed:
            if self._paused:
                await asyncio.sleep(0.5)
                continue
            
            screenshot = await self._screenshot()
            vlm_img, scale_x, scale_y = self._resize_for_vlm(screenshot)
            plan_start = time.time()
            
            # Playwright-first (only on first plan; skip for re-plan)
            steps = [] if steps_done > 0 else await self._playwright_first_plan(task)
            plan_source = "playwright"
            if not steps:
                plan_source = "vlm"
                # Build state context for re-plan: what we did, what's visible, what's next
                state_context = self._build_state_context(
                    last_step_desc, screenshot, task.objective
                )
                steps = self.vlm.plan(
                    task.objective, vlm_img, sample_data,
                    state_context=state_context,
                )
                if not steps:
                    screenshot = await self._screenshot()
                    vlm_img, _, _ = self._resize_for_vlm(screenshot)
                    state_context = self._build_state_context(
                        last_step_desc, screenshot, task.objective
                    )
                    steps = self.vlm.plan(
                        task.objective, vlm_img, sample_data,
                        state_context=state_context,
                    )
            if not steps and "click" in task.objective.lower():
                plan_source = "ocr"
                steps = self._ocr_fallback_steps(task, vlm_img)
            
            if not steps:
                break
            
            steps = [
                s for s in steps[:self.config.max_episode_steps - steps_done]
                if s is not None and (isinstance(s, dict) or (hasattr(s, "action") and hasattr(s, "target")))
            ]
            if not steps:
                break
            
            plan_sec = time.time() - plan_start
            print(f"   📋 Planned {len(steps)} steps [{plan_source}] {self._fmt_duration(plan_sec)}")
            # Show what we're about to do (helps debug re-plan loops)
            for j, s in enumerate(steps[:5], 1):
                a = s.get("action", "click") if isinstance(s, dict) else getattr(s, "action", "click")
                t = (s.get("target", "") or "").strip() or "?"
                v = s.get("value", "") if isinstance(s, dict) else getattr(s, "value", "")
                line = f"      → {j}) {a} '{t}'" + (f" → '{str(v)[:20]}...'" if v else "")
                print(line)
            if len(steps) > 5:
                print(f"      → ... +{len(steps) - 5} more")
            if steps_done > 0 and plan_source == "vlm":
                print(f"   🔄 Re-planning: task has more to do | Done so far: {steps_done} steps")
            should_replan = False
            
            for i, step in enumerate(steps):
                if self._killed or self._paused or steps_done >= self.config.max_episode_steps:
                    break
                
                while self._paused:
                    await asyncio.sleep(0.5)
                
                action = step.get("action", "click") if isinstance(step, dict) else step.action
                target = (step.get("target", "") if isinstance(step, dict) else step.target or "").strip()
                raw_value = step.get("value", "") if isinstance(step, dict) else step.value
                value = raw_value if isinstance(raw_value, str) else (str(raw_value) if raw_value else "")
                if (action or "").lower() in ("submit", "press", "button") or "click" in (action or "").lower():
                    action = "click"
                
                has_coords = "x" in step and "y" in step
                if not has_coords and not (target or "").strip():
                    print(f"   Step {steps_done+1}: {action} '{target}' → ⚠️ Skipped")
                    continue
                
                val_preview = f" → '{str(value)[:25]}...'" if value and action == "type" else ""
                print(f"   Step {steps_done+1}: {action} '{target}'{val_preview}")
                step_start = time.time()

                async def _run_step():
                    if has_coords:
                        tx, ty = int(step["x"]), int(step["y"])
                        traj = await self._execute_and_record(
                            tx, ty,
                            action_type="press_enter" if action == "press_enter" else action,
                            text=value if action == "type" else None,
                            target_label=target,
                            value_label=value,
                            task_objective=task.objective,
                            task_category=task.category.value if hasattr(task, "category") else "",
                        )
                        return (traj, tx, ty)
                    coords = await self._try_playwright_github_first_repo(target)
                    if coords:
                        target_x, target_y = coords
                        step["x"] = target_x
                        step["y"] = target_y
                        act_type = "press_enter" if action == "press_enter" else action
                        traj = await self._execute_and_record(
                            target_x, target_y,
                            action_type=act_type,
                            text=value if action == "type" else None,
                            target_label=target,
                            value_label=value,
                            task_objective=task.objective,
                            task_category=task.category.value if hasattr(task, "category") else "",
                        )
                        return (traj, target_x, target_y)
                    screenshot = await self._screenshot()
                    playwright_elements = await self._get_playwright_elements() if self._get_page() else []
                    synonyms = [a for a in _get_synonyms(target) if a.strip().lower() != target.strip().lower()]
                    refinement_max = getattr(self.config, "grounding_refinement_max", 2)
                    candidates = [target] + synonyms[:refinement_max]
                    last_traj = None
                    last_tx = last_ty = 0
                    for alt_target in candidates:
                        try:
                            grounding = await asyncio.wait_for(
                                self.grounding_harness.ground(
                                    alt_target, screenshot,
                                    task_objective=task.objective or "",
                                    task_category=task.category.value if hasattr(task, "category") else "",
                                    playwright_elements=playwright_elements,
                                ),
                                timeout=self.config.grounding_timeout,
                            )
                        except asyncio.TimeoutError:
                            grounding = GroundingHarnessResult(found=False)
                        if not grounding.found:
                            continue
                        target_x = int(grounding.x)
                        target_y = int(grounding.y)
                        act_type = "press_enter" if action == "press_enter" else action
                        traj = await self._execute_and_record(
                            target_x, target_y,
                            action_type=act_type,
                            text=value if action == "type" else None,
                            target_label=target,
                            value_label=value,
                            task_objective=task.objective,
                            task_category=task.category.value if hasattr(task, "category") else "",
                        )
                        last_traj = traj
                        last_tx, last_ty = target_x, target_y
                        if traj.success:
                            return (traj, target_x, target_y)
                        screenshot = await self._screenshot()
                    if last_traj is not None:
                        return (last_traj, last_tx, last_ty)
                    return (None, 0, 0)

                try:
                    traj, target_x, target_y = await asyncio.wait_for(
                        _run_step(),
                        timeout=getattr(self.config, "step_timeout", 90.0),
                    )
                except asyncio.TimeoutError:
                    step_sec = time.time() - step_start
                    print(f"      ⏱️ Step timeout ({getattr(self.config, 'step_timeout', 90)}s) {self._fmt_duration(step_sec)}")
                    self.progress.failed_trajectories += 1
                    self.progress.total_trajectories += 1
                    continue
                if traj is None:
                    step_sec = time.time() - step_start
                    print(f"      ❌ Element not found (target: '{target}') {self._fmt_duration(step_sec)}")
                    self.progress.failed_trajectories += 1
                    self.progress.total_trajectories += 1
                    continue

                if traj:
                    steps_done += 1
                    step_source = "playwright" if has_coords else "vlm"
                    if step_source != "playwright":
                        episode_trajectories.append(traj)
                    step_sec = time.time() - step_start
                    # Loop detector: stop after 3 identical steps with no visible change
                    step_key = (target.lower(), action)
                    if traj.success:
                        last_success = True
                        if step_source != "playwright":
                            self.progress.successful_trajectories += 1
                        print(f"      ✅ Success ({target_x}, {target_y}) {self._fmt_duration(step_sec)}")
                        # Loop detector: same step 3x even with "success" = likely stuck (e.g. re-typing Search)
                        if step_key == last_identical_key:
                            last_identical_count += 1
                            if last_identical_count >= 3:
                                print(f"   🔁 Loop detected: same step 3x in a row, stopping")
                                break
                        else:
                            last_identical_key = step_key
                            last_identical_count = 1
                        # Only re-plan when we've finished ALL steps in current batch
                        if i >= len(steps) - 1 and self._task_has_more_to_do(task, steps_done, last_action=action):
                            should_replan = True
                            last_step_desc = f"{action} '{target}'"
                            if value and action == "type":
                                v = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
                                last_step_desc += f" (typed '{v}')"
                            await asyncio.sleep(1.5)
                            break
                    else:
                        if step_key == last_identical_key:
                            last_identical_count += 1
                            if last_identical_count >= 3:
                                print(f"   🔁 Loop detected: same step 3x with no change, stopping")
                                break
                        else:
                            last_identical_key = step_key
                            last_identical_count = 1
                        self.progress.failed_trajectories += 1
                        print(f"      ⚠️ No visible change {self._fmt_duration(step_sec)}")
                        # Plan refinement on failure: re-plan so VLM can try different approach
                        if (
                            getattr(self.config, "replan_on_failure", True)
                            and failure_replan_count < getattr(self.config, "max_failure_replans", 2)
                            and last_identical_count < 3
                        ):
                            # Trust Playwright: continue to next step instead of VLM replan (avoids
                            # wrong suggestions like "type First number" after "click Multiply")
                            if plan_source == "playwright" and i < len(steps) - 1:
                                should_replan = False
                                # Don't break; continue to next Playwright step
                            else:
                                should_replan = True
                                last_step_desc = f"FAILED: {action} '{target}' at ({target_x},{target_y}) - no visible change"
                                failure_replan_count += 1
                                break
                    self.progress.total_trajectories += 1
            
            if not should_replan:
                break
        
        self.progress.episodes_completed += 1
        ep_sec = time.time() - ep_start
        if ep_sec >= 5:
            print(f"   ⏱️ Episode: {ep_sec:.1f}s")
        return episode_trajectories
    
    async def gather_data(self):
        """
        Main data gathering loop.
        Runs until target trajectories reached or killed.
        """
        self._running = True
        pending_trajectories = []
        
        # Baseline for before/after comparison
        session_start = time.time()
        baseline = {
            "successful": self.progress.successful_trajectories,
            "failed": self.progress.failed_trajectories,
            "total": self.progress.total_trajectories,
            "episodes": self.progress.episodes_completed,
        }
        
        print("\n" + "=" * 60)
        print("🤖 SMART DATA GATHERER: VLM-Guided Exploration")
        print("   Press Cmd+Shift+Escape to stop | Cmd+Shift+P to pause")
        print("   Self-healing: browser restarts on crash, progress saved every episode")
        print("=" * 60)
        print(f"\n📊 Target: {self.progress.target_trajectories} successful trajectories")
        print(f"   Current: {self.progress.successful_trajectories}")
        print(f"   Remaining: {self.progress.target_trajectories - self.progress.successful_trajectories}")
        print(f"   Grounding timeout: {self.config.grounding_timeout}s (fail fast)")
        
        self.indicator.start()
        # Init backend with retry
        for init_attempt in range(3):
            try:
                await self._init_backend()
                break
            except Exception as e:
                print(f"   🔧 Browser init failed (attempt {init_attempt + 1}/3): {e}")
                if init_attempt < 2:
                    await asyncio.sleep(5)
                else:
                    raise
        
        consecutive_errors = 0
        max_consecutive_errors = 10  # Restart browser after this many failures
        
        try:
            while self._running and not self._killed:
                # Check if target reached
                if self.progress.successful_trajectories >= self.progress.target_trajectories:
                    print(f"\n🎉 Target reached! {self.progress.successful_trajectories} trajectories collected")
                    break
                
                # Ensure backend is ready before each episode
                await self._ensure_backend()
                
                try:
                    episode_trajs = await self._run_episode()
                    pending_trajectories.extend(episode_trajs)
                    consecutive_errors = 0  # Reset on success
                except Exception as e:
                    consecutive_errors += 1
                    print(f"   🔧 Episode crash ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    self._save_progress()
                    if consecutive_errors >= max_consecutive_errors:
                        print("   🔧 Too many consecutive errors - restarting backend...")
                        await self._close_backend_safe()
                        await asyncio.sleep(5)
                        await self._init_backend()
                        consecutive_errors = 0
                    await asyncio.sleep(3)  # Back off before retry
                    continue
                
                # Save periodically and on any trajectories
                if pending_trajectories:
                    successful_in_pending = sum(1 for t in pending_trajectories if t.success)
                    if successful_in_pending >= self.config.save_every:
                        self._save_trajectories(pending_trajectories)
                        pending_trajectories = []
                
                self._save_progress()  # Save after every episode
                successful_this_ep = sum(1 for t in episode_trajs if t.success)
                self.progress.record_episode(time.time(), successful_this_ep)
                self._print_progress()
                await asyncio.sleep(1)
        
        finally:
            # Save remaining
            if pending_trajectories:
                self._save_trajectories(pending_trajectories)
            self._save_progress()
            if self._killed:
                # Fast exit: skip slow model/browser cleanup (UGround/PyTorch can take 10+ min)
                print("\n   ⚡ Fast exit (skipping model unload). Progress saved.")
                self.indicator.stop()
                import os
                os._exit(0)
            await self._close_backend_safe()
            self.indicator.stop()
            
            # Before/after comparison
            session_sec = time.time() - session_start
            p = self.progress
            delta_succ = p.successful_trajectories - baseline["successful"]
            delta_fail = p.failed_trajectories - baseline["failed"]
            delta_total = p.total_trajectories - baseline["total"]
            delta_ep = p.episodes_completed - baseline["episodes"]
            rate = delta_succ / (session_sec / 3600) if session_sec > 0 else 0
            
            print("\n" + "=" * 60)
            print("📊 Final Statistics")
            print("=" * 60)
            print("   BEFORE (session start):")
            print(f"      Successful: {baseline['successful']} | Failed: {baseline['failed']} | Total: {baseline['total']} | Episodes: {baseline['episodes']}")
            print("   AFTER (session end):")
            print(f"      Successful: {p.successful_trajectories} | Failed: {p.failed_trajectories} | Total: {p.total_trajectories} | Episodes: {p.episodes_completed}")
            print("   SESSION DELTA:")
            print(f"      +{delta_succ} successful | +{delta_fail} failed | +{delta_total} total | +{delta_ep} episodes")
            sess_rate = delta_succ / max(1, delta_total) if delta_total > 0 else 0.0
            print(f"      Session time: {session_sec/60:.1f}m | Success rate this session: {sess_rate:.1%} | Speed: {rate:.0f}/h")
            print("=" * 60)
    
    def _print_progress(self):
        """Print current progress. ETA uses rolling window (25 ep) or overall rate."""
        p = self.progress
        speed, source = p._trajectories_per_hour_impl()
        eta_str = f"{p.eta_hours:.1f}h" if p.eta_hours < 1000 else "calculating..."
        print(f"\n   📈 Progress: {p.progress_percent:.1f}% "
              f"({p.successful_trajectories}/{p.target_trajectories})")
        print(f"      Success rate: {p.success_rate:.1%} | "
              f"Speed: {speed:.0f}/h ({source}) | "
              f"ETA: {eta_str}")


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

_gatherer: Optional["SmartDataGatherer"] = None


async def main():
    """Main entrypoint for data gathering."""
    global _gatherer
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart VLM-guided data gathering")
    parser.add_argument("--target", type=int, default=5000, help="Target trajectories")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    parser.add_argument("--visible", action="store_true", help="Show browser window")
    parser.add_argument("--model", type=str, default="moondream", help="VLM: moondream (default), llava-phi3, llava, Me7war/Astria")
    parser.add_argument("--vlm-timeout", type=float, default=600, help="VLM request timeout (seconds, CPU needs ~10 min)")
    parser.add_argument("--grounding-timeout", type=float, default=30, help="Cap grounding (DOM+OCR+VLM) at N seconds (fail fast)")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU for OCR (default: on)")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Disable GPU (CPU-only)")
    parser.add_argument("--llamacpp", action="store_true", help="Use llama.cpp backend")
    parser.add_argument("--llamacpp-url", type=str, default="http://localhost:8080", help="llama.cpp URL")
    parser.add_argument("--vlm-verify", action="store_true", help="Use VLM to verify action success (slow, ~1 call per trajectory)")
    parser.add_argument("--backend", type=str, default="browser", choices=["browser", "desktop"],
                        help="Execution backend: browser (Playwright) or desktop (pyautogui/screen)")
    parser.add_argument("--grounding", type=str, default="vlm", choices=["vlm", "uground", "vision_trm", "anorha_trm"],
                        help="Grounding: vlm | uground | vision_trm | anorha_trm (unified)")
    parser.add_argument("--uground-4bit", action="store_true", help="Use 4-bit quant UGround (laptop, ~2GB VRAM)")
    parser.add_argument("--vision-trm-checkpoint", type=str, default="checkpoints/vision_trm_best.pt",
                        help="Path to Vision TRM checkpoint (for --grounding vision_trm)")
    parser.add_argument("--anorha-trm-checkpoint", type=str, default="checkpoints/anorha_trm_best.pt",
                        help="Path to Anorha TRM checkpoint (for --grounding anorha_trm)")
    parser.add_argument("--save-screenshots", action="store_true",
                        help="Save screenshots for Vision TRM training (crap-top path, fully local)")
    
    args = parser.parse_args()
    
    config = GathererConfig(
        headless=not args.visible,
        target_trajectories=args.target,
        vlm_model=args.model,
        vlm_timeout=args.vlm_timeout,
        grounding_timeout=args.grounding_timeout,
        use_gpu=args.gpu,
        use_vlm_verification=args.vlm_verify,
        vlm_backend="llamacpp" if args.llamacpp else "ollama",
        vlm_url=args.llamacpp_url if args.llamacpp else "http://localhost:11434",
        execution_backend=args.backend,
        grounding_model=args.grounding,
        uground_4bit=getattr(args, "uground_4bit", False),
        vision_trm_checkpoint=getattr(args, "vision_trm_checkpoint", "checkpoints/vision_trm_best.pt"),
        anorha_trm_checkpoint=getattr(args, "anorha_trm_checkpoint", "checkpoints/anorha_trm_best.pt"),
        save_screenshots=getattr(args, "save_screenshots", False),
    )
    
    _gatherer = SmartDataGatherer(config)
    try:
        await _gatherer.gather_data()
    finally:
        await _gatherer._close_backend_safe()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Progress saved.")
        if _gatherer:
            try:
                asyncio.run(_gatherer._close_backend_safe())
            except Exception:
                pass
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise
