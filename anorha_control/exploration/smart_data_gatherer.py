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
from ..models.local_llm import LocalLLM, TaskPlanner
from ..utils.overlay import get_indicator
from .task_curriculum import TaskCurriculum, Difficulty

try:
    import torch
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
    """Complete trajectory data for TRM training."""
    task_id: str
    target: Dict[str, int]  # {x, y}
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    screen_size: List[int] = field(default_factory=lambda: [1920, 1080])
    task_type: str = ""  # click, type, scroll
    source: str = "vlm"  # vlm, human, random
    timestamp: str = ""
    
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
    
    # Training targets
    target_trajectories: int = 5000
    target_success_rate: float = 0.80
    
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
    
    @property
    def trajectories_per_hour(self) -> float:
        if self.elapsed_hours == 0:
            return 0
        return self.total_trajectories / self.elapsed_hours
    
    @property
    def eta_hours(self) -> float:
        if self.trajectories_per_hour == 0 or self.success_rate == 0:
            return float('inf')
        remaining = self.target_trajectories - self.successful_trajectories
        return remaining / (self.trajectories_per_hour * max(0.01, self.success_rate))
    
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
    
    # VLM settings
    vlm_model: str = "qwen3-vl:2b"
    vlm_backend: str = "ollama"
    vlm_url: str = "http://localhost:11434"
    
    # Data settings
    data_dir: Path = Path("data/trajectories")
    save_every: int = 10  # Save after N successful trajectories
    target_trajectories: int = 100000  # Target: 100k for good TRM training
    
    # Episode settings
    max_episode_steps: int = 15
    action_delay: float = 0.5
    max_difficulty: Difficulty = Difficulty.MEDIUM
    
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
            backend_type=self.config.vlm_backend
        )
        
        # Check VLM availability
        self._check_vlm_connection()
        
        # Task curriculum
        self.curriculum = TaskCurriculum(max_difficulty=self.config.max_difficulty)
        
        # Playwright
        self._browser = None
        self._page = None
        
        # Control
        self._running = False
        self._paused = False
        self._killed = False
        
        # Overlay for hotkeys
        self.indicator = get_indicator(on_kill=self._on_kill, on_pause=self._on_pause)
        
        # Load existing progress
        self._load_progress()
    
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
                    return True
            except:
                pass
            
            print(f"[VLM] ⚠️ llama.cpp server not responding at {url}")
            print(f"      Start it with: llama-server.exe -m <model.gguf> --port 8080 -ngl 99")
            print(f"      Or use Ollama: remove --llamacpp flag")
            
        else:  # ollama
            try:
                r = requests.get(f"{url}/api/tags", timeout=2)
                if r.status_code == 200:
                    print(f"[VLM] ✅ Ollama running at {url}")
                    return True
            except:
                pass
            
            print(f"[VLM] ⚠️ Ollama not responding at {url}")
            print(f"      Start it with: ollama serve")
            print(f"      Then run: ollama pull qwen3-vl:2b")
        
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
    
    async def _init_browser(self):
        """Initialize Playwright browser."""
        from playwright.async_api import async_playwright
        
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
            ]
        )
        self._context = await self._browser.new_context(
            viewport={"width": self.config.viewport_width, "height": self.config.viewport_height},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        )
        self._page = await self._context.new_page()
    
    async def _close_browser(self):
        """Close browser gracefully."""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except:
            pass
    
    async def _screenshot(self) -> Image.Image:
        """Capture screenshot."""
        buffer = await self._page.screenshot()
        import io
        return Image.open(io.BytesIO(buffer))
    
    async def _navigate(self, url: str) -> bool:
        """Navigate to URL."""
        try:
            await self._page.goto(url, timeout=30000, wait_until="domcontentloaded")
            await asyncio.sleep(1)  # Wait for page to settle
            return True
        except Exception as e:
            print(f"   ⚠️ Navigation failed: {e}")
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
        text: str = None
    ) -> Optional[TrajectoryData]:
        """
        Execute an action and record the trajectory.
        """
        # Get current mouse position (center of viewport as starting point)
        start_x = random.randint(100, self.config.viewport_width - 100)
        start_y = random.randint(100, self.config.viewport_height - 100)
        
        # Record trajectory
        trajectory = await self._record_trajectory(
            (start_x, start_y),
            (target_x, target_y),
            duration_ms=random.randint(300, 700)
        )
        
        # Execute the actual action
        before_screenshot = await self._screenshot()
        
        if action_type == "click":
            await self._page.mouse.click(target_x, target_y)
        elif action_type == "type" and text:
            await self._page.mouse.click(target_x, target_y)
            await asyncio.sleep(0.2)
            await self._page.keyboard.type(text)
        
        await asyncio.sleep(self.config.action_delay)
        after_screenshot = await self._screenshot()
        
        # Verify action success
        success = self._check_action_success(before_screenshot, after_screenshot)
        
        # Create trajectory data
        traj_data = TrajectoryData(
            task_id=f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            target={"x": target_x, "y": target_y},
            trajectory=trajectory,
            success=success,
            screen_size=[self.config.viewport_width, self.config.viewport_height],
            task_type=action_type,
            source="vlm",
            timestamp=datetime.now().isoformat()
        )
        
        return traj_data
    
    def _check_action_success(self, before: Image.Image, after: Image.Image) -> bool:
        """Check if action caused meaningful change."""
        from ..utils.hashing import phash_image
        
        hash_before = phash_image(before)
        hash_after = phash_image(after)
        
        # If hashes are different, something changed
        return hash_before != hash_after
    
    async def _run_episode(self) -> List[TrajectoryData]:
        """
        Run one VLM-guided exploration episode.
        Returns list of trajectory data collected.
        """
        episode_trajectories = []
        task = self.curriculum.sample_task()
        
        print(f"\n📍 Episode {self.progress.episodes_completed + 1}")
        print(f"   Site: {task.site}")
        print(f"   Task: {task.objective}")
        
        # Navigate
        if not await self._navigate(task.site):
            return []
        
        # Take screenshot and plan with VLM
        screenshot = await self._screenshot()
        sample_data = task.sample_data or {}
        
        steps = self.vlm.plan(task.objective, screenshot, sample_data)
        
        if not steps:
            print("   ⚠️ VLM returned no steps, trying fallback")
            # Fallback: try to find clickable elements
            steps = [{"action": "click", "target": "main button or link", "value": ""}]
        
        print(f"   📋 Planned {len(steps)} steps")
        
        # Execute each step
        for i, step in enumerate(steps[:self.config.max_episode_steps]):
            if self._killed or self._paused:
                break
            
            while self._paused:
                await asyncio.sleep(0.5)
            
            action = step.get("action", "click") if isinstance(step, dict) else step.action
            target = step.get("target", "") if isinstance(step, dict) else step.target
            value = step.get("value", "") if isinstance(step, dict) else step.value
            
            print(f"   Step {i+1}: {action} '{target}'")
            
            # Get fresh screenshot
            screenshot = await self._screenshot()
            
            # Ground the element
            grounding = self.vlm.locate(target, screenshot)
            
            if not grounding.found:
                print(f"      ❌ Element not found")
                self.progress.failed_trajectories += 1
                self.progress.total_trajectories += 1
                continue
            
            # Execute and record
            traj = await self._execute_and_record(
                grounding.x, grounding.y,
                action_type=action,
                text=value if action == "type" else None
            )
            
            if traj:
                episode_trajectories.append(traj)
                
                if traj.success:
                    self.progress.successful_trajectories += 1
                    print(f"      ✅ Success ({grounding.x}, {grounding.y})")
                else:
                    self.progress.failed_trajectories += 1
                    print(f"      ⚠️ No visible change")
                
                self.progress.total_trajectories += 1
        
        self.progress.episodes_completed += 1
        return episode_trajectories
    
    async def gather_data(self):
        """
        Main data gathering loop.
        Runs until target trajectories reached or killed.
        """
        self._running = True
        pending_trajectories = []
        
        print("\n" + "=" * 60)
        print("🤖 SMART DATA GATHERER: VLM-Guided Exploration")
        print("   Press Cmd+Shift+Escape to stop")
        print("   Press Cmd+Shift+P to pause/resume")
        print("=" * 60)
        print(f"\n📊 Target: {self.progress.target_trajectories} successful trajectories")
        print(f"   Current: {self.progress.successful_trajectories}")
        print(f"   Remaining: {self.progress.target_trajectories - self.progress.successful_trajectories}")
        
        self.indicator.start()
        await self._init_browser()
        
        try:
            while self._running and not self._killed:
                # Check if target reached
                if self.progress.successful_trajectories >= self.progress.target_trajectories:
                    print(f"\n🎉 Target reached! {self.progress.successful_trajectories} trajectories collected")
                    break
                
                # Run episode
                episode_trajs = await self._run_episode()
                pending_trajectories.extend(episode_trajs)
                
                # Save periodically
                successful_in_pending = sum(1 for t in pending_trajectories if t.success)
                if successful_in_pending >= self.config.save_every:
                    self._save_trajectories(pending_trajectories)
                    pending_trajectories = []
                
                # Print progress
                self._print_progress()
                
                # Small delay between episodes
                await asyncio.sleep(1)
        
        finally:
            # Save remaining
            if pending_trajectories:
                self._save_trajectories(pending_trajectories)
            
            self._save_progress()
            await self._close_browser()
            self.indicator.stop()
            
            print("\n" + "=" * 60)
            print("📊 Final Statistics:")
            for key, value in self.progress.to_dict().items():
                print(f"   {key}: {value}")
            print("=" * 60)
    
    def _print_progress(self):
        """Print current progress."""
        p = self.progress
        print(f"\n   📈 Progress: {p.progress_percent:.1f}% "
              f"({p.successful_trajectories}/{p.target_trajectories})")
        print(f"      Success rate: {p.success_rate:.1%} | "
              f"Speed: {p.trajectories_per_hour:.0f}/h | "
              f"ETA: {p.eta_hours:.1f}h")


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

async def main():
    """Main entrypoint for data gathering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart VLM-guided data gathering")
    parser.add_argument("--target", type=int, default=5000, help="Target trajectories")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    parser.add_argument("--visible", action="store_true", help="Show browser window")
    parser.add_argument("--model", type=str, default="qwen3-vl:2b", help="VLM model")
    parser.add_argument("--llamacpp", action="store_true", help="Use llama.cpp backend")
    parser.add_argument("--llamacpp-url", type=str, default="http://localhost:8080", help="llama.cpp URL")
    
    args = parser.parse_args()
    
    config = GathererConfig(
        headless=not args.visible,
        target_trajectories=args.target,
        vlm_model=args.model,
        vlm_backend="llamacpp" if args.llamacpp else "ollama",
        vlm_url=args.llamacpp_url if args.llamacpp else "http://localhost:11434",
    )
    
    gatherer = SmartDataGatherer(config)
    await gatherer.gather_data()


if __name__ == "__main__":
    asyncio.run(main())
