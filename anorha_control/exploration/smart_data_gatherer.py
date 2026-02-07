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
    target_trajectories: int = 100000  # Target: 100k for good TRM training
    
    # Episode settings
    max_episode_steps: int = 15
    action_delay: float = 0.2  # Snappy like human (was 0.5)
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
            backend_type=self.config.vlm_backend,
            use_ocr_gpu=self.config.use_gpu,
            timeout=self.config.vlm_timeout,
        )
        
        # Check VLM availability
        self._check_vlm_connection()
        
        # Task curriculum
        self.curriculum = TaskCurriculum(max_difficulty=self.config.max_difficulty)
        
        # Playwright
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        
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
        
        # Load existing progress
        self._load_progress()
        
        if self.config.use_gpu:
            print("[DataGatherer] 🚀 GPU acceleration enabled for OCR/Post-processing")
    
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
    
    async def _init_browser(self):
        """Initialize Playwright browser."""
        from playwright.async_api import async_playwright
        
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
                "--window-position=0,0",  # Keep it top-left
                "--no-default-browser-check",
                "--no-first-run",
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
        except Exception:
            pass
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
    
    async def _ensure_browser(self) -> bool:
        """Ensure browser is alive; restart if needed. Returns True if ready."""
        try:
            if self._page and not self._page.is_closed():
                return True
        except Exception:
            pass
        # Browser dead - restart
        print("   🔧 Browser restart...")
        await self._close_browser()
        await asyncio.sleep(2)
        await self._init_browser()
        return True
    
    async def _screenshot(self) -> Image.Image:
        """Capture screenshot. Raises if page is dead."""
        if not self._page:
            raise RuntimeError("No browser page")
        buffer = await self._page.screenshot()
        import io
        return Image.open(io.BytesIO(buffer))
    
    def _is_page_crash(self, e: Exception) -> bool:
        """Check if error indicates page crash (needs browser restart)."""
        msg = str(e).lower()
        return "crashed" in msg or "page closed" in msg
    
    async def _navigate(self, url: str) -> bool:
        """Navigate to URL. Restarts browser on crash, retries on failure."""
        for attempt in range(3):
            try:
                await self._page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await asyncio.sleep(1)  # Wait for page to settle
                return True
            except Exception as e:
                print(f"   ⚠️ Navigation failed (attempt {attempt + 1}/3): {str(e)[:80]}...")
                if self._is_page_crash(e):
                    print("   🔧 Page crashed - restarting browser...")
                    await self._close_browser()
                    await asyncio.sleep(3)
                    await self._init_browser()
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
        text: str = None
    ) -> Optional[TrajectoryData]:
        """
        Execute an action and record the trajectory.
        """
        # Get current mouse position (center of viewport as starting point)
        start_x = random.randint(100, self.config.viewport_width - 100)
        start_y = random.randint(100, self.config.viewport_height - 100)
        
        # Record trajectory (snappy: 150-350ms like human)
        trajectory = await self._record_trajectory(
            (start_x, start_y),
            (target_x, target_y),
            duration_ms=random.randint(150, 350)
        )
        
        # Execute the actual action
        before_screenshot = await self._screenshot()
        
        if action_type == "click":
            await self._page.mouse.click(target_x, target_y)
        elif action_type == "press_enter":
            await self._page.mouse.click(target_x, target_y)
            await asyncio.sleep(0.05)
            await self._page.keyboard.press("Enter")
        elif action_type == "type" and text:
            await self._page.mouse.click(target_x, target_y)
            await asyncio.sleep(0.05)
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
        """
        Check if action caused meaningful change. Uses perceptual hash.
        - False positives: animations, ads, timers can trigger phash change
        - False negatives: small typing, slow navigation may not change phash in 0.2s
        Similarity < 0.97 = meaningful change (filters tiny/noise changes).
        """
        from ..utils.hashing import phash_image, hash_similarity
        hash_before = phash_image(before)
        hash_after = phash_image(after)
        sim = hash_similarity(hash_before, hash_after)
        return sim < 0.97  # Meaningful change threshold
    
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
    
    async def _playwright_first_plan(self, task) -> List[Dict[str, Any]]:
        """
        Playwright-first: DOM-based planning. Fast, no VLM.
        Returns steps with optional "bbox" (x,y,w,h) for direct execution.
        """
        if not self._page:
            return []
        
        steps = []
        obj_lower = task.objective.lower()
        sample_data = task.sample_data or {}
        
        # Skip form fallback for calculator (needs button clicks, not form fill)
        if "solve" in obj_lower or "calculator" in obj_lower or "expression" in (sample_data or {}):
            return []
        
        # Search task: find search input, type query
        if "search" in obj_lower or "query" in sample_data:
            query = sample_data.get("query") or sample_data.get("q") or ""
            if not query and "search" in obj_lower:
                # Extract from objective: "Search for 'X'" or "Search for a book"
                if " for " in task.objective:
                    parts = task.objective.split(" for ", 1)
                    if len(parts) > 1:
                        query = parts[1].split(",")[0].split(" and ")[0].strip().strip("'\"").strip()
                if not query:
                    query = "book" if "book" in obj_lower else "search"
            if query:
                search_el = await self._page.query_selector(
                    'input[type="search"], input[name*="search"], input[placeholder*="search"], '
                    'input[placeholder*="Search"], input[aria-label*="search"], input[id*="search"]'
                )
                if search_el:
                    try:
                        box = await search_el.bounding_box()
                        if box:
                            cx = int(box["x"] + box["width"] / 2)
                            cy = int(box["y"] + box["height"] / 2)
                            steps.append({"action": "type", "target": "search", "value": query, "x": cx, "y": cy})
                            # Add Enter or search button click
                            submit = await self._page.query_selector('button[type="submit"], input[type="submit"], [aria-label*="search" i]')
                            if submit:
                                sbox = await submit.bounding_box()
                                if sbox:
                                    steps.append({"action": "click", "target": "search submit", "x": int(sbox["x"]+sbox["width"]/2), "y": int(sbox["y"]+sbox["height"]/2)})
                            else:
                                steps.append({"action": "press_enter", "target": "", "value": "", "x": cx, "y": cy})
                            print(f"   📋 Playwright search: type '{query[:20]}...' in search box")
                    except Exception:
                        pass
        
        if not steps and sample_data:
            steps = await self._playwright_form_fallback_steps(task)
            if steps:
                pass
        
        # Generic input fallback: "type X into input" without sample_data
        if not steps and ("type" in obj_lower or "input" in obj_lower or "number" in obj_lower):
            steps = await self._generic_input_fallback(task)
        
        return steps
    
    async def _playwright_form_fallback_steps(self, task) -> List[Dict[str, Any]]:
        """
        Use Playwright to find form fields and create type steps from sample_data.
        Works when VLM fails but we have labeled inputs (name, email, address, etc.).
        """
        if not self._page or not task.sample_data:
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
            inputs = await self._page.query_selector_all("input:not([type=hidden]):not([type=submit]):not([type=button]), textarea")
            for el in inputs:
                if not el:
                    continue
                placeholder = await el.get_attribute("placeholder") or ""
                name = await el.get_attribute("name") or ""
                id_attr = await el.get_attribute("id") or ""
                
                # Try <label for="id"> if we have id
                label_text = ""
                if id_attr:
                    label_el = await self._page.query_selector(f'label[for="{id_attr}"]')
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
                        try:
                            box = await el.bounding_box()
                            if box:
                                cx, cy = int(box["x"] + box["width"]/2), int(box["y"] + box["height"]/2)
                                steps.append({"action": "type", "target": display_label or s_key, "value": s_val, "x": cx, "y": cy})
                            else:
                                steps.append({"action": "type", "target": display_label or s_key, "value": s_val})
                        except Exception:
                            steps.append({"action": "type", "target": display_label or s_key, "value": s_val})
                        used_keys.add(s_key)
                        break
            
            if steps:
                submit_btn = await self._page.query_selector("button[type=submit], input[type=submit]")
                submit_text = await submit_btn.inner_text() if submit_btn else None
                try:
                    if submit_btn:
                        box = await submit_btn.bounding_box()
                        if box:
                            cx, cy = int(box["x"] + box["width"]/2), int(box["y"] + box["height"]/2)
                            steps.append({"action": "click", "target": (submit_text or "Submit").strip(), "value": "", "x": cx, "y": cy})
                        else:
                            steps.append({"action": "click", "target": (submit_text or "Submit").strip(), "value": ""})
                    else:
                        steps.append({"action": "click", "target": "Submit", "value": ""})
                except Exception:
                    steps.append({"action": "click", "target": (submit_text or "Submit").strip(), "value": ""})
                print(f"   📋 Playwright form fallback: {len(steps)} steps from DOM")
        
        except Exception as e:
            print(f"   ⚠️ Playwright form fallback failed: {e}")
        
        return steps
    
    async def _generic_input_fallback(self, task) -> List[Dict[str, Any]]:
        """
        Fallback when task is "type into input" but we have no sample_data.
        Finds first visible input and creates a type step (e.g. "123" for numbers).
        """
        if not self._page:
            return []
        obj_lower = task.objective.lower()
        try:
            inputs = await self._page.query_selector_all(
                "input:not([type=hidden]):not([type=submit]):not([type=button]), textarea"
            )
            if not inputs:
                return []
            el = inputs[0]
            box = await el.bounding_box()
            if not box:
                return []
            cx = int(box["x"] + box["width"] / 2)
            cy = int(box["y"] + box["height"] / 2)
            # Use "123" for number tasks, else "test"
            value = "123" if "number" in obj_lower else "test"
            steps = [{"action": "type", "target": "input", "value": value, "x": cx, "y": cy}]
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
            print(f"   🔧 Episode error (recovering): {e}")
            self.progress.episodes_completed += 1
            self._save_progress()
            return []
    
    async def _run_episode_impl(self) -> List[TrajectoryData]:
        """Inner episode logic - may raise on browser/VLM errors."""
        episode_trajectories = []
        task = self.curriculum.sample_task()
        
        print(f"\n📍 Episode {self.progress.episodes_completed + 1}")
        print(f"   Site: {task.site}")
        print(f"   Task: {task.objective}")
        
        ep_start = time.time()
        
        # Navigate (retry once on failure)
        if not await self._navigate(task.site):
            return []
        
        screenshot = await self._screenshot()
        vlm_img, scale_x, scale_y = self._resize_for_vlm(screenshot)
        sample_data = task.sample_data or {}
        
        # Playwright-first: DOM-based (search, forms) - fast, no VLM
        steps = await self._playwright_first_plan(task)
        if not steps:
            # VLM: visual planning for complex tasks (retry once if garbage)
            steps = self.vlm.plan(task.objective, vlm_img, sample_data)
            if not steps:
                screenshot = await self._screenshot()
                vlm_img, _, _ = self._resize_for_vlm(screenshot)
                steps = self.vlm.plan(task.objective, vlm_img, sample_data)
        if not steps and "click" in task.objective.lower():
            # OCR fallback: find task-relevant text and click it
            steps = self._ocr_fallback_steps(task, vlm_img)
        if not steps:
            print("   ⚠️ VLM returned no steps. Skipping.")
            self.progress.episodes_completed += 1
            return []
        
        print(f"   📋 Planned {len(steps)} steps")
        
        # Execute each step (filter None and invalid steps from OCR/fallback)
        steps = [
            s for s in steps[:self.config.max_episode_steps]
            if s is not None and (isinstance(s, dict) or (hasattr(s, "action") and hasattr(s, "target")))
        ]
        for i, step in enumerate(steps):
            if self._killed or self._paused:
                break
            
            while self._paused:
                await asyncio.sleep(0.5)
            
            action = step.get("action", "click") if isinstance(step, dict) else step.action
            target = step.get("target", "") if isinstance(step, dict) else step.target
            value = step.get("value", "") if isinstance(step, dict) else step.value
            
            # Skip steps with no target and no coords
            has_coords = "x" in step and "y" in step
            if not has_coords and not (target or "").strip():
                print(f"   Step {i+1}: {action} '{target}' → ⚠️ Skipped (empty target, no coords)")
                continue
            
            print(f"   Step {i+1}: {action} '{target}'")
            
            # Direct coords from Playwright (skip VLM/OCR)
            if has_coords:
                target_x = int(step["x"])
                target_y = int(step["y"])
            else:
                # Ground: OCR first, then VLM
                screenshot = await self._screenshot()
                vlm_img, scale_x, scale_y = self._resize_for_vlm(screenshot)
                coords = self.vlm.find_text(target, vlm_img)
                if coords:
                    grounding = GroundingResult(found=True, x=coords[0], y=coords[1])
                else:
                    grounding = self.vlm.locate(target, vlm_img)
                if not grounding.found:
                    print(f"      ❌ Element not found")
                    self.progress.failed_trajectories += 1
                    self.progress.total_trajectories += 1
                    continue
                target_x = int(grounding.x * scale_x)
                target_y = int(grounding.y * scale_y)
            
            # Execute and record
            act_type = "press_enter" if action == "press_enter" else action
            traj = await self._execute_and_record(
                target_x, target_y,
                action_type=act_type,
                text=value if action == "type" else None
            )
            
            if traj:
                episode_trajectories.append(traj)
                
                if traj.success:
                    self.progress.successful_trajectories += 1
                    print(f"      ✅ Success ({target_x}, {target_y})")
                else:
                    self.progress.failed_trajectories += 1
                    print(f"      ⚠️ No visible change")
                
                self.progress.total_trajectories += 1
        
        self.progress.episodes_completed += 1
        ep_sec = time.time() - ep_start
        if ep_sec >= 5:  # Log VLM-heavy episodes (Playwright-only is usually <2s)
            print(f"   ⏱️ Episode: {ep_sec:.1f}s")
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
        print("   Press Cmd+Shift+Escape to stop | Cmd+Shift+P to pause")
        print("   Self-healing: browser restarts on crash, progress saved every episode")
        print("=" * 60)
        print(f"\n📊 Target: {self.progress.target_trajectories} successful trajectories")
        print(f"   Current: {self.progress.successful_trajectories}")
        print(f"   Remaining: {self.progress.target_trajectories - self.progress.successful_trajectories}")
        
        self.indicator.start()
        # Init browser with retry
        for init_attempt in range(3):
            try:
                await self._init_browser()
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
                
                # Ensure browser is alive before each episode
                await self._ensure_browser()
                
                try:
                    episode_trajs = await self._run_episode()
                    pending_trajectories.extend(episode_trajs)
                    consecutive_errors = 0  # Reset on success
                except Exception as e:
                    consecutive_errors += 1
                    print(f"   🔧 Episode crash ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    self._save_progress()
                    if consecutive_errors >= max_consecutive_errors:
                        print("   🔧 Too many consecutive errors - restarting browser...")
                        await self._close_browser()
                        await asyncio.sleep(5)
                        await self._init_browser()
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
                self._print_progress()
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
    parser.add_argument("--model", type=str, default="moondream", help="VLM: moondream (default), llava-phi3, llava, Me7war/Astria")
    parser.add_argument("--vlm-timeout", type=float, default=600, help="VLM request timeout (seconds, CPU needs ~10 min)")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU for OCR (default: on)")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Disable GPU (CPU-only)")
    parser.add_argument("--llamacpp", action="store_true", help="Use llama.cpp backend")
    parser.add_argument("--llamacpp-url", type=str, default="http://localhost:8080", help="llama.cpp URL")
    
    args = parser.parse_args()
    
    config = GathererConfig(
        headless=not args.visible,
        target_trajectories=args.target,
        vlm_model=args.model,
        vlm_timeout=args.vlm_timeout,
        use_gpu=args.gpu,
        vlm_backend="llamacpp" if args.llamacpp else "ollama",
        vlm_url=args.llamacpp_url if args.llamacpp else "http://localhost:11434",
    )
    
    gatherer = SmartDataGatherer(config)
    await gatherer.gather_data()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Progress saved.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise
