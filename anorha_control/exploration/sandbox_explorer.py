"""
Sandbox Explorer - Playwright-based isolated browser exploration.
Runs in a headless/visible browser window without touching your real mouse.
Now with structured curriculum and strategic planning!
"""
import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import queue
import base64
import io

import torch
from PIL import Image

from ..config import config
from ..models.vision_encoder import VisionEncoder
from ..models.trm import TRM
from ..models.local_llm import LocalLLM, TaskPlanner, TaskStep
from ..models.strategic_planner import StrategicPlanner, PlanStep
from ..utils.hashing import phash_image
from ..knowledge.database import ExperienceDB, Experience
from .task_curriculum import TaskCurriculum, Task, TaskCategory, Difficulty
from ..utils.overlay import get_indicator


@dataclass
class SandboxConfig:
    """Configuration for sandbox exploration."""
    headless: bool = False  # Show browser window
    viewport_width: int = 1280
    viewport_height: int = 800
    epsilon: float = 0.2  # Lower epsilon - rely more on curriculum
    max_episode_steps: int = 20
    action_delay: float = 0.8
    screenshot_dir: Path = Path("data/screenshots_sandbox")
    use_curriculum: bool = True  # Use structured curriculum
    use_strategic_planner: bool = True  # Use GLM 4.7 Flash for complex tasks
    max_difficulty: Difficulty = Difficulty.MEDIUM
    
    def __post_init__(self):
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)


class SandboxExplorer:
    """
    Playwright-based exploration with structured curriculum.
    
    Architecture:
      GLM 4.7 Flash (Strategic) -> Qwen3-VL (Tactical) -> TRM (Precision)
    
    Training Estimates:
      - Basic proficiency: 10K-20K experiences (~17-33 hours)
      - High accuracy: 50K-100K experiences (~83-166 hours)
    """
    
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        trm: TRM,
        db: ExperienceDB,
        config: SandboxConfig = None,
        planner: TaskPlanner = None,
        strategic_planner: StrategicPlanner = None,
    ):
        self.vision_encoder = vision_encoder
        self.trm = trm
        self.db = db
        self.config = config or SandboxConfig()
        self.planner = planner or TaskPlanner(LocalLLM())
        
        # Strategic planner for complex tasks
        self.strategic_planner = strategic_planner
        if self.config.use_strategic_planner and not strategic_planner:
            self.strategic_planner = StrategicPlanner()
        
        # Task curriculum
        self.curriculum = TaskCurriculum(max_difficulty=self.config.max_difficulty)
        self.current_task: Optional[Task] = None
        self.current_plan: List[PlanStep] = []
        
        # Playwright browser (initialized async)
        self._browser = None
        self._context = None
        self._page = None
        
        # State tracking
        self.states_visited: set = set()
        self.experience_buffer: deque = deque(maxlen=10000)
        
        # Stats
        self.episode_count = 0
        self.total_actions = 0
        self.total_successes = 0
        
        # Current state
        self.current_site = None
        self.current_instruction = "explore the page"
        
        # Control & Overlay
        self._running = False
        self._paused = False
        self._killed = False
        
        # Visual overlay for hotkeys/status
        self.indicator = get_indicator(on_kill=self._on_kill, on_pause=self._on_pause)
        
        # Training queue for async trainer
        self.training_queue = queue.Queue()
    
    def _on_kill(self):
        """Handle kill switch activation."""
        print("\n🛑 Kill switch triggered!")
        self._killed = True
        self._running = False
        # Browser cleanup happens in stop()
    
    def _on_pause(self, paused: bool):
        """Handle pause toggle."""
        self._paused = paused


    
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
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        self._page = await self._context.new_page()
        
        # Inject ad blocking CSS on every page load
        await self._page.add_init_script("""
            // Block ads via CSS
            const style = document.createElement('style');
            style.textContent = `
                /* Hide common ad containers */
                [class*="ad-"], [class*="ads-"], [class*="advert"],
                [id*="ad-"], [id*="ads-"], [id*="advert"],
                [class*="sponsor"], [id*="sponsor"],
                [class*="banner"], [id*="banner"],
                [class*="promo"], [id*="promo"],
                iframe[src*="ad"], iframe[src*="doubleclick"],
                iframe[src*="googlesyndication"], iframe[src*="amazon-adsystem"],
                [class*="google-ad"], [class*="googleAd"],
                [data-ad], [data-ads], [data-advert],
                ins.adsbygoogle, .adsbygoogle,
                [class*="Ad__"], [class*="__ad"],
                [class*="outbrain"], [class*="taboola"],
                [aria-label*="advertisement"], [aria-label*="Advertisement"],
                div[style*="position: fixed"][style*="z-index: 9"],
                .overlay-ad, .popup-ad, .modal-ad {
                    display: none !important;
                    visibility: hidden !important;
                    height: 0 !important;
                    width: 0 !important;
                    opacity: 0 !important;
                    pointer-events: none !important;
                }
            `;
            document.head.appendChild(style);
            
            // Block ad scripts
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    mutation.addedNodes.forEach((node) => {
                        if (node.tagName === 'SCRIPT' || node.tagName === 'IFRAME') {
                            const src = node.src || '';
                            if (src.includes('ad') || src.includes('doubleclick') || 
                                src.includes('googlesyndication') || src.includes('amazon-adsystem')) {
                                node.remove();
                            }
                        }
                    });
                });
            });
            observer.observe(document.body || document.documentElement, {childList: true, subtree: true});
        """)
        
        print("[Sandbox] Browser initialized with ad blocking")

    
    async def _close_browser(self):
        """Close browser gracefully with error handling."""
        try:
            if self._browser:
                await self._browser.close()
        except Exception as e:
            print(f"[Sandbox] Browser close warning: {e}")
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            print(f"[Sandbox] Playwright stop warning: {e}")
    
    async def _screenshot(self) -> Image.Image:
        """Capture screenshot of the browser viewport with timeout protection."""
        try:
            screenshot_bytes = await self._page.screenshot(timeout=10000)  # 10s max
            return Image.open(io.BytesIO(screenshot_bytes))
        except Exception as e:
            print(f"[Sandbox] Screenshot error: {e}")
            # Return a blank image as fallback
            return Image.new("RGB", (self.config.viewport_width, self.config.viewport_height), (255, 255, 255))
    
    def _save_screenshot(self, image: Image.Image, prefix: str) -> str:
        """Save screenshot and return path."""
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}.png"
        path = self.config.screenshot_dir / filename
        image.save(path)
        return str(path)
    
    async def _navigate(self, url: str) -> bool:
        """Navigate to URL with robust error handling.
        
        Returns:
            True if navigation succeeded, False otherwise
        """
        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self.current_site = url
            print(f"[Sandbox] Navigated to: {url}")
            return True
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                print(f"[Sandbox] ⚠️ Timeout navigating to {url} - skipping")
            elif "net::" in error_msg.lower():
                print(f"[Sandbox] ⚠️ Network error for {url} - skipping")
            else:
                print(f"[Sandbox] ⚠️ Navigation error: {e}")
            return False

    
    async def _get_clickable_elements(self) -> List[Dict[str, Any]]:
        """Get all clickable elements with their positions."""
        elements = await self._page.evaluate("""
            () => {
                const clickable = document.querySelectorAll('a, button, input, select, textarea, [onclick], [role="button"]');
                return Array.from(clickable).slice(0, 50).map(el => {
                    const rect = el.getBoundingClientRect();
                    return {
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        text: (el.innerText || el.value || el.placeholder || '').slice(0, 50),
                        x: rect.x + rect.width / 2,
                        y: rect.y + rect.height / 2,
                        width: rect.width,
                        height: rect.height,
                        visible: rect.width > 0 && rect.height > 0
                    };
                }).filter(el => el.visible && el.x > 0 && el.y > 0);
            }
        """)
        return elements
    
    def _compute_reward(self, before: Image.Image, after: Image.Image, state_hash_after: str) -> tuple[float, bool]:
        """Compute reward for an action."""
        from ..utils.screen import compute_visual_difference
        
        visual_diff = compute_visual_difference(before, after)
        is_new = state_hash_after not in self.states_visited
        novelty = 0.3 if is_new else 0.0
        transition = 0.5 if visual_diff > 0.05 else 0.0
        
        reward = transition + novelty
        success = visual_diff > 0.05
        
        return reward, success
    
    async def _ensure_viewport(self):
        """
        Detect and repair viewport changes.
        Called at the start of each step to handle window resizes.
        """
        try:
            current_size = self._page.viewport_size
            if current_size:
                expected_w = self.config.viewport_width
                expected_h = self.config.viewport_height
                actual_w = current_size.get('width', expected_w)
                actual_h = current_size.get('height', expected_h)
                
                if actual_w != expected_w or actual_h != expected_h:
                    print(f"   ⚠️ Viewport changed: {actual_w}x{actual_h} → adapting")
                    # Update config to match actual viewport
                    self.config.viewport_width = actual_w
                    self.config.viewport_height = actual_h
        except Exception as e:
            # Viewport check is non-critical
            pass
    
    async def _attempt_recovery(self, failed_count: int) -> bool:
        """
        Try recovery strategies when stuck after multiple failures.
        Returns True if recovery was attempted.
        """
        strategies = [
            ("scroll_down", self._recovery_scroll_down),
            ("scroll_up", self._recovery_scroll_up),
            ("go_back", self._recovery_go_back),
            ("click_random", self._recovery_click_random),
        ]
        
        # Cycle through strategies based on failure count
        strategy_idx = (failed_count - 1) % len(strategies)
        name, action_fn = strategies[strategy_idx]
        
        print(f"   🔄 Recovery attempt #{failed_count}: {name}")
        try:
            await action_fn()
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            print(f"   ❌ Recovery failed: {e}")
            return False
    
    async def _recovery_scroll_down(self):
        """Recovery: scroll down to reveal more content."""
        await self._page.mouse.wheel(0, 300)
    
    async def _recovery_scroll_up(self):
        """Recovery: scroll up to return to previous view."""
        await self._page.mouse.wheel(0, -300)
    
    async def _recovery_go_back(self):
        """Recovery: navigate back to previous page."""
        await self._page.go_back()
    
    async def _recovery_click_random(self):
        """Recovery: click a random interactive element."""
        elements = await self._get_page_elements()
        if elements:
            elem = random.choice(elements)
            x, y = int(elem.get("x", 500)), int(elem.get("y", 400))
            await self._page.mouse.click(x, y)
            print(f"   🖱️ Random click @ ({x}, {y})")
    
    async def _execute_action(self, action: Dict[str, Any]):
        """Execute action in the browser."""
        action_type = action.get("action_type", 0)
        x = int(action["x"])
        y = int(action["y"])
        
        actions = ["click", "right_click", "double_click", "type", "scroll"]
        action_name = actions[min(action_type, len(actions) - 1)]
        
        try:
            if action_name == "click":
                await self._page.mouse.click(x, y)
                print(f"   🖱️ Click @ ({x}, {y})")
                
            elif action_name == "double_click":
                await self._page.mouse.dblclick(x, y)
                print(f"   🖱️ Double-click @ ({x}, {y})")
                
            elif action_name == "type":
                # Click first, then type
                await self._page.mouse.click(x, y)
                await asyncio.sleep(0.2)
                
                # Use provided text from step, or fall back to sample data
                text = action.get("text_to_type", "")
                if not text and self.current_task and self.current_task.sample_data:
                    # Try to find relevant text from sample_data
                    data = self.current_task.sample_data
                    # Priority order for common fields
                    for key in ["username", "email", "user", "name", "password", "query", "first", "address"]:
                        if key in data and data[key]:
                            text = str(data[key])
                            break
                
                if not text:
                    # Last resort fallback
                    sample_texts = ["test@example.com", "John Doe", "123 Main St"]
                    text = random.choice(sample_texts)
                    
                await self._page.keyboard.type(text, delay=50)
                print(f"   ⌨️ Type '{text}' @ ({x}, {y})")
                
            elif action_name == "scroll":
                await self._page.mouse.wheel(0, 300)
                print(f"   📜 Scroll down")
                
            elif action_name == "right_click":
                await self._page.mouse.click(x, y, button="right")
                print(f"   🖱️ Right-click @ ({x}, {y})")
                
        except Exception as e:
            print(f"   ❌ Action error: {e}")
    
    def _generate_action(
        self, 
        elements: List[Dict[str, Any]], 
        vision_embedding: torch.Tensor,
        step_instruction: str = "",
    ) -> Dict[str, Any]:
        """
        Generate action based on elements, TRM prediction, and task context.
        
        Hierarchy:
        1. TRM prediction (if available and trained)
        2. Task-specific targeting
        3. Element matching based on instruction
        4. Random exploration (epsilon)
        """
        # Get TRM prediction
        with torch.no_grad():
            trm_out = self.trm(vision_embedding)
            trm_x = trm_out["coords"][0, 0].item()
            trm_y = trm_out["coords"][0, 1].item()
            trm_action_type = trm_out["action_type"][0].argmax().item()
        
        # Convert TRM normalized coords to pixel coords
        trm_px = int(trm_x * self.config.viewport_width)
        trm_py = int(trm_y * self.config.viewport_height)
        
        # Random exploration (epsilon greedy)
        if random.random() < self.config.epsilon:
            if elements:
                elem = random.choice(elements)
                x, y = int(elem["x"]), int(elem["y"])
            else:
                x = random.randint(100, self.config.viewport_width - 100)
                y = random.randint(100, self.config.viewport_height - 100)
            action_type = random.choices([0, 3, 4], weights=[0.5, 0.3, 0.2])[0]
            return {"x": x, "y": y, "action_type": action_type, "source": "random", "trm_x": trm_px, "trm_y": trm_py}
        
        # Task-specific action selection based on instruction keywords
        step_lower = step_instruction.lower()
        
        # Typing tasks
        if any(kw in step_lower for kw in ["type", "enter", "fill", "input", "username", "password", "email"]):
            input_elements = [e for e in elements if e["tag"] in ["input", "textarea"]]
            if input_elements:
                # Find matching input
                for elem in input_elements:
                    elem_text = elem.get("text", "").lower()
                    elem_type = elem.get("type", "").lower()
                    if any(kw in elem_text or kw in elem_type for kw in ["user", "email", "pass", "name", "search", "query"]):
                        return {"x": int(elem["x"]), "y": int(elem["y"]), "action_type": 3, "source": "input_match", "trm_x": trm_px, "trm_y": trm_py}
                # Fallback to first visible input
                elem = input_elements[0]
                return {"x": int(elem["x"]), "y": int(elem["y"]), "action_type": 3, "source": "input_fallback", "trm_x": trm_px, "trm_y": trm_py}
        
        # Click tasks - try to match target description
        if any(kw in step_lower for kw in ["click", "button", "link", "submit", "login", "search"]):
            for elem in elements:
                elem_text = elem.get("text", "").lower()
                # Check if element matches instruction keywords
                if any(kw in elem_text for kw in step_lower.split()):
                    return {"x": int(elem["x"]), "y": int(elem["y"]), "action_type": 0, "source": "text_match", "trm_x": trm_px, "trm_y": trm_py}
            # Use TRM prediction for click
            return {"x": trm_px, "y": trm_py, "action_type": 0, "source": "trm", "trm_x": trm_px, "trm_y": trm_py}
        
        # Scroll tasks
        if any(kw in step_lower for kw in ["scroll", "down", "page"]):
            return {"x": self.config.viewport_width // 2, "y": self.config.viewport_height // 2, "action_type": 4, "source": "scroll", "trm_x": trm_px, "trm_y": trm_py}
        
        # Default: use TRM prediction or random element
        if elements:
            elem = random.choice(elements)
            return {"x": int(elem["x"]), "y": int(elem["y"]), "action_type": 0, "source": "element", "trm_x": trm_px, "trm_y": trm_py}
        else:
            return {"x": trm_px, "y": trm_py, "action_type": trm_action_type, "source": "trm", "trm_x": trm_px, "trm_y": trm_py}

    
    async def explore_episode(self) -> List[Experience]:
        """
        Run one exploration episode with CLOSED-LOOP VLM supervision.
        
        For each step:
        1. VLM provides pixel-level target coordinates (ground truth)
        2. TRM predicts where it thinks the target is
        3. Compare TRM vs VLM distance → this IS the training signal
        4. Execute the closer/better prediction
        5. Store experience WITH VLM ground truth for supervised learning
        """
        experiences = []
        
        print(f"\n📍 Episode {self.episode_count + 1}")
        print(f"   Site: {self.current_site}")
        print(f"   Task: {self.current_instruction}")
        
        # VLM Planning - get step list with task context
        current_plan = []
        if self.planner and self.planner.llm.available:
            print("   🧠 VLM Planning...")
            planning_img = await self._screenshot()
            
            # Pass sample_data for context-aware planning
            sample_data = self.current_task.sample_data if self.current_task else {}
            current_plan = self.planner.plan_task_with_vision(
                self.current_instruction, 
                planning_img,
                sample_data=sample_data
            )
            if current_plan:
                # Show plan with values for debugging
                plan_summary = []
                for s in current_plan[:5]:
                    if s.value:
                        plan_summary.append(f"{s.action}({s.target})='{s.value}'")
                    else:
                        plan_summary.append(s.target)
                print(f"   📝 Plan: {plan_summary}...")
        
        max_steps = min(self.config.max_episode_steps, max(5, len(current_plan) + 2))
        consecutive_failures = 0
        
        for step_idx in range(max_steps):
            if not self._running or consecutive_failures >= 3:
                if consecutive_failures >= 3:
                    print("   ⚠️ 3 consecutive failures - ending episode")
                break
            
            while self._paused:
                await asyncio.sleep(0.5)
            
            # Determine target for this step
            step_value = None  # Text to type for this step
            if step_idx < len(current_plan):
                step = current_plan[step_idx]
                step_instruction = f"{step.action} {step.target}"
                target_description = step.target
                step_value = step.value  # Preserve the value from VLM planning
            else:
                step_instruction = self.current_instruction
                target_description = self.current_instruction
            
            # Show step with value if typing
            if step_value:
                print(f"\n   Step {step_idx + 1}/{max_steps}: {step_instruction} → type '{step_value}'")
            else:
                print(f"\n   Step {step_idx + 1}/{max_steps}: {step_instruction}")
            
            # Check and adapt to viewport changes
            await self._ensure_viewport()
            
            # Capture screenshot
            before_img = await self._screenshot()
            before_path = self._save_screenshot(before_img, "before")
            state_hash_before = phash_image(before_img)
            state_hash_before = phash_image(before_img)
            
            # Get elements for context
            elements = await self._get_clickable_elements()
            embedding = self.vision_encoder.encode_image(before_img)
            
            # =========================================================
            # PHASE 1: Get VLM target (ground truth for training)
            # =========================================================
            vlm_target = None
            if self.planner and self.planner.llm.available:
                vlm_target = self.planner.llm.locate_target(
                    target_description, 
                    before_img,
                    self.config.viewport_width,
                    self.config.viewport_height
                )
            
            # =========================================================
            # PHASE 2: Get TRM prediction
            # =========================================================
            with torch.no_grad():
                trm_out = self.trm(embedding)
                trm_x = trm_out["coords"][0, 0].item()
                trm_y = trm_out["coords"][0, 1].item()
                trm_action_type = trm_out["action_type"][0].argmax().item()
            
            trm_px = int(trm_x * self.config.viewport_width)
            trm_py = int(trm_y * self.config.viewport_height)
            
            # =========================================================
            # PHASE 3: Compare TRM vs VLM and decide execution
            # =========================================================
            import math
            
            if vlm_target and vlm_target.get("found"):
                vlm_x, vlm_y = vlm_target["x"], vlm_target["y"]
                distance = math.sqrt((trm_px - vlm_x)**2 + (trm_py - vlm_y)**2)
                
                # Distance-based reward (KEY training signal)
                if distance < 15:
                    distance_reward = 1.0
                elif distance < 30:
                    distance_reward = 0.8
                elif distance < 60:
                    distance_reward = 0.5
                elif distance < 100:
                    distance_reward = 0.2
                else:
                    distance_reward = 0.0
                
                # Execute TRM if close enough, otherwise use VLM (but store correction)
                if distance < 30:
                    exec_x, exec_y = trm_px, trm_py
                    source = "trm"
                    print(f"   🎯 VLM: ({vlm_x}, {vlm_y}) | TRM: ({trm_px}, {trm_py}) | dist={distance:.0f}px ✓")
                else:
                    exec_x, exec_y = vlm_x, vlm_y
                    source = "vlm_correction"
                    print(f"   🎯 VLM: ({vlm_x}, {vlm_y}) | TRM: ({trm_px}, {trm_py}) | dist={distance:.0f}px → VLM override")
            else:
                # VLM couldn't find target - use element matching as ground truth
                # This lets us train even without VLM
                action = self._generate_action(elements, embedding, step_instruction)
                exec_x, exec_y = action["x"], action["y"]
                source = action["source"]
                
                # Use matched element position as ground truth for training
                # This gives partial reward for learning element types
                if source in ["text_match", "input_match", "button_match", "element"]:
                    vlm_x, vlm_y = exec_x, exec_y  # Element IS the ground truth
                    distance = math.sqrt((trm_px - vlm_x)**2 + (trm_py - vlm_y)**2)
                    
                    # Partial distance reward (lower than VLM but still useful)
                    if distance < 20:
                        distance_reward = 0.6
                    elif distance < 50:
                        distance_reward = 0.3
                    elif distance < 100:
                        distance_reward = 0.1
                    else:
                        distance_reward = 0.0
                    print(f"   📍 Element: ({vlm_x}, {vlm_y}) | TRM: ({trm_px}, {trm_py}) | dist={distance:.0f}px [{source}]")
                else:
                    # Random/fallback - no ground truth
                    vlm_x, vlm_y = None, None
                    distance = None
                    distance_reward = 0.0
                    print(f"   ⚠️ No ground truth - using {source}: ({exec_x}, {exec_y})")
            
            # =========================================================
            # PHASE 4: Execute action
            # =========================================================
            action_type = 0  # Click by default
            if any(kw in step_instruction.lower() for kw in ["type", "enter", "fill", "input"]):
                action_type = 3
            
            action = {
                "x": exec_x, 
                "y": exec_y, 
                "action_type": action_type, 
                "source": source, 
                "trm_x": trm_px, 
                "trm_y": trm_py,
                "vlm_x": vlm_x,
                "vlm_y": vlm_y,
                "text_to_type": step_value,  # Pass VLM-planned text for typing
            }
            
            await self._execute_action(action)
            await asyncio.sleep(self.config.action_delay)
            
            # =========================================================
            # PHASE 5: Capture result and compute reward
            # =========================================================
            after_img = await self._screenshot()
            after_path = self._save_screenshot(after_img, "after")
            state_hash_after = phash_image(after_img)
            
            # Visual change reward
            visual_reward, visual_success = self._compute_reward(before_img, after_img, state_hash_after)
            
            # Combined reward: distance to VLM target + visual change
            if distance is not None:
                # Weighted combination
                reward = 0.7 * distance_reward + 0.3 * visual_reward
                success = distance < 30 or visual_success
            else:
                reward = visual_reward
                success = visual_success
            
            # Track consecutive failures - try recovery before giving up
            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                # Try recovery if failing too much (but not yet at limit)
                if consecutive_failures == 3:
                    print("   🔄 Trying recovery strategies...")
                    await self._attempt_recovery(consecutive_failures)
                    # Reset counter to give one more chance
                    consecutive_failures = 1
            
            status = "✓" if success else "✗"
            print(f"   {status} reward={reward:.2f} (dist_r={distance_reward:.2f} vis_r={visual_reward:.2f})")
            
            # =========================================================
            # PHASE 6: Store experience WITH ground truth
            # =========================================================
            task_category = self.current_task.category.value if self.current_task else "unknown"
            task_name = self.current_task.name if self.current_task else "exploration"
            
            exp = Experience(
                screenshot_before_path=before_path,
                screenshot_after_path=after_path,
                action_x=exec_x / self.config.viewport_width,   # Normalized executed action
                action_y=exec_y / self.config.viewport_height,
                action_type=action_type,
                reward=reward,
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
                instruction=step_instruction,
                success=success,
                metadata={
                    "step": step_idx,
                    "source": source,
                    "site": self.current_site,
                    "category": task_category,
                    "task_name": task_name,
                    "success": success,
                    # GROUND TRUTH for training
                    "vlm_target_x": vlm_x / self.config.viewport_width if vlm_x else None,
                    "vlm_target_y": vlm_y / self.config.viewport_height if vlm_y else None,
                    "trm_predicted_x": trm_px / self.config.viewport_width,
                    "trm_predicted_y": trm_py / self.config.viewport_height,
                    "distance_to_target": distance,
                    "distance_reward": distance_reward,
                },
            )

            experiences.append(exp)
            
            # Store
            self.states_visited.add(state_hash_before)
            await self.db.add_experience(exp)
            self.experience_buffer.append(exp)
            
            self.total_actions += 1
            if success:
                self.total_successes += 1
        
        self.episode_count += 1
        return experiences

    
    async def explore_forever(self):
        """Main exploration loop with structured curriculum."""
        self._running = True
        
        print("\n" + "=" * 60)
        print("🔒 ANORHA-CONTROL: Sandbox Explorer Active")
        print("   Browser-only mode - your mouse is free!")
        print("   Press Cmd+Shift+Escape to stop")
        print("   Press Cmd+Shift+P to pause/resume")
        print("=" * 60)
        
        # Start visual overlay and hotkeys
        self.indicator.start()
        
        # Show training estimates
        estimates = self.curriculum.get_training_estimate()

        print(f"\n📊 Training Estimates:")
        print(f"   Basic proficiency: {estimates['basic_proficiency']['experiences']} experiences")
        print(f"   High accuracy: {estimates['high_accuracy']['experiences']} experiences")
        
        await self._init_browser()
        
        try:
            while self._running:
                # Sample a task from curriculum
                if self.config.use_curriculum:
                    self.current_task = self.curriculum.sample_task()
                    site = self.current_task.site
                    objective = self.current_task.objective
                    max_steps = self.current_task.max_steps
                else:
                    site = random.choice(self.curriculum.get_all_sites())
                    objective = "explore the page and interact with elements"
                    max_steps = self.config.max_episode_steps
                
                # Navigate to task site (skip if fails)
                nav_success = await self._navigate(site)
                if not nav_success:
                    print(f"   ⏭️ Skipping task due to navigation failure")
                    await asyncio.sleep(1)
                    continue
                    
                self.current_instruction = objective

                
                # Strategic planning for complex tasks
                if self.strategic_planner and self.current_task:
                    if self.current_task.category in [TaskCategory.LONGHORIZON, TaskCategory.ECOMMERCE]:
                        print("   🎯 Strategic planning (GLM)...")
                        self.current_plan = self.strategic_planner.plan_objective(
                            objective, site, max_steps=max_steps
                        )
                        if self.current_plan:
                            print(f"   📋 Strategy: {len(self.current_plan)} steps")
                
                # Run episode
                await self.explore_episode()
                
                # Mark success if we completed the estimated steps
                if self.current_task and self.total_successes > 0:
                    self.curriculum.mark_success(self.current_task)
                
                # Log progress
                if self.episode_count % 3 == 0:
                    stats = await self.db.get_stats()
                    print(f"\n📊 Progress: {stats.get('total_actions', 0)} actions, "
                          f"{self.total_successes}/{self.total_actions} successes")
                
                # Increase difficulty after enough successes
                curriculum_stats = self.curriculum.get_stats()
                if curriculum_stats['total_successes'] >= 50 and self.curriculum.max_difficulty == Difficulty.EASY:
                    self.curriculum.increase_difficulty()
                elif curriculum_stats['total_successes'] >= 200 and self.curriculum.max_difficulty == Difficulty.MEDIUM:
                    self.curriculum.increase_difficulty()
                
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self._close_browser()
            self.indicator.stop()

            
            # Show final stats with category breakdown
            stats = await self.db.get_stats()
            category_stats = await self.db.get_category_stats()
            curriculum_stats = self.curriculum.get_stats()
            
            print(f"\n✅ Saved {stats.get('total_actions', 0)} experiences to {self.db.db_path}")
            print(f"\n📊 Final Stats:")
            print(f"   Episodes: {self.episode_count}")
            print(f"   Success rate: {self.total_successes}/{self.total_actions} ({100*self.total_successes/max(1,self.total_actions):.1f}%)")
            
            print(f"\n📂 Experiences by Category:")
            for cat, count in sorted(category_stats.get('by_category', {}).items(), key=lambda x: -x[1]):
                successes = category_stats.get('successes_by_category', {}).get(cat, 0)
                rate = 100 * successes / max(1, count)
                print(f"   {cat}: {count} ({successes} successful, {rate:.1f}%)")
            
            print(f"\n🌐 Top Sites:")
            top_sites = sorted(category_stats.get('by_site', {}).items(), key=lambda x: -x[1])[:10]
            for site, count in top_sites:
                if site and site != "unknown":
                    # Truncate long URLs
                    display_site = site[:50] + "..." if len(site) > 50 else site
                    print(f"   {display_site}: {count}")
            
            print(f"\n🎯 Session Task Types: {curriculum_stats['unique_tasks_completed']}")



    
    def pause(self):
        """Pause exploration."""
        self._paused = True
        print("⏸️ PAUSED")
    
    def resume(self):
        """Resume exploration."""
        self._paused = False
        print("▶️ RESUMED")
    
    def stop(self):
        """Stop exploration."""
        self._running = False
