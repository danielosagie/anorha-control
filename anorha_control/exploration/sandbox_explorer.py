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
        print("[Sandbox] Browser initialized")
    
    async def _close_browser(self):
        """Close browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
    
    async def _screenshot(self) -> Image.Image:
        """Capture screenshot of the browser viewport."""
        screenshot_bytes = await self._page.screenshot()
        return Image.open(io.BytesIO(screenshot_bytes))
    
    def _save_screenshot(self, image: Image.Image, prefix: str) -> str:
        """Save screenshot and return path."""
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}.png"
        path = self.config.screenshot_dir / filename
        image.save(path)
        return str(path)
    
    async def _navigate(self, url: str):
        """Navigate to URL."""
        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=15000)
            self.current_site = url
            print(f"[Sandbox] Navigated to: {url}")
        except Exception as e:
            print(f"[Sandbox] Navigation error: {e}")
    
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
                sample_texts = ["hello world", "test@example.com", "John Doe", "123 Main St", "New York"]
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
        """Run one exploration episode."""
        experiences = []
        
        print(f"\n📍 Episode {self.episode_count + 1}")
        print(f"   Site: {self.current_site}")
        print(f"   Task: {self.current_instruction}")
        
        # VLM Planning
        current_plan = []
        if self.planner and self.planner.llm.available:
            print("   🧠 VLM Planning...")
            planning_img = await self._screenshot()
            current_plan = self.planner.plan_task_with_vision(self.current_instruction, planning_img)
            if current_plan:
                print(f"   📝 Plan: {[s.target for s in current_plan[:3]]}...")
        
        max_steps = max(self.config.max_episode_steps, len(current_plan))
        
        for step_idx in range(max_steps):
            if not self._running:
                break
            
            while self._paused:
                await asyncio.sleep(0.5)
            
            # Determine instruction for this step
            if step_idx < len(current_plan):
                step = current_plan[step_idx]
                step_instruction = f"{step.action} {step.target}"
            else:
                step_instruction = self.current_instruction
            
            print(f"\n   Step {step_idx + 1}/{max_steps}: {step_instruction}")
            
            # Capture before
            before_img = await self._screenshot()
            before_path = self._save_screenshot(before_img, "before")
            state_hash_before = phash_image(before_img)
            
            # Get elements and embedding
            elements = await self._get_clickable_elements()
            embedding = self.vision_encoder.encode_image(before_img)
            
            # Generate action with task context
            action = self._generate_action(elements, embedding, step_instruction)
            
            # Log TRM vs actual action
            print(f"   🤖 TRM: ({action.get('trm_x', 0):.0f}, {action.get('trm_y', 0):.0f}) | Action: ({action['x']}, {action['y']}) [{action['source']}]")
            
            # Execute
            await self._execute_action(action)
            await asyncio.sleep(self.config.action_delay)

            
            # Capture after
            after_img = await self._screenshot()
            after_path = self._save_screenshot(after_img, "after")
            state_hash_after = phash_image(after_img)
            
            # Compute reward
            reward, success = self._compute_reward(before_img, after_img, state_hash_after)
            
            status = "✓" if success else "✗"
            print(f"   {status} reward={reward:.2f}")
            
            # Create experience
            exp = Experience(
                screenshot_before_path=before_path,
                screenshot_after_path=after_path,
                action_x=action["x"] / self.config.viewport_width,
                action_y=action["y"] / self.config.viewport_height,
                action_type=action["action_type"],
                reward=reward,
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
                instruction=step_instruction,
                success=success,
                metadata={"step": step_idx, "source": action["source"], "site": self.current_site},
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
                
                # Navigate to task site
                await self._navigate(site)
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

            
            # Show final stats
            stats = await self.db.get_stats()
            curriculum_stats = self.curriculum.get_stats()
            print(f"\n✅ Saved {stats.get('total_actions', 0)} experiences to {self.db.db_path}")
            print(f"📊 Final stats:")
            print(f"   Episodes: {self.episode_count}")
            print(f"   Success rate: {self.total_successes}/{self.total_actions}")
            print(f"   Task types completed: {curriculum_stats['unique_tasks_completed']}")
            print(f"   By category: {curriculum_stats['by_category']}")


    
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
