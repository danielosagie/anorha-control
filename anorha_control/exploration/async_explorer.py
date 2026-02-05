"""
Async Explorer - Real mouse control with browser and task generation
Controls the REAL cursor and opens real browser windows
"""
import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import queue
import torch

from PIL import Image

from ..config import config
from ..models.vision_encoder import VisionEncoder, load_vision_encoder
from ..models.trm import TRM, load_trm
from ..utils.screen import ScreenCapture, compute_visual_difference, detect_ui_elements
from ..utils.mouse import click, double_click, right_click, scroll, smooth_move_to, get_position, get_screen_size
from ..utils.hashing import phash_image, StateHasher
from ..utils.overlay import get_indicator
from ..knowledge.database import ExperienceDB, Experience
from .tasks import BrowserLauncher, ExplorationSession, generate_task, EXPLORATION_SITES


@dataclass
class ExplorationConfig:
    """Configuration for exploration."""
    epsilon: float = 0.3  # Random action probability
    max_episode_steps: int = 10
    novelty_bonus: float = 0.3
    transition_threshold: float = 0.05
    screenshot_dir: Path = Path("data/screenshots")
    action_delay: float = 0.8  # Delay between actions
    movement_duration: float = 0.3  # Mouse movement duration
    sites: List[str] = field(default_factory=lambda: EXPLORATION_SITES.copy())
    change_site_interval: int = 5  # Change site every N episodes
    
    def __post_init__(self):
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)


class RealMouseExplorer:
    """
    Autonomous explorer that controls the REAL mouse cursor.
    Opens real browser windows and explores with random tasks.
    """
    
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        trm: TRM,
        db: ExperienceDB,
        config: ExplorationConfig = None,
    ):
        self.vision_encoder = vision_encoder
        self.trm = trm
        self.db = db
        self.config = config or ExplorationConfig()
        
        # Browser and task session
        self.session = ExplorationSession(sites=self.config.sites)
        
        # State tracking
        self.state_hasher = StateHasher()
        self.states_visited: set = set()
        self.actions_tried: Dict[str, set] = {}
        
        # Experience buffer for training
        self.experience_buffer: deque = deque(maxlen=10000)
        self.training_queue: queue.Queue = queue.Queue(maxsize=1000)
        
        # Screen capture
        self.screen = ScreenCapture()
        
        # Stats
        self.episode_count = 0
        self.total_actions = 0
        self.total_successes = 0
        
        # Current task
        self.current_task = None
        self.current_instruction = "explore the page"
        
        # Control
        self._running = False
        self._killed = False
        
        # Overlay indicator
        self.indicator = get_indicator(on_kill=self._on_kill)
    
    def _on_kill(self):
        """Handle kill switch activation."""
        print("\n🛑 Kill switch triggered!")
        self._killed = True
        self._running = False
    
    def _save_screenshot(self, image: Image.Image, prefix: str) -> str:
        """Save screenshot and return path."""
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}.png"
        path = self.config.screenshot_dir / filename
        image.save(path)
        return str(path)
    
    def _compute_reward(
        self,
        before: Image.Image,
        after: Image.Image,
        state_hash_after: str,
    ) -> tuple[float, bool]:
        """Compute reward for an action."""
        visual_diff = compute_visual_difference(before, after)
        
        is_new = state_hash_after not in self.states_visited
        novelty = self.config.novelty_bonus if is_new else 0.0
        
        transition = 0.5 if visual_diff > self.config.transition_threshold else 0.0
        
        reward = transition + novelty
        success = visual_diff > self.config.transition_threshold
        
        return reward, success
    
    def _generate_action(
        self,
        vision_embedding: torch.Tensor,
        elements: List[Dict[str, Any]],
        state_hash: str,
    ) -> Dict[str, Any]:
        """Generate action using curiosity + model prediction."""
        tried = self.actions_tried.get(state_hash, set())
        
        # Random exploration with epsilon probability
        if random.random() < self.config.epsilon or not elements:
            # Focus on center of screen (more likely to hit content)
            x = 0.2 + random.random() * 0.6
            y = 0.2 + random.random() * 0.6
            action_type = random.choice([0, 4])  # click or scroll
            return {"x": x, "y": y, "action_type": action_type, "source": "random"}
        
        # Try untried elements first (curiosity)
        untried_elements = []
        for elem in elements:
            elem_hash = f"{elem['x']:.2f}_{elem['y']:.2f}"
            if elem_hash not in tried:
                untried_elements.append(elem)
        
        if untried_elements:
            elem = random.choice(untried_elements)
            return {
                "x": elem["x"],
                "y": elem["y"],
                "action_type": 0,  # click
                "source": "curiosity",
                "element": elem,
            }
        
        # Use model prediction
        with torch.no_grad():
            output = self.trm(vision_embedding)
            coords = output["coords"][0].cpu()
            action_idx = output["action_type"][0].argmax().item()
        
        return {
            "x": coords[0].item(),
            "y": coords[1].item(),
            "action_type": action_idx,
            "source": "model",
        }
    
    def _capture_state(self) -> tuple[Image.Image, torch.Tensor, str]:
        """Capture current state: screenshot, embedding, hash."""
        screenshot = self.screen.capture()
        embedding = self.vision_encoder.encode_image(screenshot)
        state_hash = phash_image(screenshot)
        return screenshot, embedding, state_hash
    
    def _execute_action(self, action: Dict[str, Any], screen_size: tuple):
        """Execute an action with the REAL mouse."""
        from ..utils.mouse import type_text, press_enter
        
        x = int(action["x"] * screen_size[0])
        y = int(action["y"] * screen_size[1])
        action_type = action["action_type"]
        
        actions = ["click", "right_click", "double_click", "type", "scroll"]
        action_name = actions[action_type]
        
        # Show click indicator (red dot)
        self.indicator.show_click(x, y)
        
        # Move smoothly with bezier curve
        smooth_move_to(x, y, duration=self.config.movement_duration)
        
        # Small pause before action
        time.sleep(0.1)
        
        # Execute the action
        if action_name == "click":
            click(x, y, smooth=False)  # Already moved
        elif action_name == "right_click":
            right_click(x, y, smooth=False)
        elif action_name == "double_click":
            double_click(x, y, smooth=False)
        elif action_name == "type":
            # Type some random text (for search boxes, etc)
            sample_texts = ["hello", "test", "search", "python", "machine learning"]
            text = random.choice(sample_texts)
            type_text(text)
            press_enter()
        elif action_name == "scroll":
            scroll(-3, x=x, y=y)  # Scroll down
    
    async def explore_episode(self) -> List[Experience]:
        """Run one exploration episode."""
        experiences = []
        screen_size = self.screen.screen_size
        
        # Get current task
        task_str = self.current_instruction
        
        print(f"\n📍 Episode {self.episode_count + 1}")
        print(f"   Task: {task_str}")
        
        for step in range(self.config.max_episode_steps):
            if self._killed:
                break
            
            print(f"   Step {step + 1}/{self.config.max_episode_steps}")
            
            # Capture before state
            before_img, before_emb, state_hash_before = self._capture_state()
            before_path = self._save_screenshot(before_img, "before")
            
            # Detect UI elements
            elements = detect_ui_elements(before_img)
            
            # Generate action
            action = self._generate_action(before_emb, elements, state_hash_before)
            
            # Execute action with REAL mouse
            self._execute_action(action, screen_size)
            
            # Wait for screen to update
            await asyncio.sleep(self.config.action_delay)
            
            # Capture after state
            after_img, after_emb, state_hash_after = self._capture_state()
            after_path = self._save_screenshot(after_img, "after")
            
            # Compute reward
            reward, success = self._compute_reward(before_img, after_img, state_hash_after)
            
            status = "✓" if success else "✗"
            print(f"   {status} reward={reward:.2f}")
            
            # Create experience with instruction
            exp = Experience(
                screenshot_before_path=before_path,
                screenshot_after_path=after_path,
                action_x=action["x"],
                action_y=action["y"],
                action_type=action["action_type"],
                reward=reward,
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
                instruction=task_str,  # Store the task instruction
                success=success,
                metadata={"step": step, "elements_count": len(elements), "source": action["source"]},
            )
            experiences.append(exp)
            
            # Track state
            self.states_visited.add(state_hash_before)
            action_hash = f"{action['x']:.2f}_{action['y']:.2f}"
            if state_hash_before not in self.actions_tried:
                self.actions_tried[state_hash_before] = set()
            self.actions_tried[state_hash_before].add(action_hash)
            
            # Update counts
            self.total_actions += 1
            if success:
                self.total_successes += 1
            
            # Store in database
            await self.db.add_experience(exp)
            
            # Add to buffer and queue
            self.experience_buffer.append(exp)
            try:
                self.training_queue.put_nowait(exp)
            except queue.Full:
                pass
            
            # Update knowledge
            await self.db.update_knowledge(state_hash_before, {
                "action": {"x": action["x"], "y": action["y"], "type": action["action_type"]},
                "outcome": state_hash_after,
                "reward": reward,
                "success": success,
                "instruction": task_str,
            })
        
        self.episode_count += 1
        return experiences
    
    async def explore_forever(self):
        """Main exploration loop - runs until killed."""
        self._running = True
        self._killed = False
        
        # Activate green border and kill switch
        self.indicator.start()
        
        print("\n" + "=" * 60)
        print("🟢 ANORHA-CONTROL: Real Mouse Explorer Active")
        print("   Press Cmd+Shift+Escape to STOP")
        print("=" * 60)
        
        # Open browser with first site
        print("\n🌐 Opening browser...")
        self.current_task = self.session.start_session()
        self.current_instruction = self.session.get_task_instruction()
        
        # Focus browser and wait for it to load
        self.session.browser.focus_browser()
        await asyncio.sleep(2)
        
        try:
            while self._running and not self._killed:
                # Run episode
                experiences = await self.explore_episode()
                
                # Change site periodically
                if self.episode_count % self.config.change_site_interval == 0:
                    print("\n🌐 Switching to new site...")
                    self.current_task = self.session.next_task()
                    self.current_instruction = self.session.get_task_instruction()
                    self.session.browser.focus_browser()
                    await asyncio.sleep(2)
                else:
                    # Just get new task on same site
                    self.current_task = self.session.next_task()
                    self.current_instruction = self.session.get_task_instruction()
                
                # Log progress
                if self.episode_count % 5 == 0:
                    stats = await self.db.get_stats()
                    print(f"\n📊 Progress: {stats['total_actions']} actions, "
                          f"{stats['total_successes']} successes, "
                          f"{stats['unique_states']} states, "
                          f"{stats['success_rate']:.1%} rate")
                
                # Small delay between episodes
                await asyncio.sleep(0.5)
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.indicator.stop()
    
    def stop(self):
        """Stop exploration."""
        self._running = False
        self._killed = True
        self.indicator.stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current exploration stats."""
        return {
            "episode_count": self.episode_count,
            "total_actions": self.total_actions,
            "total_successes": self.total_successes,
            "success_rate": self.total_successes / max(1, self.total_actions),
            "unique_states": len(self.states_visited),
            "buffer_size": len(self.experience_buffer),
            "queue_size": self.training_queue.qsize(),
            "current_site": self.session.browser.current_site,
            "current_task": self.current_instruction,
        }


# Keep old name for compatibility
AsyncExplorer = RealMouseExplorer


# Quick test
if __name__ == "__main__":
    async def test():
        print("Testing RealMouseExplorer...")
        print("This will control your REAL mouse and open browser!")
        print("Press Cmd+Shift+Escape to stop at any time.")
        
        time.sleep(3)  # Give user time to read
        
        # Load models
        vision = load_vision_encoder()
        trm = load_trm()
        
        # Create database
        db = ExperienceDB(Path("data/experiences.db"))
        await db.connect()
        
        # Create explorer
        exp_config = ExplorationConfig(
            max_episode_steps=3,  # Short episodes for testing
        )
        explorer = RealMouseExplorer(vision, trm, db, exp_config)
        
        try:
            # Run exploration
            await explorer.explore_forever()
        finally:
            explorer.stop()
            await db.close()
    
    asyncio.run(test())
