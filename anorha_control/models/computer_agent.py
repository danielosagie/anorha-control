"""
ComputerAgent - Unified interface for local computer use.

This is the FINAL ASSEMBLY that combines:
1. Orchestrator (high-level LLM) - breaks tasks into steps
2. VLM Subsystems - sees and understands screens
3. TRM - controls mouse with precision
4. Direct execution - keyboard, scroll, etc.

Usage:
    agent = ComputerAgent()
    await agent.execute("Login to Gmail with admin@example.com / password123")
    
The agent handles:
- Task decomposition
- Element grounding
- Mouse movement via TRM
- Keyboard input (direct)
- Verification and retry
"""
import asyncio
import io
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from PIL import Image
import time

# Local imports
from .vlm_subsystems import VLMSubsystems, GroundingResult, VerificationResult
from .trm import TRM, load_trm
from .local_llm import LocalLLM, TaskStep


@dataclass
class AgentConfig:
    """Configuration for ComputerAgent."""
    # VLM settings
    vlm_model: str = "llava"  # Most compatible; Alt: Me7war/Astria
    vlm_backend: str = "ollama"  # "ollama" or "llamacpp"
    vlm_url: str = "http://localhost:11434"
    
    # Orchestrator LLM (for high-level planning)
    orchestrator_model: str = "qwen3:4b"
    
    # TRM settings
    trm_checkpoint: str = None  # Path to trained vision TRM
    trm_use_text_encoder: bool = False  # Use text encoder for semantic instruction conditioning
    trajectory_trm_checkpoint: str = None  # Path to TrajectoryTRM for smooth movement
    use_trajectory_smoothing: bool = False  # Use TrajectoryTRM for smooth path to target
    use_trajectory_smoothing: bool = False  # Use TrajectoryTRM for smooth path to target
    trm_fallback_to_vlm: bool = True  # Use VLM if TRM not confident
    trm_confidence_threshold: float = 0.7
    
    # Execution settings
    trajectory_max_steps: int = 15  # Max TrajectoryTRM steps for smooth path
    max_retries: int = 3
    action_delay_ms: int = 100
    verify_actions: bool = True
    
    # Screen settings
    viewport_width: int = 1920
    viewport_height: int = 1080


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    action: str
    target: str = ""
    coordinates: tuple = (0, 0)
    source: str = ""  # "trm", "vlm", "direct"
    error: str = ""
    duration_ms: float = 0


class ComputerAgent:
    """
    The Final Assembly: A unified interface for local computer use.
    
    Architecture:
        ┌──────────────────────────────────────────┐
        │              ComputerAgent               │
        ├──────────────────────────────────────────┤
        │  Orchestrator (LLM)                      │
        │    ↓ breaks task into steps              │
        │  VLM Subsystems                          │
        │    ├─ ElementGrounder → coordinates      │
        │    ├─ TextReader → OCR                   │
        │    ├─ StateVerifier → success/fail       │
        │    └─ ActionPlanner → micro-steps        │
        │  TRM (trained model)                     │
        │    ↓ executes mouse movements            │
        │  Direct Execution (keyboard, scroll)     │
        └──────────────────────────────────────────┘
    
    Example:
        agent = ComputerAgent(config)
        
        # High-level task
        await agent.execute(
            "Search for 'python tutorials' on Google",
            sample_data={"query": "python tutorials"}
        )
        
        # Low-level action
        await agent.click("search button")
        await agent.type("hello world")
    """
    
    def __init__(
        self,
        config: AgentConfig = None,
        page = None,  # Playwright page
        screenshot_fn: Callable = None,
    ):
        self.config = config or AgentConfig()
        self.page = page
        self._screenshot_fn = screenshot_fn
        
        # Initialize components
        self._init_vlm()
        self._init_trm()
        self._init_trajectory_trm()
        self._init_orchestrator()
        
        print(f"[ComputerAgent] Initialized")
        print(f"  VLM: {self.config.vlm_model} via {self.config.vlm_backend}")
        print(f"  TRM: {'loaded' if self.trm else 'not loaded'}")
        print(f"  TrajectoryTRM: {'loaded' if self.trajectory_trm else 'not loaded'}")
        print(f"  Orchestrator: {self.config.orchestrator_model}")
    
    def _init_vlm(self):
        """Initialize VLM subsystems."""
        self.vlm = VLMSubsystems(
            model=self.config.vlm_model,
            base_url=self.config.vlm_url,
            backend_type=self.config.vlm_backend
        )
    
    def _init_trm(self):
        """Initialize TRM model."""
        self.trm = None
        if self.config.trm_checkpoint:
            try:
                self.trm = load_trm(
                    self.config.trm_checkpoint,
                    use_text_encoder=self.config.trm_use_text_encoder,
                )
                print(f"[ComputerAgent] TRM loaded from {self.config.trm_checkpoint}")
            except Exception as e:
                print(f"[ComputerAgent] TRM load failed: {e}")
    
    def _init_trajectory_trm(self):
        """Initialize TrajectoryTRM for smooth mouse movement."""
        self.trajectory_trm = None
        if self.config.trajectory_trm_checkpoint and self.config.use_trajectory_smoothing:
            try:
                from ..training.trm_training import load_trajectory_trm
                self.trajectory_trm = load_trajectory_trm(self.config.trajectory_trm_checkpoint)
                print(f"[ComputerAgent] TrajectoryTRM loaded from {self.config.trajectory_trm_checkpoint}")
            except Exception as e:
                print(f"[ComputerAgent] TrajectoryTRM load failed: {e}")
    
    def _generate_trajectory(self, target_x: int, target_y: int) -> Optional[List[tuple]]:
        """Generate smooth path from viewport center to target using TrajectoryTRM."""
        if not self.trajectory_trm:
            return None
        import torch
        w, h = self.config.viewport_width, self.config.viewport_height
        cur_x_norm = 0.5
        cur_y_norm = 0.5
        tar_x_norm = target_x / w
        tar_y_norm = target_y / h
        path = []
        vx, vy = 0.0, 0.0
        dt_norm = 0.05
        for _ in range(self.config.trajectory_max_steps):
            dev = next(self.trajectory_trm.parameters()).device if list(self.trajectory_trm.parameters()) else "cpu"
            inp = torch.tensor(
                [[cur_x_norm, cur_y_norm, tar_x_norm, tar_y_norm, vx, vy]],
                dtype=torch.float32,
                device=dev,
            )
            with torch.no_grad():
                coords, click_prob = self.trajectory_trm(inp)
            next_x_norm = coords[0, 0].item()
            next_y_norm = coords[0, 1].item()
            click_p = click_prob[0, 0].item()
            px = int(next_x_norm * w)
            py = int(next_y_norm * h)
            path.append((px, py))
            dist = ((next_x_norm - tar_x_norm) ** 2 + (next_y_norm - tar_y_norm) ** 2) ** 0.5
            if click_p > 0.5 or dist < 0.02:
                break
            vx = (next_x_norm - cur_x_norm) / dt_norm
            vy = (next_y_norm - cur_y_norm) / dt_norm
            cur_x_norm, cur_y_norm = next_x_norm, next_y_norm
        return path if path else None
    
    def _init_orchestrator(self):
        """Initialize orchestrator LLM."""
        self.orchestrator = LocalLLM(model=self.config.orchestrator_model)
    
    async def screenshot(self) -> Image.Image:
        """Capture current screen state."""
        if self._screenshot_fn:
            return await self._screenshot_fn()
        elif self.page:
            buffer = await self.page.screenshot()
            return Image.open(io.BytesIO(buffer))
        else:
            raise RuntimeError("No screenshot method available")
    
    # =========================================================================
    # HIGH-LEVEL EXECUTION
    # =========================================================================
    
    async def execute(
        self,
        task: str,
        sample_data: Dict[str, Any] = None,
        max_steps: int = 20
    ) -> List[ActionResult]:
        """
        Execute a high-level task.
        
        Args:
            task: Natural language task description
            sample_data: Data to use (credentials, form values)
            max_steps: Maximum steps before giving up
            
        Returns:
            List of ActionResults for each step
        """
        results = []
        screenshot = await self.screenshot()
        
        # Step 1: VLM plans atomic steps
        steps = self.vlm.plan(task, screenshot, sample_data)
        
        if not steps:
            # Fallback: use orchestrator for high-level breakdown
            steps = self.orchestrator.plan(task)
        
        print(f"[ComputerAgent] Planned {len(steps)} steps for: {task}")
        
        # Step 2: Execute each step
        for i, step in enumerate(steps[:max_steps]):
            action = step.get("action", "click") if isinstance(step, dict) else step.action
            target = step.get("target", "") if isinstance(step, dict) else step.target
            value = step.get("value", "") if isinstance(step, dict) else step.value
            
            print(f"  Step {i+1}: {action} '{target}'" + (f" → '{value}'" if value else ""))
            
            # Execute based on action type
            if action == "click":
                result = await self.click(target)
            elif action == "type":
                result = await self.type(value, target)
            elif action == "scroll":
                result = await self.scroll(value or "down")
            else:
                result = ActionResult(success=False, action=action, error=f"Unknown action: {action}")
            
            results.append(result)
            
            # Verify if enabled
            if self.config.verify_actions and not result.success:
                # Try recovery
                recovered = await self._attempt_recovery(action, target)
                if recovered:
                    results.append(recovered)
            
            await asyncio.sleep(self.config.action_delay_ms / 1000)
        
        return results
    
    # =========================================================================
    # LOW-LEVEL ACTIONS
    # =========================================================================
    
    async def click(self, target: str, retry: int = 0) -> ActionResult:
        """
        Click on a target element.
        
        Uses TRM for precision if available, falls back to VLM grounding.
        """
        start = time.time()
        screenshot = await self.screenshot()
        
        # Step 1: Ground the element
        grounding = self.vlm.locate(target, screenshot)
        
        if not grounding.found:
            return ActionResult(
                success=False,
                action="click",
                target=target,
                error="Element not found"
            )
        
        # Step 2: Get precise coordinates
        x, y = grounding.x, grounding.y
        source = "vlm"
        
        # Use TRM if available and confident
        if self.trm:
            try:
                # Encode screen for TRM
                from .vision_encoder import VisionEncoder
                encoder = VisionEncoder()
                embedding = encoder.encode_image(screenshot)
                
                # TRM prediction (pass target for instruction conditioning if text encoder set)
                pred = self.trm.predict(
                    embedding,
                    instruction_text=target if getattr(self.trm, "text_encoder", None) else None,
                    screen_size=(self.config.viewport_width, self.config.viewport_height)
                )
                
                if pred["confidence"] > self.config.trm_confidence_threshold:
                    x, y = pred["x"], pred["y"]
                    source = "trm"
            except Exception as e:
                print(f"[ComputerAgent] TRM error: {e}")
        
        # Step 3: Execute click (with optional trajectory smoothing)
        if self.page:
            if self.trajectory_trm:
                path = self._generate_trajectory(x, y)
                if path:
                    for px, py in path[:-1]:
                        await self.page.mouse.move(px, py)
                        await asyncio.sleep(0.02)
                    x, y = path[-1]
                    source = "trajectory_trm"
            await self.page.mouse.click(x, y)
        
        duration = (time.time() - start) * 1000
        
        # Step 4: Verify
        if self.config.verify_actions:
            after = await self.screenshot()
            verify = self.vlm.verify(f"clicked {target}", screenshot, after)
            success = verify.success
        else:
            success = True
        
        return ActionResult(
            success=success,
            action="click",
            target=target,
            coordinates=(x, y),
            source=source,
            duration_ms=duration
        )
    
    async def type(self, text: str, target: str = None) -> ActionResult:
        """
        Type text. If target specified, click it first.
        
        This is DIRECT execution - no TRM needed for keyboard.
        """
        start = time.time()
        
        # Click target first if specified
        if target:
            click_result = await self.click(target)
            if not click_result.success:
                return ActionResult(
                    success=False,
                    action="type",
                    target=target,
                    error=f"Could not click target: {click_result.error}"
                )
        
        # Type the text directly
        if self.page:
            await self.page.keyboard.type(text)
        
        duration = (time.time() - start) * 1000
        
        return ActionResult(
            success=True,
            action="type",
            target=text[:20] + "..." if len(text) > 20 else text,
            source="direct",
            duration_ms=duration
        )
    
    async def scroll(self, direction: str = "down", amount: int = 300) -> ActionResult:
        """
        Scroll the page. DIRECT execution.
        """
        if self.page:
            delta = amount if direction == "down" else -amount
            await self.page.mouse.wheel(0, delta)
        
        return ActionResult(
            success=True,
            action="scroll",
            target=direction,
            source="direct"
        )
    
    async def _attempt_recovery(self, action: str, target: str) -> Optional[ActionResult]:
        """
        Attempt recovery after failed action.
        
        Strategies:
        1. Scroll to find element
        2. Wait and retry
        3. Try alternative target description
        """
        print(f"[ComputerAgent] Attempting recovery for {action} '{target}'")
        
        # Strategy 1: Scroll and retry
        await self.scroll("down", 200)
        await asyncio.sleep(0.3)
        
        if action == "click":
            result = await self.click(target)
            if result.success:
                return result
        
        # Strategy 2: Scroll up
        await self.scroll("up", 400)
        await asyncio.sleep(0.3)
        
        if action == "click":
            result = await self.click(target)
            if result.success:
                return result
        
        return None


# =============================================================================
# FINAL ASSEMBLY EXPLAINED
# =============================================================================

"""
# How The Final Assembly Works

## Data Flow

1. USER REQUEST
   "Book a flight from SF to Tokyo"
   
2. ORCHESTRATOR (high-level LLM, e.g., qwen3:4b)
   Breaks into sub-tasks:
   - "Navigate to Google Flights"
   - "Enter 'San Francisco' in departure"
   - "Enter 'Tokyo' in destination"
   - "Click search"
   
3. VLM SUBSYSTEMS (for each sub-task)
   a. ActionPlanner: "Enter SF" → [click field, type "San Francisco"]
   b. ElementGrounder: "departure field" → {x: 300, y: 200}
   c. TextReader (optional): Extract visible text for context
   d. StateVerifier: "Did we type correctly?" → {success: true}
   
4. EXECUTION LAYER
   a. TRM: Move mouse smoothly to (300, 200)
   b. Direct: Click, then type "San Francisco"
   
5. VERIFICATION
   - VLM compares before/after screenshots
   - If failed, AgentRecovery kicks in (scroll, retry)

## Training Requirements

### TRM Training (one-time, ~$15-30 on Modal)
- Data: 10,000+ mouse trajectories from human demos
- Format: [(t, x, y, vx, vy, click), ...] with target coordinates
- Output: ~5M parameter model, <50MB checkpoint
- Accuracy goal: 95% within 20px of target

### VLM Fine-tuning (optional, ~$50-100 on Modal)
- Data: UI-Vision dataset (element grounding annotations)
- Base model: Qwen3-VL-1B or Florence-2
- Focus: Precise coordinate extraction, element type classification

### Orchestrator (no training needed)
- Use existing LLM (qwen3:4b, llama3.2, etc.)
- Prompt engineering for task decomposition

## File Locations

- vlm_subsystems.py: VLM components (ElementGrounder, TextReader, StateVerifier, ActionPlanner)
- trm.py: TRM model architecture
- trm_training.py: Training pipeline for Modal/local GPU
- computer_agent.py: This file - the unified interface

## Usage Example

```python
from anorha_control.models.computer_agent import ComputerAgent, AgentConfig

config = AgentConfig(
    vlm_model="llava",
    trm_checkpoint="checkpoints/trm_v1.pt",
    verify_actions=True
)

agent = ComputerAgent(config, page=playwright_page)

# High-level execution
results = await agent.execute(
    "Login with admin@example.com and password123",
    sample_data={"email": "admin@example.com", "password": "password123"}
)

# Or low-level
await agent.click("login button")
await agent.type("hello", "search box")
```
"""


# Quick test
if __name__ == "__main__":
    print("ComputerAgent module loaded successfully.")
    print("\nArchitecture:")
    print("  1. Orchestrator breaks task → sub-tasks")
    print("  2. VLM subsystems ground elements → coordinates")
    print("  3. TRM executes precise mouse movements")
    print("  4. Direct execution for keyboard/scroll")
    print("  5. Verification loop ensures success")
