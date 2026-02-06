"""
Local LLM integration via Ollama.
Uses Qwen3-0.6B for fast local task planning.
Falls back to rule-based planning if Ollama unavailable.
"""
import subprocess
import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import requests
from PIL import Image
import io
import base64


@dataclass
class TaskStep:
    """A single step in a task plan."""
    action: str  # click, type, scroll, wait
    target: Optional[str] = None  # What to target (button, field, etc.)
    value: Optional[str] = None  # Value to type
    reason: str = ""  # Why this step


class LocalLLM:
    """
    Local LLM client using Ollama.
    Optimized for speed with non-thinking mode.
    """
    
    def __init__(
        self,
        model: str = "qwen3-vl:2b",  # 2B is fast on GPU
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,  # Reasonable timeout for 2B model
        keep_alive: str = "24h",  # Keep model loaded for 24 hours
    ):
        self.model = model
        self.base_url = base_url
        self.keep_alive = keep_alive
        
        # Auto-detect large models and increase timeout
        if any(size in model.lower() for size in ["7b", "8b", "13b", "19b", "70b"]):
            self.timeout = max(timeout, 180.0)  # 3 minutes for large models
            print(f"[LocalLLM] Large model detected ({model}), timeout={self.timeout}s")
        else:
            self.timeout = timeout
        
        self._available: Optional[bool] = None
        print(f"[LocalLLM] Using model: {model} (keep_alive={keep_alive})")
    
    @property
    def available(self) -> bool:
        """Check if Ollama is available."""
        if self._available is None:
            self._available = self._check_available()
        return self._available
    
    def _check_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate(
        self,
        prompt: str,
        system: str = None,
        images: Optional[List[str]] = None,  # Base64 encoded images
        temperature: float = 0.3,
        max_tokens: int = 256,
        thinking: bool = False,
    ) -> str:
        """
        Generate text from the LLM, optionally with vision.
        """
        if not self.available:
            return ""
        
        # Add thinking control to prompt if supported
        if not thinking:
            prompt = prompt + "\n\n/no_think"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,  # Keep model loaded in VRAM
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        
        if images:
            payload["images"] = images
        
        try:
            print(f"[LocalLLM] Calling {self.model} (images={len(images) if images else 0})...")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "").strip()
                if text:
                    print(f"[LocalLLM] Got response: {len(text)} chars")
                else:
                    print(f"[LocalLLM] Empty response from {self.model}")
                return text
            else:
                print(f"[LocalLLM] API error: status={response.status_code}, body={response.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"[LocalLLM] Timeout after {self.timeout}s - increase timeout or use smaller model")
        except Exception as e:
            print(f"[LocalLLM] Error: {type(e).__name__}: {e}")
        
        return ""
    
    def plan_task_with_vision(
        self, 
        instruction: str, 
        screenshot: Image.Image
    ) -> List[TaskStep]:
        """
        Plan a task using the VLM's vision capabilities.
        
        Args:
            instruction: What to do
            screenshot: PIL Image of the current screen
            
        Returns:
            List of TaskStep objects
        """
        # Convert PIL to base64
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        system = """You are a GUI automation assistant with vision. 
Analyze the image and the task. Output a JSON array of steps.
Each step has: action (click/type/scroll/wait), target (description of what to interact with), value (for type), reason.
Be extremely precise based on the visual evidence.
Output ONLY valid JSON."""

        prompt = f"Analyze this screen and plan the task: {instruction}"
        
        response = self.generate(
            prompt, 
            system=system, 
            images=[img_base64], 
            temperature=0.1
        )
        
        # Parse JSON
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                steps_json = json.loads(response[start:end])
                return [
                    TaskStep(
                        action=s.get("action", "click"),
                        target=s.get("target"),
                        value=s.get("value"),
                        reason=s.get("reason", ""),
                    )
                    for s in steps_json
                ]
        except json.JSONDecodeError:
            print(f"[LocalLLM] Failed to parse steps from: {response[:100]}...")
            
        return []
    
    def plan_task(self, instruction: str, screen_description: str = "") -> List[TaskStep]:
        """
        Plan a task given an instruction.
        
        Args:
            instruction: What to do (e.g., "log into email")
            screen_description: Optional description of current screen
            
        Returns:
            List of TaskStep objects
        """
        system = """You are a GUI automation assistant. Given a task, output a JSON array of steps.
Each step has: action (click/type/scroll/wait), target (what to interact with), value (for type), reason.
Be concise. Output ONLY valid JSON, no explanation."""

        prompt = f"Task: {instruction}"
        if screen_description:
            prompt += f"\n\nScreen: {screen_description}"
        
        prompt += "\n\nOutput steps as JSON array:"
        
        response = self.generate(prompt, system=system, temperature=0.1)
        
        # Parse JSON
        try:
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                steps_json = json.loads(response[start:end])
                return [
                    TaskStep(
                        action=s.get("action", "click"),
                        target=s.get("target"),
                        value=s.get("value"),
                        reason=s.get("reason", ""),
                    )
                    for s in steps_json
                ]
        except json.JSONDecodeError:
            pass
        
        return []
    
    def describe_screen(self, elements: List[Dict[str, Any]]) -> str:
        """
        Generate a description of the screen given detected elements.
        
        Args:
            elements: List of detected UI elements
            
        Returns:
            Natural language description
        """
        if not elements:
            return "Screen with no detected elements"
        
        # Summarize elements
        element_types = {}
        for el in elements:
            el_type = el.get("type", "element")
            element_types[el_type] = element_types.get(el_type, 0) + 1
        
        summary = ", ".join(f"{count} {t}s" for t, count in element_types.items())
        return f"Screen with {summary}"
    
    def locate_target(
        self, 
        target_description: str, 
        screenshot: Image.Image,
        viewport_width: int = 1280,
        viewport_height: int = 800,
    ) -> Optional[Dict[str, Any]]:
        """
        Use VLM to locate a target element and return its pixel coordinates.
        
        This is the KEY method for supervised training - it provides ground truth
        pixel coordinates for the TRM to learn from.
        
        Args:
            target_description: What to find (e.g., "Login button", "username field")
            screenshot: PIL Image of the current screen
            viewport_width: Viewport width for coordinate scaling
            viewport_height: Viewport height for coordinate scaling
            
        Returns:
            Dict with {x, y, confidence, action_type} or None if not found
        """
        # Convert PIL to base64
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        system = f"""You are a precise UI element locator. 
The image is {viewport_width}x{viewport_height} pixels.
Find the element described and output its CENTER coordinates as JSON.
Output ONLY: {{"x": <pixel_x>, "y": <pixel_y>, "confidence": <0.0-1.0>, "found": true/false}}
If element not found, set found=false and x=0, y=0.
Be VERY precise with pixel coordinates based on the visual."""

        prompt = f"Find this element and output its center pixel coordinates: {target_description}"
        
        response = self.generate(
            prompt, 
            system=system, 
            images=[img_base64], 
            temperature=0.0,  # Deterministic
            max_tokens=150,  # More room for response
        )
        
        # Debug: log raw response
        if response:
            print(f"[LocalLLM] locate_target raw: {response[:200]}")
        else:
            print(f"[LocalLLM] locate_target: No response from VLM")
            return None
        
        # Parse JSON response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                if result.get("found", False):
                    x = int(result.get("x", 0))
                    y = int(result.get("y", 0))
                    print(f"[LocalLLM] locate_target: Found at ({x}, {y})")
                    return {
                        "x": x,
                        "y": y,
                        "confidence": float(result.get("confidence", 0.5)),
                        "found": True,
                    }
                else:
                    print(f"[LocalLLM] locate_target: Element not found by VLM")
            else:
                print(f"[LocalLLM] locate_target: No JSON in response")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[LocalLLM] locate_target parse error: {e}")
        
        return None
    
    def verify_action(
        self, 
        action_description: str,
        before_screenshot: Image.Image,
        after_screenshot: Image.Image,
    ) -> Dict[str, Any]:
        """
        Use VLM to verify if an action succeeded by comparing before/after screenshots.
        
        Args:
            action_description: What was attempted (e.g., "click Login button")
            before_screenshot: Screenshot before action
            after_screenshot: Screenshot after action
            
        Returns:
            Dict with {success: bool, reason: str, next_action: str}
        """
        # Convert to base64
        buf_before = io.BytesIO()
        before_screenshot.save(buf_before, format="PNG")
        img_before = base64.b64encode(buf_before.getvalue()).decode("utf-8")
        
        buf_after = io.BytesIO()
        after_screenshot.save(buf_after, format="PNG")
        img_after = base64.b64encode(buf_after.getvalue()).decode("utf-8")
        
        system = """You verify if a UI action succeeded.
Compare the two images: first is BEFORE, second is AFTER the action.
Output ONLY JSON: {"success": true/false, "reason": "brief explanation", "next_action": "what to do next or empty"}"""

        prompt = f"Did this action succeed? Action: {action_description}"
        
        response = self.generate(
            prompt, 
            system=system, 
            images=[img_before, img_after], 
            temperature=0.0,
            max_tokens=150,
        )
        
        # Parse response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Default: assume success if screen changed
        return {"success": True, "reason": "Unable to verify", "next_action": ""}



class TaskPlanner:
    """
    Plans tasks using LLM or rule-based fallback.
    """
    
    def __init__(self, llm: Optional[LocalLLM] = None):
        self.llm = llm or LocalLLM()
        
        # Rule-based templates
        self._templates = {
            "login": [
                TaskStep("click", "username field", reason="Focus on username"),
                TaskStep("type", "username field", reason="Enter username"),
                TaskStep("click", "password field", reason="Focus on password"),
                TaskStep("type", "password field", reason="Enter password"),
                TaskStep("click", "login button", reason="Submit login"),
            ],
            "search": [
                TaskStep("click", "search field", reason="Focus on search"),
                TaskStep("type", "search field", reason="Enter search query"),
                TaskStep("click", "search button", reason="Submit search"),
            ],
            "scroll": [
                TaskStep("scroll", "page", reason="Scroll to see more content"),
            ],
            "click": [
                TaskStep("click", "target element", reason="Click the target"),
            ],
        }
    
    def plan(self, instruction: str, screen_description: str = "") -> List[TaskStep]:
        """
        Create a plan for the given instruction.
        Uses LLM if available, otherwise falls back to rules.
        """
        # Try LLM first
        if self.llm and self.llm.available:
            steps = self.llm.plan_task(instruction, screen_description)
            if steps:
                return steps
        
        # Fall back to rule-based
        return self._rule_based_plan(instruction)
    
    def _rule_based_plan(self, instruction: str) -> List[TaskStep]:
        """Simple rule-based planning."""
        instruction_lower = instruction.lower()
        
        for keyword, template in self._templates.items():
            if keyword in instruction_lower:
                return template
        
        # Default: just click
        return [TaskStep("click", instruction, reason="Execute instruction")]
    
    def plan_task_with_vision(self, instruction: str, screenshot: Image.Image) -> List[TaskStep]:
        """
        Plan a task using vision.
        Delegates to LLM if available, otherwise falls back to rules.
        """
        if self.llm and self.llm.available:
            steps = self.llm.plan_task_with_vision(instruction, screenshot)
            if steps:
                return steps
        
        # Fall back to rule-based
        return self._rule_based_plan(instruction)



# CLI for testing
if __name__ == "__main__":
    print("Testing LocalLLM + TaskPlanner...")
    
    llm = LocalLLM()
    print(f"Ollama available: {llm.available}")
    
    if llm.available:
        # Test generation
        print("\n--- Generation Test ---")
        start = time.time()
        response = llm.generate("What is 2+2? Answer briefly.")
        elapsed = time.time() - start
        print(f"Response ({elapsed:.2f}s): {response}")
        
        # Test task planning
        print("\n--- Task Planning Test ---")
        start = time.time()
        steps = llm.plan_task("Log into my Gmail account")
        elapsed = time.time() - start
        print(f"Planned in {elapsed:.2f}s:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step.action} -> {step.target}: {step.reason}")
    else:
        print("\nOllama not available. Testing rule-based fallback...")
    
    # Test planner with fallback
    planner = TaskPlanner(llm)
    print("\n--- Planner Test ---")
    for task in ["login to the website", "search for python tutorials", "scroll down"]:
        steps = planner.plan(task)
        print(f"\nTask: {task}")
        for step in steps:
            print(f"  - {step.action}: {step.target}")
