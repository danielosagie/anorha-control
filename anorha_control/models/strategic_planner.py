"""
Strategic Planner - High-level task planning using GLM 4.7 Flash.
Breaks complex objectives into step-by-step plans.
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests


@dataclass
class PlanStep:
    """A step in the strategic plan."""
    step_num: int
    action: str  # What to do
    target: str  # What to target
    checkpoint: str  # How to verify success
    fallback: str = ""  # What to do if step fails


class StrategicPlanner:
    """
    Uses GLM 4.7 Flash for high-level task decomposition.
    Only called at episode start for complex, multi-step tasks.
   
    - qwen3:7b - Fallback if GLM unavailable
    """
    
    def __init__(
        self,
        model: str = "qwen3:4b",  # Fast text planner (GPU-optimized)
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,  # Reasonable for 8B
        keep_alive: str = "24h",  # Keep model loaded for 24 hours
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._available: Optional[bool] = None
        self._fallback_model = "qwen3-vl:4b"  # VLM fallback
        print(f"[StrategicPlanner] Using model: {model} (keep_alive={keep_alive})")
    
    @property
    def available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if self._available is None:
            self._available = self._check_available()
        return self._available
    
    def _check_available(self) -> bool:
        """Check server and model availability."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                return False
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check if primary or fallback model exists
            if any(self.model in name for name in model_names):
                return True
            if any(self._fallback_model in name for name in model_names):
                print(f"[StrategicPlanner] {self.model} not found, using fallback: {self._fallback_model}")
                self.model = self._fallback_model
                return True
            
            print(f"[StrategicPlanner] Neither {self.model} nor {self._fallback_model} found")
            return False
        except Exception as e:
            print(f"[StrategicPlanner] Server check failed: {e}")
            return False
    
    def _generate(self, prompt: str, system: str = None) -> str:
        """Generate text from the LLM."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,  # Keep model loaded
            "options": {
                "temperature": 0.3,
                "num_predict": 1024,
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"[StrategicPlanner] Generation error: {e}")
        
        return ""
    
    def plan_objective(
        self,
        objective: str,
        site: str,
        context: str = "",
        max_steps: int = 15,
    ) -> List[PlanStep]:
        """
        Create a strategic plan for a complex objective.
        
        Args:
            objective: What to accomplish (e.g., "Buy the cheapest laptop")
            site: Starting website
            context: Additional context (e.g., current page state)
            max_steps: Maximum plan steps
            
        Returns:
            List of PlanStep objects
        """
        if not self.available:
            return self._fallback_plan(objective)
        
        system = """You are an expert web automation planner. 
Given an objective, create a step-by-step plan that a mouse/keyboard controller can execute.

Output a JSON array of steps. Each step has:
- step_num: integer
- action: click, type, scroll, wait, navigate
- target: description of what to interact with
- checkpoint: how to verify this step succeeded
- fallback: what to do if step fails

Be specific about UI elements. Output ONLY valid JSON."""

        prompt = f"""Objective: {objective}
Starting site: {site}
{f'Context: {context}' if context else ''}

Create a {max_steps}-step maximum plan as a JSON array:"""

        response = self._generate(prompt, system)
        
        # Parse JSON
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                steps_json = json.loads(response[start:end])
                return [
                    PlanStep(
                        step_num=s.get("step_num", i + 1),
                        action=s.get("action", "click"),
                        target=s.get("target", ""),
                        checkpoint=s.get("checkpoint", ""),
                        fallback=s.get("fallback", ""),
                    )
                    for i, s in enumerate(steps_json)
                ]
        except json.JSONDecodeError:
            print(f"[StrategicPlanner] Failed to parse: {response[:200]}...")
        
        return self._fallback_plan(objective)
    
    def verify_checkpoint(
        self,
        checkpoint: str,
        screenshot_description: str,
    ) -> bool:
        """
        Verify if a checkpoint has been reached.
        
        Args:
            checkpoint: Expected state description
            screenshot_description: Description of current screen
            
        Returns:
            True if checkpoint appears to be met
        """
        if not self.available:
            return True  # Assume success if planner unavailable
        
        system = """You verify if a checkpoint has been reached.
Respond with only "YES" or "NO"."""

        prompt = f"""Checkpoint to verify: {checkpoint}

Current screen state: {screenshot_description}

Has the checkpoint been reached? Answer YES or NO only:"""

        response = self._generate(prompt, system)
        return "YES" in response.upper()
    
    def suggest_recovery(
        self,
        failed_action: str,
        current_state: str,
        original_objective: str,
    ) -> str:
        """
        Suggest a recovery action when something fails.
        
        Args:
            failed_action: What action failed
            current_state: Current screen state
            original_objective: What we're trying to accomplish
            
        Returns:
            Suggested recovery action
        """
        if not self.available:
            return "scroll down and try again"
        
        system = """You suggest recovery actions when automation fails.
Be concise. Suggest one specific action."""

        prompt = f"""Failed action: {failed_action}
Current state: {current_state}
Original objective: {original_objective}

Suggest one specific recovery action:"""

        response = self._generate(prompt, system)
        return response if response else "scroll down and try again"
    
    def _fallback_plan(self, objective: str) -> List[PlanStep]:
        """Generate a simple fallback plan when LLM is unavailable."""
        objective_lower = objective.lower()
        
        if "login" in objective_lower:
            return [
                PlanStep(1, "click", "username/email field", "cursor in field"),
                PlanStep(2, "type", "username", "text entered"),
                PlanStep(3, "click", "password field", "cursor in field"),
                PlanStep(4, "type", "password", "text entered"),
                PlanStep(5, "click", "login/submit button", "page changed"),
            ]
        elif "search" in objective_lower:
            return [
                PlanStep(1, "click", "search box", "cursor in search"),
                PlanStep(2, "type", "search query", "text entered"),
                PlanStep(3, "click", "search button", "results displayed"),
            ]
        elif "checkout" in objective_lower or "buy" in objective_lower:
            return [
                PlanStep(1, "scroll", "page", "more items visible"),
                PlanStep(2, "click", "add to cart button", "cart updated"),
                PlanStep(3, "click", "cart icon", "cart page opened"),
                PlanStep(4, "click", "checkout button", "checkout page"),
                PlanStep(5, "type", "form fields", "form filled"),
                PlanStep(6, "click", "submit/complete", "order confirmed"),
            ]
        else:
            return [
                PlanStep(1, "scroll", "page", "more content visible"),
                PlanStep(2, "click", "relevant element", "page changed"),
            ]


# CLI for testing
if __name__ == "__main__":
    print("=== Strategic Planner Test ===\n")
    
    planner = StrategicPlanner()
    print(f"Model: {planner.model}")
    print(f"Available: {planner.available}")
    
    if planner.available:
        print("\n--- Planning Test ---")
        objective = "Find and purchase the cheapest laptop under $500"
        site = "https://www.amazon.com"
        
        print(f"Objective: {objective}")
        print(f"Site: {site}")
        print("\nGenerating plan...")
        
        plan = planner.plan_objective(objective, site)
        
        print(f"\nPlan ({len(plan)} steps):")
        for step in plan:
            print(f"  {step.step_num}. {step.action}: {step.target}")
            print(f"     Checkpoint: {step.checkpoint}")
    else:
        print("\n--- Fallback Plan Test ---")
        plan = planner._fallback_plan("login to the website")
        print(f"Fallback plan ({len(plan)} steps):")
        for step in plan:
            print(f"  {step.step_num}. {step.action}: {step.target}")
