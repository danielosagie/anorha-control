"""
Anorha SDK Client - Calls Anorha server for LLM tool use.
"""
import base64
from typing import Optional, Dict, Any
import requests


class AnorhaClient:
    """
    Client for Anorha SDK server. Use with LLM tool calling.
    
    Example:
        client = AnorhaClient("http://localhost:8765")
        result = client.click("login button")
        img_b64 = client.screenshot()["image_base64"]
    """
    
    def __init__(self, base_url: str = "http://localhost:8765", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def goto(self, url: str) -> Dict[str, Any]:
        """Navigate to URL."""
        r = requests.post(
            f"{self.base_url}/goto",
            json={"url": url},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
    
    def click(self, target: str) -> Dict[str, Any]:
        """Click on element by semantic description."""
        r = requests.post(
            f"{self.base_url}/click",
            json={"target": target},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
    
    def type_text(self, text: str, target: Optional[str] = None) -> Dict[str, Any]:
        """Type text. Optionally focus target first."""
        payload = {"text": text}
        if target is not None:
            payload["target"] = target
        r = requests.post(
            f"{self.base_url}/type",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
    
    def scroll(self, direction: str, amount: int = 300) -> Dict[str, Any]:
        """Scroll the page."""
        r = requests.post(
            f"{self.base_url}/scroll",
            json={"direction": direction, "amount": amount},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
    
    def screenshot(self) -> Dict[str, Any]:
        """Capture screenshot. Returns base64 image."""
        r = requests.get(f"{self.base_url}/screenshot", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    
    def press_key(self, key: str) -> Dict[str, Any]:
        """Press a key."""
        r = requests.post(
            f"{self.base_url}/press_key",
            json={"key": key},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name (for LLM tool calling).
        Maps tool name to client method.
        """
        if name == "browser_goto":
            return self.goto(arguments["url"])
        if name == "browser_click":
            return self.click(arguments["target"])
        if name == "browser_type":
            return self.type_text(arguments["text"], arguments.get("target"))
        if name == "browser_scroll":
            return self.scroll(
                arguments["direction"],
                arguments.get("amount", 300),
            )
        if name == "browser_screenshot":
            return self.screenshot()
        if name == "browser_press_key":
            return self.press_key(arguments["key"])
        raise ValueError(f"Unknown tool: {name}")
