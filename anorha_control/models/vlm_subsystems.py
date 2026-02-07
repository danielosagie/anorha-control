"""
VLM Subsystems - Specialized vision models for computer use agent.

This module breaks down the monolithic VLM into 4 fast, specialized components:
1. ElementGrounder - "Where is X?" → coordinates
2. TextReader - "What text is visible?" → OCR
3. StateVerifier - "Did action work?" → success/fail
4. ActionPlanner - "What's next?" → step breakdown

Each can use different backends:
- Ollama (Qwen3-VL-1B, etc.)
- llama.cpp server (GGUF models)
- Direct Python (EasyOCR, etc.)
"""
import requests
import json
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import base64
import io

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[VLM] EasyOCR not installed, OCR will use VLM fallback")


@dataclass
class GroundingResult:
    """Result of element grounding."""
    found: bool
    x: int = 0
    y: int = 0
    confidence: float = 0.0
    element_type: str = ""  # button, input, link, etc.
    

@dataclass  
class OCRResult:
    """Result of text extraction."""
    text: str
    bbox: Tuple[int, int, int, int] = None  # x1, y1, x2, y2
    confidence: float = 0.0


@dataclass
class VerificationResult:
    """Result of state verification."""
    success: bool
    reason: str = ""
    changed: bool = False  # Did the screen change?


class VLMBackend:
    """Base class for VLM backends (Ollama, llama.cpp, etc.)"""
    
    def __init__(self, model: str, base_url: str, timeout: float = 120.0):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._available = None
    
    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._check_available()
        return self._available
    
    def _check_available(self) -> bool:
        raise NotImplementedError
    
    def generate(self, prompt: str, image: Image.Image, max_tokens: int = 256, json_mode: bool = False, system: str = None) -> str:
        raise NotImplementedError
    
    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL image to base64 with cross-platform compatibility."""
        buffer = io.BytesIO()
        # Convert to RGB if needed (Windows screenshots can be RGBA or L)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Save as JPEG for balance between quality and speed
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()



# Models that need -nothink variant (thinking mode breaks JSON output)
_THINKING_MODELS = ("qwen3-vl", "qwen3-vl:2b", "qwen3-vl:4b", "qwen3-vl:8b")


def _resolve_ollama_model(base_url: str, model: str) -> str:
    """Prefer -nothink variant when base model uses thinking mode."""
    if not any(m in model for m in ["qwen3-vl", "qwen3-vl:2b"]):
        return model
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        if r.status_code != 200:
            return model
        tags = [m.get("name", "") for m in r.json().get("models", [])]
        for t in tags:
            if "qwen3-vl-nothink" in t:
                return t
    except Exception:
        pass
    return model


class OllamaBackend(VLMBackend):
    """Ollama backend for VLM."""
    _nothink_warned = False
    
    def _check_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def __init__(self, model: str, base_url: str, timeout: float = 120.0):
        resolved = _resolve_ollama_model(base_url, model)
        if resolved != model:
            print(f"[VLM] Using {resolved} (qwen3-vl thinking mode disabled)")
        super().__init__(resolved, base_url, timeout)
    
    def generate(self, prompt: str, image: Image.Image, max_tokens: int = 256, json_mode: bool = False, system: str = None) -> str:
        if not self.available:
            return ""
        
        img_base64 = self._image_to_base64(image)
        
        num_ctx = 4096
        options = {
            "num_predict": max_tokens, 
            "temperature": 0.1,
            "num_ctx": num_ctx
        }
        
        messages = [{"role": "user", "content": prompt, "images": [img_base64]}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options
        }
        
        if json_mode:
            payload["format"] = "json"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                err = response.text or ""
                print(f"[Ollama] HTTP {response.status_code}: {err[:200]}")
                if response.status_code == 500 and ("expert_weights_scale" in err or "error loading model" in err.lower()):
                    print(f"[Ollama] 💡 Model may need newer Ollama. Try: ollama update && ollama pull llava")
                return ""
                
            result = response.json()
            message = result.get("message", {})
            content = message.get("content", "")
            
            # Fallback: if content empty but thinking has JSON array, extract it
            if not content and message.get("thinking"):
                import re
                array_match = re.search(r'\[[\s\S]*?\]', message["thinking"])
                if array_match:
                    raw = array_match.group()
                    if raw != "[img]" and len(raw) > 10:  # Skip placeholder tokens
                        content = raw
                        print(f"[Ollama Debug] Extracted JSON from thinking fallback")
            
            if not content and message.get("thinking"):
                if not OllamaBackend._nothink_warned:
                    OllamaBackend._nothink_warned = True
                    if "nothink" in self.model.lower() or "qwen3-vl" in self.model.lower():
                        print("\n[VLM] ⚠️ Model returning empty content. Try: ollama pull llava\n")
                    else:
                        print("\n[VLM] ⚠️ Model using thinking mode - output empty. Create no-think variant:")
                        print("   ollama create qwen3-vl-nothink -f Modelfile.qwen3-vl-nothink\n")
            elif not content:
                print(f"[Ollama Debug] Empty content in response")
            
            return content
        except Exception as e:
            print(f"[Ollama] Error: {e}")
            return ""


class LlamaCppBackend(VLMBackend):
    """llama.cpp server backend for GGUF models."""
    
    def _check_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, image: Image.Image, max_tokens: int = 256, json_mode: bool = False, system: str = None) -> str:
        if not self.available:
            print(f"[llama.cpp] Server not available at {self.base_url}")
            return ""
        
        img_base64 = self._image_to_base64(image)
        
        # llama.cpp uses OpenAI-compatible API with multimodal content
        payload = {
            "model": self.model,  # Explicitly send model name (required by some server versions)
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                err_text = response.text
                print(f"[llama.cpp] HTTP {response.status_code}: {err_text[:200]}")
                if "image input is not supported" in err_text or "mmproj" in err_text:
                    if not getattr(LlamaCppBackend, "_warned_vision", False):
                        LlamaCppBackend._warned_vision = True
                        print("\n[llama.cpp] ⚠️ VISION NOT SUPPORTED: Your server is running a text-only model.")
                        print("   Options:")
                        print("   1. Use Ollama instead: omit --llamacpp and run: ollama pull qwen3-vl:2b")
                        print("   2. Start llama.cpp with a VLM: pass --mmproj <path> to load the vision encoder")
                        print("   3. Use a merged GGUF that includes vision (e.g. Qwen3-VL-*-Merged-*.gguf)\n")
                return ""
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                print(f"[llama.cpp] Empty response. Full result: {str(result)[:300]}")
            
            return content
        except Exception as e:
            print(f"[llama.cpp] Error: {e}")
            return ""


# =============================================================================
# SPECIALIZED VLM SUBSYSTEMS
# =============================================================================

class ElementGrounder:
    """
    Fast element grounding: "Where is login button?" → {x, y}
    
    Uses VLM to locate UI elements with pixel coordinates.
    Target latency: 200-500ms
    """
    
    def __init__(self, backend: VLMBackend = None):
        if backend is None:
            # Default to Qwen2.5-VL-7B via Ollama (avoid qwen3-vl loops)
            backend = OllamaBackend("qwen2.5-vl:7b", "http://localhost:11434", timeout=30.0)
        self.backend = backend
    
    
    def locate(self, target: str, screenshot: Image.Image) -> GroundingResult:
        """
        Locate an element on screen.
        
        Args:
            target: Description of element to find (e.g., "login button")
            screenshot: Current screen image
            
        Returns:
            GroundingResult with coordinates if found
        """
        prompt = f"""Look at this screenshot and find the exact location of: "{target}"

If found, respond with ONLY a JSON object:
{{"found": true, "x": <pixel_x>, "y": <pixel_y>, "type": "<element_type>"}}

If not found:
{{"found": false}}

Be precise with pixel coordinates. The image is {screenshot.width}x{screenshot.height} pixels. /nothink"""
        
        # Use JSON mode for structural enforcement
        response = self.backend.generate(prompt, screenshot, max_tokens=2500, json_mode=True)
        return self._parse_grounding_response(response)
    
    def _parse_grounding_response(self, response: str) -> GroundingResult:
        """Parse VLM response into GroundingResult."""
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                if data.get("found"):
                    return GroundingResult(
                        found=True,
                        x=int(data.get("x", 0)),
                        y=int(data.get("y", 0)),
                        confidence=0.8,
                        element_type=data.get("type", "unknown")
                    )
        except Exception:
            pass
        return GroundingResult(found=False)


class TextReader:
    """
    Fast text extraction: "What text is on screen?" → list of (text, bbox)
    
    Uses EasyOCR for speed, falls back to VLM.
    Target latency: 100-300ms
    """
    
    def __init__(self, use_gpu: bool = False):
        self.reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(['en'], gpu=use_gpu)
                print("[TextReader] Using EasyOCR")
            except Exception as e:
                print(f"[TextReader] EasyOCR init failed: {e}")
    
    def extract(self, screenshot: Image.Image) -> List[OCRResult]:
        """
        Extract all visible text from screenshot.
        
        Returns:
            List of OCRResult with text and bounding boxes
        """
        if self.reader is None:
            return []
        
        import numpy as np
        img_array = np.array(screenshot)
        
        try:
            results = self.reader.readtext(img_array)
            return [
                OCRResult(
                    text=text,
                    bbox=(int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])),
                    confidence=conf
                )
                for bbox, text, conf in results
                if conf > 0.3
            ]
        except Exception as e:
            print(f"[TextReader] OCR error: {e}")
            return []
    
    def find_text(self, target: str, screenshot: Image.Image) -> Optional[Tuple[int, int]]:
        """
        Find specific text and return its center coordinates.
        """
        results = self.extract(screenshot)
        target_lower = target.lower()
        
        for result in results:
            if target_lower in result.text.lower():
                if result.bbox:
                    x = (result.bbox[0] + result.bbox[2]) // 2
                    y = (result.bbox[1] + result.bbox[3]) // 2
                    return (x, y)
        return None


class StateVerifier:
    """
    State verification: "Did the action work?" → success/fail
    
    Compares before/after screenshots or analyzes current state.
    Target latency: 200-500ms (only called after actions)
    """
    
    def __init__(self, backend: VLMBackend = None):
        if backend is None:
            backend = OllamaBackend("qwen2.5-vl:7b", "http://localhost:11434", timeout=30.0)
        self.backend = backend
    
    def verify_action(
        self, 
        action: str, 
        before: Image.Image, 
        after: Image.Image
    ) -> VerificationResult:
        """
        Verify if an action succeeded by comparing before/after screenshots.
        
        Args:
            action: Description of action taken (e.g., "clicked login button")
            before: Screenshot before action
            after: Screenshot after action
            
        Returns:
            VerificationResult with success status
        """
        prompt = f"""I just performed this action: "{action}"

Look at the current screen state and determine:
1. Did the action succeed?
2. What changed?

Respond with ONLY a JSON object:
{{"success": true/false, "reason": "brief explanation"}} /nothink"""
        
        response = self.backend.generate(prompt, after, max_tokens=2500, json_mode=True)
        return self._parse_response(response)
    
    def check_state(self, expected: str, screenshot: Image.Image) -> VerificationResult:
        """
        Check if screen matches expected state.
        
        Args:
            expected: Description of expected state (e.g., "login form visible")
            screenshot: Current screen
        """
        prompt = f"""Check if this screen matches the expected state: "{expected}"

Respond with ONLY a JSON object:
{{"success": true/false, "reason": "brief explanation"}} /nothink"""
        
        response = self.backend.generate(prompt, screenshot, max_tokens=2500, json_mode=True)
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> VerificationResult:
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return VerificationResult(
                    success=data.get("success", False),
                    reason=data.get("reason", ""),
                    changed=True
                )
        except Exception:
            pass
        return VerificationResult(success=False, reason="Could not parse response")


class ActionPlanner:
    """
    Action planning: Complex task → atomic steps
    
    Target latency: 300-800ms
    """
    
    def __init__(self, backend: VLMBackend = None, sample_data: Dict[str, Any] = None):
        if backend is None:
            backend = OllamaBackend("qwen2.5-vl:7b", "http://localhost:11434", timeout=60.0)
        self.backend = backend
        self.sample_data = sample_data or {}
    
    def plan(self, task: str, screenshot: Image.Image) -> List[Dict[str, Any]]:
        """
        Create atomic action steps for a task.
        
        Args:
            task: High-level task description
            screenshot: Current screen state
            
        Returns:
            List of action dicts: [{action, target, value, reason}]
        """
        data_line = ""
        if self.sample_data:
            data_line = " Data: " + ", ".join(f"{k}={v}" for k, v in self.sample_data.items())
        
        system = "Output ONLY a JSON array. No reasoning. No thinking. Start with ["
        prompt = f"""Task: {task}{data_line}

Steps = ONE click or ONE type per step. Output JSON only:
[{{"action":"click","target":"element name","value":""}}] or {{"action":"click","target":"element"}}"""
        
        # 512 tokens: less room for thinking loops, forces faster output
        response = self.backend.generate(
            prompt, screenshot, max_tokens=512, json_mode=True,
            system=system if isinstance(self.backend, OllamaBackend) else None
        )
        return self._parse_steps(response)
    
    def _valid_step(self, s: Any) -> bool:
        """Filter out null/invalid steps from garbage JSON."""
        if s is None:
            return False
        if isinstance(s, dict):
            return "action" in s or "target" in s
        return hasattr(s, "action") and hasattr(s, "target")
    
    def _parse_steps(self, response: str) -> List[Dict[str, Any]]:
        try:
            stripped = response.strip()
            # Try full parse first (handles {"steps": [...]} from Astria)
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(stripped)
            if isinstance(obj, list):
                return [s for s in obj if self._valid_step(s)]
            if isinstance(obj, dict):
                if "steps" in obj:
                    steps = obj["steps"]
                    return [s for s in (steps if isinstance(steps, list) else []) if self._valid_step(s)]
                if "action" in obj:
                    return [obj]
            # Fallback: find array with bracket matching (avoids ] inside strings)
            return self._extract_steps_fallback(response)
        except json.JSONDecodeError:
            return self._extract_steps_fallback(response)
        except Exception as e:
            print(f"[VLM Debug] JSON parse error: {e}\nResponse: {response[:300]}")
        return []
    
    def _extract_steps_fallback(self, response: str) -> List[Dict[str, Any]]:
        """Extract steps when full parse fails (handles ] inside strings)."""
        # Try {"steps": [...]} - use raw_decode on the part after "steps":
        idx = response.find('"steps"')
        if idx == -1:
            idx = response.find("'steps'")
        if idx >= 0:
            bracket = response.find("[", idx)
            if bracket >= 0:
                try:
                    decoder = json.JSONDecoder()
                    steps, _ = decoder.raw_decode(response[bracket:])
                    return [s for s in (steps if isinstance(steps, list) else []) if self._valid_step(s)]
                except json.JSONDecodeError:
                    pass
        # Try top-level array
        bracket = response.find("[")
        if bracket >= 0:
            try:
                decoder = json.JSONDecoder()
                steps, _ = decoder.raw_decode(response[bracket:])
                return [s for s in (steps if isinstance(steps, list) else []) if self._valid_step(s)]
            except json.JSONDecodeError:
                pass
        # Single object
        brace = response.find("{")
        if brace >= 0:
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(response[brace:])
                if isinstance(obj, dict) and "action" in obj:
                    return [obj]
            except json.JSONDecodeError:
                pass
        print(f"[VLM Debug] No JSON array or object found in response:\n{response[:300]}")
        return []


# =============================================================================
# UNIFIED VLM INTERFACE
# =============================================================================

class VLMSubsystems:
    """
    Unified interface to all VLM subsystems.
    
    Usage:
        vlm = VLMSubsystems()
        
        # Ground an element
        result = vlm.locate("login button", screenshot)
        
        # Extract text
        texts = vlm.read_text(screenshot)
        
        # Verify action
        success = vlm.verify("clicked button", before, after)
        
        # Plan steps
        steps = vlm.plan("login with admin/password", screenshot)
    """
    
    def __init__(
        self,
        model: str = "llava",  # Most compatible. Alt: Me7war/Astria, youtu/youtu-vl
        base_url: str = "http://localhost:11434",
        backend_type: str = "ollama",  # "ollama" or "llamacpp"
        use_ocr_gpu: bool = False,
        timeout: float = 600.0,  # CPU inference can take 10+ min for 5k tokens
    ):
        # Create backend
        if backend_type == "llamacpp":
            backend = LlamaCppBackend(model, base_url, timeout=timeout)
        else:
            backend = OllamaBackend(model, base_url, timeout=timeout)
        
        # Initialize subsystems
        self.grounder = ElementGrounder(backend)
        self.text_reader = TextReader(use_gpu=use_ocr_gpu)
        self.verifier = StateVerifier(backend)
        self.planner = ActionPlanner(backend)
        
        print(f"[VLMSubsystems] Initialized with {backend_type} backend ({model})")
    
    def locate(self, target: str, screenshot: Image.Image) -> GroundingResult:
        """Locate an element on screen."""
        return self.grounder.locate(target, screenshot)
    
    def read_text(self, screenshot: Image.Image) -> List[OCRResult]:
        """Extract all text from screenshot."""
        return self.text_reader.extract(screenshot)
    
    def find_text(self, target: str, screenshot: Image.Image) -> Optional[Tuple[int, int]]:
        """Find text and return its center coordinates."""
        return self.text_reader.find_text(target, screenshot)
    
    def verify(self, action: str, before: Image.Image, after: Image.Image) -> VerificationResult:
        """Verify if action succeeded."""
        return self.verifier.verify_action(action, before, after)
    
    def check_state(self, expected: str, screenshot: Image.Image) -> VerificationResult:
        """Check if screen matches expected state."""
        return self.verifier.check_state(expected, screenshot)
    
    def plan(self, task: str, screenshot: Image.Image, sample_data: Dict = None) -> List[Dict]:
        """Plan atomic steps for task."""
        if sample_data:
            self.planner.sample_data = sample_data
        return self.planner.plan(task, screenshot)


# =============================================================================
# HYBRID ELEMENT DETECTION (Fast path)
# =============================================================================

class HybridElementDetector:
    """
    Hybrid approach: Element detection + VLM selection
    
    1. Fast: Extract clickable elements with bounding boxes (Playwright)
    2. Fast: VLM selects which element matches target
    
    This is ~5-10x faster than pure VLM grounding.
    """
    
    def __init__(self, vlm_subsystems: VLMSubsystems):
        self.vlm = vlm_subsystems
    
    async def locate_with_elements(
        self,
        target: str,
        screenshot: Image.Image,
        elements: List[Dict[str, Any]],  # From Playwright
    ) -> GroundingResult:
        """
        Locate element using pre-detected element list.
        
        Args:
            target: What to find
            screenshot: Current screen (for fallback)
            elements: List of {tag, text, x, y, width, height} from Playwright
        """
        # Try text matching first (instant)
        target_lower = target.lower()
        for el in elements:
            el_text = el.get("text", "").lower()
            if target_lower in el_text or el_text in target_lower:
                return GroundingResult(
                    found=True,
                    x=el["x"] + el.get("width", 0) // 2,
                    y=el["y"] + el.get("height", 0) // 2,
                    confidence=0.9,
                    element_type=el.get("tag", "unknown")
                )
        
        # Try OCR matching (fast)
        coords = self.vlm.find_text(target, screenshot)
        if coords:
            return GroundingResult(
                found=True,
                x=coords[0],
                y=coords[1],
                confidence=0.85,
                element_type="text"
            )
        
        # Fall back to full VLM (slow but thorough)
        return self.vlm.locate(target, screenshot)


# Quick test
if __name__ == "__main__":
    print("Testing VLM Subsystems...")
    
    vlm = VLMSubsystems()
    
    # Create test image
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([300, 250, 500, 290], fill='blue')
    draw.text((350, 260), "Login", fill='white')
    
    # Test grounding
    print("\nTesting element grounding...")
    result = vlm.locate("Login button", img)
    print(f"  Found: {result.found}, ({result.x}, {result.y})")
    
    # Test text reading
    print("\nTesting text reading...")
    texts = vlm.read_text(img)
    print(f"  Found {len(texts)} text regions")
    
    print("\nVLM Subsystems test complete!")
