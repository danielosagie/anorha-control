"""
Robust Grounding Harness - Multi-strategy element grounding for 97%+ accuracy.

When the VLM plans a step with target "Submit", we need to find its screen coordinates.
Single-strategy (OCR or VLM) often fails. This harness chains multiple approaches:

1. Playwright DOM (when available): text match on buttons/links/inputs
2. OCR exact match: target in visible text
3. OCR fuzzy + synonyms: "Submit" → ["Start", "Go", "OK", "Click to start"]
4. VLM locate with task context: "Task: X. Find: Y"
5. VLM locate with alternative phrasings: "main button", "primary action"

Each fallback adds resilience. Target: 97% grounding success.
"""
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from PIL import Image

# Common target → alternatives (for OCR/VLM retry when exact fails)
TARGET_SYNONYMS: Dict[str, List[str]] = {
    "submit": ["Submit", "Start", "Go", "OK", "Click to start", "Click", "Login", "Sign in", "Enter"],
    "start": ["Start", "Click to start", "Begin", "Go", "Submit", "Click"],
    "login": ["Login", "Sign in", "Submit", "Log in"],
    "sign in": ["Sign in", "Login", "Log in"],
    "search": ["Search", "Find", "Google Search", "Go"],
    "add to cart": ["Add to cart", "Add", "Cart", "Buy"],
    "click": ["Click", "Submit", "Start", "Go"],
    "ok": ["OK", "Ok", "Submit", "Confirm", "Yes"],
    "next": ["Next", "Continue", "Submit", "Go"],
    "continue": ["Continue", "Next", "Proceed", "Go"],
    # Human Benchmark and similar
    "click to start": ["Click to start", "Start", "Click", "Begin"],
    "wait for green": ["Wait for green", "Click", "Start", "Click to start"],
    "username": ["Username", "User name", "Login", "Email"],
    "password": ["Password", "Pass"],
}


def _get_synonyms(target: str) -> List[str]:
    """Return target and its synonyms for multi-attempt matching."""
    target = target.strip()
    if not target:
        return []
    seen = {target.lower()}
    candidates = [target]
    # Add from map
    key = target.lower().replace(" ", "_")
    for k, alts in TARGET_SYNONYMS.items():
        if k in key or key in k or target.lower() in k:
            for a in alts:
                if a.lower() not in seen:
                    seen.add(a.lower())
                    candidates.append(a)
    return candidates


def _fuzzy_match_score(a: str, b: str) -> float:
    """Simple similarity: 1.0 = exact, 0.0 = no match. No deps."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.9
    # Word overlap
    wa, wb = set(a.split()), set(b.split())
    if not wa:
        return 0.0
    overlap = len(wa & wb) / len(wa)
    return overlap


@dataclass
class GroundingHarnessResult:
    found: bool
    x: int = 0
    y: int = 0
    confidence: float = 0.0
    source: str = ""  # "dom" | "ocr" | "ocr_fuzzy" | "vlm" | "vlm_retry"


class GroundingHarness:
    """
    Robust grounding: chain DOM → OCR → OCR fuzzy → VLM/UGround/VisionTRM/AnorhaTRM → VLM retry.
    Use anorha_trm_backend for unified model (grounding + trajectory, crap-top).
    Use uground_backend for GUI-specialized grounding (77%+ vs 16% GPT-4) when available.
    Use vision_trm_backend for vision-only grounding (crap-top).
    """
    
    def __init__(self, vlm_subsystems, uground_backend=None, vision_trm_backend=None, anorha_trm_backend=None):
        self.vlm = vlm_subsystems
        self.uground = uground_backend  # Optional: UGroundBackend for locate
        self.vision_trm = vision_trm_backend  # Optional: VisionTRMBackend (crap-top)
        self.anorha_trm = anorha_trm_backend  # Optional: AnorhaTRMBackend (unified grounding+trajectory)
    
    def ground_sync(
        self,
        target: str,
        screenshot: Image.Image,
        task_objective: str = "",
        task_category: str = "",
        playwright_elements: Optional[List[Dict[str, Any]]] = None,
    ) -> GroundingHarnessResult:
        """
        Sync grounding: finds screen coordinates for target. Use ground_sync via
        asyncio.to_thread when timeout via asyncio.wait_for is needed.
        """
        target = (target or "").strip()
        if not target:
            return GroundingHarnessResult(found=False)
        
        # 1. Playwright DOM text match (fastest, most accurate when available)
        if playwright_elements:
            result = self._match_dom_elements(target, playwright_elements)
            if result.found:
                return result
        
        # 2. OCR: run extract once, use for exact + fuzzy (avoids 3+ OCR passes)
        try:
            results = self.vlm.read_text(screenshot)
        except Exception:
            results = []
        coords = self._find_in_ocr_results(target, results)
        if coords:
            return GroundingHarnessResult(
                found=True, x=coords[0], y=coords[1],
                confidence=0.9, source="ocr"
            )
        # 3. OCR synonyms (reuse results)
        for alt in _get_synonyms(target):
            if alt == target:
                continue
            coords = self._find_in_ocr_results(alt, results)
            if coords:
                return GroundingHarnessResult(
                    found=True, x=coords[0], y=coords[1],
                    confidence=0.8, source="ocr_fuzzy"
                )
        # 4. OCR fuzzy: reuse results
        result = self._ocr_fuzzy_match_from_results(target, results)
        if result.found:
            return result
        
        # 5. VLM locate with task context
        result = self._vlm_locate_with_context(target, screenshot, task_objective, task_category)
        if result.found:
            return result
        
        # 6. VLM locate with alternative phrasings
        for phrase in self._alternative_phrasings(target):
            result = self._vlm_locate_with_context(phrase, screenshot, task_objective, task_category)
            if result.found:
                result.source = "vlm_retry"
                return result
        
        return GroundingHarnessResult(found=False)

    async def ground(
        self,
        target: str,
        screenshot: Image.Image,
        task_objective: str = "",
        task_category: str = "",
        playwright_elements: Optional[List[Dict[str, Any]]] = None,
    ) -> GroundingHarnessResult:
        """Async wrapper: runs ground_sync in thread so asyncio.wait_for can timeout."""
        import asyncio
        return await asyncio.to_thread(
            self.ground_sync,
            target, screenshot, task_objective, task_category, playwright_elements,
        )
    
    def _match_dom_elements(
        self, target: str, elements: List[Dict[str, Any]]
    ) -> GroundingHarnessResult:
        """Match target against DOM element text."""
        target_lower = target.lower()
        best_score = 0.0
        best_el = None
        
        for el in elements:
            text = (el.get("text") or "").strip()
            if not text:
                continue
            text_lower = text.lower()
            # Exact or substring
            if target_lower in text_lower or text_lower in target_lower:
                score = 1.0 if target_lower == text_lower else 0.85
            else:
                score = _fuzzy_match_score(target, text)
            if score > best_score:
                best_score = score
                best_el = el
        
        if best_el and best_score >= 0.5:
            x = best_el.get("x", 0) + best_el.get("width", 0) // 2
            y = best_el.get("y", 0) + best_el.get("height", 0) // 2
            return GroundingHarnessResult(
                found=True, x=x, y=y,
                confidence=0.95, source="dom"
            )
        return GroundingHarnessResult(found=False)
    
    def _find_in_ocr_results(
        self, target: str, results: list
    ) -> Optional[Tuple[int, int]]:
        """Find target in pre-extracted OCR results."""
        target_lower = target.lower()
        for r in results:
            if not getattr(r, "text", None) or not getattr(r, "bbox", None):
                continue
            if target_lower in (r.text or "").lower():
                x = (r.bbox[0] + r.bbox[2]) // 2
                y = (r.bbox[1] + r.bbox[3]) // 2
                return (x, y)
        return None

    def _ocr_fuzzy_match_from_results(
        self, target: str, results: list
    ) -> GroundingHarnessResult:
        """Use pre-extracted OCR results, pick best fuzzy match."""
        target_lower = target.lower()
        best_score = 0.0
        best_coords = None

        for r in results:
            if not r.bbox:
                continue
            text = (r.text or "").strip()
            if not text:
                continue
            score = _fuzzy_match_score(target, text)
            if score < 0.5:
                continue
            # Prefer longer matches (more specific)
            if len(text) > len(target) and target_lower in text.lower():
                score = 0.9
            if score > best_score:
                best_score = score
                x = (r.bbox[0] + r.bbox[2]) // 2
                y = (r.bbox[1] + r.bbox[3]) // 2
                best_coords = (x, y)
        
        if best_coords and best_score >= 0.5:
            return GroundingHarnessResult(
                found=True, x=best_coords[0], y=best_coords[1],
                confidence=0.75, source="ocr_fuzzy"
            )
        return GroundingHarnessResult(found=False)
    
    def _vlm_locate_with_context(
        self,
        target: str,
        screenshot: Image.Image,
        task_objective: str,
        task_category: str = "",
    ) -> GroundingHarnessResult:
        """VLM/UGround/VisionTRM/AnorhaTRM locate - uses task context when available for disambiguation."""
        if self.anorha_trm:
            gr = self.anorha_trm.locate(target, screenshot, task_category=task_category)
            if gr.found:
                return GroundingHarnessResult(
                    found=True, x=gr.x, y=gr.y,
                    confidence=gr.confidence, source="anorha_trm"
                )
        if self.vision_trm:
            gr = self.vision_trm.locate(target, screenshot, task_category=task_category)
            if gr.found:
                return GroundingHarnessResult(
                    found=True, x=gr.x, y=gr.y,
                    confidence=gr.confidence, source="vision_trm"
                )
        if self.uground:
            gr = self.uground.locate(target, screenshot)
            if gr.found:
                return GroundingHarnessResult(
                    found=True, x=gr.x, y=gr.y,
                    confidence=gr.confidence, source="uground"
                )
        gr = self.vlm.locate(target, screenshot)
        if gr.found:
            return GroundingHarnessResult(
                found=True, x=gr.x, y=gr.y,
                confidence=gr.confidence, source="vlm"
            )
        # Retry with task context if first attempt failed
        if task_objective and len(task_objective) > 5:
            gr = self.vlm.locate_with_task_context(target, screenshot, task_objective)
            if gr.found:
                return GroundingHarnessResult(
                    found=True, x=gr.x, y=gr.y,
                    confidence=gr.confidence, source="vlm"
                )
        return GroundingHarnessResult(found=False)
    
    def _alternative_phrasings(self, target: str) -> List[str]:
        """Alternative phrasings for VLM when exact target fails."""
        t = target.lower()
        phrases = []
        if "button" not in t:
            phrases.append(f"{target} button")
        if "submit" in t or "start" in t or "go" in t:
            phrases.extend(["main button", "primary action", "clickable button", "center button"])
        if "search" in t:
            phrases.extend(["search box", "search input", "search field"])
        return phrases[:3]  # Limit retries
