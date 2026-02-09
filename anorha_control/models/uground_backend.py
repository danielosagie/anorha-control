"""
UGround Backend - GUI-specialized visual grounding (77%+ accuracy vs 16% GPT-4).

Use for data gathering: --grounding uground
Use for inference: quantized 4-bit runs on laptop (~2-3s per locate).

Requires: pip install transformers accelerate qwen-vl-utils
Optional: bitsandbytes for 4-bit quantization (laptop use)
"""
from typing import Optional, Tuple
from PIL import Image
import base64
import io
import re

from .vlm_subsystems import GroundingResult


def _parse_uground_coords(text: str, width: int, height: int) -> Optional[Tuple[int, int]]:
    """Parse UGround output: '(x, y)' or 'x, y' in range [0,1000); scale to image coords."""
    # Match (x, y) or x, y
    m = re.search(r'\(?\s*(\d+)\s*,\s*(\d+)\s*\)?', text)
    if not m:
        return None
    x_norm, y_norm = int(m.group(1)), int(m.group(2))
    if 0 <= x_norm <= 1000 and 0 <= y_norm <= 1000:
        x = int(x_norm / 1000 * width)
        y = int(y_norm / 1000 * height)
        return (x, y)
    return None


class UGroundBackend:
    """
    UGround GUI grounding - screenshot + description -> (x, y).
    Uses osunlp/UGround-V1-2B (Qwen2-VL based).
    """
    
    def __init__(
        self,
        model_id: str = "osunlp/UGround-V1-2B",
        device: str = "auto",
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
    ):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._use_flash_attention = use_flash_attention
    
    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            kwargs = {"torch_dtype": "auto", "device_map": self._device}
            if self._load_in_4bit:
                try:
                    import bitsandbytes
                    kwargs["load_in_4bit"] = True
                except ImportError:
                    print("[UGround] bitsandbytes not installed, using full precision")
            if self._use_flash_attention:
                try:
                    kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    pass
            
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
            self._model.eval()
            dev = str(next(self._model.parameters()).device) if hasattr(self._model, "parameters") and list(self._model.parameters()) else "unknown"
            print(f"[UGround] Loaded {self.model_id} (4bit={self._load_in_4bit}) on {dev}")
        except Exception as e:
            print(f"[UGround] Load failed: {e}")
            raise
    
    def preload(self):
        """Load model at startup so first grounding doesn't block mid-episode."""
        self._ensure_loaded()

    def locate(self, description: str, screenshot: Image.Image) -> GroundingResult:
        """
        Locate element on screen. Returns pixel coords.
        UGround output is in [0,1000), we scale to image size.
        """
        self._ensure_loaded()
        
        w, h = screenshot.size
        prompt = f"""Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {description}

Answer:"""
        
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": screenshot},
                {"type": "text", "text": prompt},
            ]},
        ]
        
        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs = [screenshot]
            video_inputs = []
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                pass
            inp_kw = {"text": [text], "images": image_inputs, "padding": True, "return_tensors": "pt"}
            if video_inputs:
                inp_kw["videos"] = video_inputs
            inputs = self._processor(**inp_kw)
            inputs = inputs.to(self._model.device)
            
            with __import__("torch").no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0,
                    do_sample=False,
                )
            
            out_ids = out[0][inputs.input_ids.shape[1]:]
            output_text = self._processor.decode(out_ids, skip_special_tokens=True).strip()
            
            coords = _parse_uground_coords(output_text, w, h)
            if coords:
                return GroundingResult(
                    found=True,
                    x=coords[0],
                    y=coords[1],
                    confidence=0.9,
                    element_type="unknown",
                )
        except Exception as e:
            print(f"[UGround] Inference error: {e}")
        
        return GroundingResult(found=False)


def get_uground_available() -> bool:
    """Check if UGround can be loaded (transformers + model)."""
    try:
        from transformers import Qwen2VLForConditionalGeneration
        return True
    except ImportError:
        return False
