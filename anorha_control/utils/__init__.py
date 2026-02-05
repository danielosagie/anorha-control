"""Utils package"""
from .screen import ScreenCapture, capture, compute_visual_difference, detect_ui_elements
from .mouse import click, double_click, right_click, type_text, scroll, smooth_move_to, SafeController
from .hashing import phash_image, dhash_image, hash_similarity, StateHasher
from .overlay import get_indicator, ControlIndicator, KillSwitch

__all__ = [
    "ScreenCapture", "capture", "compute_visual_difference", "detect_ui_elements",
    "click", "double_click", "right_click", "type_text", "scroll", "smooth_move_to", "SafeController",
    "phash_image", "dhash_image", "hash_similarity", "StateHasher",
    "get_indicator", "ControlIndicator", "KillSwitch",
]
