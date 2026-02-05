"""
Anorha-Control: TRM-based autonomous GUI control
"""
from .config import Config, config
from .models import VisionEncoder, load_vision_encoder, TRM, load_trm
from .utils import (
    ScreenCapture, capture, 
    click, double_click, right_click, type_text, scroll, SafeController,
    StateHasher,
    get_indicator,
)
from .knowledge import ExperienceDB, Experience
from .exploration import AsyncExplorer, RealMouseExplorer, ExplorationConfig
from .training import AsyncTrainer

__version__ = "0.1.0"
__all__ = [
    # Config
    "Config", "config",
    # Models
    "VisionEncoder", "load_vision_encoder", "TRM", "load_trm",
    # Utils
    "ScreenCapture", "capture",
    "click", "double_click", "right_click", "type_text", "scroll", "SafeController",
    "StateHasher", "get_indicator",
    # Knowledge
    "ExperienceDB", "Experience",
    # Exploration
    "AsyncExplorer", "RealMouseExplorer", "ExplorationConfig",
    # Training
    "AsyncTrainer",
]
