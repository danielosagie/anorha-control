"""
Anorha-Control Configuration
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

@dataclass
class Config:
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "checkpoints")
    db_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "experiences.db")
    
    # Vision Encoder
    vision_model: str = "mobilevitv2_050"  # ~2M params, outputs 256d
    vision_input_size: int = 256
    vision_output_dim: int = 256
    freeze_vision: bool = True
    
    # TRM Model
    trm_hidden_dim: int = 256
    trm_num_layers: int = 2
    trm_num_heads: int = 4
    trm_dropout: float = 0.1
    
    # Actions
    action_types: list = field(default_factory=lambda: ["click", "right_click", "double_click", "type", "scroll"])
    
    # Exploration
    exploration_epsilon: float = 0.3
    max_episode_steps: int = 10
    experience_buffer_size: int = 10000
    training_queue_size: int = 1000
    
    # Rewards
    novelty_bonus: float = 0.3
    transition_threshold: float = 0.1  # MSE threshold for "screen changed"
    
    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    checkpoint_interval: int = 100  # batches
    
    # Screen
    screen_region: tuple = None  # None = full screen, or (x, y, w, h)
    
    # Browser
    browser_headless: bool = False
    browser_viewport: tuple = (1280, 720)
    
    # Safety
    safe_mode: bool = True  # Log actions instead of executing
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "screenshots").mkdir(exist_ok=True)


# Global config instance
config = Config()
