"""
Unified Anorha TRM - One model for grounding and trajectory smoothing.

Architecture:
- vision_encoder + text_encoder + grounding_head -> (target_x, target_y)
- trajectory_branch (MLP) -> (next_x, next_y, click)

Data from single gather run with --save-screenshots.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from PIL import Image
from itertools import cycle

from .vision_trm_training import VisionGroundingDataset, CATEGORY_TO_IDX
from .trm_training import TrajectoryDataset, TrainingConfig


@dataclass
class UnifiedConfig:
    """Combined config for AnorhaTRM."""
    # Vision (grounding)
    img_size: int = 224
    max_label_len: int = 32
    char_vocab_size: int = 128
    embed_dim: int = 64
    task_category_vocab_size: int = 9  # 8 TaskCategory + unknown
    vision_channels: List[int] = None
    # Shared
    hidden_dim: int = 256
    dropout: float = 0.15
    # Trajectory
    trajectory_hidden_dim: int = 256
    # Training
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    grounding_ratio: float = 0.5  # fraction of steps that are grounding
    # Checkpointing
    save_every: int = 10
    eval_every: int = 5
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.vision_channels is None:
            self.vision_channels = [32, 64, 128, 256]


class AnorhaTRM(nn.Module):
    """
    Unified model: grounding (screenshot, target) -> (x,y) + trajectory (cur, target, vel) -> (next, click).
    """
    def __init__(self, config: UnifiedConfig, char_vocab_size: int):
        super().__init__()
        self.config = config
        # --- Grounding branch ---
        ch = config.vision_channels
        layers = []
        in_c = 3
        for c in ch:
            layers.extend([
                nn.Conv2d(in_c, c, 3, stride=2, padding=1),
                nn.BatchNorm2d(c),
                nn.GELU(),
                nn.Dropout2d(config.dropout * 0.5),
            ])
            in_c = c
        self.vision_encoder = nn.Sequential(*layers)
        self.vision_out_dim = ch[-1] * 14 * 14
        self.char_embed = nn.Embedding(
            max(char_vocab_size, config.char_vocab_size),
            config.embed_dim,
            padding_idx=0,
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(config.embed_dim * config.max_label_len, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.task_embed = nn.Embedding(config.task_category_vocab_size, config.embed_dim)
        fusion_dim = self.vision_out_dim + config.hidden_dim + config.embed_dim
        self.grounding_head = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 2),
            nn.Sigmoid(),
        )
        # --- Trajectory branch ---
        traj_dim = config.trajectory_hidden_dim
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(6, traj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(traj_dim, traj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(traj_dim, traj_dim),
        )
        self.coord_head = nn.Sequential(
            nn.Linear(traj_dim, traj_dim // 2),
            nn.GELU(),
            nn.Linear(traj_dim // 2, 2),
            nn.Sigmoid(),
        )
        self.click_head = nn.Sequential(
            nn.Linear(traj_dim, traj_dim // 2),
            nn.GELU(),
            nn.Linear(traj_dim // 2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def ground(
        self,
        image: torch.Tensor,
        label_ids: torch.Tensor,
        task_category_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(screenshot, target_label[, task_category_id]) -> (x, y) normalized [0,1]."""
        B = image.size(0)
        v = self.vision_encoder(image)
        v = v.view(B, -1)
        emb = self.char_embed(label_ids)
        emb = emb.view(B, -1)
        lab = self.text_encoder(emb)
        if task_category_id is not None:
            t_emb = self.task_embed(task_category_id).squeeze(1)
            fused = torch.cat([v, lab, t_emb], dim=1)
        else:
            t_emb = torch.zeros(B, self.config.embed_dim, device=image.device, dtype=image.dtype)
            fused = torch.cat([v, lab, t_emb], dim=1)
        return self.grounding_head(fused)

    def trajectory_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(cur_x, cur_y, target_x, target_y, vx, vy) -> (next_x, next_y), click_prob."""
        features = self.trajectory_encoder(x)
        coords = self.coord_head(features)
        click = self.click_head(features)
        return coords, click


class UnifiedDataset:
    """
    Wraps VisionGroundingDataset and TrajectoryDataset.
    Both built from the same trajectory files.
    """
    def __init__(self, data_path: str, config: UnifiedConfig, data_dir: Path = None):
        path = Path(data_path)
        dd = data_dir or (path.parent if path.is_file() else path)
        self.grounding_dataset = VisionGroundingDataset(
            data_path,
            data_dir=dd,
            img_size=config.img_size,
            max_label_len=config.max_label_len,
            augment=True,
        )
        traj_config = TrainingConfig(
            hidden_dim=config.trajectory_hidden_dim,
            dropout=config.dropout,
            batch_size=config.batch_size,
            augment=True,
            normalize=True,
            include_failed=False,
        )
        self.trajectory_dataset = TrajectoryDataset(data_path, traj_config, mode="train")
        self.config = config

    def get_char_to_idx(self) -> Dict[str, int]:
        return self.grounding_dataset.get_char_to_idx()

    def get_char_vocab_size(self) -> int:
        return self.grounding_dataset.get_char_vocab_size()


class UnifiedTRMTrainer:
    """Train AnorhaTRM with alternating grounding and trajectory batches."""

    def __init__(self, config: UnifiedConfig, char_vocab_size: int):
        self.config = config
        self.model = AnorhaTRM(config, char_vocab_size).to(config.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        self.best_loss = float("inf")
        print(f"[UnifiedTRMTrainer] Params: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(
        self,
        train_path: str,
        data_dir: Path = None,
        val_split: float = 0.1,
        epochs: int = None,
    ) -> Tuple[Dict[str, List[float]], UnifiedDataset]:
        path = Path(train_path)
        dd = data_dir or (path.parent if path.is_file() else path)
        dataset = UnifiedDataset(train_path, self.config, data_dir=dd)

        if len(dataset.grounding_dataset) == 0:
            raise ValueError("No grounding samples. Run gatherer with --save-screenshots.")
        if len(dataset.trajectory_dataset) == 0:
            raise ValueError("No trajectory samples in data.")

        g_loader = DataLoader(
            dataset.grounding_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        t_loader = DataLoader(
            dataset.trajectory_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        epochs = epochs or self.config.epochs
        history = {"train_loss": [], "grounding_loss": [], "trajectory_loss": []}

        for ep in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_g = 0.0
            epoch_t = 0.0
            n_steps = 0

            g_iter = cycle(g_loader) if len(g_loader) > 0 else iter([])
            t_iter = cycle(t_loader) if len(t_loader) > 0 else iter([])
            max_steps = max(len(g_loader), len(t_loader)) * 2

            pbar = tqdm(range(max_steps), desc=f"Epoch {ep+1}/{epochs}")
            for step in pbar:
                if np.random.random() < self.config.grounding_ratio:
                    try:
                        batch = next(g_iter)
                    except StopIteration:
                        g_iter = cycle(g_loader)
                        batch = next(g_iter)
                    loss = self._grounding_step(batch)
                    epoch_g += loss
                else:
                    try:
                        batch = next(t_iter)
                    except StopIteration:
                        t_iter = cycle(t_loader)
                        batch = next(t_iter)
                    loss = self._trajectory_step(batch)
                    epoch_t += loss
                epoch_loss += loss
                n_steps += 1
                pbar.set_postfix(loss=f"{loss:.4f}")

            avg = epoch_loss / n_steps if n_steps else 0
            history["train_loss"].append(avg)
            history["grounding_loss"].append(epoch_g / n_steps if n_steps else 0)
            history["trajectory_loss"].append(epoch_t / n_steps if n_steps else 0)

            if avg < self.best_loss:
                self.best_loss = avg
                self.save("checkpoints/anorha_trm_best.pt", dataset.get_char_to_idx())
            self.scheduler.step()
            print(f"Epoch {ep+1}: loss={avg:.4f} g={epoch_g/n_steps:.4f} t={epoch_t/n_steps:.4f}")

        return history, dataset

    def _grounding_step(self, batch: Dict) -> float:
        self.optimizer.zero_grad()
        img = batch["image"].to(self.config.device)
        lab = batch["label_ids"].to(self.config.device)
        tx = batch["target_x"].to(self.config.device)
        ty = batch["target_y"].to(self.config.device)
        cat_id = batch.get("task_category_id")
        if cat_id is not None:
            cat_id = cat_id.to(self.config.device)
        pred = self.model.ground(img, lab, task_category_id=cat_id)
        tgt = torch.stack([tx, ty], dim=1)
        loss = nn.functional.mse_loss(pred, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def _trajectory_step(self, batch: Dict) -> float:
        self.optimizer.zero_grad()
        inp = batch["input"].to(self.config.device)
        tgt = batch["output"].to(self.config.device)
        pred_coords, pred_click = self.model.trajectory_step(inp)
        coord_loss = nn.functional.mse_loss(pred_coords, tgt[:, :2])
        click_loss = nn.functional.binary_cross_entropy(
            pred_click.squeeze(), tgt[:, 2], reduction="mean"
        )
        loss = 0.7 * coord_loss + 0.3 * click_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def save(self, path: str, char_to_idx: Dict[str, int] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_loss": self.best_loss,
            "char_to_idx": char_to_idx or {},
        }, path)
        print(f"[UnifiedTRMTrainer] Saved to {path}")


def load_anorha_trm(
    checkpoint_path: str,
    device: str = None,
) -> Tuple[AnorhaTRM, UnifiedConfig, Dict[str, int]]:
    """Load AnorhaTRM for inference. Returns (model, config, char_to_idx)."""
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
    cfg = ckpt.get("config")
    if cfg is None or not isinstance(cfg, UnifiedConfig):
        cfg = UnifiedConfig()
    char_to_idx = ckpt.get("char_to_idx") or {"<pad>": 0, "<unk>": 1}
    char_vocab = max(len(char_to_idx), cfg.char_vocab_size)
    model = AnorhaTRM(cfg, char_vocab)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    return model, cfg, char_to_idx


class AnorhaTRMBackend:
    """
    Backend for AnorhaTRM grounding. Same interface as UGroundBackend.
    Uses unified model's ground() for locate.
    """
    def __init__(self, checkpoint_path: str = "checkpoints/anorha_trm_best.pt", device: str = None):
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._config = None
        self._char_to_idx = None
        self._device = device

    def _ensure_loaded(self):
        if self._model is not None:
            return
        self._model, self._config, self._char_to_idx = load_anorha_trm(
            self.checkpoint_path, device=self._device
        )
        print(f"[AnorhaTRM] Loaded from {self.checkpoint_path}")

    def locate(
        self,
        description: str,
        screenshot,
        task_category: str = "",
    ) -> "GroundingResult":
        """Same interface as UGroundBackend.locate -> GroundingResult."""
        from ..models.vlm_subsystems import GroundingResult
        self._ensure_loaded()
        found, x, y, conf = anorha_trm_ground(
            self._model, self._config, self._char_to_idx,
            description, screenshot, task_category=task_category, device=self._device
        )
        return GroundingResult(found=found, x=x, y=y, confidence=conf, element_type="anorha_trm")


def anorha_trm_ground(
    model: AnorhaTRM,
    config: UnifiedConfig,
    char_to_idx: Dict[str, int],
    target: str,
    screenshot,
    task_category: str = "",
    device: str = None,
) -> Tuple[bool, int, int, float]:
    """Run grounding. Returns (found, x, y, confidence)."""
    from PIL import Image
    dev = device or next(model.parameters()).device
    img = screenshot.convert("RGB").resize((config.img_size, config.img_size), Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(dev)
    unk = char_to_idx.get("<unk>", 1)
    pad = char_to_idx.get("<pad>", 0)
    ids = []
    for c in target[: config.max_label_len]:
        ids.append(char_to_idx.get(c, unk))
    while len(ids) < config.max_label_len:
        ids.append(pad)
    lab_t = torch.tensor([ids], dtype=torch.long, device=dev)
    cat_key = (task_category or "").strip().lower() or "unknown"
    cat_idx = CATEGORY_TO_IDX.get(cat_key, CATEGORY_TO_IDX["unknown"])
    cat_t = torch.tensor([cat_idx], dtype=torch.long, device=dev)
    with torch.no_grad():
        pred = model.ground(img_t, lab_t, task_category_id=cat_t)
    x_norm, y_norm = pred[0, 0].item(), pred[0, 1].item()
    w, h = screenshot.size
    x = int(x_norm * w)
    y = int(y_norm * h)
    return True, x, y, 0.8


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train unified Anorha TRM")
    parser.add_argument("--data", type=str, default="data/trajectories")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    config = UnifiedConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
    )
    p = Path(args.data)
    data_dir = p if p.is_dir() else p.parent
    dataset = UnifiedDataset(args.data, config, data_dir=data_dir)
    if len(dataset.grounding_dataset) == 0 or len(dataset.trajectory_dataset) == 0:
        print("Need both grounding and trajectory data. Run gatherer with --save-screenshots.")
        exit(1)
    trainer = UnifiedTRMTrainer(config, dataset.get_char_vocab_size())
    history, dataset = trainer.train(args.data, data_dir=data_dir, epochs=args.epochs)
    trainer.save("checkpoints/anorha_trm_final.pt", dataset.get_char_to_idx())
