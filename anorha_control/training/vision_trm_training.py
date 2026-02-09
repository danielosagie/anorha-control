"""
Vision TRM Training - Lightweight grounding for crap-top laptops.

Trains a small model: (screenshot, target_label) -> (x, y) normalized.
Fully local, no API calls. Use when UGround is too heavy.

Data: trajectories with screenshot_path and target_label (from --save-screenshots).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance


# Task category embedding: 8 TaskCategory values + unknown for backward compat
CATEGORY_TO_IDX = {
    "precision": 0, "typing": 1, "forms": 2, "navigation": 3,
    "ecommerce": 4, "longhorizon": 5, "realworld": 6, "desktop": 7,
    "unknown": 8,
}


@dataclass
class VisionTRMConfig:
    """Config for Vision TRM (grounding model)."""
    # Image
    img_size: int = 224  # Resize to NxN for training
    # Text
    max_label_len: int = 32
    char_vocab_size: int = 128  # ASCII + unk
    embed_dim: int = 64
    # Model
    vision_channels: List[int] = None  # [32, 64, 128, 256]
    hidden_dim: int = 256
    dropout: float = 0.2
    task_category_vocab_size: int = 9  # 8 TaskCategory + unknown
    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.vision_channels is None:
            self.vision_channels = [32, 64, 128, 256]


class VisionGroundingDataset(Dataset):
    """
    Dataset for (screenshot, target_label) -> (x, y) grounding.
    Loads trajectories that have screenshot_path and target_label.
    """
    def __init__(
        self,
        data_path: str,
        data_dir: Path = None,
        img_size: int = 224,
        max_label_len: int = 32,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir or "data/trajectories")
        self.img_size = img_size
        self.max_label_len = max_label_len
        self.augment = augment
        self.samples = []
        self._char_to_idx = {"<pad>": 0, "<unk>": 1}
        self._build_vocab_and_load(data_path)

    def _build_vocab_and_load(self, data_path: str):
        path = Path(data_path)
        trajectories = []
        if path.is_file():
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                trajectories = data
            elif isinstance(data, dict) and "target" in data and "trajectory" in data:
                trajectories = [data]
            elif isinstance(data, dict) and "trajectories" in data:
                trajectories = data["trajectories"]
        elif path.is_dir():
            for f in path.glob("*.json"):
                with open(f) as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    trajectories.extend(data)
                elif isinstance(data, dict) and "target" in data and "trajectory" in data:
                    trajectories.append(data)
                elif isinstance(data, dict) and "trajectories" in data:
                    trajectories.extend(data["trajectories"])
        else:
            raise ValueError(f"Data path not found: {data_path}")

        # Build char vocab from target_labels
        for traj in trajectories:
            if not isinstance(traj, dict):
                continue
            label = (traj.get("target_label") or "").strip()
            if not label:
                continue
            for c in label:
                if c not in self._char_to_idx:
                    self._char_to_idx[c] = len(self._char_to_idx)
        # Cap vocab
        while len(self._char_to_idx) > 120:
            break

        for traj in trajectories:
            if not isinstance(traj, dict) or not traj.get("success", False):
                continue
            sp = (traj.get("screenshot_path") or "").strip()
            label = (traj.get("target_label") or "").strip()
            if not sp or not label:
                continue
            target = traj.get("target", {})
            if not target:
                continue
            # Resolve screenshot path: data_dir is parent of trajectories, screenshots are in data_dir/screenshots
            # trajectory JSON is in data_dir, screenshot_path is relative to data_dir
            base = self.data_dir if (self.data_dir / sp).exists() else self.data_dir.parent
            img_path = base / sp
            if not img_path.exists():
                continue
            normalized = traj.get("normalized", True)
            tx = target.get("x", 0.5)
            ty = target.get("y", 0.5)
            if not normalized and traj.get("screen_size"):
                w, h = traj["screen_size"][0], traj["screen_size"][1]
                if w and h:
                    tx, ty = tx / w, ty / h
            cat = (traj.get("task_category") or "").strip().lower()
            cat_idx = CATEGORY_TO_IDX.get(cat, CATEGORY_TO_IDX["unknown"])
            self.samples.append({
                "img_path": str(img_path),
                "target_label": label,
                "target_x": float(tx),
                "target_y": float(ty),
                "task_category_id": cat_idx,
            })
        print(f"[VisionGroundingDataset] Loaded {len(self.samples)} samples, vocab size {len(self._char_to_idx)}")

    def get_char_to_idx(self) -> Dict[str, int]:
        """Return char vocab for saving/loading at inference."""
        return dict(self._char_to_idx)

    def _encode_label(self, label: str) -> torch.Tensor:
        ids = []
        for c in label[:self.max_label_len]:
            ids.append(self._char_to_idx.get(c, self._char_to_idx["<unk>"]))
        while len(ids) < self.max_label_len:
            ids.append(self._char_to_idx["<pad>"])
        return torch.tensor(ids, dtype=torch.long)

    def get_char_vocab_size(self) -> int:
        return max(len(self._char_to_idx), 2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        tx, ty = float(s["target_x"]), float(s["target_y"])

        if self.augment and random.random() < 0.5:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                tx = 1.0 - tx
            if random.random() < 0.5:
                enh = ImageEnhance.Brightness(img)
                img = enh.enhance(random.uniform(0.9, 1.1))
            if random.random() < 0.5:
                enh = ImageEnhance.Contrast(img)
                img = enh.enhance(random.uniform(0.9, 1.1))

        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC -> CHW
        label_t = self._encode_label(s["target_label"])
        cat_idx = s.get("task_category_id", CATEGORY_TO_IDX["unknown"])
        return {
            "image": img_t,
            "label_ids": label_t,
            "target_x": torch.tensor(tx, dtype=torch.float32),
            "target_y": torch.tensor(ty, dtype=torch.float32),
            "task_category_id": torch.tensor(cat_idx, dtype=torch.long),
        }


class VisionTRM(nn.Module):
    """
    Lightweight vision + text -> (x, y) grounding.
    Small CNN + char embedding + FC. Crap-top friendly.
    """
    def __init__(self, config: VisionTRMConfig, char_vocab_size: int):
        super().__init__()
        self.config = config
        # Vision: simple CNN
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
        self.vision = nn.Sequential(*layers)
        # Spatial size after 4 stride-2: 224/16 = 14
        self.vision_out_dim = ch[-1] * 14 * 14
        # Text: char embedding
        self.char_embed = nn.Embedding(
            max(char_vocab_size, config.char_vocab_size),
            config.embed_dim,
            padding_idx=0,
        )
        self.label_encoder = nn.Sequential(
            nn.Linear(config.embed_dim * config.max_label_len, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.task_embed = nn.Embedding(config.task_category_vocab_size, config.embed_dim)
        fusion_dim = self.vision_out_dim + config.hidden_dim + config.embed_dim
        # Combine
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 2),
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

    def forward(
        self,
        image: torch.Tensor,
        label_ids: torch.Tensor,
        task_category_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W]
            label_ids: [B, L]
            task_category_id: [B] optional, defaults to unknown (8)
        Returns:
            coords: [B, 2] in [0, 1]
        """
        B = image.size(0)
        v = self.vision(image)
        v = v.view(B, -1)
        emb = self.char_embed(label_ids)  # [B, L, D]
        emb = emb.view(B, -1)
        lab = self.label_encoder(emb)
        if task_category_id is not None:
            t_emb = self.task_embed(task_category_id).squeeze(1)
        else:
            t_emb = torch.zeros(B, self.config.embed_dim, device=image.device, dtype=image.dtype)
        fused = torch.cat([v, lab, t_emb], dim=1)
        coords = self.fusion(fused)
        return coords


class VisionTRMTrainer:
    """Train Vision TRM for grounding."""

    def __init__(self, config: VisionTRMConfig, char_vocab_size: int):
        self.config = config
        self.model = VisionTRM(config, char_vocab_size).to(config.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        self.best_loss = float("inf")
        print(f"[VisionTRMTrainer] Params: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(
        self,
        train_path: str,
        data_dir: Path = None,
        val_split: float = 0.1,
        epochs: int = None,
    ) -> Tuple[Dict[str, List[float]], VisionGroundingDataset]:
        path = Path(train_path)
        dd = data_dir or (path.parent if path.is_file() else path)
        dataset = VisionGroundingDataset(
            train_path,
            data_dir=dd,
            img_size=self.config.img_size,
            max_label_len=self.config.max_label_len,
        )
        if len(dataset) == 0:
            raise ValueError("No samples with screenshot_path and target_label. Run with --save-screenshots.")
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)
        epochs = epochs or self.config.epochs
        history = {"train_loss": [], "val_loss": []}

        for ep in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}"):
                loss = self._step(batch)
                train_loss += loss
            avg_train = train_loss / len(train_loader)
            history["train_loss"].append(avg_train)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    img = batch["image"].to(self.config.device)
                    lab = batch["label_ids"].to(self.config.device)
                    tx = batch["target_x"].to(self.config.device)
                    ty = batch["target_y"].to(self.config.device)
                    cat_id = batch.get("task_category_id")
                    if cat_id is not None:
                        cat_id = cat_id.to(self.config.device)
                    pred = self.model(img, lab, task_category_id=cat_id)
                    tgt = torch.stack([tx, ty], dim=1)
                    val_loss += nn.functional.mse_loss(pred, tgt).item()
            avg_val = val_loss / len(val_loader)
            history["val_loss"].append(avg_val)

            if avg_val < self.best_loss:
                self.best_loss = avg_val
                self.save("checkpoints/vision_trm_best.pt", dataset.get_char_to_idx())
            self.scheduler.step()
            print(f"Epoch {ep+1}: train={avg_train:.4f} val={avg_val:.4f}")
        return history, dataset

    def _step(self, batch: Dict) -> float:
        self.optimizer.zero_grad()
        img = batch["image"].to(self.config.device)
        lab = batch["label_ids"].to(self.config.device)
        tx = batch["target_x"].to(self.config.device)
        ty = batch["target_y"].to(self.config.device)
        cat_id = batch.get("task_category_id")
        if cat_id is not None:
            cat_id = cat_id.to(self.config.device)
        pred = self.model(img, lab, task_category_id=cat_id)
        tgt = torch.stack([tx, ty], dim=1)
        loss = nn.functional.mse_loss(pred, tgt)
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
        print(f"[VisionTRMTrainer] Saved to {path}")


def load_vision_trm(
    checkpoint_path: str,
    device: str = None,
) -> Tuple[VisionTRM, VisionTRMConfig, Dict[str, int]]:
    """Load Vision TRM for inference. Returns (model, config, char_to_idx)."""
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
    cfg = ckpt.get("config")
    if cfg is None or not isinstance(cfg, VisionTRMConfig):
        cfg = VisionTRMConfig()
    char_to_idx = ckpt.get("char_to_idx") or {"<pad>": 0, "<unk>": 1}
    char_vocab = max(len(char_to_idx), cfg.char_vocab_size)
    model = VisionTRM(cfg, char_vocab)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    return model, cfg, char_to_idx


class VisionTRMBackend:
    """
    Backend for Vision TRM grounding. Same interface as UGroundBackend.
    Fully local, crap-top friendly. Use when UGround is too heavy.
    """
    def __init__(self, checkpoint_path: str = "checkpoints/vision_trm_best.pt", device: str = None):
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._config = None
        self._char_to_idx = None
        self._device = device

    def _ensure_loaded(self):
        if self._model is not None:
            return
        self._model, self._config, self._char_to_idx = load_vision_trm(
            self.checkpoint_path, device=self._device
        )
        print(f"[VisionTRM] Loaded from {self.checkpoint_path}")

    def locate(
        self,
        description: str,
        screenshot: Image.Image,
        task_category: str = "",
    ):
        """Same interface as UGroundBackend.locate -> GroundingResult."""
        from ..models.vlm_subsystems import GroundingResult
        self._ensure_loaded()
        found, x, y, conf = vision_trm_locate(
            self._model, self._config, self._char_to_idx,
            description, screenshot, task_category=task_category, device=self._device
        )
        return GroundingResult(found=found, x=x, y=y, confidence=conf, element_type="vision_trm")


def vision_trm_locate(
    model: VisionTRM,
    config: VisionTRMConfig,
    char_to_idx: Dict[str, int],
    target: str,
    screenshot: Image.Image,
    task_category: str = "",
    device: str = None,
) -> Tuple[bool, int, int, float]:
    """
    Run Vision TRM grounding. Same interface as UGround.
    Returns (found, x, y, confidence).
    """
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
        pred = model(img_t, lab_t, task_category_id=cat_t)
    x_norm, y_norm = pred[0, 0].item(), pred[0, 1].item()
    w, h = screenshot.size
    x = int(x_norm * w)
    y = int(y_norm * h)
    return True, x, y, 0.8


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Vision TRM for grounding")
    parser.add_argument("--data", type=str, default="data/trajectories", help="Path to trajectories")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    config = VisionTRMConfig(
        img_size=args.img_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
    )
    dataset = VisionGroundingDataset(
        args.data,
        data_dir=Path(args.data).parent,
        img_size=config.img_size,
    )
    if len(dataset) == 0:
        print("No data. Run gatherer with --save-screenshots first.")
        exit(1)
    p = Path(args.data)
    data_dir = p if p.is_dir() else p.parent
    trainer = VisionTRMTrainer(config, dataset.get_char_vocab_size())
    history, dataset = trainer.train(args.data, data_dir=data_dir, epochs=args.epochs)
    trainer.save("checkpoints/vision_trm_final.pt", dataset.get_char_to_idx())
