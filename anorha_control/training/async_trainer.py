"""
Async Trainer - Continuous training on collected experiences
Uses REINFORCE / policy gradient to learn from exploration
"""
import asyncio
import time
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ..config import config as global_config
from ..models.vision_encoder import VisionEncoder
from ..models.trm import TRM
from ..knowledge.database import ExperienceDB, Experience


class ExperienceDataset(Dataset):
    """Dataset from collected experiences."""
    
    def __init__(
        self,
        experiences: List[Experience],
        vision_encoder: VisionEncoder,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.experiences = experiences
        self.vision_encoder = vision_encoder
        self.device = device
    
    def __len__(self):
        return len(self.experiences)
    
    def __getitem__(self, idx):
        exp = self.experiences[idx]
        
        # Load and encode image
        try:
            img = Image.open(exp.screenshot_before_path).convert("RGB")
            vision_emb = self.vision_encoder.preprocess(img)
        except:
            # Fallback to random embedding if image not found
            vision_emb = torch.randn(3, 256, 256)
        
        return {
            "vision_emb": vision_emb,
            "action_x": torch.tensor(exp.action_x, dtype=torch.float32),
            "action_y": torch.tensor(exp.action_y, dtype=torch.float32),
            "action_type": torch.tensor(exp.action_type, dtype=torch.long),
            "reward": torch.tensor(exp.reward, dtype=torch.float32),
            "success": torch.tensor(float(exp.success), dtype=torch.float32),
        }


class AsyncTrainer:
    """
    Asynchronous trainer that continuously learns from collected experiences.
    
    Uses policy gradient / REINFORCE:
    - Maximize reward-weighted log probability of actions
    - Train on successful experiences more heavily
    """
    
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        trm: TRM,
        training_queue: queue.Queue,
        db: ExperienceDB,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        checkpoint_dir: Path = None,
        checkpoint_interval: int = 100,
        device: str = None,
    ):
        self.vision_encoder = vision_encoder
        self.trm = trm
        self.training_queue = training_queue
        self.db = db
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.checkpoint_interval = checkpoint_interval
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move TRM to device
        self.trm = self.trm.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.trm.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
        )
        
        # Buffer for batch accumulation
        self.batch_buffer: List[Experience] = []
        
        # Stats
        self.total_batches = 0
        self.total_updates = 0
        self.running_loss = 0.0
        self.best_loss = float("inf")
        
        # Running flag
        self._running = False
        
        # Ensure checkpoint dir exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Uses a combination of:
        1. Coordinate regression loss (MSE)
        2. Action type classification loss (CrossEntropy)
        3. Reward-weighted policy gradient loss
        """
        vision_embs = batch["vision_emb"].to(self.device)
        target_x = batch["action_x"].to(self.device)
        target_y = batch["action_y"].to(self.device)
        target_actions = batch["action_type"].to(self.device)
        rewards = batch["reward"].to(self.device)
        successes = batch["success"].to(self.device)
        
        # Encode images through vision encoder
        with torch.no_grad():
            vision_encoded = self.vision_encoder(vision_embs)
        
        # Forward pass through TRM
        outputs = self.trm(vision_encoded)
        
        pred_coords = outputs["coords"]  # [B, 2]
        pred_actions = outputs["action_type"]  # [B, num_actions]
        
        # Coordinate loss (MSE)
        target_coords = torch.stack([target_x, target_y], dim=1)
        coord_loss = F.mse_loss(pred_coords, target_coords, reduction="none")
        
        # Weight by success (successful actions matter more)
        weights = 1.0 + successes * 2.0  # 1.0 for failures, 3.0 for successes
        coord_loss = (coord_loss.mean(dim=1) * weights).mean()
        
        # Action type loss (CrossEntropy)
        action_loss = F.cross_entropy(pred_actions, target_actions, reduction="none")
        action_loss = (action_loss * weights).mean()
        
        # Policy gradient term (REINFORCE)
        # Maximize log prob of actions that got high rewards
        action_log_probs = F.log_softmax(pred_actions, dim=1)
        selected_log_probs = action_log_probs.gather(1, target_actions.unsqueeze(1)).squeeze(1)
        pg_loss = -(rewards * selected_log_probs).mean()
        
        # Combined loss
        total_loss = coord_loss + action_loss + 0.1 * pg_loss
        
        return {
            "total_loss": total_loss,
            "coord_loss": coord_loss,
            "action_loss": action_loss,
            "pg_loss": pg_loss,
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.trm.train()
        
        # Forward and compute loss
        losses = self._compute_loss(batch)
        total_loss = losses["total_loss"]
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.trm.parameters(), max_norm=1.0)
        
        # Update
        self.optimizer.step()
        self.scheduler.step()
        
        self.total_updates += 1
        
        return {k: v.item() for k, v in losses.items()}
    
    def _collate_experiences(self, experiences: List[Experience]) -> Dict[str, torch.Tensor]:
        """Collate experiences into a batch."""
        batch_data = []
        for exp in experiences:
            try:
                img = Image.open(exp.screenshot_before_path).convert("RGB")
                vision_tensor = self.vision_encoder.preprocess(img)
            except:
                vision_tensor = torch.randn(3, 256, 256)
            
            batch_data.append({
                "vision_emb": vision_tensor,
                "action_x": torch.tensor(exp.action_x, dtype=torch.float32),
                "action_y": torch.tensor(exp.action_y, dtype=torch.float32),
                "action_type": torch.tensor(exp.action_type, dtype=torch.long),
                "reward": torch.tensor(exp.reward, dtype=torch.float32),
                "success": torch.tensor(float(exp.success), dtype=torch.float32),
            })
        
        # Stack into batch
        return {
            key: torch.stack([d[key] for d in batch_data])
            for key in batch_data[0].keys()
        }
    
    def save_checkpoint(self, name: str = None):
        """Save model checkpoint."""
        if name is None:
            name = f"trm_checkpoint_{self.total_updates}.pt"
        
        path = self.checkpoint_dir / name
        torch.save({
            "model_state_dict": self.trm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "total_updates": self.total_updates,
            "running_loss": self.running_loss,
        }, path)
        
        print(f"[Trainer] Saved checkpoint: {path}")
        return path
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.trm.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.total_updates = checkpoint.get("total_updates", 0)
        self.running_loss = checkpoint.get("running_loss", 0.0)
        print(f"[Trainer] Loaded checkpoint: {path}")
    
    async def train_on_queue(self):
        """Train continuously on experiences from queue."""
        self._running = True
        print(f"[Trainer] Starting training loop...")
        
        while self._running:
            try:
                # Get experience from queue (with timeout)
                exp = self.training_queue.get(timeout=0.1)
                self.batch_buffer.append(exp)
                
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            
            # Train when batch is ready
            if len(self.batch_buffer) >= self.batch_size:
                batch = self._collate_experiences(self.batch_buffer)
                losses = self.train_step(batch)
                
                self.total_batches += 1
                self.running_loss = 0.9 * self.running_loss + 0.1 * losses["total_loss"]
                
                # Clear buffer
                self.batch_buffer = []
                
                # Log progress
                if self.total_batches % 10 == 0:
                    print(f"[Trainer] Batch {self.total_batches}: "
                          f"loss={self.running_loss:.4f}, "
                          f"lr={self.scheduler.get_last_lr()[0]:.2e}")
                
                # Checkpoint
                if self.total_batches % self.checkpoint_interval == 0:
                    self.save_checkpoint()
                    
                    # Save best
                    if self.running_loss < self.best_loss:
                        self.best_loss = self.running_loss
                        self.save_checkpoint("trm_best.pt")
    
    async def train_on_db(self, num_epochs: int = 10, only_successes: bool = True):
        """
        Train on experiences from database.
        Used for batch training after collection.
        """
        print(f"[Trainer] Training on database experiences...")
        
        # Load experiences
        if only_successes:
            experiences = await self.db.get_successful_experiences(limit=10000)
        else:
            experiences = await self.db.get_recent_experiences(limit=10000)
        
        if not experiences:
            print("[Trainer] No experiences found in database")
            return
        
        print(f"[Trainer] Loaded {len(experiences)} experiences")
        
        for epoch in range(num_epochs):
            # Shuffle
            random.shuffle(experiences)
            
            epoch_loss = 0.0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(experiences), self.batch_size):
                batch_exps = experiences[i:i + self.batch_size]
                if len(batch_exps) < self.batch_size:
                    continue
                
                batch = self._collate_experiences(batch_exps)
                losses = self.train_step(batch)
                
                epoch_loss += losses["total_loss"]
                num_batches += 1
            
            avg_loss = epoch_loss / max(1, num_batches)
            print(f"[Trainer] Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}")
            
            # Save checkpoint each epoch
            self.save_checkpoint(f"trm_epoch_{epoch + 1}.pt")
        
        self.save_checkpoint("trm_final.pt")
    
    def stop(self):
        """Stop training loop."""
        self._running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training stats."""
        return {
            "total_batches": self.total_batches,
            "total_updates": self.total_updates,
            "running_loss": self.running_loss,
            "best_loss": self.best_loss,
            "buffer_size": len(self.batch_buffer),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }


# Quick test
if __name__ == "__main__":
    async def test():
        print("Testing AsyncTrainer...")
        
        from ..models.vision_encoder import load_vision_encoder
        from ..models.trm import load_trm
        
        # Load models
        vision = load_vision_encoder()
        trm = load_trm()
        
        # Create database
        db = ExperienceDB(Path("/tmp/test_training.db"))
        await db.connect()
        
        # Create trainer
        training_queue = queue.Queue()
        trainer = AsyncTrainer(vision, trm, training_queue, db)
        
        print(f"Trainer initialized. Device: {trainer.device}")
        print(f"TRM parameters: {sum(p.numel() for p in trm.parameters()):,}")
        
        # Test with fake batch
        fake_batch = {
            "vision_emb": torch.randn(8, 3, 256, 256),
            "action_x": torch.rand(8),
            "action_y": torch.rand(8),
            "action_type": torch.randint(0, 5, (8,)),
            "reward": torch.rand(8),
            "success": torch.randint(0, 2, (8,)).float(),
        }
        
        losses = trainer.train_step(fake_batch)
        print(f"Loss: {losses}")
        
        await db.close()
    
    asyncio.run(test())
