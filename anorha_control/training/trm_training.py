"""
TRM Training Pipeline - Train the Tiny Recursive Model for mouse control.

This module provides:
1. Data format specification for trajectories
2. Local GPU training
3. Modal cloud training
4. Hyperparameter configuration
5. Evaluation metrics

Data Format:
    Each trajectory is a sequence of mouse states:
    {
        "task_id": "login_001",
        "target": {"x": 500, "y": 300},
        "trajectory": [
            {"t": 0, "x": 100, "y": 100, "vx": 0, "vy": 0, "click": false},
            {"t": 50, "x": 200, "y": 180, "vx": 2000, "vy": 1600, "click": false},
            {"t": 100, "x": 480, "y": 290, "vx": -400, "vy": 200, "click": false},
            {"t": 150, "x": 500, "y": 300, "vx": 0, "vy": 0, "click": true}
        ],
        "success": true,
        "screen_size": [1920, 1080]
    }
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


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Model
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    warmup_steps: int = 500
    
    # Data
    sequence_length: int = 10  # History of positions
    augment: bool = True
    normalize: bool = True
    
    # Checkpointing
    save_every: int = 10
    eval_every: int = 5
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TrajectoryDataset(Dataset):
    """
    Dataset for mouse trajectory training.
    
    Each sample contains:
    - Input: (current_x, current_y, target_x, target_y, vx, vy, history...)
    - Output: (next_x, next_y, should_click)
    """
    
    def __init__(
        self,
        data_path: str,
        config: TrainingConfig,
        mode: str = "train"
    ):
        self.config = config
        self.mode = mode
        self.samples = []
        
        # Load trajectories
        self._load_data(data_path)
        print(f"[TrajectoryDataset] Loaded {len(self.samples)} samples for {mode}")
    
    def _load_data(self, data_path: str):
        """Load and preprocess trajectory data."""
        path = Path(data_path)
        
        if path.is_file():
            with open(path) as f:
                trajectories = json.load(f)
        elif path.is_dir():
            trajectories = []
            for file in path.glob("*.json"):
                with open(file) as f:
                    trajectories.extend(json.load(f))
        else:
            raise ValueError(f"Data path not found: {data_path}")
        
        # Convert trajectories to training samples
        for traj in trajectories:
            if not traj.get("success", False):
                continue  # Only train on successful trajectories
            
            target = traj["target"]
            screen_size = traj.get("screen_size", [1920, 1080])
            trajectory = traj["trajectory"]
            
            # Create samples from trajectory points
            for i in range(len(trajectory) - 1):
                current = trajectory[i]
                next_point = trajectory[i + 1]
                
                # Normalize coordinates
                if self.config.normalize:
                    cur_x = current["x"] / screen_size[0]
                    cur_y = current["y"] / screen_size[1]
                    tar_x = target["x"] / screen_size[0]
                    tar_y = target["y"] / screen_size[1]
                    next_x = next_point["x"] / screen_size[0]
                    next_y = next_point["y"] / screen_size[1]
                    vx = current.get("vx", 0) / screen_size[0]
                    vy = current.get("vy", 0) / screen_size[1]
                else:
                    cur_x, cur_y = current["x"], current["y"]
                    tar_x, tar_y = target["x"], target["y"]
                    next_x, next_y = next_point["x"], next_point["y"]
                    vx, vy = current.get("vx", 0), current.get("vy", 0)
                
                # Input: current pos, target pos, velocity
                input_vec = [cur_x, cur_y, tar_x, tar_y, vx, vy]
                
                # Output: next pos, click
                output_vec = [next_x, next_y, float(next_point.get("click", False))]
                
                self.samples.append({
                    "input": torch.tensor(input_vec, dtype=torch.float32),
                    "output": torch.tensor(output_vec, dtype=torch.float32)
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Data augmentation for training
        if self.mode == "train" and self.config.augment:
            sample = self._augment(sample)
        
        return sample
    
    def _augment(self, sample: Dict) -> Dict:
        """Apply data augmentation."""
        inp = sample["input"].clone()
        out = sample["output"].clone()
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            inp[0] = 1.0 - inp[0]  # cur_x
            inp[2] = 1.0 - inp[2]  # tar_x
            inp[4] = -inp[4]       # vx
            out[0] = 1.0 - out[0]  # next_x
        
        # Random vertical flip
        if np.random.random() > 0.5:
            inp[1] = 1.0 - inp[1]  # cur_y
            inp[3] = 1.0 - inp[3]  # tar_y
            inp[5] = -inp[5]       # vy
            out[1] = 1.0 - out[1]  # next_y
        
        # Small position noise
        noise = torch.randn(2) * 0.01
        inp[0:2] += noise
        out[0:2] += noise
        
        return {"input": inp, "output": out}


class TrajectoryTRM(nn.Module):
    """
    TRM specialized for trajectory prediction.
    
    Input: (current_x, current_y, target_x, target_y, vx, vy)
    Output: (next_x, next_y, click_probability)
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(6, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Coordinate prediction head
        self.coord_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 2),
            nn.Sigmoid()  # Normalized coords [0, 1]
        )
        
        # Click prediction head
        self.click_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 6] input vector
            
        Returns:
            coords: [B, 2] predicted coordinates
            click: [B, 1] click probability
        """
        features = self.encoder(x)
        coords = self.coord_head(features)
        click = self.click_head(features)
        return coords, click


class TRMTrainer:
    """
    Trainer class for TRM.
    
    Usage:
        trainer = TRMTrainer(config)
        trainer.train("data/trajectories.json", epochs=100)
        trainer.save("checkpoints/trm_v1.pt")
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model = TrajectoryTRM(self.config).to(self.config.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.epochs
        )
        
        # Loss functions
        self.coord_loss = nn.MSELoss()
        self.click_loss = nn.BCELoss()
        
        self.best_loss = float('inf')
        self.step = 0
        
        print(f"[TRMTrainer] Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"[TRMTrainer] Device: {self.config.device}")
    
    def train(
        self,
        train_path: str,
        val_path: str = None,
        epochs: int = None
    ) -> Dict[str, List[float]]:
        """
        Train the TRM model.
        
        Args:
            train_path: Path to training data
            val_path: Optional path to validation data
            epochs: Override config epochs
            
        Returns:
            Training history dict
        """
        epochs = epochs or self.config.epochs
        
        # Create datasets
        train_dataset = TrajectoryDataset(train_path, self.config, mode="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_path:
            val_dataset = TrajectoryDataset(val_path, self.config, mode="val")
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        history = {"train_loss": [], "val_loss": [], "coord_loss": [], "click_loss": []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            epoch_coord_loss = 0.0
            epoch_click_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                loss, coord_l, click_l = self._train_step(batch)
                epoch_loss += loss
                epoch_coord_loss += coord_l
                epoch_click_loss += click_l
                pbar.set_postfix(loss=f"{loss:.4f}")
            
            avg_loss = epoch_loss / len(train_loader)
            history["train_loss"].append(avg_loss)
            history["coord_loss"].append(epoch_coord_loss / len(train_loader))
            history["click_loss"].append(epoch_click_loss / len(train_loader))
            
            # Validation
            if val_loader and (epoch + 1) % self.config.eval_every == 0:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save("checkpoints/trm_best.pt")
            
            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save(f"checkpoints/trm_epoch_{epoch+1}.pt")
            
            self.scheduler.step()
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, lr={self.scheduler.get_last_lr()[0]:.6f}")
        
        return history
    
    def _train_step(self, batch: Dict) -> Tuple[float, float, float]:
        """Single training step."""
        self.optimizer.zero_grad()
        
        inputs = batch["input"].to(self.config.device)
        targets = batch["output"].to(self.config.device)
        
        # Forward
        pred_coords, pred_click = self.model(inputs)
        
        # Loss
        coord_loss = self.coord_loss(pred_coords, targets[:, :2])
        click_loss = self.click_loss(pred_click.squeeze(), targets[:, 2])
        
        # Combined loss (coords more important than click)
        total_loss = 0.7 * coord_loss + 0.3 * click_loss
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.step += 1
        return total_loss.item(), coord_loss.item(), click_loss.item()
    
    def _validate(self, loader: DataLoader) -> float:
        """Validation pass."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                inputs = batch["input"].to(self.config.device)
                targets = batch["output"].to(self.config.device)
                
                pred_coords, pred_click = self.model(inputs)
                coord_loss = self.coord_loss(pred_coords, targets[:, :2])
                click_loss = self.click_loss(pred_click.squeeze(), targets[:, 2])
                total_loss += (0.7 * coord_loss + 0.3 * click_loss).item()
        
        return total_loss / len(loader)
    
    def save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "step": self.step,
            "best_loss": self.best_loss
        }, path)
        print(f"[TRMTrainer] Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        print(f"[TRMTrainer] Loaded checkpoint from {path}")
    
    def evaluate(self, test_path: str) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dict with metrics: avg_distance, click_accuracy, success_rate
        """
        test_dataset = TrajectoryDataset(test_path, self.config, mode="test")
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        self.model.eval()
        distances = []
        click_correct = 0
        click_total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                inputs = batch["input"].to(self.config.device)
                targets = batch["output"].to(self.config.device)
                
                pred_coords, pred_click = self.model(inputs)
                
                # Distance in normalized space
                dist = torch.sqrt(((pred_coords - targets[:, :2]) ** 2).sum(dim=1))
                distances.extend(dist.cpu().numpy())
                
                # Click accuracy
                pred_click_binary = (pred_click.squeeze() > 0.5).float()
                click_correct += (pred_click_binary == targets[:, 2]).sum().item()
                click_total += len(targets)
        
        avg_distance = np.mean(distances)
        click_accuracy = click_correct / click_total if click_total > 0 else 0
        
        # Success rate: distance < 0.05 (5% of screen)
        success_rate = np.mean([d < 0.05 for d in distances])
        
        return {
            "avg_distance": avg_distance,
            "click_accuracy": click_accuracy,
            "success_rate": success_rate
        }


# =============================================================================
# MODAL TRAINING FUNCTION
# =============================================================================

def train_on_modal(
    data_path: str,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-4
):
    """
    Train TRM on Modal cloud GPU.
    
    Usage:
        modal run trm_training.py::train_on_modal --data-path data/trajectories.json
    """
    try:
        import modal
        
        app = modal.App("trm-training")
        
        # Define image with dependencies
        image = modal.Image.debian_slim(python_version="3.11").pip_install(
            "torch",
            "numpy",
            "tqdm"
        )
        
        @app.function(
            gpu="A10G",  # Budget-friendly GPU
            timeout=3600,
            image=image
        )
        def _train():
            config = TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            trainer = TRMTrainer(config)
            history = trainer.train(data_path)
            trainer.save("trm_final.pt")
            return history
        
        with app.run():
            return _train.remote()
            
    except ImportError:
        print("Modal not installed. Run: pip install modal")
        print("Falling back to local training...")
        config = TrainingConfig(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        trainer = TRMTrainer(config)
        return trainer.train(data_path)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TRM model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--modal", action="store_true", help="Use Modal cloud GPU")
    parser.add_argument("--eval", type=str, help="Evaluate on test set instead of training")
    
    args = parser.parse_args()
    
    if args.modal:
        history = train_on_modal(args.data, args.epochs, args.batch_size, args.lr)
    elif args.eval:
        config = TrainingConfig()
        trainer = TRMTrainer(config)
        trainer.load(args.data)
        metrics = trainer.evaluate(args.eval)
        print(f"\nEvaluation Results:")
        print(f"  Avg Distance: {metrics['avg_distance']:.4f}")
        print(f"  Click Accuracy: {metrics['click_accuracy']:.2%}")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
    else:
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        trainer = TRMTrainer(config)
        history = trainer.train(args.data)
        trainer.save("checkpoints/trm_final.pt")
