"""
Modal training script for cloud GPU training.
Uses real TrajectoryTRM + TrajectoryDataset from trm_training.

Usage:
  modal run anorha_control.training.modal_train --data-path data/trajectories --epochs 50 --gpu T4
  # Or mount locally: modal run -m anorha_control.training.modal_train --data-path ./data/trajectories
"""
import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("anorha-control-training")

# Image with our package + deps (install from local or git)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "timm",
        "pillow",
        "numpy",
        "tqdm",
    )
    .copy_local_dir(".", remote_path="/app")
    .pip_install("-e", "/app")
)

@app.function(
    image=image,
    gpu="T4",  # T4 fits $30 budget (~$0.59/hr); use A100 for speed
    timeout=7200,  # 2 hours max
    secrets=[modal.Secret.from_name("wandb-secret")] if os.environ.get("WANDB_API_KEY") else [],
)
def train_trm(
    data_path: str = "/app/data/trajectories",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    checkpoint_path: str = None,
):
    """
    Train TrajectoryTRM (path smoothing) on Modal.
    Uses real TRMTrainer + TrajectoryDataset; expects trajectory JSON format.
    """
    import sys
    sys.path.insert(0, "/app")
    from anorha_control.training.trm_training import (
        TRMTrainer,
        TrainingConfig,
        load_trajectory_trm,
    )

    import torch
    from pathlib import Path

    print(f"Training on {torch.cuda.get_device_name(0)}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")

    cfg = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        device="cuda",
    )
    trainer = TRMTrainer(cfg)
    history = trainer.train(data_path, val_path=None, epochs=epochs)
    out_path = "/tmp/trm_trajectory.pt"
    trainer.save(out_path)
    print(f"Saved to {out_path}")
    return {"best_loss": trainer.best_loss, "epochs": epochs, "history": history}


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret")] if os.environ.get("WANDB_API_KEY") else [],
)
def train_anorha_trm(
    data_path: str = "/app/data/trajectories",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
):
    """
    Train AnorhaTRM (grounding + trajectory) on Modal.
    Requires trajectories with screenshot_path, target_label, and trajectory sequences.
    """
    import sys
    sys.path.insert(0, "/app")
    from pathlib import Path
    from anorha_control.training.unified_trm import (
        UnifiedTRMTrainer,
        UnifiedConfig,
        UnifiedDataset,
    )

    import torch

    print(f"Training on {torch.cuda.get_device_name(0)}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")

    p = Path(data_path)
    data_dir = p if p.is_dir() else p.parent

    config = UnifiedConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        device="cuda",
    )
    dataset = UnifiedDataset(data_path, config, data_dir=data_dir)
    if len(dataset.grounding_dataset) == 0 or len(dataset.trajectory_dataset) == 0:
        raise ValueError(
            "Need both grounding and trajectory data. Run gatherer with --save-screenshots."
        )
    trainer = UnifiedTRMTrainer(config, dataset.get_char_vocab_size())
    history, dataset = trainer.train(data_path, data_dir=data_dir, epochs=epochs)
    out_path = "/tmp/anorha_trm_best.pt"
    trainer.save(out_path, dataset.get_char_to_idx())
    print(f"Saved to {out_path}")
    return {"best_loss": trainer.best_loss, "epochs": epochs, "history": history}


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret")] if os.environ.get("WANDB_API_KEY") else [],
)
def train_vision_trm(
    data_path: str = "/app/data/trajectories",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
):
    """
    Train VisionTRM (grounding only) on Modal.
    Requires trajectories with screenshot_path and target_label.
    """
    import sys
    sys.path.insert(0, "/app")
    from pathlib import Path
    from anorha_control.training.vision_trm_training import (
        VisionTRMTrainer,
        VisionTRMConfig,
        VisionGroundingDataset,
    )

    import torch

    print(f"Training on {torch.cuda.get_device_name(0)}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")

    p = Path(data_path)
    data_dir = p if p.is_dir() else p.parent

    config = VisionTRMConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        device="cuda",
    )
    dataset = VisionGroundingDataset(
        data_path,
        data_dir=data_dir,
        img_size=config.img_size,
        max_label_len=config.max_label_len,
    )
    if len(dataset) == 0:
        raise ValueError(
            "No grounding samples. Run gatherer with --save-screenshots first."
        )
    trainer = VisionTRMTrainer(config, dataset.get_char_vocab_size())
    history, dataset = trainer.train(data_path, data_dir=data_dir, epochs=epochs)
    out_path = "/tmp/vision_trm_best.pt"
    trainer.save(out_path, dataset.get_char_to_idx())
    print(f"Saved to {out_path}")
    return {"best_loss": trainer.best_loss, "epochs": epochs, "history": history}


@app.function(image=image, gpu="A100", timeout=7200)
def train_with_maml(
    data_path: str,
    meta_lr: float = 1e-3,
    inner_lr: float = 1e-2,
    inner_steps: int = 5,
    tasks_per_batch: int = 4,
    epochs: int = 50,
):
    """
    MAML meta-learning training.
    
    Args:
        data_path: Path to training data
        meta_lr: Outer loop learning rate
        inner_lr: Inner loop learning rate (adaptation)
        inner_steps: Number of gradient steps for adaptation
        tasks_per_batch: Number of tasks per meta-batch
        epochs: Number of meta-epochs
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from copy import deepcopy
    import json
    from tqdm import tqdm
    import random
    
    print(f"MAML Training on {torch.cuda.get_device_name(0)}")
    
    # Load and cluster experiences into tasks
    with open(data_path) as f:
        experiences = json.load(f)
    
    # Simple clustering by action type (replace with better clustering)
    tasks = {}
    for exp in experiences:
        action_type = exp.get("action_type", 0)
        if action_type not in tasks:
            tasks[action_type] = []
        tasks[action_type].append(exp)
    
    task_list = list(tasks.values())
    print(f"Created {len(task_list)} tasks")
    
    # Model
    class SimpleTRM(nn.Module):
        def __init__(self, hidden=256):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(256, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.coord_head = nn.Linear(hidden, 2)
            self.action_head = nn.Linear(hidden, 5)
        
        def forward(self, x):
            h = self.encoder(x)
            coords = torch.sigmoid(self.coord_head(h))
            actions = self.action_head(h)
            return {"coords": coords, "action_type": actions}
    
    model = SimpleTRM().cuda()
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
    
    # MAML training loop
    for epoch in range(epochs):
        meta_loss = 0
        
        # Sample tasks
        sampled_tasks = random.sample(task_list, min(tasks_per_batch, len(task_list)))
        
        for task_data in sampled_tasks:
            if len(task_data) < 10:
                continue
            
            # Split into support (adapt) and query (evaluate)
            random.shuffle(task_data)
            support = task_data[:5]
            query = task_data[5:10]
            
            # Clone model for inner loop
            adapted_model = deepcopy(model)
            inner_optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)
            
            # Inner loop: adapt to task
            for _ in range(inner_steps):
                x = torch.randn(len(support), 256).cuda()
                target_coords = torch.tensor([[e["action_x"], e["action_y"]] for e in support]).cuda()
                target_actions = torch.tensor([e["action_type"] for e in support]).cuda()
                
                output = adapted_model(x)
                loss = nn.MSELoss()(output["coords"], target_coords) + \
                       nn.CrossEntropyLoss()(output["action_type"], target_actions)
                
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()
            
            # Outer loop: evaluate on query set
            x = torch.randn(len(query), 256).cuda()
            target_coords = torch.tensor([[e["action_x"], e["action_y"]] for e in query]).cuda()
            target_actions = torch.tensor([e["action_type"] for e in query]).cuda()
            
            output = adapted_model(x)
            query_loss = nn.MSELoss()(output["coords"], target_coords) + \
                        nn.CrossEntropyLoss()(output["action_type"], target_actions)
            
            meta_loss += query_loss
        
        # Meta update
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Meta-loss = {meta_loss.item():.4f}")
    
    # Save meta-learned model
    torch.save(model.state_dict(), "/tmp/maml_model.pt")
    print("MAML training complete!")
    
    return {"final_loss": meta_loss.item(), "epochs": epochs}


@app.local_entrypoint()
def main(
    data_path: str = "/app/data/trajectories",
    epochs: int = 50,
    model: str = "anorha_trm",
    maml: bool = False,
    gpu: str = "T4",
):
    """
    Local entrypoint for Modal training.

    Usage (run from project root with data/trajectories/):
        modal run anorha_control.training.modal_train --model anorha_trm --epochs 100
        modal run anorha_control.training.modal_train --model vision_trm --epochs 50
        modal run anorha_control.training.modal_train --model trm --epochs 50
    """
    # Resolve data path for container (copy_local_dir puts project at /app)
    if not data_path.startswith("/"):
        data_path = f"/app/{data_path}"

    if maml:
        result = train_with_maml.remote(data_path=data_path, epochs=epochs)
    elif model == "anorha_trm":
        result = train_anorha_trm.remote(
            data_path=data_path,
            epochs=epochs,
        )
    elif model == "vision_trm":
        result = train_vision_trm.remote(
            data_path=data_path,
            epochs=epochs,
        )
    elif model == "trm":
        result = train_trm.remote(data_path=data_path, epochs=epochs)
    else:
        raise ValueError(f"Unknown model: {model}. Use anorha_trm, vision_trm, or trm.")

    print(f"Training complete: {result}")
