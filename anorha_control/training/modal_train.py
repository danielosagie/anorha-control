"""
Modal training script for cloud GPU training.
Exports training job to Modal for A100/H100 GPU usage.
"""
import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("anorha-control-training")

# Define GPU image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "torchvision", 
    "timm",
    "pillow",
    "numpy",
    "aiosqlite",
    "tqdm",
)


@app.function(
    image=image,
    gpu="A100",  # Use A100 for fast training
    timeout=3600,  # 1 hour max
    secrets=[modal.Secret.from_name("wandb-secret")] if os.environ.get("WANDB_API_KEY") else [],
)
def train_trm(
    data_path: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    checkpoint_path: str = None,
):
    """
    Train TRM on Modal with A100 GPU.
    
    Args:
        data_path: Path to training data (JSON or tar.gz)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_path: Optional checkpoint to resume from
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import json
    from tqdm import tqdm
    
    print(f"Training on {torch.cuda.get_device_name(0)}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    
    # Load data
    with open(data_path) as f:
        experiences = json.load(f)
    
    print(f"Loaded {len(experiences)} experiences")
    
    # Create dataset
    class ExperienceDataset(Dataset):
        def __init__(self, experiences):
            self.experiences = experiences
        
        def __len__(self):
            return len(self.experiences)
        
        def __getitem__(self, idx):
            exp = self.experiences[idx]
            return {
                "action_x": torch.tensor(exp["action_x"], dtype=torch.float32),
                "action_y": torch.tensor(exp["action_y"], dtype=torch.float32),
                "action_type": torch.tensor(exp["action_type"], dtype=torch.long),
                "reward": torch.tensor(exp["reward"], dtype=torch.float32),
            }
    
    dataset = ExperienceDataset(experiences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Simple model for demonstration (replace with actual TRM)
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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Create dummy input (normally would be vision embeddings)
            x = torch.randn(len(batch["action_x"]), 256).cuda()
            
            # Forward
            output = model(x)
            
            # Losses
            target_coords = torch.stack([batch["action_x"], batch["action_y"]], dim=1).cuda()
            coord_loss = nn.MSELoss()(output["coords"], target_coords)
            action_loss = nn.CrossEntropyLoss()(output["action_type"], batch["action_type"].cuda())
            
            # Weight by reward
            reward = batch["reward"].cuda().unsqueeze(1)
            loss = (coord_loss + action_loss) * reward.mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "/tmp/best_model.pt")
    
    print(f"Training complete! Best loss: {best_loss:.4f}")
    return {"best_loss": best_loss, "epochs": epochs}


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
    data_path: str,
    epochs: int = 100,
    maml: bool = False,
):
    """
    Local entrypoint for Modal training.
    
    Usage:
        modal run modal_train.py --data-path data/export.json --epochs 100
        modal run modal_train.py --data-path data/export.json --maml  # For MAML
    """
    if maml:
        result = train_with_maml.remote(data_path=data_path, epochs=epochs)
    else:
        result = train_trm.remote(data_path=data_path, epochs=epochs)
    
    print(f"Training complete: {result}")
