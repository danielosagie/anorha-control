"""
Tiny Recursive Model (TRM)
Predicts click coordinates from vision embeddings + instruction embeddings
Uses recursive refinement with cross-attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class RecursiveRefinementLayer(nn.Module):
    """
    Single layer of recursive refinement.
    Uses self-attention on the coordinate query and cross-attention to vision features.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention on query
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention to vision features
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, query: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, 1, hidden_dim] - The coordinate query
            vision_features: [B, N, hidden_dim] - Vision + instruction features
        
        Returns:
            Refined query: [B, 1, hidden_dim]
        """
        # Self-attention
        q = self.self_attn_norm(query)
        query = query + self.self_attn(q, q, q)[0]
        
        # Cross-attention to vision
        q = self.cross_attn_norm(query)
        query = query + self.cross_attn(q, vision_features, vision_features)[0]
        
        # FFN
        query = query + self.ffn(self.ffn_norm(query))
        
        return query


class TRM(nn.Module):
    """
    Tiny Recursive Model for GUI control.
    
    Takes vision embeddings and instruction embedding,
    outputs normalized (x, y) coordinates and action type.
    
    Architecture:
    - Vision projection
    - Instruction embedding (learnable for now, can be replaced with text encoder)
    - Recursive refinement layers
    - Coordinate prediction head
    - Action type prediction head
    """
    
    def __init__(
        self,
        vision_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        num_action_types: int = 5,  # click, right_click, double_click, type, scroll
        dropout: float = 0.1,
        num_instructions: int = 100,  # Learnable instruction embeddings for now
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Vision projection (if dims don't match)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim) if vision_dim != hidden_dim else nn.Identity()
        
        # Learnable instruction embeddings (will be replaced with text encoder later)
        self.instruction_embeddings = nn.Embedding(num_instructions, hidden_dim)
        
        # Learnable coordinate query (refined through layers)
        self.coord_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Recursive refinement layers
        self.layers = nn.ModuleList([
            RecursiveRefinementLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Coordinate prediction head (outputs normalized x, y)
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )
        
        # Action type prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_action_types),
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        vision_embedding: torch.Tensor,
        instruction_id: torch.Tensor = None,
        instruction_embedding: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            vision_embedding: [B, vision_dim] from vision encoder
            instruction_id: [B] integer IDs for learnable embeddings
            instruction_embedding: [B, hidden_dim] optional external instruction embedding
        
        Returns:
            dict with:
                - coords: [B, 2] normalized (x, y) in [0, 1]
                - action_type: [B, num_action_types] logits
                - confidence: [B, 1] confidence score
        """
        B = vision_embedding.size(0)
        
        # Project vision to hidden dim
        vision_feat = self.vision_proj(vision_embedding)  # [B, hidden_dim]
        vision_feat = vision_feat.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Get instruction embedding
        if instruction_embedding is not None:
            instr_feat = instruction_embedding.unsqueeze(1)  # [B, 1, hidden_dim]
        elif instruction_id is not None:
            instr_feat = self.instruction_embeddings(instruction_id).unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            # Default: use first instruction embedding
            instr_feat = self.instruction_embeddings.weight[0:1].unsqueeze(0).expand(B, -1, -1)
        
        # Combine vision + instruction as context
        context = torch.cat([vision_feat, instr_feat], dim=1)  # [B, 2, hidden_dim]
        
        # Initialize query
        query = self.coord_query.expand(B, -1, -1)  # [B, 1, hidden_dim]
        
        # Recursive refinement
        for layer in self.layers:
            query = layer(query, context)
        
        # Predictions
        query_flat = query.squeeze(1)  # [B, hidden_dim]
        
        coords = self.coord_head(query_flat)  # [B, 2]
        action_type = self.action_head(query_flat)  # [B, num_action_types]
        confidence = self.confidence_head(query_flat)  # [B, 1]
        
        return {
            "coords": coords,
            "action_type": action_type,
            "confidence": confidence,
        }
    
    def predict(
        self,
        vision_embedding: torch.Tensor,
        instruction_id: torch.Tensor = None,
        screen_size: tuple = (1920, 1080),
    ) -> dict:
        """
        Make a prediction for actual use.
        
        Returns:
            dict with:
                - x, y: Pixel coordinates
                - action: Action type string
                - confidence: Confidence score
        """
        self.eval()
        with torch.no_grad():
            output = self(vision_embedding, instruction_id)
        
        # Convert normalized coords to pixels
        coords = output["coords"][0].cpu()
        x = int(coords[0].item() * screen_size[0])
        y = int(coords[1].item() * screen_size[1])
        
        # Get action type
        action_idx = output["action_type"][0].argmax().item()
        actions = ["click", "right_click", "double_click", "type", "scroll"]
        action = actions[action_idx]
        
        confidence = output["confidence"][0, 0].item()
        
        return {
            "x": x,
            "y": y,
            "action": action,
            "confidence": confidence,
        }


def load_trm(
    checkpoint_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> TRM:
    """Load TRM model, optionally from checkpoint."""
    model = TRM(**kwargs)
    
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    
    model = model.to(device)
    return model


# Quick test
if __name__ == "__main__":
    print("Testing TRM...")
    model = TRM()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    vision_emb = torch.randn(2, 256)  # Batch of 2
    output = model(vision_emb)
    
    print(f"Coords shape: {output['coords'].shape}")  # [2, 2]
    print(f"Action type shape: {output['action_type'].shape}")  # [2, 5]
    print(f"Confidence shape: {output['confidence'].shape}")  # [2, 1]
    
    # Test prediction
    pred = model.predict(vision_emb[:1])
    print(f"\nPrediction: x={pred['x']}, y={pred['y']}, action={pred['action']}, conf={pred['confidence']:.3f}")
