"""
Text encoder for instruction conditioning.
Converts text instructions to embeddings for the TRM.
"""
import torch
import torch.nn as nn
from typing import List, Optional
import hashlib


class SimpleTextEncoder(nn.Module):
    """
    Simple text encoder using learned embeddings.
    Maps common UI-related words to embeddings.
    
    For production, replace with SentenceTransformers or similar.
    """
    
    VOCAB = [
        # Actions
        "click", "tap", "press", "hover", "scroll", "type", "find", "locate",
        # Elements
        "button", "link", "input", "field", "text", "menu", "dropdown", "checkbox",
        "radio", "tab", "icon", "image", "card", "modal", "popup", "form",
        "search", "navigation", "header", "footer", "sidebar", "content",
        # Descriptors
        "blue", "green", "red", "orange", "purple", "white", "black", "gray",
        "large", "small", "big", "tiny", "main", "primary", "secondary",
        "top", "bottom", "left", "right", "center", "first", "last",
        # Common words
        "the", "a", "an", "on", "in", "at", "to", "for", "with", "and", "or",
        # UI-specific
        "login", "signup", "register", "submit", "send", "cancel", "close",
        "open", "save", "delete", "edit", "add", "remove", "next", "back",
        "home", "settings", "profile", "account", "cart", "checkout",
        # Padding/Unknown
        "<pad>", "<unk>",
    ]
    
    def __init__(self, embed_dim: int = 256, max_tokens: int = 20):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens
        
        # Build vocab
        self.word2idx = {w: i for i, w in enumerate(self.VOCAB)}
        self.pad_idx = self.word2idx["<pad>"]
        self.unk_idx = self.word2idx["<unk>"]
        
        # Embeddings
        self.embedding = nn.Embedding(len(self.VOCAB), embed_dim, padding_idx=self.pad_idx)
        
        # Simple pooling (can upgrade to transformer later)
        self.pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[self.pad_idx].zero_()
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token indices."""
        words = text.lower().replace(",", " ").replace(".", " ").split()
        tokens = []
        for w in words[:self.max_tokens]:
            tokens.append(self.word2idx.get(w, self.unk_idx))
        # Pad
        while len(tokens) < self.max_tokens:
            tokens.append(self.pad_idx)
        return tokens
    
    def forward(self, text: str) -> torch.Tensor:
        """
        Encode a single text instruction.
        
        Args:
            text: Instruction string like "click the blue button"
            
        Returns:
            Tensor of shape [1, embed_dim]
        """
        tokens = self.tokenize(text)
        token_tensor = torch.tensor([tokens], dtype=torch.long)
        
        if next(self.parameters()).is_cuda:
            token_tensor = token_tensor.cuda()
        
        # Get embeddings [1, max_tokens, embed_dim]
        embeds = self.embedding(token_tensor)
        
        # Mean pool (ignoring padding)
        mask = (token_tensor != self.pad_idx).float().unsqueeze(-1)
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Project
        return self.pool(pooled)
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of instructions."""
        all_tokens = [self.tokenize(t) for t in texts]
        token_tensor = torch.tensor(all_tokens, dtype=torch.long)
        
        if next(self.parameters()).is_cuda:
            token_tensor = token_tensor.cuda()
        
        embeds = self.embedding(token_tensor)
        mask = (token_tensor != self.pad_idx).float().unsqueeze(-1)
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return self.pool(pooled)


# Quick test
if __name__ == "__main__":
    encoder = SimpleTextEncoder()
    print(f"Vocab size: {len(encoder.VOCAB)}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test encoding
    texts = [
        "click the blue button",
        "find the search field",
        "scroll to the bottom",
        "hover over the menu",
    ]
    
    for text in texts:
        emb = encoder(text)
        print(f"'{text}' → shape {emb.shape}")
    
    # Batch
    batch_emb = encoder.encode_batch(texts)
    print(f"\nBatch: {batch_emb.shape}")
