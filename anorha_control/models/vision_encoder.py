"""
Modern Vision Encoder using SigLIP2 (Google, Feb 2025)

Supports:
- Single image encoding (for basic TRM)
- Temporal frame buffering (for video-like context)

SigLIP2 advantages over MobileViTV2:
- Better localization (critical for click prediction)
- Dense feature extraction
- Multilingual UI understanding
- Variable resolution support (NaFlex)
"""
import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, List
from collections import deque

# Try to import transformers for SigLIP2
try:
    from transformers import AutoProcessor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[Vision] Warning: transformers not installed, falling back to timm")

import timm
from torchvision import transforms


class SigLIP2Encoder(nn.Module):
    """
    Vision encoder using Google's SigLIP2 (Feb 2025).
    
    Features:
    - 768d embeddings (vs 256d from MobileViTV2)
    - Better semantic understanding
    - Improved localization for UI elements
    - Optional temporal frame buffer for video context
    """
    
    def __init__(
        self, 
        model_name: str = "google/siglip2-base-patch16-224",
        freeze: bool = True,
        temporal_buffer_size: int = 0,  # 0 = single image, >0 = buffer N frames
        device: str = "cpu",
    ):
        super().__init__()
        
        self.device = device
        self.temporal_buffer_size = temporal_buffer_size
        self.frozen = freeze
        
        if HAS_TRANSFORMERS:
            print(f"[Vision] Loading SigLIP2: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.output_dim = self.model.config.vision_config.hidden_size  # 768 for base
            self.use_siglip = True
        else:
            # Fallback to timm CLIP
            print("[Vision] Falling back to timm ViT-B/16")
            self.model = timm.create_model(
                "vit_base_patch16_clip_224",
                pretrained=True,
                num_classes=0,
            ).to(device)
            self.output_dim = 768
            self.use_siglip = False
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ])
        
        # Freeze model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Temporal buffer for video context
        if temporal_buffer_size > 0:
            self.frame_buffer = deque(maxlen=temporal_buffer_size)
            # Temporal aggregation layer
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=self.output_dim,
                num_heads=8,
                batch_first=True,
            ).to(device)
            self.temporal_norm = nn.LayerNorm(self.output_dim).to(device)
        else:
            self.frame_buffer = None
            self.temporal_attn = None
        
        print(f"[Vision] Encoder ready: {self.output_dim}d output, temporal={temporal_buffer_size}")
    
    def _encode_single(self, image: Image.Image) -> torch.Tensor:
        """Encode a single PIL Image."""
        if self.use_siglip:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            return outputs
        else:
            x = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.model(x)
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a PIL Image.
        
        If temporal buffering is enabled, adds to buffer and returns
        temporally-aggregated embedding.
        
        Returns:
            Tensor of shape [1, output_dim]
        """
        embedding = self._encode_single(image)
        
        if self.frame_buffer is not None:
            # Add to buffer
            self.frame_buffer.append(embedding.squeeze(0))
            
            if len(self.frame_buffer) >= 2:
                # Aggregate temporal features
                stacked = torch.stack(list(self.frame_buffer)).unsqueeze(0)  # [1, T, D]
                
                # Self-attention over temporal dimension
                attn_out, _ = self.temporal_attn(stacked, stacked, stacked)
                
                # Take the last (most recent) frame's attended representation
                temporal_embedding = self.temporal_norm(attn_out[:, -1, :])
                return temporal_embedding
        
        return embedding
    
    def encode_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode a batch of PIL Images."""
        if self.use_siglip:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                return self.model.get_image_features(**inputs)
        else:
            tensors = [self.transform(img) for img in images]
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                return self.model(batch)
    
    def clear_buffer(self):
        """Clear the temporal frame buffer."""
        if self.frame_buffer is not None:
            self.frame_buffer.clear()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for pre-processed tensors."""
        if self.frozen:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)


# Legacy VisionEncoder for backward compatibility
class VisionEncoder(nn.Module):
    """
    Frozen vision encoder using MobileViTv2.
    DEPRECATED: Use SigLIP2Encoder for new code.
    """
    
    def __init__(self, model_name: str = "mobilevitv2_050", pretrained: bool = True, freeze: bool = True):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
        self.output_dim = self.backbone.num_features
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        self.frozen = freeze
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        x = self.preprocess(image).unsqueeze(0)
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        return self(x)
    
    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        tensors = [self.preprocess(img) for img in images]
        batch = torch.stack(tensors)
        if next(self.parameters()).is_cuda:
            batch = batch.cuda()
        return self(batch)


def load_vision_encoder(
    model_type: str = "siglip2",  # "siglip2" or "mobilevit" (legacy)
    temporal_frames: int = 0,  # 0 for single image, >0 for video context
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load vision encoder.
    
    Args:
        model_type: "siglip2" (recommended) or "mobilevit" (legacy)
        temporal_frames: Number of frames to buffer for temporal context (0 = single image)
        device: "cuda" or "cpu"
    
    Returns:
        Encoder instance
    """
    if model_type == "siglip2":
        encoder = SigLIP2Encoder(
            model_name="google/siglip2-base-patch16-224",
            freeze=True,
            temporal_buffer_size=temporal_frames,
            device=device,
        )
    else:
        # Legacy MobileViTV2
        encoder = VisionEncoder(model_name="mobilevitv2_050", pretrained=True, freeze=True)
        encoder = encoder.to(device)
    
    encoder.eval()
    return encoder


# Quick test
if __name__ == "__main__":
    print("Testing SigLIP2 Vision Encoder...")
    
    # Test single image mode
    print("\n--- Single Image Mode ---")
    encoder = load_vision_encoder(model_type="siglip2", temporal_frames=0)
    print(f"Output dim: {encoder.output_dim}")
    
    # Create test image
    test_img = Image.new("RGB", (1280, 800), color="white")
    embedding = encoder.encode_image(test_img)
    print(f"Embedding shape: {embedding.shape}")
    
    # Test temporal mode
    print("\n--- Temporal Mode (5 frames) ---")
    encoder_temporal = load_vision_encoder(model_type="siglip2", temporal_frames=5)
    
    # Simulate 5 frames
    for i in range(5):
        img = Image.new("RGB", (1280, 800), color=(i * 50, 100, 100))
        emb = encoder_temporal.encode_image(img)
        print(f"  Frame {i+1}: {emb.shape}")
    
    print("\n✅ Vision encoder tests passed!")
