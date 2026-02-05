"""
Frozen Vision Encoder using MobileViTv2
Outputs 256-dimensional embeddings from screenshots
"""
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms


class VisionEncoder(nn.Module):
    """
    Frozen vision encoder using MobileViTv2.
    Takes screenshots, outputs 256d embeddings.
    ~2M params, runs fast on GTX 1650.
    """
    
    def __init__(self, model_name: str = "mobilevitv2_050", pretrained: bool = True, freeze: bool = True):
        super().__init__()
        
        # Load pretrained model without classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier, get features
        )
        
        # Get output dimension
        self.output_dim = self.backbone.num_features
        
        # Freeze if requested (default for exploration)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        self.frozen = freeze
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to model input tensor."""
        return self.transform(image)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor of shape [B, 3, 256, 256]
            
        Returns:
            Tensor of shape [B, output_dim] (256 for mobilevitv2_050)
        """
        if self.frozen:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a PIL Image to embedding.
        
        Args:
            image: PIL Image (any size)
            
        Returns:
            Tensor of shape [1, output_dim]
        """
        x = self.preprocess(image).unsqueeze(0)
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        return self(x)
    
    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode a batch of PIL Images."""
        tensors = [self.preprocess(img) for img in images]
        batch = torch.stack(tensors)
        if next(self.parameters()).is_cuda:
            batch = batch.cuda()
        return self(batch)


def load_vision_encoder(
    model_name: str = "mobilevitv2_050",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> VisionEncoder:
    """Load the frozen vision encoder."""
    encoder = VisionEncoder(model_name=model_name, pretrained=True, freeze=True)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


# Quick test
if __name__ == "__main__":
    from PIL import ImageGrab
    
    print("Loading vision encoder...")
    encoder = load_vision_encoder()
    print(f"Model loaded. Output dim: {encoder.output_dim}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test with screenshot
    print("\nCapturing screenshot...")
    screenshot = ImageGrab.grab()
    print(f"Screenshot size: {screenshot.size}")
    
    embedding = encoder.encode_image(screenshot)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding sample: {embedding[0, :5]}")
