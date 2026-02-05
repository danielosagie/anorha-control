"""
Perceptual hashing for screen state deduplication
"""
import hashlib
from typing import Union

import numpy as np
from PIL import Image
import imagehash
import torch


def phash_image(image: Image.Image, hash_size: int = 8) -> str:
    """
    Compute perceptual hash of image.
    Similar images will have similar hashes.
    
    Returns:
        Hex string of hash
    """
    return str(imagehash.phash(image, hash_size=hash_size))


def dhash_image(image: Image.Image, hash_size: int = 8) -> str:
    """
    Compute difference hash (faster than phash).
    Good for detecting if screen changed.
    """
    return str(imagehash.dhash(image, hash_size=hash_size))


def hash_similarity(hash1: str, hash2: str) -> float:
    """
    Compute similarity between two hashes.
    Returns value in [0, 1] where 1 = identical.
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    max_diff = len(h1.hash) ** 2
    diff = h1 - h2
    return 1.0 - (diff / max_diff)


def quick_hash(image: Image.Image) -> str:
    """
    Quick hash using downsampled grayscale.
    Fast but less accurate for similarity.
    """
    small = image.resize((8, 8)).convert("L")
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return hex(int(bits, 2))[2:].zfill(16)


def hash_embedding(embedding: Union[torch.Tensor, np.ndarray]) -> str:
    """
    Hash a feature embedding for state lookup.
    Useful for knowledge base indexing.
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    
    # Flatten and convert to bytes
    flat = embedding.flatten().astype(np.float32)
    return hashlib.md5(flat.tobytes()).hexdigest()


def bucketize_embedding(embedding: Union[torch.Tensor, np.ndarray], num_buckets: int = 256) -> str:
    """
    Create a discretized bucket key from embedding.
    Groups similar embeddings together.
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    
    # Normalize
    flat = embedding.flatten()
    flat = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)
    
    # Bucket first 16 dimensions
    buckets = (flat[:16] * (num_buckets - 1)).astype(int)
    return "-".join(str(b) for b in buckets)


class StateHasher:
    """
    Track unique screen states for exploration.
    """
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.threshold = similarity_threshold
        self.seen_hashes: dict[str, int] = {}  # hash -> count
    
    def is_new_state(self, image: Image.Image) -> tuple[bool, str]:
        """
        Check if this is a new screen state.
        
        Returns:
            (is_new, hash_str)
        """
        h = phash_image(image)
        
        # Check against existing hashes
        for existing_hash in self.seen_hashes:
            sim = hash_similarity(h, existing_hash)
            if sim >= self.threshold:
                self.seen_hashes[existing_hash] += 1
                return False, existing_hash
        
        # New state
        self.seen_hashes[h] = 1
        return True, h
    
    def state_count(self) -> int:
        """Number of unique states seen."""
        return len(self.seen_hashes)
    
    def clear(self):
        """Reset state tracker."""
        self.seen_hashes.clear()


# Quick test
if __name__ == "__main__":
    from PIL import ImageGrab
    
    print("Testing hashing...")
    
    # Capture two screenshots
    img1 = ImageGrab.grab()
    img2 = ImageGrab.grab()
    
    h1 = phash_image(img1)
    h2 = phash_image(img2)
    
    print(f"Hash 1: {h1}")
    print(f"Hash 2: {h2}")
    print(f"Similarity: {hash_similarity(h1, h2):.4f}")
    
    # Quick hash
    q1 = quick_hash(img1)
    print(f"Quick hash: {q1}")
    
    # State hasher
    hasher = StateHasher()
    is_new, h = hasher.is_new_state(img1)
    print(f"Is new: {is_new}, hash: {h}")
    
    is_new, h = hasher.is_new_state(img2)
    print(f"Is new: {is_new}, hash: {h}")
    
    print(f"Unique states: {hasher.state_count()}")
