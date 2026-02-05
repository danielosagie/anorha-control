"""
Screen capture utilities using mss (cross-platform, fast)
"""
import io
import time
from pathlib import Path
from typing import Optional, Tuple

import mss
import mss.tools
import numpy as np
from PIL import Image
import cv2


class ScreenCapture:
    """Fast screen capture using mss."""
    
    def __init__(self, monitor: int = 0):
        """
        Args:
            monitor: Monitor index (0 = all monitors, 1 = primary, 2+ = others)
        """
        self.sct = mss.mss()
        self.monitor = monitor
    
    def capture(self, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """
        Capture screen or region.
        
        Args:
            region: Optional (x, y, width, height) tuple
            
        Returns:
            PIL Image in RGB format
        """
        if region:
            monitor = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
        else:
            monitor = self.sct.monitors[self.monitor]
        
        screenshot = self.sct.grab(monitor)
        
        # Convert to PIL Image (mss returns BGRA)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img
    
    def capture_numpy(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture as numpy array (BGR format for OpenCV)."""
        if region:
            monitor = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
        else:
            monitor = self.sct.monitors[self.monitor]
        
        screenshot = self.sct.grab(monitor)
        return np.array(screenshot)[:, :, :3]  # Drop alpha
    
    def capture_and_save(self, path: Path, region: Optional[Tuple[int, int, int, int]] = None) -> Path:
        """Capture and save to disk."""
        img = self.capture(region)
        img.save(path)
        return path
    
    @property
    def screen_size(self) -> Tuple[int, int]:
        """Get current monitor size."""
        mon = self.sct.monitors[self.monitor]
        return (mon["width"], mon["height"])


def compute_visual_difference(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute normalized visual difference between two images.
    Returns value in [0, 1] where 0 = identical, 1 = completely different.
    """
    # Resize to same size for comparison
    size = (64, 64)
    arr1 = np.array(img1.resize(size).convert("RGB"), dtype=np.float32) / 255.0
    arr2 = np.array(img2.resize(size).convert("RGB"), dtype=np.float32) / 255.0
    
    # MSE
    mse = np.mean((arr1 - arr2) ** 2)
    return float(mse)


def detect_ui_elements(image: Image.Image) -> list[dict]:
    """
    Simple UI element detection using edge detection + contours.
    Returns list of potential clickable regions.
    """
    # Convert to grayscale
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    elements = []
    h, w = gray.shape
    
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        
        # Filter: reasonable size for UI elements
        if area < 100 or area > (w * h * 0.5):
            continue
        if cw < 20 or ch < 15:
            continue
        
        # Normalize coordinates
        center_x = (x + cw / 2) / w
        center_y = (y + ch / 2) / h
        
        elements.append({
            "x": center_x,
            "y": center_y,
            "width": cw / w,
            "height": ch / h,
            "bbox": (x, y, cw, ch),
            "area": area,
        })
    
    # Sort by area (larger elements first)
    elements.sort(key=lambda e: e["area"], reverse=True)
    
    return elements[:50]  # Top 50 elements


# Global instance for convenience
_screen = None

def get_screen() -> ScreenCapture:
    global _screen
    if _screen is None:
        _screen = ScreenCapture()
    return _screen


def capture() -> Image.Image:
    """Quick capture of full screen."""
    return get_screen().capture()


# Quick test
if __name__ == "__main__":
    print("Testing screen capture...")
    screen = ScreenCapture()
    print(f"Screen size: {screen.screen_size}")
    
    t0 = time.perf_counter()
    img = screen.capture()
    t1 = time.perf_counter()
    print(f"Capture time: {(t1-t0)*1000:.1f}ms")
    print(f"Image size: {img.size}")
    
    # Test element detection
    elements = detect_ui_elements(img)
    print(f"Detected {len(elements)} UI elements")
    
    # Test visual diff
    img2 = screen.capture()
    diff = compute_visual_difference(img, img2)
    print(f"Visual diff (same screen): {diff:.6f}")
