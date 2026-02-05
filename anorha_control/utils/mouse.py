"""
Mouse and keyboard control utilities.
Human-like movements using bezier curves.
"""
import math
import random
import time
from typing import Optional, Tuple

import pyautogui


# Disable PyAutoGUI failsafe (we have our own kill switch)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.02  # Small delay between actions


def get_screen_size() -> Tuple[int, int]:
    """Get screen dimensions."""
    return pyautogui.size()


def get_position() -> Tuple[int, int]:
    """Get current cursor position."""
    return pyautogui.position()


def bezier_curve(p0, p1, p2, p3, t):
    """Calculate point on cubic bezier curve."""
    return (
        (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0],
        (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1],
    )


def smooth_move_to(x: int, y: int, duration: float = 0.3, steps: int = 30):
    """
    Move mouse smoothly using bezier curves.
    This creates human-like movement patterns.
    """
    start_x, start_y = pyautogui.position()
    
    # Calculate distance
    dx = x - start_x
    dy = y - start_y
    dist = math.sqrt(dx**2 + dy**2)
    
    # Create random control points for natural curve
    offset = min(dist * 0.3, 100)
    
    ctrl1 = (
        start_x + dx * 0.25 + random.uniform(-offset, offset),
        start_y + dy * 0.25 + random.uniform(-offset, offset),
    )
    ctrl2 = (
        start_x + dx * 0.75 + random.uniform(-offset, offset),
        start_y + dy * 0.75 + random.uniform(-offset, offset),
    )
    
    start = (start_x, start_y)
    end = (x, y)
    
    step_delay = duration / steps
    
    for i in range(1, steps + 1):
        t = i / steps
        # Ease in-out
        t = t * t * (3 - 2 * t)
        
        px, py = bezier_curve(start, ctrl1, ctrl2, end, t)
        pyautogui.moveTo(int(px), int(py), _pause=False)
        time.sleep(step_delay)
    
    # Ensure we end exactly at target
    pyautogui.moveTo(x, y, _pause=False)


def click(x: int = None, y: int = None, smooth: bool = True, button: str = "left"):
    """
    Click at position with optional smooth movement.
    """
    if x is not None and y is not None:
        if smooth:
            smooth_move_to(x, y)
        else:
            pyautogui.moveTo(x, y)
    
    pyautogui.click(button=button)


def double_click(x: int = None, y: int = None, smooth: bool = True):
    """Double click at position."""
    if x is not None and y is not None:
        if smooth:
            smooth_move_to(x, y)
        else:
            pyautogui.moveTo(x, y)
    
    pyautogui.doubleClick()


def right_click(x: int = None, y: int = None, smooth: bool = True):
    """Right click at position."""
    click(x, y, smooth, button="right")


def type_text(text: str, interval: float = 0.05, human_like: bool = True):
    """
    Type text with optional human-like delays.
    """
    if human_like:
        for char in text:
            pyautogui.write(char, interval=0)
            # Random delay between keystrokes
            time.sleep(random.uniform(0.03, 0.12))
    else:
        pyautogui.write(text, interval=interval)


def press_key(key: str):
    """Press a single key."""
    pyautogui.press(key)


def hotkey(*keys):
    """Press a key combination."""
    pyautogui.hotkey(*keys)


def scroll(clicks: int, x: int = None, y: int = None):
    """
    Scroll the mouse wheel.
    Positive = up, negative = down.
    """
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.scroll(clicks)


def drag_to(x: int, y: int, duration: float = 0.5, button: str = "left"):
    """Drag mouse to position."""
    pyautogui.drag(x - pyautogui.position()[0], y - pyautogui.position()[1], 
                   duration=duration, button=button)


# Keyboard helpers
def cmd_key(key: str):
    """Press Cmd+key."""
    pyautogui.hotkey("command", key)


def type_in_field(text: str, clear_first: bool = True):
    """Type in a text field, optionally clearing first."""
    if clear_first:
        pyautogui.hotkey("command", "a")  # Select all
        time.sleep(0.1)
    type_text(text)


def press_enter():
    """Press Enter key."""
    pyautogui.press("return")


def press_tab():
    """Press Tab key."""
    pyautogui.press("tab")


def press_escape():
    """Press Escape key."""
    pyautogui.press("escape")


class SafeController:
    """
    Controller that can log actions instead of executing them.
    Useful for testing without actually clicking.
    """
    
    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode
        self.action_log = []
    
    def execute_action(self, action: str, x: int = None, y: int = None, **kwargs):
        """Execute or log an action."""
        timestamp = time.time()
        log_entry = {
            "action": action,
            "x": x,
            "y": y,
            "time": timestamp,
            **kwargs
        }
        self.action_log.append(log_entry)
        
        if self.safe_mode:
            print(f"[SAFE] Would {action} at ({x}, {y})")
            return
        
        # Execute the action for real
        if action == "click":
            click(x, y)
        elif action == "double_click":
            double_click(x, y)
        elif action == "right_click":
            right_click(x, y)
        elif action == "scroll":
            scroll(kwargs.get("clicks", -3), x, y)
        elif action == "type":
            type_text(kwargs.get("text", ""))
        elif action == "press_key":
            press_key(kwargs.get("key", ""))
    
    def clear_log(self):
        """Clear the action log."""
        self.action_log = []


# Quick test
if __name__ == "__main__":
    print("Testing mouse control...")
    print(f"Screen size: {get_screen_size()}")
    print(f"Current position: {get_position()}")
    
    # Move to center
    screen = get_screen_size()
    center = (screen[0] // 2, screen[1] // 2)
    
    print(f"Moving to center: {center}")
    smooth_move_to(*center)
    
    print("Mouse test complete!")
