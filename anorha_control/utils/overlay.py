"""
Visual cursor indicator - shows a red circle where the agent will click.
Also handles the green border overlay and kill switch.
"""
import threading
import time
import subprocess
from typing import Callable, Optional
import pyautogui

from pynput import keyboard


import sys
import os

# Anorha green color
ANORHA_GREEN = "#8cc63f"
AGENT_CURSOR_RED = "#ff3b30"


def check_accessibility():
    """Check if accessibility permissions are granted."""
    if sys.platform != "darwin":
        return True # Windows/Linux don't have the same accessibility prompt model
    
    # Try to get mouse position - if it works, we have some access
    try:
        x, y = pyautogui.position()
        return True
    except Exception as e:
        return False


def request_accessibility():
    """Open System Settings to Accessibility preferences."""
    if sys.platform != "darwin":
        return

    print("\n" + "=" * 60)
    print("⚠️  ACCESSIBILITY PERMISSIONS REQUIRED")
    print("=" * 60)
    print("""
To control the mouse and keyboard, you need to grant accessibility access:

1. Open System Settings → Privacy & Security → Accessibility
2. Click the '+' button
3. Add your Terminal app (Terminal, iTerm, VS Code, etc.)
4. Make sure the checkbox is ENABLED

Opening System Settings for you...
""")
    # Open accessibility settings
    subprocess.run([
        "open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    ])
    input("Press Enter after granting permission...")


class CursorIndicator:
    """
    Shows a red circle at the cursor position to indicate agent control.
    Uses a transparent overlay window.
    """
    
    def __init__(self, size: int = 30, color: str = AGENT_CURSOR_RED):
        self.size = size
        self.color = color
        self._process: Optional[subprocess.Popen] = None
    
    def show_at(self, x: int, y: int, duration: float = 0.5):
        """
        Show a brief red circle indicator at position.
        Uses AppleScript to draw a quick visual.
        """
        # Create a visual flash using a simple approach
        # We'll use a Python window that appears briefly
        self._flash_indicator(x, y, duration)
    
    def _flash_indicator(self, x: int, y: int, duration: float):
        """Flash a visual indicator using terminal cursor."""
        # For now, print a visual log
        print(f"   🔴 Click: ({x}, {y})")


class ScreenBorder:
    """
    Creates a green border around the screen using a shell script approach.
    More reliable than AppKit for non-main-thread usage.
    """
    
    def __init__(self, color: str = ANORHA_GREEN, width: int = 8):
        self.color = color
        self.width = width
        self._process: Optional[subprocess.Popen] = None
        self._script_path = os.path.join(os.path.expanduser("~"), "anorha_border.py")
    
    def _create_border_script(self):
        """Create a Python script for the border overlay."""
        script = f'''#!/usr/bin/env python3
import tkinter as tk
import sys

def create_border():
    # Get screen dimensions
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    
    # Create borderless window for each edge
    borders = []
    border_width = {self.width}
    color = "{self.color}"
    
    positions = [
        (0, 0, width, border_width),           # Top
        (0, height - border_width, width, border_width),  # Bottom
        (0, 0, border_width, height),          # Left
        (width - border_width, 0, border_width, height),  # Right
    ]
    
    for x, y, w, h in positions:
        border = tk.Tk()
        border.overrideredirect(True)  # No window decorations
        border.attributes("-topmost", True)  # Always on top
        border.attributes("-alpha", 1.0)
        border.geometry(f"{{w}}x{{h}}+{{x}}+{{y}}")
        border.configure(bg=color)
        
        # Transparent click-through on Windows
        if sys.platform == "win32":
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            WS_EX_TRANSPARENT = 0x20
            hwnd = ctypes.windll.user32.GetParent(border.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)

        borders.append(border)
    
    # Controls hint label at bottom center
    hint = tk.Tk()
    hint.overrideredirect(True)
    hint.attributes("-topmost", True)
    hint.attributes("-alpha", 0.85)
    hint_w, hint_h = 400, 40
    hint_x = (width - hint_w) // 2
    hint_y = height - 60
    hint.geometry(f"{{hint_w}}x{{hint_h}}+{{hint_x}}+{{hint_y}}")
    hint.configure(bg="#1a1a1a")
    
    # Rounded corners via canvas
    canvas = tk.Canvas(hint, width=hint_w, height=hint_h, bg="#1a1a1a", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    
    # Draw rounded rectangle
    radius = 15
    canvas.create_arc(0, 0, radius*2, radius*2, start=90, extent=90, fill="#1a1a1a", outline="#1a1a1a")
    canvas.create_arc(hint_w-radius*2, 0, hint_w, radius*2, start=0, extent=90, fill="#1a1a1a", outline="#1a1a1a")
    canvas.create_arc(0, hint_h-radius*2, radius*2, hint_h, start=180, extent=90, fill="#1a1a1a", outline="#1a1a1a")
    canvas.create_arc(hint_w-radius*2, hint_h-radius*2, hint_w, hint_h, start=270, extent=90, fill="#1a1a1a", outline="#1a1a1a")
    
    # Text
    ctrl_key = "Ctrl" if sys.platform != "darwin" else "Cmd"
    hint_text = f"STOP: {{ctrl_key}}+Shift+Esc  |  PAUSE: {{ctrl_key}}+Shift+P"
    canvas.create_text(hint_w//2, hint_h//2, text=hint_text, fill="{self.color}", font=("Arial", 12, "bold"))
    
    # Make click-through on Windows
    if sys.platform == "win32":
        import ctypes
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x80000
        WS_EX_TRANSPARENT = 0x20
        hwnd = ctypes.windll.user32.GetParent(hint.winfo_id())
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
    
    borders.append(hint)
    
    # Keep windows alive
    def keep_alive():
        for b in borders:
            try:
                b.update()
            except:
                pass
        borders[0].after(50, keep_alive)
    
    print("BORDER_READY", flush=True)
    keep_alive()
    borders[0].mainloop()

if __name__ == "__main__":
    try:
        create_border()
    except KeyboardInterrupt:
        pass
'''
        with open(self._script_path, 'w') as f:
            f.write(script)

    
    def show(self):
        """Show the green border."""
        self._create_border_script()
        python_cmd = sys.executable or "python3"
        self._process = subprocess.Popen(
            [python_cmd, self._script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for ready signal
        time.sleep(0.5)
        print(f"[Border] Green border activated ({self.color})")
    
    def hide(self):
        """Hide the border."""
        if self._process:
            self._process.terminate()
            self._process = None
        print("[Border] Green border deactivated")


class KillSwitch:
    """Global hotkey listener for kill switch."""
    
    def __init__(self, callback: Callable):
        self.callback = callback
        # Cross-platform hotkey
        if sys.platform == "darwin":
            self.hotkey = "<cmd>+<shift>+<esc>"
        else:
            self.hotkey = "<ctrl>+<shift>+<esc>"
            
        self._listener: Optional[keyboard.GlobalHotKeys] = None
    
    def start(self):
        """Start listening."""
        def on_activate():
            print(f"\n🛑 KILL SWITCH ({self.hotkey})")
            self.callback()
        
        try:
            self._listener = keyboard.GlobalHotKeys({self.hotkey: on_activate})
            self._listener.start()
            print(f"[KillSwitch] Listening for {self.hotkey}")
        except Exception as e:
            print(f"[KillSwitch] Warning: Could not start hotkey listener: {e}")
    
    def stop(self):
        """Stop listening."""
        if self._listener:
            self._listener.stop()
            self._listener = None


class ControlIndicator:
    """Combined overlay + cursor + kill switch."""
    
    def __init__(self, on_kill: Callable = None):
        self.border = ScreenBorder()
        self.cursor = CursorIndicator()
        self.on_kill = on_kill or (lambda: None)
        self.kill_switch = KillSwitch(self._handle_kill)
        self._running = False
    
    def _handle_kill(self):
        self.stop()
        self.on_kill()
    
    def start(self):
        """Activate everything."""
        self._running = True
        
        # Check accessibility first
        if not check_accessibility():
            request_accessibility()
        
        self.border.show()
        self.kill_switch.start()
    
    def stop(self):
        """Deactivate."""
        if not self._running:
            return
            
        self._running = False
        self.border.hide()
        self.kill_switch.stop()
    
    def show_click(self, x: int, y: int):
        """Show cursor indicator at click position."""
        self.cursor.show_at(x, y)
    
    @property
    def is_running(self) -> bool:
        return self._running


def get_indicator(on_kill: Callable = None) -> ControlIndicator:
    """Get the control indicator."""
    return ControlIndicator(on_kill)


# Quick test
if __name__ == "__main__":
    print(f"Testing Overlay on {sys.platform}...")
    
    def on_kill():
        print("Killed!")
    
    indicator = get_indicator(on_kill)
    indicator.start()
    
    # Show some click indicators
    for i in range(3):
        try:
            x, y = pyautogui.position()
            indicator.show_click(x + i * 50, y)
            time.sleep(1)
        except KeyboardInterrupt:
            break
    
    indicator.stop()
