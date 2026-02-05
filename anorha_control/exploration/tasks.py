"""
Browser launcher and random task generator for exploration.
Opens browser windows and generates exploration tasks.
"""
import subprocess
import random
import time
from dataclasses import dataclass
from typing import List, Optional


# Websites to explore (diverse for training)
EXPLORATION_SITES = [
    # Search engines
    "https://google.com",
    "https://bing.com",
    "https://duckduckgo.com",
    
    # Social/Content
    "https://reddit.com",
    "https://twitter.com",
    "https://youtube.com",
    "https://medium.com",
    
    # Shopping  
    "https://amazon.com",
    "https://ebay.com",
    "https://etsy.com",
    
    # Productivity
    "https://github.com",
    "https://notion.so",
    "https://trello.com",
    
    # News/Info
    "https://wikipedia.org",
    "https://cnn.com",
    "https://nytimes.com",
    
    # Tools
    "https://figma.com",
    "https://canva.com",
]

# Elements to look for
ELEMENT_TYPES = [
    "button", "link", "search field", "text input", 
    "menu", "dropdown", "checkbox", "tab", "icon",
    "navigation", "header", "footer", "sidebar",
    "image", "card", "modal", "popup"
]

# Actions to try
ACTIONS = [
    "click", "hover over", "scroll to", "find"
]

# Descriptors
DESCRIPTORS = [
    "blue", "green", "red", "large", "small",
    "main", "primary", "secondary", "top", "bottom",
    "left", "right", "first", "last", "center"
]


@dataclass
class ExplorationTask:
    """A generated exploration task."""
    instruction: str
    site: str
    element_type: str
    action: str
    difficulty: int  # 1-3


def generate_task(site: Optional[str] = None) -> ExplorationTask:
    """Generate a random exploration task."""
    if site is None:
        site = random.choice(EXPLORATION_SITES)
    
    element = random.choice(ELEMENT_TYPES)
    action = random.choice(ACTIONS)
    descriptor = random.choice(DESCRIPTORS) if random.random() > 0.5 else ""
    
    # Generate instruction
    if descriptor:
        instruction = f"{action} the {descriptor} {element}"
    else:
        instruction = f"{action} a {element}"
    
    # Difficulty based on specificity
    difficulty = 1
    if descriptor:
        difficulty = 2
    if "find" in action or random.random() > 0.7:
        difficulty = 3
    
    return ExplorationTask(
        instruction=instruction,
        site=site,
        element_type=element,
        action=action,
        difficulty=difficulty,
    )


def generate_task_batch(n: int = 10) -> List[ExplorationTask]:
    """Generate a batch of exploration tasks."""
    return [generate_task() for _ in range(n)]


import sys
import webbrowser


class BrowserLauncher:
    """
    Launches and manages browser windows for exploration.
    Uses the system default browser via command line.
    Cross-platform: Windows, macOS, Linux.
    """
    
    def __init__(self):
        self.current_site: Optional[str] = None
        self._browser_pid: Optional[int] = None
    
    def open_site(self, url: str):
        """Open a URL in the default browser."""
        # Cross-platform: use webbrowser module
        webbrowser.open(url)
        self.current_site = url
        time.sleep(2)  # Wait for browser to open
        print(f"[Browser] Opened: {url}")
    
    def open_random_site(self) -> str:
        """Open a random site from the exploration list."""
        site = random.choice(EXPLORATION_SITES)
        self.open_site(site)
        return site
    
    def focus_browser(self):
        """Bring browser to foreground."""
        if sys.platform == "darwin":
            # macOS
            for browser in ["Google Chrome", "Safari", "Firefox", "Arc"]:
                try:
                    subprocess.run([
                        "osascript", "-e",
                        f'tell application "{browser}" to activate'
                    ], capture_output=True)
                    return
                except:
                    continue
        elif sys.platform == "win32":
            # Windows: browsers usually focus themselves when opening
            pass
        else:
            # Linux: try wmctrl if available
            try:
                subprocess.run(["wmctrl", "-a", "browser"], capture_output=True)
            except:
                pass
    
    def new_tab(self, url: Optional[str] = None):
        """Open a new tab."""
        if url:
            self.open_site(url)
        else:
            # Platform-specific keyboard shortcut
            if sys.platform == "darwin":
                subprocess.run([
                    "osascript", "-e",
                    'tell application "System Events" to keystroke "t" using command down'
                ], capture_output=True)
            # On Windows, opening a new URL usually opens in new tab
    
    def close(self):
        """Clean up (no-op for now)."""
        pass
    
    def close_tab(self):
        """Close current tab."""
        if sys.platform == "darwin":
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to keystroke "w" using command down'
            ], capture_output=True)
    
    def navigate_back(self):
        """Go back in browser history."""
        if sys.platform == "darwin":
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to keystroke "[" using command down'
            ], capture_output=True)
    
    def scroll_page(self, direction: str = "down"):
        """Scroll the page."""
        if sys.platform == "darwin":
            subprocess.run([
                "osascript", "-e",
                f'tell application "System Events" to key code 125'  # Down arrow
            ], capture_output=True)



class ExplorationSession:
    """
    Manages a full exploration session with browser and tasks.
    """
    
    def __init__(self, sites: Optional[List[str]] = None):
        self.browser = BrowserLauncher()
        self.sites = sites or EXPLORATION_SITES
        self.tasks_completed = 0
        self.current_task: Optional[ExplorationTask] = None
    
    def start_session(self):
        """Start exploration session by opening browser."""
        site = self.browser.open_random_site()
        self.current_task = generate_task(site)
        print(f"[Session] Started with task: {self.current_task.instruction}")
        return self.current_task
    
    def next_task(self) -> ExplorationTask:
        """Generate and switch to next task."""
        self.tasks_completed += 1
        
        # Change site every 5 tasks
        if self.tasks_completed % 5 == 0:
            site = self.browser.open_random_site()
            self.current_task = generate_task(site)
        else:
            self.current_task = generate_task(self.browser.current_site)
        
        print(f"[Session] Task {self.tasks_completed}: {self.current_task.instruction}")
        return self.current_task
    
    def get_task_instruction(self) -> str:
        """Get current task instruction for the model."""
        if self.current_task:
            return self.current_task.instruction
        return "explore the page"


# Quick test
if __name__ == "__main__":
    print("Testing Browser + Task Generation...")
    
    # Generate some tasks
    tasks = generate_task_batch(5)
    for t in tasks:
        print(f"  [{t.difficulty}] {t.instruction} on {t.site}")
    
    print("\nStarting session (will open browser)...")
    session = ExplorationSession()
    session.start_session()
    
    time.sleep(3)
    
    for _ in range(3):
        session.next_task()
        time.sleep(2)
    
    print(f"\nCompleted {session.tasks_completed} tasks")
