"""
Execution Backend - Abstraction for browser vs desktop control.

Enables full computer use: browser tasks (Playwright) and desktop tasks
(file explorer, downloads, search) share the same interface.
"""
import asyncio
import io
import os
import platform
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from PIL import Image

from ..utils.screen import ScreenCapture
from ..utils.mouse import click, type_text, smooth_move_to

try:
    import pyautogui
except ImportError:
    pyautogui = None


class ExecutionBackend(ABC):
    """Abstract interface for executing actions (browser or desktop)."""

    @abstractmethod
    async def screenshot(self) -> Image.Image:
        """Capture current screen state."""
        ...

    @abstractmethod
    async def navigate(self, url_or_context: str) -> bool:
        """Navigate to URL (browser) or open context (desktop). Returns True on success."""
        ...

    @abstractmethod
    async def click(self, x: int, y: int):
        """Click at pixel coordinates."""
        ...

    @abstractmethod
    async def type_text(self, text: str, at: Optional[Tuple[int, int]] = None):
        """Type text. If at is provided, click there first."""
        ...

    @abstractmethod
    async def press_key(self, key: str):
        """Press a key (e.g. Enter, Tab)."""
        ...

    @abstractmethod
    def viewport_size(self) -> Tuple[int, int]:
        """Return (width, height) of the viewport/screen."""
        ...

    async def ensure_ready(self) -> bool:
        """Ensure backend is ready (e.g. browser not crashed). Default: True."""
        return True


class BrowserBackend(ExecutionBackend):
    """Playwright-based browser control."""

    def __init__(self, viewport_width: int = 1280, viewport_height: int = 800, headless: bool = True):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def init(self):
        """Initialize Playwright browser."""
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                f"--window-size={self.viewport_width},{self.viewport_height}",
                "--window-position=0,0",
                "--no-default-browser-check",
                "--no-first-run",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        )
        self._page = await self._context.new_page()

    async def close(self):
        """Close browser."""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None

    @property
    def page(self):
        return self._page

    async def screenshot(self) -> Image.Image:
        if not self._page:
            raise RuntimeError("Browser not initialized")
        buffer = await self._page.screenshot()
        return Image.open(io.BytesIO(buffer))

    async def navigate(self, url_or_context: str) -> bool:
        if not self._page or not url_or_context or url_or_context == "desktop":
            return True
        for attempt in range(3):
            try:
                await self._page.goto(url_or_context, timeout=30000, wait_until="domcontentloaded")
                await asyncio.sleep(1)
                return True
            except Exception as e:
                if "crashed" in str(e).lower() or "page closed" in str(e).lower():
                    await self.close()
                    await asyncio.sleep(3)
                    await self.init()
                elif attempt < 2:
                    await asyncio.sleep(2)
        return False

    async def click(self, x: int, y: int):
        await self._page.mouse.click(x, y)

    async def type_text(self, text: str, at: Optional[Tuple[int, int]] = None):
        if at:
            await self._page.mouse.click(at[0], at[1])
            await asyncio.sleep(0.05)
        await self._page.keyboard.type(text)

    async def press_key(self, key: str):
        await self._page.keyboard.press(key)

    def viewport_size(self) -> Tuple[int, int]:
        return (self.viewport_width, self.viewport_height)

    async def ensure_ready(self) -> bool:
        try:
            if self._page and not self._page.is_closed():
                return True
        except Exception:
            pass
        await self.close()
        await asyncio.sleep(2)
        await self.init()
        return True


class DesktopBackend(ExecutionBackend):
    """Real desktop control via pyautogui and screen capture."""

    def __init__(self, monitor: int = 0):
        self._screen = ScreenCapture(monitor=monitor)

    async def screenshot(self) -> Image.Image:
        return self._screen.capture()

    async def navigate(self, url_or_context: str) -> bool:
        if not url_or_context or url_or_context == "desktop":
            return True
        ctx_lower = url_or_context.lower()
        if "download" in ctx_lower or "file" in ctx_lower:
            return await self._open_file_explorer(url_or_context)
        return True

    async def _open_file_explorer(self, context: str) -> bool:
        """Open File Explorer. On Windows: Win+E. Optional path via context."""
        if not pyautogui:
            return False
        try:
            system = platform.system()
            if system == "Windows":
                pyautogui.hotkey("win", "e")
                await asyncio.sleep(1.5)
                if "download" in context.lower():
                    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
                    pyautogui.typewrite(downloads, interval=0.05)
                    await asyncio.sleep(0.3)
                    pyautogui.press("enter")
                    await asyncio.sleep(1)
            elif system == "Darwin":
                pyautogui.hotkey("command", "space")
                await asyncio.sleep(0.5)
                pyautogui.typewrite("Finder", interval=0.05)
                await asyncio.sleep(0.3)
                pyautogui.press("enter")
                await asyncio.sleep(1)
            return True
        except Exception:
            return False

    async def click(self, x: int, y: int):
        smooth_move_to(x, y, duration=0.2)
        await asyncio.sleep(0.05)
        click(x, y, smooth=False)

    async def type_text(self, text: str, at: Optional[Tuple[int, int]] = None):
        if at:
            smooth_move_to(at[0], at[1])
            await asyncio.sleep(0.05)
            click(at[0], at[1], smooth=False)
            await asyncio.sleep(0.05)
        type_text(text)

    async def press_key(self, key: str):
        if not pyautogui:
            return
        key_map = {"enter": "return", "return": "return"}
        pyautogui.press(key_map.get(key.lower(), key))

    def viewport_size(self) -> Tuple[int, int]:
        return self._screen.screen_size
