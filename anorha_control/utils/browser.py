"""
Browser automation using Playwright
Provides both headless and visual browsing for exploration
"""
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

from playwright.async_api import async_playwright, Browser, Page, BrowserContext


class BrowserController:
    """
    Async browser controller using Playwright.
    Can be used alongside mouse control for real clicking,
    or with Playwright's built-in actions for faster exploration.
    """
    
    def __init__(
        self,
        headless: bool = False,
        viewport: tuple = (1280, 720),
        user_data_dir: Optional[Path] = None,
    ):
        self.headless = headless
        self.viewport = {"width": viewport[0], "height": viewport[1]}
        self.user_data_dir = user_data_dir
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
    
    async def start(self):
        """Start browser instance."""
        self._playwright = await async_playwright().start()
        
        # Use Chromium
        if self.user_data_dir:
            # Persistent context (keeps cookies, storage)
            self._context = await self._playwright.chromium.launch_persistent_context(
                str(self.user_data_dir),
                headless=self.headless,
                viewport=self.viewport,
            )
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
            )
            self._context = await self._browser.new_context(viewport=self.viewport)
            self._page = await self._context.new_page()
        
        return self
    
    async def stop(self):
        """Stop browser instance."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
    
    @property
    def page(self) -> Page:
        if not self._page:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page
    
    async def goto(self, url: str, wait_until: str = "domcontentloaded"):
        """Navigate to URL."""
        await self.page.goto(url, wait_until=wait_until)
        await asyncio.sleep(0.5)  # Small delay for dynamic content
    
    async def screenshot(self, path: Optional[Path] = None) -> bytes:
        """Take screenshot, optionally save to path."""
        return await self.page.screenshot(path=path)
    
    async def screenshot_pil(self):
        """Take screenshot and return as PIL Image."""
        from PIL import Image
        import io
        
        screenshot_bytes = await self.page.screenshot()
        return Image.open(io.BytesIO(screenshot_bytes))
    
    async def get_viewport_size(self) -> tuple:
        """Get current viewport size."""
        size = self.page.viewport_size
        return (size["width"], size["height"])
    
    async def click_at(self, x: int, y: int):
        """Click at viewport coordinates."""
        await self.page.mouse.click(x, y)
    
    async def type_text(self, text: str, delay: int = 50):
        """Type text with delay between keys."""
        await self.page.keyboard.type(text, delay=delay)
    
    async def press_key(self, key: str):
        """Press a key."""
        await self.page.keyboard.press(key)
    
    async def scroll(self, delta_y: int):
        """Scroll page (negative = up, positive = down)."""
        await self.page.mouse.wheel(0, delta_y)
    
    async def get_clickable_elements(self) -> List[Dict[str, Any]]:
        """
        Get all clickable elements with their bounding boxes.
        Useful for curiosity-driven exploration.
        """
        elements = await self.page.evaluate("""
            () => {
                const clickable = document.querySelectorAll(
                    'a, button, input, select, textarea, [onclick], [role="button"], [role="link"]'
                );
                
                return Array.from(clickable).map(el => {
                    const rect = el.getBoundingClientRect();
                    return {
                        tag: el.tagName.toLowerCase(),
                        text: el.innerText?.slice(0, 50) || '',
                        type: el.type || '',
                        href: el.href || '',
                        x: rect.x + rect.width / 2,
                        y: rect.y + rect.height / 2,
                        width: rect.width,
                        height: rect.height,
                        visible: rect.width > 0 && rect.height > 0,
                    };
                }).filter(el => el.visible);
            }
        """)
        return elements
    
    async def get_page_info(self) -> Dict[str, Any]:
        """Get current page information."""
        return {
            "url": self.page.url,
            "title": await self.page.title(),
        }
    
    async def wait_for_navigation(self, timeout: int = 5000):
        """Wait for page navigation."""
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=timeout)
        except:
            pass  # Timeout is okay
    
    async def random_clickable_element(self) -> Optional[Dict[str, Any]]:
        """Get a random clickable element for exploration."""
        import random
        elements = await self.get_clickable_elements()
        if elements:
            return random.choice(elements)
        return None


class BrowserPool:
    """
    Pool of browser instances for parallel exploration.
    """
    
    def __init__(self, size: int = 2, **browser_kwargs):
        self.size = size
        self.browser_kwargs = browser_kwargs
        self.browsers: List[BrowserController] = []
        self._available: asyncio.Queue = asyncio.Queue()
    
    async def start(self):
        """Start all browsers in pool."""
        for _ in range(self.size):
            browser = BrowserController(**self.browser_kwargs)
            await browser.start()
            self.browsers.append(browser)
            await self._available.put(browser)
    
    async def stop(self):
        """Stop all browsers in pool."""
        for browser in self.browsers:
            await browser.stop()
        self.browsers.clear()
    
    async def acquire(self) -> BrowserController:
        """Get an available browser."""
        return await self._available.get()
    
    async def release(self, browser: BrowserController):
        """Return browser to pool."""
        await self._available.put(browser)


# Quick test
if __name__ == "__main__":
    async def test():
        print("Testing browser controller...")
        browser = BrowserController(headless=False)
        await browser.start()
        
        try:
            await browser.goto("https://google.com")
            print(f"Page info: {await browser.get_page_info()}")
            
            elements = await browser.get_clickable_elements()
            print(f"Found {len(elements)} clickable elements")
            
            screenshot = await browser.screenshot_pil()
            print(f"Screenshot size: {screenshot.size}")
            
            await asyncio.sleep(3)
        finally:
            await browser.stop()
    
    asyncio.run(test())
