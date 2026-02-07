"""
Anorha SDK Server - HTTP API wrapping ComputerAgent for LLM tool use.
"""
import asyncio
import base64
import io
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from ..exploration.execution_backend import BrowserBackend
from ..models.computer_agent import ComputerAgent, AgentConfig


# Shared state
_backend: Optional[BrowserBackend] = None
_agent: Optional[ComputerAgent] = None


class GotoRequest(BaseModel):
    url: str


class ClickRequest(BaseModel):
    target: str


class TypeRequest(BaseModel):
    text: str
    target: Optional[str] = None


class ScrollRequest(BaseModel):
    direction: str
    amount: int = 300


class PressKeyRequest(BaseModel):
    key: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start browser and agent on startup."""
    global _backend, _agent
    _backend = BrowserBackend(viewport_width=1280, viewport_height=800, headless=True)
    await _backend.init()
    
    async def screenshot_fn():
        return await _backend.screenshot()
    
    config = AgentConfig(
        vlm_model="moondream",
        verify_actions=False,
        viewport_width=1280,
        viewport_height=800,
    )
    _agent = ComputerAgent(config, page=_backend.page, screenshot_fn=screenshot_fn)
    yield
    await _backend.close()
    _backend = None
    _agent = None


app = FastAPI(
    title="Anorha SDK",
    description="Browser control API for LLM tool use",
    lifespan=lifespan,
)


@app.post("/goto")
async def goto(req: GotoRequest):
    """Navigate to URL."""
    if not _backend:
        raise HTTPException(503, "Server not ready")
    ok = await _backend.navigate(req.url)
    return {"success": ok, "url": req.url}


@app.post("/click")
async def click(req: ClickRequest):
    """Click on element by semantic description."""
    if not _agent:
        raise HTTPException(503, "Server not ready")
    result = await _agent.click(req.target)
    return {
        "success": result.success,
        "target": req.target,
        "coordinates": list(result.coordinates),
        "source": result.source,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


@app.post("/type")
async def type_text(req: TypeRequest):
    """Type text, optionally focusing target first."""
    if not _agent:
        raise HTTPException(503, "Server not ready")
    result = await _agent.type(req.text, req.target)
    return {
        "success": result.success,
        "target": req.target or "(focused)",
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


@app.post("/scroll")
async def scroll(req: ScrollRequest):
    """Scroll the page."""
    if not _agent:
        raise HTTPException(503, "Server not ready")
    result = await _agent.scroll(req.direction, req.amount)
    return {"success": result.success, "direction": req.direction}


@app.get("/screenshot")
async def screenshot():
    """Capture screenshot. Returns base64 JPEG."""
    if not _backend:
        raise HTTPException(503, "Server not ready")
    img = await _backend.screenshot()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"success": True, "image_base64": b64}


@app.post("/press_key")
async def press_key(req: PressKeyRequest):
    """Press a key."""
    if not _backend or not _backend.page:
        raise HTTPException(503, "Server not ready")
    await _backend.press_key(req.key)
    return {"success": True, "key": req.key}


def run_server(host: str = "127.0.0.1", port: int = 8765):
    """Run the SDK server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
