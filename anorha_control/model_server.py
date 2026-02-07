"""
Model Server CLI - Unified interface for llama.cpp and Ollama.

Usage:
    # Start llama.cpp server with model shorthand
    python -m anorha_control.model_server start qwen3-vl:1b
    python -m anorha_control.model_server start qwen3-vl:1b --port 8080
    
    # Stop the server
    python -m anorha_control.model_server stop
    
    # List available models
    python -m anorha_control.model_server list
    
    # Check server status
    python -m anorha_control.model_server status

Supports:
    - llama.cpp (GGUF models, fast with GPU)
    - Ollama (easy setup, slower)
"""
import argparse
import subprocess
import os
import sys
import json
import signal
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import platform

# =============================================================================
# MODEL REGISTRY - Shorthand to model file/name mapping
# =============================================================================

# =============================================================================
# MODEL REGISTRY - Shorthand to model file/name mapping
# =============================================================================

MODELS = {
    # VLM models (for grounding/planning) - Official Qwen3-VL from HuggingFace
    # IMPORTANT: These require BOTH main gguf AND mmproj (vision encoder)
    "qwen3-vl:2b": {
        "gguf": "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-Qwen3VL-2B-Instruct-F16.gguf",  # Vision encoder - REQUIRED!
        "hf_repo": "Qwen/Qwen3-VL-2B-Instruct-GGUF",
        "ollama": "qwen3-vl:2b",
        "type": "vlm"
    },
    "qwen3-vl:8b": {
        "gguf": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        "hf_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "ollama": "qwen3-vl:8b",
        "type": "vlm"
    },
    # LLM models (for planning/orchestration) - no mmproj needed
    "qwen3:4b": {
        "gguf": None,
        "ollama": "qwen3:4b",
        "type": "llm"
    },
    "llama3.2:3b": {
        "gguf": None,
        "ollama": "llama3.2:3b",
        "type": "llm"
    },
}

# Aliases (Ollama-only)
MODELS["qwen-vl"] = MODELS["qwen3-vl:2b"]
MODELS["qwen-vl-2b"] = MODELS["qwen3-vl:2b"]
MODELS["vlm"] = MODELS["qwen3-vl:2b"]
MODELS["astria"] = {"ollama": "Me7war/Astria", "type": "vlm"}
MODELS["youtu-vl"] = {"ollama": "youtu/youtu-vl", "type": "vlm"}
# Smaller VLMs: moondream (1.4B, 1.7GB), llava-phi3 (3.8B, 2.9GB) - often better JSON than llava
MODELS["moondream"] = {"ollama": "moondream", "type": "vlm"}
MODELS["llava-phi3"] = {"ollama": "llava-phi3", "type": "vlm"}


# =============================================================================
# PATH DETECTION
# =============================================================================

def find_llamacpp() -> Optional[Path]:
    """Find llama.cpp installation on Windows/Mac/Linux."""
    possible_paths = []
    
    if platform.system() == "Windows":
        # Common Windows locations
        possible_paths = [
            Path("B:/llama-cpp"),
            Path("C:/llama-cpp"),
            Path("D:/llama-cpp"),
            Path.home() / "llama-cpp",
            Path.home() / "llama.cpp",
            Path("C:/Program Files/llama-cpp"),
        ]
        exe_name = "llama-server.exe"
    else:
        # Mac/Linux
        possible_paths = [
            Path("/usr/local/bin"),  # Homebrew
            Path("/opt/homebrew/bin"),  # M1 Homebrew
            Path.home() / "llama.cpp/build/bin",
            Path.home() / "llama-cpp",
        ]
        exe_name = "llama-server"
    
    # Check each path
    for base in possible_paths:
        if not base.exists():
            continue
        # Check for exe directly
        exe = base / exe_name
        if exe.exists():
            return exe
        # Check in subdirectories (Windows releases have nested structure)
        for sub in ["bin", "build/bin", ""]:
            exe = base / sub / exe_name if sub else base / exe_name
            if exe.exists():
                return exe
        # List directory to find any llama-server variant
        try:
            for f in base.iterdir():
                if f.name.startswith("llama-server") or f.name.startswith("llama-cli"):
                    return f
        except:
            pass
    
    return None


def find_models_dir() -> Path:
    """Find or create models directory."""
    possible = [
        Path.home() / "models",
        Path("models"),
        Path.home() / ".cache/huggingface/hub",
    ]
    
    for p in possible:
        if p.exists():
            return p
    
    # Create default
    default = Path.home() / "models"
    default.mkdir(parents=True, exist_ok=True)
    return default


def get_model_path(model_key: str) -> Optional[Path]:
    """Get path to GGUF model file."""
    model = MODELS.get(model_key)
    if not model or not model.get("gguf"):
        return None
    
    models_dir = find_models_dir()
    gguf_name = model["gguf"]
    
    # Check direct path
    direct = models_dir / gguf_name
    if direct.exists():
        return direct
    
    # Check in subdirectory (HF download structure)
    for subdir in models_dir.iterdir():
        if subdir.is_dir():
            candidate = subdir / gguf_name
            if candidate.exists():
                return candidate
    
    return None


def get_mmproj_path(model_key: str) -> Optional[Path]:
    """Get path to mmproj (vision encoder) file for VLM models."""
    model = MODELS.get(model_key)
    if not model or not model.get("mmproj"):
        return None
    
    models_dir = find_models_dir()
    mmproj_name = model["mmproj"]
    
    # Check direct path
    direct = models_dir / mmproj_name
    if direct.exists():
        return direct
    
    # Check in subdirectory (HF download structure)
    for subdir in models_dir.iterdir():
        if subdir.is_dir():
            candidate = subdir / mmproj_name
            if candidate.exists():
                return candidate
    
    return None



# =============================================================================
# SERVER MANAGEMENT
# =============================================================================

class ServerManager:
    """Manage llama-server and Ollama processes."""
    
    PID_FILE = Path.home() / ".anorha" / "server.pid"
    STATE_FILE = Path.home() / ".anorha" / "server.json"
    
    def __init__(self):
        (Path.home() / ".anorha").mkdir(exist_ok=True)
    
    def _save_state(self, pid: int, port: int, backend: str, model: str):
        """Save server state for later management."""
        self.STATE_FILE.write_text(json.dumps({
            "pid": pid,
            "port": port,
            "backend": backend,
            "model": model
        }))
        self.PID_FILE.write_text(str(pid))
    
    def _load_state(self) -> Optional[Dict[str, Any]]:
        """Load saved server state."""
        if self.STATE_FILE.exists():
            try:
                return json.loads(self.STATE_FILE.read_text())
            except:
                pass
        return None
    
    def _clear_state(self):
        """Clear saved state."""
        if self.STATE_FILE.exists():
            self.STATE_FILE.unlink()
        if self.PID_FILE.exists():
            self.PID_FILE.unlink()
    
    def start(
        self,
        model_key: str,
        port: int = 8080,
        gpu_layers: int = 99,
        prefer_ollama: bool = False,
    ) -> bool:
        """
        Start model server.
        
        Tries llama.cpp first (faster), falls back to Ollama.
        """
        model = MODELS.get(model_key)
        if not model:
            print(f"❌ Unknown model: {model_key}")
            print(f"   Available: {', '.join(MODELS.keys())}")
            return False
        
        # Check if already running
        if self._is_running(port):
            print(f"⚠️ Server already running on port {port}")
            return True
        
        # Try llama.cpp first (if GGUF available)
        if not prefer_ollama and model.get("gguf"):
            llamacpp = find_llamacpp()
            model_path = get_model_path(model_key)
            mmproj_path = get_mmproj_path(model_key) if model.get("mmproj") else None
            
            if llamacpp and model_path:
                # Check for required mmproj
                if model.get("mmproj") and not mmproj_path:
                    print(f"⚠️ Vision encoder (mmproj) not found: {model['mmproj']}")
                    print(f"   Download with: hf download {model['hf_repo']} --local-dir ~/models/")
                else:
                    return self._start_llamacpp(llamacpp, model_path, mmproj_path, port, gpu_layers, model_key)
            elif not model_path:
                print(f"⚠️ GGUF model not found. Download with:")
                print(f"   hf download {model['hf_repo']} --local-dir ~/models/")
        
        # Fall back to Ollama
        if model.get("ollama"):
            return self._start_ollama(model["ollama"], port, model_key)
        
        print(f"❌ No available backend for {model_key}")
        return False
    
    def _start_llamacpp(
        self,
        exe: Path,
        model_path: Path,
        mmproj_path: Optional[Path],
        port: int,
        gpu_layers: int,
        model_key: str,
    ) -> bool:
        """Start llama-server process."""
        cmd = [
            str(exe),
            "-m", str(model_path),
            "--port", str(port),
            "-ngl", str(gpu_layers),
        ]
        
        # Add vision encoder for VLM models
        if mmproj_path:
            cmd.extend(["--mmproj", str(mmproj_path)])
        
        print(f"🚀 Starting llama.cpp server...")
        print(f"   Model: {model_path.name}")
        if mmproj_path:
            print(f"   Vision: {mmproj_path.name}")
        print(f"   Port: {port}")
        print(f"   GPU layers: {gpu_layers}")
        
        try:
            # Start in background
            if platform.system() == "Windows":
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
            
            self._save_state(process.pid, port, "llamacpp", model_key)
            print(f"✅ Server started (PID: {process.pid})")
            print(f"   API: http://localhost:{port}/v1/chat/completions")
            return True
        except Exception as e:
            print(f"❌ Failed to start: {e}")
            return False
    
    def _start_ollama(self, model_name: str, port: int, model_key: str) -> bool:
        """Start Ollama with model."""
        # Check if Ollama is running
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                print(f"✅ Ollama already running")
            else:
                print(f"⚠️ Start Ollama first: ollama serve")
                return False
        except:
            print(f"⚠️ Ollama not running. Start it with: ollama serve")
            return False
        
        # Pull model if needed
        print(f"🔄 Ensuring model is available: {model_name}")
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
        except:
            pass  # Model might already exist
        
        self._save_state(-1, 11434, "ollama", model_key)
        print(f"✅ Using Ollama on port 11434")
        print(f"   Model: {model_name}")
        return True
    
    def stop(self) -> bool:
        """Stop the running server."""
        state = self._load_state()
        if not state:
            print("⚠️ No server state found")
            return False
        
        if state["backend"] == "ollama":
            print("ℹ️ Ollama runs as a service - use 'ollama stop' if needed")
            self._clear_state()
            return True
        
        pid = state.get("pid")
        if pid:
            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
                else:
                    os.kill(pid, signal.SIGTERM)
                print(f"✅ Stopped server (PID: {pid})")
            except:
                print(f"⚠️ Process {pid} not found (may have already stopped)")
        
        self._clear_state()
        return True
    
    def status(self) -> Dict[str, Any]:
        """Get server status."""
        state = self._load_state()
        
        result = {
            "running": False,
            "backend": None,
            "model": None,
            "port": None,
            "url": None
        }
        
        # Check llama.cpp (port 8080)
        if self._is_running(8080):
            result.update({
                "running": True,
                "backend": "llamacpp",
                "port": 8080,
                "url": "http://localhost:8080"
            })
        
        # Check Ollama (port 11434)
        if self._is_running(11434, endpoint="/api/tags"):
            result.update({
                "running": True,
                "backend": "ollama",
                "port": 11434,
                "url": "http://localhost:11434"
            })
        
        if state:
            result["model"] = state.get("model")
        
        return result
    
    def _is_running(self, port: int, endpoint: str = "/health") -> bool:
        """Check if server is running on port."""
        try:
            r = requests.get(f"http://localhost:{port}{endpoint}", timeout=2)
            return r.status_code == 200
        except:
            return False


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_start(args):
    """Start model server."""
    manager = ServerManager()
    success = manager.start(
        args.model,
        port=args.port,
        gpu_layers=args.gpu_layers,
        prefer_ollama=args.ollama,
    )
    return 0 if success else 1


def cmd_stop(args):
    """Stop model server."""
    manager = ServerManager()
    return 0 if manager.stop() else 1


def cmd_status(args):
    """Show server status."""
    manager = ServerManager()
    status = manager.status()
    
    if status["running"]:
        print(f"✅ Server running")
        print(f"   Backend: {status['backend']}")
        print(f"   Port: {status['port']}")
        print(f"   Model: {status['model'] or 'unknown'}")
        print(f"   URL: {status['url']}")
    else:
        print("❌ No server running")
        print("   Start with: python -m anorha_control.model_server start qwen3-vl:1b")
    
    return 0


def cmd_list(args):
    """List available models."""
    print("📦 Available Models:\n")
    
    vlm_models = [(k, v) for k, v in MODELS.items() if v.get("type") == "vlm" and ":" in k]
    llm_models = [(k, v) for k, v in MODELS.items() if v.get("type") == "llm" and ":" in k]
    
    print("VLM (Vision-Language) models:")
    for key, model in vlm_models:
        gguf = "✅ GGUF" if model.get("gguf") else "❌ GGUF"
        ollama = "✅ Ollama" if model.get("ollama") else "❌ Ollama"
        print(f"  {key:20s} {gguf:12s} {ollama}")
    
    print("\nLLM (Text) models:")
    for key, model in llm_models:
        gguf = "✅ GGUF" if model.get("gguf") else "❌ GGUF"
        ollama = "✅ Ollama" if model.get("ollama") else "❌ Ollama"
        print(f"  {key:20s} {gguf:12s} {ollama}")
    
    print("\nAliases: vlm, qwen-vl, qwen-vl-1b")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Model Server CLI - Manage llama.cpp and Ollama",
        prog="python -m anorha_control.model_server"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Start command
    start_p = subparsers.add_parser("start", help="Start model server")
    start_p.add_argument("model", help="Model shorthand (e.g., qwen3-vl:1b)")
    start_p.add_argument("--port", type=int, default=8080, help="Server port")
    start_p.add_argument("--gpu-layers", "-ngl", type=int, default=99, help="GPU layers")
    start_p.add_argument("--ollama", action="store_true", help="Prefer Ollama backend")
    start_p.set_defaults(func=cmd_start)
    
    # Stop command
    stop_p = subparsers.add_parser("stop", help="Stop model server")
    stop_p.set_defaults(func=cmd_stop)
    
    # Status command
    status_p = subparsers.add_parser("status", help="Check server status")
    status_p.set_defaults(func=cmd_status)
    
    # List command
    list_p = subparsers.add_parser("list", help="List available models")
    list_p.set_defaults(func=cmd_list)
    
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
