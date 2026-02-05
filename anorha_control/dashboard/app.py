"""
Dashboard for monitoring exploration and training.
Uses Gradio for quick web UI.
"""
import gradio as gr
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from PIL import Image
import io
import base64

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anorha_control.knowledge.database import ExperienceDB
from anorha_control.knowledge.embeddings import EmbeddingStore


class DashboardState:
    """Shared state for the dashboard."""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.db: Optional[ExperienceDB] = None
        self.embeddings: Optional[EmbeddingStore] = None
        self._loop = None
    
    async def init(self):
        """Initialize database connections."""
        self.db = ExperienceDB(self.data_dir / "experiences.db")
        await self.db.connect()
        
        self.embeddings = EmbeddingStore(
            self.data_dir / "embeddings.db",
            embedding_dim=256,
        )
    
    def get_loop(self):
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        if self.db is None:
            return {"error": "Database not connected"}
        
        stats = await self.db.get_stats()
        
        if self.embeddings:
            emb_stats = self.embeddings.stats()
            stats["embeddings"] = emb_stats["total_embeddings"]
        
        return stats
    
    async def get_recent_experiences(self, limit: int = 20) -> List[Dict]:
        """Get recent experiences for display."""
        if self.db is None:
            return []
        
        return await self.db.get_recent_experiences(limit)
    
    async def export_data(self, min_reward: float = 0.0) -> str:
        """Export filtered training data."""
        if self.db is None:
            return "Database not connected"
        
        experiences = await self.db.get_experiences_by_reward(min_reward)
        
        export_path = self.data_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(export_path, "w") as f:
            json.dump([e.__dict__ for e in experiences], f, indent=2, default=str)
        
        return f"Exported {len(experiences)} experiences to {export_path}"


# Global state
state = DashboardState()


def run_async(coro):
    """Run async function in sync context."""
    loop = state.get_loop()
    return loop.run_until_complete(coro)


def get_stats_display():
    """Get formatted stats for display."""
    try:
        stats = run_async(state.get_stats())
        
        if "error" in stats:
            return stats["error"]
        
        return f"""## 📊 Exploration Stats

| Metric | Value |
|--------|-------|
| **Total Actions** | {stats.get('total_actions', 0):,} |
| **Successes** | {stats.get('total_successes', 0):,} |
| **Success Rate** | {stats.get('success_rate', 0):.1%} |
| **Unique States** | {stats.get('unique_states', 0):,} |
| **Embeddings** | {stats.get('embeddings', 0):,} |
"""
    except Exception as e:
        return f"Error loading stats: {e}"


def get_recent_experiences_display():
    """Get recent experiences for display."""
    try:
        experiences = run_async(state.get_recent_experiences(10))
        
        if not experiences:
            return "No experiences recorded yet. Run exploration first."
        
        rows = []
        for exp in experiences:
            status = "✓" if exp.get("success") else "✗"
            action_type = ["click", "right_click", "double_click", "type", "scroll"][exp.get("action_type", 0)]
            rows.append(f"| {status} | {action_type} | ({exp.get('action_x', 0):.2f}, {exp.get('action_y', 0):.2f}) | {exp.get('reward', 0):.2f} |")
        
        table = "| Status | Action | Position | Reward |\n|--------|--------|----------|--------|\n" + "\n".join(rows)
        
        return f"""## 🕹️ Recent Experiences

{table}
"""
    except Exception as e:
        return f"Error loading experiences: {e}"


def export_training_data(min_reward: float):
    """Export training data with minimum reward filter."""
    try:
        result = run_async(state.export_data(min_reward))
        return result
    except Exception as e:
        return f"Export failed: {e}"


def refresh_all():
    """Refresh all displays."""
    return get_stats_display(), get_recent_experiences_display()


def create_dashboard():
    """Create the Gradio dashboard."""
    
    with gr.Blocks(title="Anorha-Control Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🎮 Anorha-Control Dashboard

Monitor exploration progress, view experiences, and export training data.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                stats_display = gr.Markdown(get_stats_display())
                refresh_btn = gr.Button("🔄 Refresh Stats", variant="primary")
            
            with gr.Column(scale=2):
                experiences_display = gr.Markdown(get_recent_experiences_display())
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📦 Export Training Data")
                min_reward_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Minimum Reward Filter",
                )
                export_btn = gr.Button("📥 Export Data", variant="secondary")
                export_result = gr.Textbox(label="Export Result", interactive=False)
        
        # Event handlers
        refresh_btn.click(
            fn=refresh_all,
            outputs=[stats_display, experiences_display],
        )
        
        export_btn.click(
            fn=export_training_data,
            inputs=[min_reward_slider],
            outputs=[export_result],
        )
    
    return demo


def launch_dashboard(port: int = 7860, share: bool = False):
    """Launch the dashboard."""
    # Initialize state
    try:
        run_async(state.init())
        print(f"[Dashboard] Connected to database")
    except Exception as e:
        print(f"[Dashboard] Warning: Could not init database: {e}")
    
    demo = create_dashboard()
    demo.launch(server_port=port, share=share)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Anorha-Control Dashboard")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    print(f"Starting dashboard on port {args.port}...")
    launch_dashboard(port=args.port, share=args.share)
