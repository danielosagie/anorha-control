#!/usr/bin/env python3
"""
Main entry point for anorha-control exploration and training.

Usage:
    # Explore (controls REAL mouse!)
    python -m anorha_control.main explore
    
    # Train on collected experiences  
    python -m anorha_control.main train --epochs 10
"""
import asyncio
import argparse
from pathlib import Path

import torch


async def run_explore(args):
    """Run exploration loop with REAL mouse control."""
    from anorha_control import (
        load_vision_encoder, load_trm,
        ExperienceDB
    )
    from anorha_control.exploration import RealMouseExplorer, ExplorationConfig
    from anorha_control.training import AsyncTrainer
    
    print("=" * 60)
    print("🖱️  ANORHA-CONTROL: Real Mouse Explorer")
    print("=" * 60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load models
    print("\nLoading vision encoder...")
    # Use SigLIP2 by default (768d), fallback to mobilevit (256d) with --legacy-vision
    use_siglip = not getattr(args, 'legacy_vision', False)
    temporal_frames = getattr(args, 'temporal_frames', 0)
    
    vision = load_vision_encoder(
        model_type="siglip2" if use_siglip else "mobilevit",
        temporal_frames=temporal_frames,
        device=device
    )
    print(f"  Vision encoder: {vision.output_dim}d output (temporal={temporal_frames})")
    
    print("Loading TRM...")
    checkpoint_path = args.checkpoint if args.checkpoint else None
    trm = load_trm(
        checkpoint_path=checkpoint_path, 
        device=device,
        vision_dim=vision.output_dim,  # Match vision encoder output
        hidden_dim=512 if use_siglip else 256,  # Larger hidden for SigLIP2
    )
    print(f"  TRM: {sum(p.numel() for p in trm.parameters()):,} params")
    
    # Database
    db_path = Path(args.db) if args.db else Path("data/experiences.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDatabase: {db_path}")
    db = ExperienceDB(db_path)
    await db.connect()
    
    # Exploration config
    exp_config = ExplorationConfig(
        epsilon=args.epsilon,
        max_episode_steps=args.steps,
        screenshot_dir=Path("data/screenshots"),
    )
    
    print(f"\nConfig:")
    print(f"  Epsilon: {exp_config.epsilon}")
    print(f"  Steps per episode: {exp_config.max_episode_steps}")
    
    print("\n⚠️  WARNING: This will control your REAL mouse!")
    print("   Press Cmd+Shift+Escape at any time to STOP")
    print("   Press Cmd+Shift+P to PAUSE/RESUME")
    print("\n   Starting in 3 seconds...")
    
    # Launch dashboard in background if requested
    dashboard_thread = None
    if getattr(args, 'with_dashboard', False):
        import threading
        from anorha_control.dashboard.app import launch_dashboard
        
        def run_dashboard():
            launch_dashboard(port=7860, share=False)
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        print("\n📊 Dashboard starting at http://localhost:7860")
    
    await asyncio.sleep(3)

    
    from anorha_control.models.local_llm import LocalLLM, TaskPlanner
    
    # ... (skipping some loading)
    
    # Planner (VLM)
    planner_model = args.planner_model or "qwen3-vl:2b"
    print(f"\nInitializing planner: {planner_model}")
    llm = LocalLLM(model=planner_model)
    planner = TaskPlanner(llm)
    
    # Create explorer (sandbox or real mouse)
    if args.headless:
        from anorha_control.exploration import SandboxExplorer, SandboxConfig
        sandbox_config = SandboxConfig(
            headless=False,  # Show browser window (headless=True would hide it)
            epsilon=args.epsilon,
            max_episode_steps=args.steps,
        )
        explorer = SandboxExplorer(vision, trm, db, sandbox_config, planner=planner)
        print("\n🔒 Running in SANDBOX mode - your mouse is free!")
    else:
        explorer = RealMouseExplorer(vision, trm, db, exp_config, planner=planner)
    
    # Create trainer (if concurrent training enabled)
    trainer = None
    if args.train_concurrent:
        trainer = AsyncTrainer(
            vision, trm, explorer.training_queue, db,
            checkpoint_dir=Path("checkpoints"),
        )
    
    try:
        if trainer:
            await asyncio.gather(
                explorer.explore_forever(),
                trainer.train_on_queue(),
            )
        else:
            await explorer.explore_forever()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Ctrl+C received, stopping...")
    finally:
        explorer.stop()
        if trainer:
            trainer.stop()
        
        # Print final stats with persistence feedback
        stats = await db.get_stats()
        db_path_str = str(db_path.absolute())
        print(f"\n✅ Data saved to: {db_path_str}")
        print(f"📊 Final stats:")
        print(f"  Total actions: {stats.get('total_actions', 0)}")
        print(f"  Successes: {stats.get('total_successes', 0)}")
        print(f"  Unique states: {stats.get('unique_states', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
        
        await db.close()



async def run_train(args):
    """Run training on collected experiences."""
    from anorha_control import load_vision_encoder, load_trm, ExperienceDB
    from anorha_control.training import AsyncTrainer
    import queue
    
    print("=" * 60)
    print("🏋️  ANORHA-CONTROL: Training")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load models
    print("\nLoading models...")
    vision = load_vision_encoder(device=device)
    
    checkpoint = args.checkpoint if args.checkpoint else None
    trm = load_trm(checkpoint_path=checkpoint, device=device)
    print(f"TRM: {sum(p.numel() for p in trm.parameters()):,} params")
    
    # Database
    db_path = Path(args.db) if args.db else Path("data/experiences.db")
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("   Run exploration first to collect experiences.")
        return
    
    db = ExperienceDB(db_path)
    await db.connect()
    
    # Check experience count
    total = await db.count_experiences()
    successful = await db.count_successful()
    print(f"\nExperiences: {total} total, {successful} successful")
    
    if total == 0:
        print("❌ No experiences found. Run exploration first.")
        await db.close()
        return
    
    # Create trainer
    training_queue = queue.Queue()
    trainer = AsyncTrainer(
        vision, trm, training_queue, db,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_dir=Path("checkpoints"),
    )
    
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"   LR: {args.lr}, Batch: {args.batch_size}")
    
    await trainer.train_on_db(
        num_epochs=args.epochs,
        only_successes=not args.all_experiences,
    )
    
    print("\n✅ Training complete!")
    await db.close()


def run_test(args):
    """Test models and utilities."""
    import torch
    from anorha_control import load_vision_encoder, load_trm, ScreenCapture
    
    print("=" * 60)
    print("🧪 ANORHA-CONTROL: System Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Test vision encoder
    print("\n1. Vision Encoder:")
    vision = load_vision_encoder(device=device)
    print(f"   Output: {vision.output_dim}d")
    print(f"   Params: {sum(p.numel() for p in vision.parameters()):,}")
    
    # Test TRM
    print("\n2. TRM Model:")
    trm = load_trm(device=device)
    print(f"   Params: {sum(p.numel() for p in trm.parameters()):,}")
    
    # Test screenshot
    print("\n3. Screen Capture:")
    screen = ScreenCapture()
    screenshot = screen.capture()
    print(f"   Screen: {screen.screen_size}")
    
    # Test full pipeline
    print("\n4. Pipeline:")
    embedding = vision.encode_image(screenshot)
    print(f"   Embedding: {embedding.shape}")
    
    prediction = trm.predict(embedding, screen_size=screen.screen_size)
    print(f"   Prediction: x={prediction['x']}, y={prediction['y']}")
    print(f"   Action: {prediction['action']}")
    
    # Test overlay
    print("\n5. Overlay (Cmd+Shift+Escape to stop):")
    from anorha_control.utils.overlay import get_indicator
    indicator = get_indicator()
    indicator.start()
    
    import time
    time.sleep(3)
    indicator.stop()
    
    print("\n✅ All tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Anorha-Control: Real Mouse GUI Control")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Explore command
    explore_parser = subparsers.add_parser("explore", help="Run real mouse exploration")
    explore_parser.add_argument("--train-concurrent", action="store_true", help="Train while exploring")
    explore_parser.add_argument("--epsilon", type=float, default=0.3, help="Random action probability")
    explore_parser.add_argument("--steps", type=int, default=10, help="Steps per episode")
    explore_parser.add_argument("--db", type=str, help="Database path")
    explore_parser.add_argument("--checkpoint", type=str, help="Load TRM checkpoint")
    explore_parser.add_argument("--planner-model", type=str, default="qwen3-vl:2b", help="VL model for planning")
    explore_parser.add_argument("--headless", action="store_true", help="Run in sandboxed browser mode (doesn't control your mouse)")
    explore_parser.add_argument("--with-dashboard", action="store_true", help="Launch Gradio dashboard in background")
    explore_parser.add_argument("--legacy-vision", action="store_true", help="Use legacy MobileViTV2 encoder (256d) instead of SigLIP2 (768d)")
    explore_parser.add_argument("--temporal-frames", type=int, default=0, help="Number of frames for temporal video context (0=single image)")

    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train on collected experiences")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--all-experiences", action="store_true", help="Train on all")
    train_parser.add_argument("--db", type=str, help="Database path")
    train_parser.add_argument("--checkpoint", type=str, help="Continue from checkpoint")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test system components")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="View database statistics")
    stats_parser.add_argument("--db", type=str, help="Database path")
    
    # Gather command (VLM-guided data gathering)
    gather_parser = subparsers.add_parser("gather", help="VLM-guided data gathering for TRM training")
    gather_parser.add_argument("--target", type=int, default=5000, help="Target trajectories to collect")
    gather_parser.add_argument("--visible", action="store_true", help="Show browser window")
    gather_parser.add_argument("--model", type=str, default="qwen3-vl:2b", help="VLM model")
    gather_parser.add_argument("--llamacpp", action="store_true", help="Use llama.cpp backend")
    gather_parser.add_argument("--llamacpp-url", type=str, default="http://localhost:8080", help="llama.cpp server URL")
    gather_parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard", "expert"])
    gather_parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for OCR/Post-processing")
    
    # Server command (model server management)
    server_parser = subparsers.add_parser("server", help="Manage model servers (llama.cpp/Ollama)")
    # Note: server command delegates to model_server.py which has its own subparsers
    
    args = parser.parse_args()
    
    if args.command == "explore":
        asyncio.run(run_explore(args))
    elif args.command == "train":
        asyncio.run(run_train(args))
    elif args.command == "test":
        run_test(args)
    elif args.command == "stats":
        asyncio.run(run_stats(args))
    elif args.command == "gather":
        asyncio.run(run_gather(args))
    elif args.command == "server":
        # Delegate to model_server module
        from anorha_control import model_server
        sys.exit(model_server.main())


async def run_gather(args):
    """Run VLM-guided data gathering for TRM training."""
    from anorha_control.exploration import SmartDataGatherer, GathererConfig
    from .exploration.task_curriculum import Difficulty
    
    print("=" * 60)
    print("🤖 ANORHA-CONTROL: Smart Data Gatherer")
    print("   VLM-guided exploration for TRM training data")
    print("=" * 60)
    
    # Map difficulty string to enum
    difficulty_map = {
        "easy": Difficulty.EASY,
        "medium": Difficulty.MEDIUM,
        "hard": Difficulty.HARD,
        "expert": Difficulty.EXPERT
    }
    
    config = GathererConfig(
        headless=not args.visible,
        target_trajectories=args.target,
        vlm_model=args.model,
        vlm_backend="llamacpp" if args.llamacpp else "ollama",
        vlm_url=args.llamacpp_url if args.llamacpp else "http://localhost:11434",
        max_difficulty=difficulty_map.get(args.difficulty, Difficulty.MEDIUM),
        use_gpu=getattr(args, 'gpu', False)
    )
    
    print(f"\nConfiguration:")
    print(f"   Target: {config.target_trajectories} trajectories")
    print(f"   VLM: {config.vlm_model} via {config.vlm_backend}")
    print(f"   Difficulty: {args.difficulty}")
    print(f"   Data dir: {config.data_dir}")
    
    gatherer = SmartDataGatherer(config)
    await gatherer.gather_data()


async def run_stats(args):
    """Show database statistics."""
    from anorha_control.knowledge.database import ExperienceDB
    from pathlib import Path
    
    db_path = Path(args.db) if args.db else Path("data/experiences.db")
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"📂 Database: {db_path}")
    
    db = ExperienceDB(db_path)
    await db.connect()
    
    try:
        stats = await db.get_stats()
        category_stats = await db.get_category_stats()
        
        print(f"\n📊 Overall Stats:")
        print(f"   Total actions: {stats.get('total_actions', 0)}")
        print(f"   Total successes: {stats.get('total_successes', 0)}")
        print(f"   Success rate: {100*stats.get('success_rate', 0):.1f}%")
        print(f"   Unique states: {stats.get('unique_states', 0)}")
        
        print(f"\n📂 Experiences by Category:")
        for cat, count in sorted(category_stats.get('by_category', {}).items(), key=lambda x: -x[1]):
            successes = category_stats.get('successes_by_category', {}).get(cat, 0)
            rate = 100 * successes / max(1, count)
            print(f"   {cat}: {count} ({successes} successful, {rate:.1f}%)")
        
        print(f"\n🌐 Top Sites:")
        top_sites = sorted(category_stats.get('by_site', {}).items(), key=lambda x: -x[1])[:15]
        for site, count in top_sites:
            if site and site != "unknown":
                display_site = site[:50] + "..." if len(site) > 50 else site
                print(f"   {display_site}: {count}")
        
        print(f"\n📈 Total experiences: {category_stats.get('total_experiences', 0)}")
        
    finally:
        await db.close()


if __name__ == "__main__":
    main()

