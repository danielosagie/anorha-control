# Anorha-Control

**TRM-based autonomous GUI control with unsupervised exploration**

A vision-based computer control system using a Tiny Recursive Model (TRM) that learns to click, type, and navigate through curiosity-driven exploration.

## Quick Start

```bash
# Install dependencies
~/.local/bin/uv sync

# Install playwright browser
~/.local/bin/uv run playwright install chromium

# Test the system
~/.local/bin/uv run python -m anorha_control.main test
```

## Usage

### Safe Mode Exploration (Logs Only)
```bash
~/.local/bin/uv run python -m anorha_control.main explore --safe
```

### Live Exploration (Will Click!)
```bash
~/.local/bin/uv run python -m anorha_control.main explore
```

### Train on Collected Experiences
```bash
~/.local/bin/uv run python -m anorha_control.main train --epochs 10
```

### Explore + Train Concurrently
```bash
~/.local/bin/uv run python -m anorha_control.main explore --train-concurrent
```

## Architecture

```
Vision Encoder (MobileViTv2, frozen)
        ↓
    256d embedding
        ↓
  TRM (2 recursive layers)
        ↓
  (x, y) coords + action type
```

### Components

- **Vision Encoder**: Frozen MobileViTv2 (1.1M params, 256d output)
- **TRM**: Tiny Recursive Model with cross-attention (2.3M params)
- **AsyncExplorer**: Curiosity-driven exploration with novelty rewards
- **AsyncTrainer**: REINFORCE policy gradient training
- **ExperienceDB**: SQLite persistence for experiences

## Project Structure

```
anorha_control/
├── models/
│   ├── vision_encoder.py   # Frozen MobileViTv2
│   └── trm.py              # Tiny Recursive Model
├── exploration/
│   └── async_explorer.py   # Curiosity-driven explorer
├── training/
│   └── async_trainer.py    # REINFORCE trainer
├── knowledge/
│   └── database.py         # SQLite experience storage
├── utils/
│   ├── screen.py           # mss screen capture
│   ├── mouse.py            # pyautogui with bezier curves
│   ├── browser.py          # Playwright async browser
│   └── hashing.py          # Perceptual hashing
├── config.py               # Configuration
└── main.py                 # CLI entry point
```

## How It Works

1. **Capture**: Screenshot of current screen
2. **Encode**: Vision encoder → 256d embedding
3. **Predict**: TRM predicts (x, y) + action type
4. **Execute**: Click/type/scroll with human-like movement
5. **Observe**: Compute reward (did screen change? new state?)
6. **Learn**: REINFORCE on successful transitions

The explorer uses **curiosity-driven exploration**:
- Random actions with ε probability
- Prioritizes untried UI elements
- Rewards novel states and successful transitions

## Training on Modal (Future)

For larger-scale training, you can deploy to Modal's T4 GPUs:

```python
# modal_train.py (coming soon)
import modal

app = modal.App("anorha-control")

@app.function(gpu="T4")
def train_on_modal():
    ...
```

## License

MIT
