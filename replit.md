# Cookie-Run-AI-v2

An AI-powered simulation of the first stage ("The Witch's Oven") of the game Cookie Run. A neural network "imagines" and generates the game's visual frames in real-time based on user input (Jump, Slide, or None).

## Architecture

- **World Model**: Two-stage deep learning architecture
  - **VQ-VAE**: Compresses 128×256 game frames into a discrete 16×32 grid of integer tokens
  - **RSSM** (Recurrent State-Space Model): Predicts the next game state in latent space
- **Web Framework**: FastAPI with WebSocket for real-time frame delivery
- **Frontend**: Vanilla HTML/CSS/JavaScript

## Project Structure

```
main.py              # Entry point, runs FastAPI on port 5000
server.py            # FastAPI app, WebSocket endpoint
wrapper.py           # Bridges AI models and server
config/
  map_config.py      # Config loader
  oven_of_witch.yaml # Hyperparameters and model paths
modules/             # PyTorch neural network architectures
  vqvae.py / vqvae_network.py  # VQ-VAE
  rssm.py / rssm_network.py    # RSSM
  blocks.py / utils.py
static/              # Frontend (index.html, javascript.js, style.css)
samples/oven_of_witch/  # Sample PNG frames for environment reset
model_params/        # Pre-trained weights (NOT in repo, must be downloaded)
  vqvae_ep30.pth
  rssm_ep100.pth
```

## Setup Requirements

The app requires pre-trained model weights that are NOT included in the repository. Download from the [GitHub Releases page](https://github.com/Jeon-ChanYoung/Cookie-Run-AI-v2/releases) and place in `model_params/`:
- `model_params/vqvae_ep30.pth`
- `model_params/rssm_ep100.pth`

The server starts and serves on port 5000 regardless. Without weights, it shows an instructions page.

## Running

```bash
python main.py
```

Runs on `http://0.0.0.0:5000`

## Dependencies

- Python 3.12
- torch (CPU build) + numpy, Pillow, opencv-python-headless
- fastapi + uvicorn[standard]
- pyyaml (loaded via uvicorn standard)

## Controls

- **↑ / W / Space**: Jump
- **↓ / S**: Slide (hold)
- **R**: Reset the simulation

## Deployment

- Target: VM (always-running, maintains model state in memory)
- Run: `python main.py`
