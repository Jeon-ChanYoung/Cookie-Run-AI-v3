from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from modules.rssm import RSSM
from modules.vqvae import VQVAE
from wrapper import Wrapper


def create_app(config):
    app = FastAPI(title="Cookie Run Game Server")

    static_path = "static"
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    models_ready = False
    vqvae = None
    rssm = None

    import os
    if os.path.exists(config.vqvae_path) and os.path.exists(config.rssm_path):
        print("🔄 Loading resources...")
        vqvae = VQVAE(config).to(config.device)
        vqvae.load_vqvae(config.vqvae_path)
        vqvae.change_train_mode(train=False)

        codebook_weight = vqvae.quantizer.embedding.clone().detach()

        rssm = RSSM(config, codebook_weight=codebook_weight).to(config.device)
        rssm.load_rssm(config.rssm_path)
        rssm.change_train_mode(train=False)
        print("✅ Resources loaded.")
        models_ready = True
    else:
        print("⚠️  Model weights not found. Please download them from the GitHub Releases page.")
        print(f"   Expected: {config.vqvae_path}")
        print(f"   Expected: {config.rssm_path}")
        print("   Place them in: model_params/")

    @app.get("/", response_class=HTMLResponse)
    async def read_root():
        if not models_ready:
            return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Cookie Run AI - Setup Required</title>
<style>
  body { font-family: sans-serif; background: #1a1a2e; color: #eee; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
  .box { background: #16213e; border-radius: 12px; padding: 40px; max-width: 560px; text-align: center; box-shadow: 0 4px 24px rgba(0,0,0,0.5); }
  h1 { color: #e94560; margin-bottom: 8px; }
  code { background: #0f3460; padding: 4px 10px; border-radius: 4px; font-size: 0.9em; display: block; margin: 8px 0; text-align: left; }
  a { color: #e94560; }
</style>
</head>
<body>
<div class="box">
  <h1>Cookie Run AI</h1>
  <p>Model weights are missing. Download them from the <a href="https://github.com/Jeon-ChanYoung/Cookie-Run-AI-v2/releases" target="_blank">GitHub Releases page</a> and place them here:</p>
  <code>model_params/vqvae_ep30.pth</code>
  <code>model_params/rssm_ep100.pth</code>
  <p>Then restart the server.</p>
</div>
</body>
</html>""", status_code=503)
        html_file = f"{static_path}/index.html"
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        if not models_ready:
            await websocket.send_json({
                "status": "error",
                "message": "Model weights not loaded. Please add model_params/vqvae_ep30.pth and model_params/rssm_ep100.pth, then restart the server.",
            })
            await websocket.close()
            return

        wrapper = Wrapper(
            config=config,
            vqvae=vqvae,
            rssm=rssm,
        )

        try:
            img = wrapper.reset()
            img_base64 = wrapper.image_to_base64(img)

            await websocket.send_json({
                "status": "success",
                "image": img_base64,
                "current_action": "reset",
            })

            while True:
                data = await websocket.receive_json()
                action_type = data.get("type")
                action = data.get("action", "none")

                if action_type == "reset":
                    img = wrapper.reset()
                elif action_type == "action":
                    img = wrapper.step(action)
                else:
                    continue

                img_base64 = wrapper.image_to_base64(img)

                await websocket.send_json({
                    "status": "success",
                    "image": img_base64,
                    "current_action": action,
                })

        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "status": "error",
                    "message": str(e),
                })
            except:
                pass

    return app