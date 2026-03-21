import asyncio
import time
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

    print("🔄 Loading resources...")
    vqvae = VQVAE(config).to(config.device)
    vqvae.load_vqvae(config.vqvae_path)
    vqvae.change_train_mode(train=False)

    codebook_weight = vqvae.quantizer.codebook.clone().detach()
    rssm = RSSM(config, codebook_weight=codebook_weight).to(config.device)
    rssm.load_rssm(config.rssm_path)
    rssm.change_train_mode(train=False)
    print("✅ Resources loaded.")

    TARGET_FPS = getattr(config, "target_fps", 10)

    @app.get("/", response_class=HTMLResponse)
    async def read_root():
        with open(f"{static_path}/index.html", "r", encoding="utf-8") as f:
            return f.read()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        wrapper = Wrapper(config, vqvae, rssm)

        current_action = "none"
        jump_pending = False
        should_reset = True
        running = True

        async def receiver():
            nonlocal current_action, jump_pending, should_reset, running
            try:
                while running:
                    data = await websocket.receive_json()
                    msg_type = data.get("type")
                    if msg_type == "reset":
                        should_reset = True
                    elif msg_type == "action":
                        action = data.get("action", "none")
                        if action == "jump":
                            jump_pending = True
                        else:
                            current_action = action
            except (WebSocketDisconnect, Exception):
                running = False

        async def sender():
            nonlocal current_action, jump_pending, should_reset, running
            loop = asyncio.get_event_loop()
            interval = 1.0 / TARGET_FPS
            next_frame = time.monotonic()

            try:
                while running:
                    now = time.monotonic()
                    wait = next_frame - now
                    if wait > 0:
                        await asyncio.sleep(wait)

                    next_frame += interval

                    now = time.monotonic()
                    if next_frame < now - interval:
                        next_frame = now

                    if should_reset:
                        should_reset = False
                        current_action = "none"
                        jump_pending = False
                        action_id = 3
                        img_bytes = await loop.run_in_executor(
                            None, wrapper.reset_to_bytes
                        )
                    else:
                        if jump_pending:
                            jump_pending = False
                            action = "jump"
                        else:
                            action = current_action

                        action_id = wrapper.action_map.get(action, 0)
                        img_bytes = await loop.run_in_executor(
                            None, wrapper.step_to_bytes, action
                        )

                    await websocket.send_bytes(
                        bytes([action_id]) + img_bytes
                    )

            except Exception:
                running = False

        recv_task = asyncio.create_task(receiver())
        send_task = asyncio.create_task(sender())

        try:
            await asyncio.wait(
                [recv_task, send_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in [recv_task, send_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    return app