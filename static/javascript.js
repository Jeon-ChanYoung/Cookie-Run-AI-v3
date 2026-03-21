let ws = null;
let isSliding = false;
let fpsCounter = 0;
let lastFpsTime = performance.now();
let hudActionTimeout = null;

const HUD_JUMP_DURATION = 200;
const ACTION_NAMES = ["none", "jump", "slide", "reset"];

// #################### WebSocket ####################

function connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws`);
    ws.binaryType = "arraybuffer";              

    ws.onopen = () => console.log("WebSocket connected");

    ws.onmessage = (event) => {
        if (!(event.data instanceof ArrayBuffer)) return;

        const actionId = new Uint8Array(event.data, 0, 1)[0];
        const jpegData = new Uint8Array(event.data, 1);

        const action = ACTION_NAMES[actionId] || "none";
        if (action === "jump") {
            displayHUDAction("jump", HUD_JUMP_DURATION);
        } else if (!hudActionTimeout) {
            displayHUDAction(action);
        }

        const blob = new Blob([jpegData], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        const img = document.getElementById("game-image");
        const prev = img.src;
        img.src = url;
        if (prev && prev.startsWith("blob:")) URL.revokeObjectURL(prev);

        countFPS();
    };

    ws.onclose = () => {
        console.log("Disconnected, reconnecting...");
        setTimeout(connectWebSocket, 1000);
    };

    ws.onerror = (err) => console.error("WebSocket error:", err);
}

function sendAction(action) {
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "action", action }));
    }
}

function sendReset() {
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "reset" }));
    }
}

// #################### HUD / FPS ####################

function displayHUDAction(action, duration = 0) {
    document.getElementById("hud-action").textContent = action;
    if (hudActionTimeout) { clearTimeout(hudActionTimeout); hudActionTimeout = null; }
    if (duration > 0) {
        hudActionTimeout = setTimeout(() => { hudActionTimeout = null; }, duration);
    }
}

function countFPS() {
    fpsCounter++;
    const now = performance.now();
    if (now - lastFpsTime >= 500) {
        document.getElementById("hud-fps").textContent = Math.round(fpsCounter * 2);
        fpsCounter = 0;
        lastFpsTime = now;
    }
}

function handleJump() {
    sendAction("jump");
    const btn = document.getElementById("btn-jump");
    btn.classList.add("active");
    setTimeout(() => btn.classList.remove("active"), 100);
}

function startSlide() {
    if (isSliding) return;
    isSliding = true;
    sendAction("slide");
    document.getElementById("btn-slide").classList.add("active");
}

function stopSlide() {
    if (!isSliding) return;
    isSliding = false;
    sendAction("none");
    document.getElementById("btn-slide").classList.remove("active");
}

// #################### Reset ####################

function resetGame() {
    isSliding = false;
    document.getElementById("btn-slide").classList.remove("active");
    sendReset();
    const btn = document.getElementById("btn-reset");
    btn.classList.add("active");
    setTimeout(() => btn.classList.remove("active"), 100);
}

// #################### Event Listeners ####################

const jumpBtn = document.getElementById("btn-jump");
jumpBtn.addEventListener("mousedown", (e) => { e.preventDefault(); handleJump(); });
jumpBtn.addEventListener("touchstart", (e) => { e.preventDefault(); handleJump(); }, { passive: false });

const slideBtn = document.getElementById("btn-slide");
slideBtn.addEventListener("mousedown", (e) => { e.preventDefault(); startSlide(); });
slideBtn.addEventListener("mouseup", (e) => { e.preventDefault(); stopSlide(); });
slideBtn.addEventListener("mouseleave", () => { if (isSliding) stopSlide(); });
slideBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startSlide(); }, { passive: false });
slideBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopSlide(); }, { passive: false });
slideBtn.addEventListener("touchcancel", (e) => { e.preventDefault(); stopSlide(); }, { passive: false });

document.getElementById("btn-reset").addEventListener("click", (e) => { e.preventDefault(); resetGame(); });

// #################### Keyboard ####################

const slideKeys = new Set(["s", "S", "ArrowDown"]);
const jumpKeys = new Set(["w", "W", "ArrowUp", " "]);
const resetKeys = new Set(["r", "R"]);
let keySliding = false;

document.addEventListener("keydown", (e) => {
    if (resetKeys.has(e.key)) { e.preventDefault(); resetGame(); return; }
    if (jumpKeys.has(e.key) && !e.repeat) { e.preventDefault(); handleJump(); return; }
    if (slideKeys.has(e.key) && !keySliding) { e.preventDefault(); keySliding = true; startSlide(); }
});

document.addEventListener("keyup", (e) => {
    if (slideKeys.has(e.key)) { keySliding = false; stopSlide(); }
});

window.addEventListener("blur", () => { keySliding = false; stopSlide(); });
window.addEventListener("load", connectWebSocket);