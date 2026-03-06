let ws = null;
let isSliding = false;
let keySliding = false;

// ── DOM 캐싱 (반복 lookup 제거) ──
const DOM = {};

const HUD_JUMP_DURATION = 200;
let hudActionTimeout = null;

// ── FPS 측정 개선 ──
let fpsCounter = 0;
let lastFpsTime = 0;

// ── 사전 직렬화된 메시지 캐싱 ──
const MSG_CACHE = {
    reset:  '{"type":"reset"}',
    none:   '{"type":"action","action":"none"}',
    jump:   '{"type":"action","action":"jump"}',
    slide:  '{"type":"action","action":"slide"}',
};

// #################### Init ####################

function cacheDOMElements() {
    DOM.gameImage = document.getElementById('game-image');
    DOM.hudAction = document.getElementById('hud-action');
    DOM.hudFps    = document.getElementById('hud-fps');
    DOM.btnJump   = document.getElementById('btn-jump');
    DOM.btnSlide  = document.getElementById('btn-slide');
    DOM.btnReset  = document.getElementById('btn-reset');
}

// #################### WebSocket ####################

function connectWebSocket() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => console.log("WebSocket connected");

    ws.onmessage = handleMessage;

    ws.onclose = () => {
        console.log("Disconnected, reconnecting...");
        setTimeout(connectWebSocket, 1000);
    };

    ws.onerror = (e) => console.error("WebSocket error:", e);
}

function handleMessage(event) {
    const data = JSON.parse(event.data);

    if (data.status === "error") {
        alert(data.message);
        return;
    }

    // HUD 업데이트
    if (data.current_action === 'jump') {
        displayHUDAction('jump', HUD_JUMP_DURATION);
    } else if (!hudActionTimeout) {
        displayHUDAction(data.current_action);
    }

    // 이미지 업데이트 (rAF 사용)
    // src 교체는 layout/paint를 유발하므로 rAF로 배칭
    requestAnimationFrame(() => {
        DOM.gameImage.src = data.image;
        countFPS();
    });
}

function sendAction(action) {
    if (ws?.readyState === WebSocket.OPEN) {
        // ── JSON.stringify 제거: 캐싱된 문자열 직접 전송 ──
        ws.send(MSG_CACHE[action] || MSG_CACHE.none);
    }
}

// #################### HUD ####################

function displayHUDAction(action, duration = 0) {
    DOM.hudAction.textContent = action;

    if (hudActionTimeout) {
        clearTimeout(hudActionTimeout);
        hudActionTimeout = null;
    }

    if (duration > 0) {
        hudActionTimeout = setTimeout(() => {
            hudActionTimeout = null;
        }, duration);
    }
}

function countFPS() {
    fpsCounter++;
    const now = performance.now();

    if (now - lastFpsTime >= 1000) {
        DOM.hudFps.textContent = fpsCounter;
        fpsCounter = 0;
        lastFpsTime = now;
    }
}

// #################### Actions ####################

function handleJump() {
    if (isSliding) return;
    sendAction('jump');

    DOM.btnJump.classList.add('active');
    setTimeout(() => DOM.btnJump.classList.remove('active'), 100);
}

function startSlideAction() {
    if (isSliding) return;
    isSliding = true;
    sendAction('slide');
    DOM.btnSlide.classList.add('active');
}

function stopSlideAction() {
    if (!isSliding) return;
    isSliding = false;
    sendAction('none');
    DOM.btnSlide.classList.remove('active');
}

function resetGame() {
    isSliding = false;
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(MSG_CACHE.reset);
    }
    DOM.btnReset.classList.add('active');
    setTimeout(() => DOM.btnReset.classList.remove('active'), 100);
}

// #################### Event Listeners ####################

function setupEventListeners() {
    // Jump
    DOM.btnJump.addEventListener('mousedown', (e) => {
        e.preventDefault();
        handleJump();
    });
    DOM.btnJump.addEventListener('touchstart', (e) => {
        e.preventDefault();
        handleJump();
    }, { passive: false });

    // Slide
    DOM.btnSlide.addEventListener('mousedown', (e) => {
        e.preventDefault();
        startSlideAction();
    });
    DOM.btnSlide.addEventListener('mouseup', (e) => {
        e.preventDefault();
        stopSlideAction();
    });
    DOM.btnSlide.addEventListener('mouseleave', () => {
        if (isSliding) stopSlideAction();
    });
    DOM.btnSlide.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startSlideAction();
    }, { passive: false });
    DOM.btnSlide.addEventListener('touchend', (e) => {
        e.preventDefault();
        stopSlideAction();
    }, { passive: false });
    DOM.btnSlide.addEventListener('touchcancel', (e) => {
        e.preventDefault();
        stopSlideAction();
    }, { passive: false });

    // Reset
    DOM.btnReset.addEventListener('click', (e) => {
        e.preventDefault();
        resetGame();
    });

    // Keyboard
    const slideKeys = new Set(['s', 'S', 'ArrowDown']);
    const jumpKeys  = new Set(['w', 'W', 'ArrowUp', ' ']);
    const resetKeys = new Set(['r', 'R']);

    document.addEventListener('keydown', (e) => {
        if (resetKeys.has(e.key)) {
            e.preventDefault();
            resetGame();
        } else if (jumpKeys.has(e.key) && !e.repeat) {
            e.preventDefault();
            handleJump();
        } else if (slideKeys.has(e.key) && !keySliding) {
            e.preventDefault();
            keySliding = true;
            startSlideAction();
        }
    });

    document.addEventListener('keyup', (e) => {
        if (slideKeys.has(e.key)) {
            keySliding = false;
            stopSlideAction();
        }
    });

    window.addEventListener('blur', () => {
        keySliding = false;
        if (isSliding) stopSlideAction();
    });
}

// #################### Boot ####################

window.addEventListener('load', () => {
    cacheDOMElements();
    lastFpsTime = performance.now();
    setupEventListeners();
    connectWebSocket();
});