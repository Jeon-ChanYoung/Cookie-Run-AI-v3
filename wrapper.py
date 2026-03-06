import os
import cv2
import base64
import torch
import numpy as np

from modules.vqvae import VQVAE
from modules.rssm import RSSM


class Wrapper:
    def __init__(self, config, vqvae: VQVAE, rssm: RSSM):
        self.config = config
        self.vqvae = vqvae
        self.rssm = rssm

        self.vqvae.change_train_mode(train=False)
        self.rssm.change_train_mode(train=False)

        self.action_map = {'none': 0, 'jump': 1, 'slide': 2}

        self._load_samples()

        # ── 사전 생성 텐서 ──
        self._action_tensors = {
            name: self._create_action_tensor(idx)
            for name, idx in self.action_map.items()
        }
        self._zero_latent = torch.zeros(
            1, config.latent_size, device=config.device
        )

        # ── JPEG 인코딩 파라미터 사전 생성 ──
        self._jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 85]

        # ── CPU 버퍼 (BGR 순서로 직접 생성하여 cvtColor 생략) ──
        self._img_buffer_bgr = np.empty((128, 256, 3), dtype=np.uint8)

        # ── CUDA stream (GPU 파이프라인) ──
        if config.device != 'cpu' and torch.cuda.is_available():
            self._stream = torch.cuda.Stream(device=config.device)
        else:
            self._stream = None

        self.recurrent_state = None
        self.latent_state = None
        self.reset()
        print("Game state initialized")

    def _create_action_tensor(self, action_idx):
        action = torch.zeros(
            1, self.config.action_size, device=self.config.device
        )
        action[0, action_idx] = 1.0
        return action

    # ════════════════════════════════════════
    #  통합 메서드 (server에서 single thread call)
    # ════════════════════════════════════════

    def reset_and_encode(self) -> str:
        """reset + image_to_base64 를 한 번의 호출로."""
        img = self.reset()
        return self._encode_to_base64(img)

    def step_and_encode(self, action_name: str) -> str:
        """step (또는 reset) + image_to_base64 를 한 번의 호출로."""
        if action_name == "reset":
            img = self.reset()
        else:
            img = self.step(action_name)
        return self._encode_to_base64(img)

    # ════════════════════════════════════════
    #  Core Logic
    # ════════════════════════════════════════

    @torch.no_grad()
    def reset(self):
        self.recurrent_state = torch.zeros(
            1, self.config.recurrent_size, device=self.config.device
        )

        initial_image = self._single_state_sample()    # (1, 3, 128, 256)
        indices = self.vqvae.encode(initial_image)      # (1, 16, 32)
        encoded_state = self.rssm.encoder(indices)      # (1, encoded_state_size)

        action = self._action_tensors['none']

        self.recurrent_state = self.rssm.recurrent_model(
            self.recurrent_state,
            self._zero_latent,
            action
        )

        self.latent_state, _ = self.rssm.representation_model(
            self.recurrent_state,
            encoded_state
        )

        return self._render_current_frame()

    @torch.no_grad()
    def step(self, action_name: str):
        action_tensor = self._action_tensors.get(
            action_name, self._action_tensors['none']
        )

        self.recurrent_state = self.rssm.recurrent_model(
            self.recurrent_state,
            self.latent_state,
            action_tensor
        )

        self.latent_state, _ = self.rssm.transition_model(
            self.recurrent_state
        )

        return self._render_current_frame()

    # ════════════════════════════════════════
    #  Rendering (최적화 핵심)
    # ════════════════════════════════════════

    @torch.no_grad()
    def _render_current_frame(self):
        """GPU decode → BGR numpy 버퍼 반환"""
        predicted_logits = self.rssm.decoder(
            self.recurrent_state,
            self.latent_state
        )  # (1, K, 16, 32)

        predicted_indices = predicted_logits.argmax(dim=-3)

        reconstruction = self.vqvae.decode(predicted_indices)  # (1, 3, H, W)

        # GPU에서 BGR 변환까지 수행 (cv2.cvtColor 생략)
        img = reconstruction[0].clamp_(0, 1).mul_(255).byte()
        # RGB → BGR: channel flip on GPU
        img_bgr = img.flip(0)  # (3, H, W) channel 순서 반전

        # CPU 전송 + numpy 변환
        np.copyto(
            self._img_buffer_bgr,
            img_bgr.permute(1, 2, 0).cpu().numpy()
        )

        return self._img_buffer_bgr

    def _encode_to_base64(self, img_bgr: np.ndarray) -> str:
        """BGR numpy → JPEG base64 (cvtColor 불필요)"""
        _, encoded = cv2.imencode('.jpg', img_bgr, self._jpeg_params)
        b64 = base64.b64encode(encoded.tobytes()).decode('ascii')
        return f"data:image/jpeg;base64,{b64}"

    # ════════════════════════════════════════
    #  Sample Loading
    # ════════════════════════════════════════

    def _single_state_sample(self):
        idx = np.random.randint(0, len(self.sample_images))
        return self._sample_tensors[idx]

    def _load_samples(self):
        samples_dir = "samples/oven_of_witch"
        if not os.path.exists(samples_dir):
            raise FileNotFoundError(
                f"Samples directory not found: {samples_dir}"
            )

        self.sample_images = []
        self._sample_tensors = []

        for filename in sorted(os.listdir(samples_dir)):
            if not filename.startswith("oow_sample") \
               or not filename.endswith(".png"):
                continue

            file_path = os.path.join(samples_dir, filename)
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.sample_images.append(img_rgb)

            # ── 사전 텐서 변환 (매 reset마다 변환 반복 방지) ──
            img_tensor = (
                torch.from_numpy(img_rgb).float().div_(255.0)
                .permute(2, 0, 1).unsqueeze(0)
                .to(self.config.device)
            )
            self._sample_tensors.append(img_tensor)

        print(f"✅ Loaded {len(self.sample_images)} sample images"
              f" (pre-converted to tensors)")