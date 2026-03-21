import torch
import numpy as np
import os
import cv2
import time

class Wrapper:
    def __init__(self, config, vqvae, rssm):
        self.config = config
        self.vqvae = vqvae
        self.rssm = rssm

        self.action_map = {
            "none": 0, 
            "jump": 1, 
            "slide": 2
        }

        self._load_samples()

        self._action_tensors = {
            name: self._create_action_tensor(idx)
            for name, idx in self.action_map.items()
        }
        self._zero_latent = torch.zeros(
            1, config.latent_size, device=config.device
        )
        self._jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 92]

        # recording
        self.recording_dir = getattr(config, "recording_dir", "recordings")
        os.makedirs(self.recording_dir, exist_ok=True)
        self._video_writer = None
        self._frame_count = 0
        self._record_start = None


        self.reset()
        print("Game state initialized")


    def _start_recording(self, frame):
        h, w = frame.shape[:2]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.recording_dir, f"gameplay_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = self.config.video_fps
        self._video_writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        self._frame_count = 0
        self._record_start = time.monotonic()
        self._record_filename = filename
        print(f"🔴 Recording started: {filename}")

    def _save_recording(self):
        if self._video_writer is None:
            return

        self._video_writer.release()
        self._video_writer = None

        elapsed = time.monotonic() - self._record_start
        print(
            f"⬜ Recording saved: {self._record_filename} "
            f"({self._frame_count} frames, {elapsed:.1f}s)"
        )
        self._frame_count = 0
        self._record_start = None


    def _record_frame(self, img_rgb):
        if self._video_writer is None:
            return
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self._video_writer.write(img_bgr)
        self._frame_count += 1


    def _create_action_tensor(self, action_idx):
        action = torch.zeros(1, self.config.action_size, device=self.config.device)
        action[0, action_idx] = 1.0
        return action


    @torch.inference_mode()
    def reset(self):
        self._save_recording()

        self.recurrent_state = torch.zeros(
            1, self.config.recurrent_size, device=self.config.device
        )

        initial_img = self._random_sample_tensor()
        initial_indices = self.vqvae.encode(initial_img)
        encoded_state = self.rssm.encoder(initial_indices)

        self.recurrent_state = self.rssm.recurrent_model(
            self.recurrent_state,
            self._zero_latent,
            self._action_tensors["none"],
        )
        self.latent_state, _ = self.rssm.representation_model(
            self.recurrent_state, encoded_state
        )
        img = self._render()

        # 새 녹화 시작
        self._start_recording(img)
        self._record_frame(img)

        return img


    @torch.inference_mode()
    def step(self, action_name: str):
        action_tensor = self._action_tensors.get(
            action_name, self._action_tensors["none"]
        )
        self.recurrent_state = self.rssm.recurrent_model(
            self.recurrent_state, self.latent_state, action_tensor
        )
        self.latent_state, _ = self.rssm.transition_model(self.recurrent_state)
        img = self._render()

        self._record_frame(img)

        return img


    @torch.inference_mode()
    def _render(self):
        token_logits = self.rssm.decoder(self.recurrent_state, self.latent_state)
        token_indices = token_logits.argmax(dim=-3)
        recon = self.vqvae.decode(token_indices)

        img = recon[0].clamp_(0, 1).mul_(255).byte().cpu().numpy()
        return np.ascontiguousarray(img.transpose(1, 2, 0)) 


    def reset_to_bytes(self) -> bytes:
        return self._encode_jpeg(self.reset())


    def step_to_bytes(self, action_name: str) -> bytes:
        return self._encode_jpeg(self.step(action_name))


    def _encode_jpeg(self, img_rgb: np.ndarray) -> bytes:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", img_bgr, self._jpeg_params)
        return buf.tobytes()


    def _random_sample_tensor(self):
        idx = np.random.randint(0, len(self.sample_images))
        img = self.sample_images[idx]
        t = torch.from_numpy(img).float().div_(255.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.config.device)


    def _load_samples(self):
        samples_dir = "samples/oven_of_witch"
        if not os.path.exists(samples_dir):
            raise FileNotFoundError(f"Not found: {samples_dir}")


        self.sample_images = []
        for fn in sorted(os.listdir(samples_dir)):
            if fn.startswith("oow_sample") and fn.endswith(".png"):
                img = cv2.imread(os.path.join(samples_dir, fn))
                self.sample_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        print(f"✅ Loaded {len(self.sample_images)} sample images")

    def __del__(self):
        self._save_recording()