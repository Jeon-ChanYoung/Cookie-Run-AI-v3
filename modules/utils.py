import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#################### Dataset ####################

class FrameDataset(Dataset):
    def __init__(self, data):
        self.states = np.array(data["states"])  # (N, 128, 256, 3)
        print(f"FrameDataset: {len(self)} frames")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        frame = torch.from_numpy(
            self.states[idx].copy()
        ).permute(2, 0, 1).float().div_(255.0)  # (3, 128, 256)
        return frame


class IndiceSequenceDataset(Dataset):
    def __init__(self, data, vq_indices, seq_len):
        self.indices = vq_indices       # (N, 8, 16) LongTensor
        self.actions = torch.from_numpy(np.array(data["actions"])).long()
        self.episodes = data["boundaries"].tolist()
        self.seq_len = seq_len

        self.valid_starts = []
        for ep in self.episodes:
            L = ep['end'] - ep['start'] + 1
            if L < seq_len:
                continue
            for s in range(ep['start'], ep['end'] - seq_len + 2):
                self.valid_starts.append(s)
        self.valid_starts = np.array(self.valid_starts)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end   = start + self.seq_len

        return (
            self.indices[start:end].clone(),  # (T, 8, 16)Long
            F.one_hot(self.actions[start:end], num_classes=3).float()
        )


# for training RSSM
# Convert images to indices in advance to improve training speed
@torch.no_grad()
def precompute_vq_indices(vqvae, frame_dataset, batch_size=256):
    vqvae.change_train_mode(train=False)
    loader = DataLoader(frame_dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False,
                        num_workers=2, pin_memory=True)
    all_indices = []
    for frames in loader:
        frames = frames.to("cuda", non_blocking=True)
        indices = vqvae.encode(frames)  # (B, 8, 16) LongTensor
        all_indices.append(indices.cpu())
    result = torch.cat(all_indices, dim=0)  # (N, 8, 16)
    print(f"Precomputed {result.shape[0]} frames -> indices {result.shape}")
    return result


def straight_through_categorical(logits):
    gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-8)) + 1e-8)
    index = (logits + gumbel).argmax(dim=-1)
    hard = F.one_hot(index, logits.shape[-1]).float()
    probs = F.softmax(logits, dim=-1)
    return hard - probs.detach() + probs