import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from tokenizer import Tokenizer
import numpy as np
import librosa


############################################
# Audio utilities (kept same as your code)
############################################

def load_wav(path_to_audio, sr=22050):
    audio, orig_sr = torchaudio.load(path_to_audio)
    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sr)
    return audio.squeeze(0)


def amp_to_db(x, min_db=-100):
    clip_val = 10 ** (min_db / 20)
    return 20 * torch.log10(torch.clamp(x, min=clip_val))


def db_to_amp(x):
    return 10 ** (x / 20)


def normalize(x, min_db=-100., max_abs_val=4):
    x = (x - min_db) / -min_db
    x = 2 * max_abs_val * x - max_abs_val
    return torch.clip(x, min=-max_abs_val, max=max_abs_val)


def denormalize(x, min_db=-100, max_abs_val=4):
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val)
    return x * -min_db + min_db


############################################
# Mel converter
############################################

class AudioMelConversions:
    def __init__(self,
                 num_mels=80,
                 sampling_rate=22050,
                 n_fft=1024,
                 window_size=1024,
                 hop_size=256,
                 fmin=0,
                 fmax=8000,
                 center=False,
                 min_db=-100,
                 max_scaled_abs=4):
        
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.spec2mel = self._get_spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)

    def _get_spec2mel_proj(self):
        mel = librosa.filters.mel(
            sr=self.sampling_rate, 
            n_fft=self.n_fft, 
            n_mels=self.num_mels, 
            fmin=self.fmin, 
            fmax=self.fmax
        )
        return torch.from_numpy(mel)
    
    def audio2mel(self, audio, do_norm=True):

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spectrogram = torch.stft(
            input=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=torch.hann_window(self.window_size).to(audio.device),
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        spectrogram = torch.abs(spectrogram)
        
        mel = torch.matmul(self.spec2mel.to(spectrogram.device), spectrogram)

        mel = amp_to_db(mel, self.min_db)
        
        if do_norm:
            mel = normalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        return mel


############################################
# Mask for padding
############################################

def build_padding_mask(lengths):
    B = lengths.size(0)
    T = torch.max(lengths).item()
    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i]:] = 1
    return mask.bool()


############################################
# === EMOTION MAPPING (IMPORTANT) ===
############################################

EMOTION_MAP = {
    "Neutral": 0,
    "Amused": 1,
    "Angry": 2,
    "Disgusted": 3,
    "Sleepy": 4,
}


############################################
# === FIXED TTSDataset FOR EMOTION ===
############################################

class TTSDataset(Dataset):
    def __init__(self, 
                 path_to_metadata,
                 sample_rate=22050,
                 n_fft=1024, 
                 window_size=1024, 
                 hop_size=256, 
                 fmin=0,
                 fmax=8000, 
                 num_mels=80, 
                 center=False, 
                 min_db=-100, 
                 max_scaled_abs=4):
        
        self.metadata = pd.read_csv(path_to_metadata)

        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        # FIX: Transcript column for EmoV-DB
        self.transcripts = self.metadata["transcript"].tolist()
        self.paths = self.metadata["file_path"].tolist()

        # FIX: Map emotion strings → IDs
        self.emotion_ids = [EMOTION_MAP[e] for e in self.metadata["emotion"]]

        self.tokenizer = Tokenizer()
        self.audio_proc = AudioMelConversions(
            num_mels=num_mels,
            sampling_rate=sample_rate,
            n_fft=n_fft,
            window_size=window_size,
            hop_size=hop_size,
            fmin=fmin,
            fmax=fmax,
            center=center,
            min_db=min_db,
            max_scaled_abs=max_scaled_abs,
        )
        
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        transcript = self.transcripts[idx]
        emotion_id = self.emotion_ids[idx]
        path = self.paths[idx]

        audio = load_wav(path, sr=self.sample_rate)
        mel = self.audio_proc.audio2mel(audio, do_norm=True)  # (num_mels, T)

        return transcript, mel, emotion_id



############################################
# === FIXED COLLLATOR: NOW RETURNS emotion_ids ===
############################################

class TTSCollator:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def __call__(self, batch):
        # batch = [(text, mel, emotion_id), ...]

        texts = [self.tokenizer.encode(b[0]) for b in batch]
        mels = [b[1] for b in batch]
        emotion_ids = torch.tensor([b[2] for b in batch], dtype=torch.long)  # NEW

        # Lengths
        text_lens = torch.tensor([len(t) for t in texts], dtype=torch.long)
        mel_lens = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)

        # Sort by text length DESC
        text_lens, sort_idx = text_lens.sort(descending=True)
        texts = [texts[i] for i in sort_idx]
        mels = [mels[i] for i in sort_idx]
        mel_lens = mel_lens[sort_idx]
        emotion_ids = emotion_ids[sort_idx]  # match order

        # Pad text
        text_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(t) for t in texts],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        # Pad mel
        B = len(mels)
        max_T = mel_lens.max().item()
        num_mels = mels[0].shape[0]

        mel_padded = torch.zeros(B, max_T, num_mels)
        stop_tokens = torch.zeros(B, max_T)

        for i, mel in enumerate(mels):
            T = mel.shape[1]
            mel_padded[i, :T, :] = mel.transpose(0,1)
            stop_tokens[i, T-1:] = 1  # 1 = stop

        mel_padded = mel_padded  # (B, T, num_mels)

        enc_mask = build_padding_mask(text_lens)
        dec_mask = build_padding_mask(mel_lens)

        ############################################
        # FINAL OUTPUT → EXACTLY 7 items expected
        ############################################
        return (
            text_padded,      # (B, T_text)
            text_lens,        # (B,)
            mel_padded,       # (B, T_mel, num_mels)
            stop_tokens,      # (B, T_mel)
            enc_mask,         # (B, T_text)
            dec_mask,         # (B, T_mel)
            emotion_ids       # (B,)
        )


############################################
# === BatchSampler (unchanged) ===
############################################

class BatchSampler:
    def __init__(self, dataset, batch_size, drop_last=False):
        self.sampler = torch.utils.data.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self._make_batches()

    def _make_batches(self):

        indices = [i for i in self.sampler]

        if self.drop_last:
            total_size = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:total_size]

        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        random_indices = torch.randperm(len(batches))
        return [batches[i] for i in random_indices]
    
    def __iter__(self):
        for batch in self.random_batches:
            yield batch

    def __len__(self):
        return len(self.random_batches)
