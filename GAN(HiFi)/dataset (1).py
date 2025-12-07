import os
import random
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import librosa
import numpy as np

from tokenizer import Tokenizer

def load_wav(path_to_audio, sr=22050):
    audio, orig_sr = torchaudio.load(path_to_audio)

    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sr)

    return audio.squeeze(0)

def amp_to_db(x, min_db=-100):
    ### Forces min DB to be -100
    ### 20 * torch.log10(1e-5) = 20 * -5 = -100
    clip_val = 10 ** (min_db / 20)
    return 20 * torch.log10(torch.clamp(x, min=clip_val))

def db_to_amp(x):
    return 10 ** (x / 20)

def normalize(x, 
              min_db=-100., 
              max_abs_val=4):

    x = (x - min_db) / -min_db
    x = 2 * max_abs_val * x - max_abs_val
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    
    return x

def denormalize(x, 
                min_db=-100, 
                max_abs_val=4):
    
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val)
    x = x * -min_db + min_db

    return x

def pad_for_mel(x, n_fft, hop_size):
    pad = ((n_fft - hop_size) // 2)
    return F.pad(x, (pad, pad), mode="reflect")

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
        mel = librosa.filters.mel(sr=self.sampling_rate, 
                                  n_fft=self.n_fft, 
                                  n_mels=self.num_mels, 
                                  fmin=self.fmin, 
                                  fmax=self.fmax)
        return torch.from_numpy(mel)
    
    def audio2mel(self, audio, do_norm=False):

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spectrogram = torch.stft(input=audio, 
                                 n_fft=self.n_fft, 
                                 hop_length=self.hop_size, 
                                 win_length=self.window_size, 
                                 window=torch.hann_window(self.window_size).to(audio.device), 
                                 center=self.center, 
                                 pad_mode="reflect", 
                                 normalized=False, 
                                 onesided=True,
                                 return_complex=True)
        
        spectrogram = torch.abs(spectrogram)
        
        mel = torch.matmul(self.spec2mel.to(spectrogram.device), spectrogram)

        mel = amp_to_db(mel, self.min_db)
        
        if do_norm:
            mel = normalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        return mel
    
    def mel2audio(self, mel, do_denorm=False, griffin_lim_iters=60):

        if do_denorm:
            mel = denormalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        mel = db_to_amp(mel)

        spectrogram = torch.matmul(self.mel2spec.to(mel.device), mel).cpu().numpy()

        audio = librosa.griffinlim(S=spectrogram, 
                                   n_iter=griffin_lim_iters, 
                                   hop_length=self.hop_size, 
                                   win_length=self.window_size, 
                                   n_fft=self.n_fft,
                                   window="hann")

        audio *= 32767 / max(0.01, np.max(np.abs(audio)))
        
        audio = audio.astype(np.int16)

        return audio

def build_padding_mask(lengths):

    B = lengths.size(0)
    T = torch.max(lengths).item()

    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i]:] = 1

    return mask.bool()

class TTSDataset(Dataset):

    """
    Dataset to train Tacotron2
    """
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
                 normalized=False, 
                 min_db=-100, 
                 max_scaled_abs=4):
        
        self.metadata = pd.read_csv(path_to_metadata)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax 
        self.num_mels = num_mels
        self.center = center
        self.normalized = normalized
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.transcript_lengths = [len(Tokenizer().encode(t)) for t in self.metadata["normalized_transcript"]]

        self.audio_proc = AudioMelConversions(num_mels=self.num_mels, 
                                              sampling_rate=self.sample_rate, 
                                              n_fft=self.n_fft, 
                                              window_size=self.win_size, 
                                              hop_size=self.hop_size, 
                                              fmin=self.fmin, 
                                              fmax=self.fmax, 
                                              center=self.center,
                                              min_db=self.min_db, 
                                              max_scaled_abs=self.max_scaled_abs)
        
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx]
        
        path_to_audio = sample["file_path"]
        transcript = sample["normalized_transcript"]

        audio = load_wav(path_to_audio, sr=self.sample_rate)

        mel = self.audio_proc.audio2mel(audio, do_norm=True)

        # return transcript, mel.squeeze(0) # Change so we get the filename as well
        return transcript, mel.squeeze(0), path_to_audio

def TTSCollator():

    tokenizer = Tokenizer()

    def _collate_fn(batch):
        
        texts = [tokenizer.encode(b[0]) for b in batch]
        mels = [b[1] for b in batch]
        paths = [b[2] for b in batch]
        
        ### Get Lengths of Texts and Mels ###
        input_lengths = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
        output_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)

        ### Sort by Text Length (as we will be using packed tensors later) ###
        input_lengths, sorted_idx = input_lengths.sort(descending=True)
        texts = [texts[i] for i in sorted_idx]
        mels = [mels[i] for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]

        ### SORT PATHS AS WELL ###
        paths = [paths[i] for i in sorted_idx]

        ### Pad Text ###
        text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

        ### Pad Mel Sequences ###
        max_target_len = max(output_lengths).item()
        num_mels = mels[0].shape[0]
        
        ### Get gate which tells when to stop decoding. 0 is keep decoding, 1 is stop ###
        mel_padded = torch.zeros((len(mels), num_mels, max_target_len))
        gate_padded = torch.zeros((len(mels), max_target_len))

        for i, mel in enumerate(mels):
            t = mel.shape[1]
            mel_padded[i, :, :t] = mel
            gate_padded[i, t-1:] = 1
        
        mel_padded = mel_padded.transpose(1,2)

        return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths), paths # Return paths as well

    return _collate_fn

class MelDataset(Dataset):
    """
    Dataset to train HIFIGAN
    """
    def __init__(self, 
                 path_to_manifest, 
                 segment_size=8192, 
                 n_fft=1024, 
                 num_mels=80,
                 hop_size=256, 
                 window_size=1024, 
                 sampling_rate=22050,  
                 fmin=0, 
                 fmax=8000, 
                 fmax_loss=None,
                 max_audio_magnitude=0.95, 
                 min_db=-100, 
                 max_scaled_abs=4,
                 finetuning=False,
                 path_to_saved_mels=None):

        self.audio_files = list(pd.read_csv(path_to_manifest)["file_path"])

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = window_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.max_audio_magnitude = max_audio_magnitude
        self.finetuning = finetuning
        self.path_to_saved_mels = path_to_saved_mels

        if self.finetuning and self.path_to_saved_mels is None:
            raise ValueError("When finetuning provide Teacher Forcing Generations as .npy files")

        common_args = dict(
            num_mels=num_mels, 
            sampling_rate=sampling_rate,
            n_fft=n_fft, 
            window_size=window_size, 
            hop_size=hop_size, 
            fmin=fmin, 
            center=False,
            min_db=min_db, 
            max_scaled_abs=max_scaled_abs
        )

        self.audio_mel_conv = AudioMelConversions(
            **common_args,
            fmax=fmax,
        )

        self.audio_mel_conv_loss = AudioMelConversions(
            **common_args,
            fmax=None,
        )
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):

        filename = self.audio_files[index]

        audio, sampling_rate = torchaudio.load(filename)
        
        audio = audio / torch.max(audio) * self.max_audio_magnitude

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        if not self.finetuning:

            if audio.shape[1] >= self.segment_size:
                max_audio_start = audio.shape[1] - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start+self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.shape(1)), 'constant')

            audio_padded_for_mel = pad_for_mel(audio, self.n_fft, self.hop_size)

            mel = self.audio_mel_conv.audio2mel(audio_padded_for_mel, do_norm=True)
            mel_loss = self.audio_mel_conv_loss.audio2mel(audio_padded_for_mel, do_norm=True)

            
            return (mel.squeeze(0), audio, mel_loss.squeeze(0))

        else:
            
            path_to_mel = os.path.join(self.path_to_saved_mels, f"{filename.split("/")[-1].split(".")[0]}.npy")
            
            mel = torch.from_numpy(np.load(path_to_mel))

            frames_per_mel_segment = self.segment_size // self.hop_size
     
            if audio.shape[1] >= self.segment_size:
                
                ### Grab Random Chunk of the Taco Generated Mel and its Cooresponding Audio ###
                mel_start = random.randint(0, mel.shape[1] - frames_per_mel_segment - 1)
                mel = mel[:, mel_start:mel_start + frames_per_mel_segment]
                audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_mel_segment)*self.hop_size]

            else:
                mel = torch.nn.functional.pad(mel, (0, frames_per_mel_segment - mel.shape[1]), 'constant')
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.shape[1]), 'constant')
            
            ### Grab True Mel from Audio ###
            audio_padded_for_mel = pad_for_mel(audio, self.n_fft, self.hop_size)
            mel_loss = self.audio_mel_conv_loss.audio2mel(audio_padded_for_mel, do_norm=True)
            
            return mel, audio, mel_loss.squeeze(0)
            

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = MelDataset(path_to_manifest="data/train_metadata.csv", finetuning=True, path_to_saved_mels="data/taco_gen_mels")

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for a, b, c in tqdm(loader):

        pass