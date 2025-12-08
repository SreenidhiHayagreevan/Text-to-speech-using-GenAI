import os
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import librosa
from tokenizer import Tokenizer

import numpy as np

def load_wav(path_to_audio, sr=22050):
    audio, orig_sr = torchaudio.load(path_to_audio)

    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sr)

    return audio.squeeze(0)

def load_wav_from_array(audio_array, orig_sr=22050, target_sr=22050):
    """Load audio from numpy array (for HuggingFace datasets)"""
    audio = torch.tensor(audio_array, dtype=torch.float32)
    
    if len(audio.shape) > 1:
        audio = audio.squeeze()
    
    if orig_sr != target_sr:
        audio = audio.unsqueeze(0)  # Add channel dimension
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=target_sr)
        audio = audio.squeeze(0)
    
    return audio

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
                 max_scaled_abs=4,
                 use_emotions=False,
                 emotion_to_id=None):
        """
        Args:
            path_to_metadata: Path to CSV file with columns: file_path, normalized_transcript, emotion (optional)
            use_emotions: Whether to load and return emotion labels
            emotion_to_id: Dictionary mapping emotion strings to integer IDs (will be created if None)
        """
        self.metadata = pd.read_csv(path_to_metadata)
        
        # Handle different column names for transcript
        if 'normalized_transcript' not in self.metadata.columns:
            if 'text' in self.metadata.columns:
                self.metadata['normalized_transcript'] = self.metadata['text']
            elif 'transcript' in self.metadata.columns:
                self.metadata['normalized_transcript'] = self.metadata['transcript']
            else:
                raise ValueError(f"Could not find transcript column. Available columns: {self.metadata.columns.tolist()}")
        
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
        self.use_emotions = use_emotions
        
        # Handle emotion labels
        if self.use_emotions:
            if 'emotion' not in self.metadata.columns:
                raise ValueError("use_emotions=True but 'emotion' column not found in metadata")
            
            # Create emotion to ID mapping if not provided
            if emotion_to_id is None:
                unique_emotions = sorted(self.metadata['emotion'].unique())
                self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
            else:
                self.emotion_to_id = emotion_to_id
            
            self.id_to_emotion = {v: k for k, v in self.emotion_to_id.items()}
            self.num_emotions = len(self.emotion_to_id)
            
            # Map emotions to IDs
            self.metadata['emotion_id'] = self.metadata['emotion'].map(self.emotion_to_id)
            
            # Check for any unmapped emotions
            if self.metadata['emotion_id'].isna().any():
                unmapped = self.metadata[self.metadata['emotion_id'].isna()]['emotion'].unique()
                raise ValueError(f"Found unmapped emotions: {unmapped}")
        else:
            self.emotion_to_id = None
            self.num_emotions = 0

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
        
        # Check if we're using HuggingFace dataset structure (paths starting with 'huggingface_audio')
        # If paths look like actual file paths (contain '/' or end with '.wav'), use file loading
        # Otherwise, check if they're placeholder HuggingFace paths
        if len(self.metadata) > 0:
            file_paths = self.metadata['file_path'].astype(str)
            has_actual_paths = file_paths.str.contains('/').any() or file_paths.str.endswith('.wav').any()
            self.use_hf_dataset = not has_actual_paths and file_paths.str.startswith('huggingface_audio').any()
        else:
            self.use_hf_dataset = False
        
        if self.use_hf_dataset:
            # Load the HuggingFace dataset
            from datasets import load_dataset
            print("Loading HuggingFace dataset for audio access...")
            try:
                hf_ds = load_dataset("CLAPv2/EmoV_DB")
                if isinstance(hf_ds, dict):
                    if 'train' in hf_ds:
                        self.hf_dataset = hf_ds['train']
                    else:
                        self.hf_dataset = list(hf_ds.values())[0]
                else:
                    self.hf_dataset = hf_ds
                
                # Ensure audio feature is properly cast/decoded
                # Cast to Audio feature if not already done
                try:
                    from datasets import Audio
                    # This ensures audio is decoded when accessed
                    if 'audio' in self.hf_dataset.features:
                        print("Audio feature found in dataset")
                    else:
                        print("Warning: 'audio' feature not found in dataset features")
                except Exception as e:
                    print(f"Warning when checking audio feature: {e}")
                
                # Create mapping from index to dataset index
                self.hf_index_map = {}
                for idx in range(len(self.metadata)):
                    path_str = str(self.metadata.iloc[idx]['file_path'])
                    if 'huggingface_audio_' in path_str:
                        try:
                            hf_idx = int(path_str.split('_')[-1])
                        except (ValueError, IndexError):
                            hf_idx = idx
                    else:
                        hf_idx = idx
                    self.hf_index_map[idx] = min(max(hf_idx, 0), len(self.hf_dataset) - 1)
                
                # Test loading one sample to verify it works
                try:
                    test_idx = list(self.hf_index_map.values())[0] if len(self.hf_index_map) > 0 else 0
                    test_sample = self.hf_dataset[test_idx]
                    test_audio = test_sample.get('audio', None)
                    if test_audio is None:
                        raise ValueError("Test sample has no audio data")
                    print(f"Successfully loaded HuggingFace dataset with {len(self.hf_dataset)} samples")
                except Exception as e:
                    print(f"Warning: Could not verify audio loading from HuggingFace dataset: {e}")
                    print("Audio loading will be attempted at runtime")
            except Exception as e:
                print(f"Warning: Could not load HuggingFace dataset: {e}")
                print("Will try to load audio from file paths instead")
                self.use_hf_dataset = False
                self.hf_dataset = None
                self.hf_index_map = None
        else:
            self.hf_dataset = None
            self.hf_index_map = None
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):

        sample = self.metadata.iloc[idx]
        
        path_to_audio = sample["file_path"]
        transcript = sample["normalized_transcript"]

        # Load audio - handle both file paths and HuggingFace dataset
        if self.use_hf_dataset and self.hf_dataset is not None:
            try:
                hf_idx = self.hf_index_map.get(idx, idx)
                if hf_idx >= len(self.hf_dataset):
                    raise IndexError(f"HF index {hf_idx} out of range for dataset size {len(self.hf_dataset)}")
                
                hf_sample = self.hf_dataset[hf_idx]
                audio_data = hf_sample.get('audio', None)
                
                if audio_data is not None:
                    # HuggingFace Audio feature returns a dict with 'array' and 'sampling_rate'
                    # But it might need to be decoded first depending on dataset configuration
                    audio_array = None
                    orig_sr = None
                    
                    # Method 1: Direct dict access (most common)
                    if isinstance(audio_data, dict):
                        audio_array = audio_data.get('array', None)
                        orig_sr = audio_data.get('sampling_rate', None)
                    
                    # Method 2: If it's a path string, decode it
                    if audio_array is None and isinstance(audio_data, str):
                        # Audio feature might return a path, need to decode
                        try:
                            from datasets import Audio
                            decoded = Audio().decode_example(audio_data)
                            audio_array = decoded['array']
                            orig_sr = decoded['sampling_rate']
                        except Exception:
                            # If decode fails, it's probably already decoded or in wrong format
                            pass
                    
                    # Method 3: Try attribute access for Audio objects
                    if audio_array is None:
                        try:
                            audio_array = getattr(audio_data, 'array', None)
                            orig_sr = getattr(audio_data, 'sampling_rate', None)
                        except AttributeError:
                            pass
                    
                    # Method 4: Try __getitem__ access
                    if audio_array is None and hasattr(audio_data, '__getitem__'):
                        try:
                            audio_array = audio_data['array']
                            orig_sr = audio_data['sampling_rate']
                        except (KeyError, TypeError):
                            pass
                    
                    if audio_array is not None and orig_sr is not None:
                        # Convert to numpy if needed
                        if not isinstance(audio_array, np.ndarray):
                            audio_array = np.array(audio_array, dtype=np.float32)
                        audio = load_wav_from_array(audio_array, orig_sr=orig_sr, target_sr=self.sample_rate)
                    else:
                        raise ValueError(f"Could not extract audio array or sampling rate from HuggingFace audio data. "
                                       f"Type: {type(audio_data)}, Keys/attrs: {dir(audio_data) if hasattr(audio_data, '__dict__') else 'N/A'}")
                else:
                    raise ValueError(f"No audio data found in HuggingFace sample at index {hf_idx}")
            except Exception as e:
                # Don't fallback to file loading with placeholder path - raise clear error
                raise RuntimeError(f"Failed to load audio from HuggingFace dataset for metadata index {idx} "
                                 f"(HF dataset index: {self.hf_index_map.get(idx, 'N/A')}): {e}. "
                                 f"Make sure the HuggingFace dataset is properly downloaded and accessible. "
                                 f"You may need to: 1) Check your internet connection, 2) Login with 'huggingface-cli login', "
                                 f"3) Re-run prep_emovdb.py to regenerate metadata.")
        else:
            # Normal file path loading - paths like 'data/emovdb/wavs/train_4282.wav'
            if not os.path.exists(path_to_audio):
                raise FileNotFoundError(f"Audio file not found: {path_to_audio}. "
                                       f"Make sure the path in your metadata CSV is correct.")
            audio = load_wav(path_to_audio, sr=self.sample_rate)

        mel = self.audio_proc.audio2mel(audio, do_norm=True)

        if self.use_emotions:
            emotion_id = int(sample['emotion_id'])
            return transcript, mel.squeeze(0), emotion_id
        else:
            return transcript, mel.squeeze(0)

def TTSCollator(use_emotions=False):

    tokenizer = Tokenizer()

    def _collate_fn(batch):
        
        if use_emotions:
            texts = [tokenizer.encode(b[0]) for b in batch]
            mels = [b[1] for b in batch]
            emotions = [b[2] for b in batch]
        else:
            texts = [tokenizer.encode(b[0]) for b in batch]
            mels = [b[1] for b in batch]
            emotions = None
        
        ### Get Lengths of Texts and Mels ###
        input_lengths = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
        output_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)

        ### Sort by Text Length (as we will be using packed tensors later) ###
        input_lengths, sorted_idx = input_lengths.sort(descending=True)
        texts = [texts[i] for i in sorted_idx]
        mels = [mels[i] for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]
        
        if use_emotions and emotions is not None:
            emotions = [emotions[i] for i in sorted_idx]
            emotions = torch.tensor(emotions, dtype=torch.long)

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

        if use_emotions and emotions is not None:
            return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths), emotions
        else:
            return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths)


    return _collate_fn

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
