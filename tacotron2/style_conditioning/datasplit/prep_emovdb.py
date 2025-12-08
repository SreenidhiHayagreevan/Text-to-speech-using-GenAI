import os
from sklearn.model_selection import train_test_split
import pandas as pd
import librosa
from datasets import load_dataset

def random_split_and_prep_emovdb_metadata(path_to_save="data/", seed=42, test_split_pct=0.1, sort=True):
    """
    Load EmoV_DB dataset from HuggingFace and create train/test splits with emotion labels.
    The dataset structure from CLAPv2/EmoV_DB includes:
    - audio: audio data
    - text: transcript text
    - emotion: emotion label (e.g., 'happy', 'sad', 'angry', 'disgust', 'neutral')
    - speaker: speaker identifier
    """
    
    print("Loading EmoV_DB dataset from HuggingFace...")
    ds = load_dataset("CLAPv2/EmoV_DB")
    
    # Convert to pandas DataFrame for easier manipulation
    data_list = []
    
    # Process each split if available, or use the main dataset
    if isinstance(ds, dict):
        # If dataset has multiple splits, combine all or use 'train'
        if 'train' in ds:
            dataset = ds['train']
        elif len(ds) > 0:
            # Take the first available split or combine all
            dataset = list(ds.values())[0]
        else:
            raise ValueError("No dataset splits found")
    else:
        dataset = ds
    
    print(f"Processing {len(dataset)} samples...")
    
    for idx, sample in enumerate(dataset):
        # Get the audio - HuggingFace datasets stores audio as Audio object
        audio_data = sample.get('audio', None)
        
        # Create a placeholder path that we'll use to map back to the dataset index
        # We'll handle actual audio loading in dataset.py
        audio_path = f"huggingface_audio_{idx}"
        
        text = sample.get('text', '')
        if not text:
            text = sample.get('transcription', '')  # Try alternative field name
        
        emotion = sample.get('emotion', 'neutral')
        if not emotion:
            emotion = sample.get('label', 'neutral')  # Try alternative field name
        
        speaker = sample.get('speaker', 'unknown')
        if not speaker:
            speaker = sample.get('actor', 'unknown')  # Try alternative field name
        
        # Normalize emotion labels (handle variations)
        emotion = str(emotion).lower().strip()
        
        # Skip if no text
        if not text or len(text.strip()) == 0:
            continue
        
        data_list.append({
            'file_path': audio_path,
            'text': text,
            'normalized_transcript': text,  # Using same text for now, can add normalization later
            'emotion': emotion,
            'speaker': str(speaker),
        })
    
    metadata = pd.DataFrame(data_list)
    
    # If we have actual file paths, verify them
    if not metadata['file_path'].str.startswith('huggingface_audio').all():
        verify_func = lambda x: os.path.isfile(x) if isinstance(x, str) else False
        exists = metadata['file_path'].apply(verify_func)
        if not all(list(exists)):
            print(f"Warning: {sum(~exists)} files are missing")
    
    # Get durations - we'll compute from the actual dataset
    print("Computing audio durations...")
    durations = []
    for idx, row in metadata.iterrows():
        hf_idx = int(row['file_path'].split('_')[-1]) if 'huggingface_audio_' in row['file_path'] else idx
        if hf_idx < len(dataset):
            sample = dataset[hf_idx]
            audio_data = sample.get('audio', None)
            if audio_data is not None:
                if isinstance(audio_data, dict):
                    if 'array' in audio_data and 'sampling_rate' in audio_data:
                        duration = len(audio_data['array']) / audio_data['sampling_rate']
                    else:
                        duration = 0.0
                else:
                    # Audio might be a different format
                    duration = 0.0
            else:
                duration = 0.0
        else:
            duration = 0.0
        durations.append(duration)
    
    metadata['duration'] = durations
    
    # Filter out empty transcripts
    metadata = metadata[metadata['normalized_transcript'].notna()].reset_index(drop=True)
    metadata = metadata[metadata['normalized_transcript'].str.len() > 0].reset_index(drop=True)
    
    print(f"Total samples after filtering: {len(metadata)}")
    print(f"Emotions distribution:\n{metadata['emotion'].value_counts()}")
    
    # Random Split
    train_df, test_df = train_test_split(metadata, test_size=test_split_pct, random_state=seed, stratify=metadata['emotion'])
    
    # Sort from longest to shortest (so Padding is minimized)
    if sort:
        train_df = train_df.sort_values(by=["duration"], ascending=False)
        test_df = test_df.sort_values(by=["duration"], ascending=False)
    
    # Create save directory if it doesn't exist
    os.makedirs(path_to_save, exist_ok=True)
    
    train_df.to_csv(os.path.join(path_to_save, "train_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(path_to_save, "test_metadata.csv"), index=False)
    
    print(f"Saved train metadata with {len(train_df)} samples")
    print(f"Saved test metadata with {len(test_df)} samples")
    print("Done!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_save", type=str, default="data/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split_pct", type=float, default=0.1)

    args = parser.parse_args()

    random_split_and_prep_emovdb_metadata(args.path_to_save, args.seed, args.test_split_pct)

