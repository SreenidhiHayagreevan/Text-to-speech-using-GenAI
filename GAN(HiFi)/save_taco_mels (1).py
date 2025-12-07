import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tacotron2 import Tacotron2, Tacotron2Config
from dataset import TTSDataset, TTSCollator

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_manifest", nargs="+", type=str, required=True)
    parser.add_argument("--path_to_save", type=str, required=True)
    parser.add_argument("--taco_weights", type=str, required=True)

    ### Inference Config ###
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)

    ### DATASET CONFIG ###
    parser.add_argument("--sampling_rate", type=int, default=22050)
    parser.add_argument("--segment_size", type=int, default=8192)
    parser.add_argument("--num_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--min_db", type=float, default=-100.)
    parser.add_argument("--max_scaled_abs", type=float, default=4.)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction)

    return parser.parse_args()

def gen_mels(path_to_manifest,
             path_to_save):

    ### Load Model ###
    model = Tacotron2(config=Tacotron2Config())
    state_dict = torch.load(args.taco_weights)
    success = model.load_state_dict(state_dict)
    model.eval()
    model = model.to(args.device)
    print(success)

    ### Make Directory to Save Mel Spectrograms ###
    os.makedirs(path_to_save, exist_ok=True)

    for path in path_to_manifest:
        print("Processing: ", path)

        dataset = TTSDataset(path, 
                             sample_rate=args.sampling_rate, 
                             n_fft=args.n_fft, 
                             window_size=args.window_size, 
                             hop_size=args.hop_size, 
                             fmin=args.fmin, 
                             fmax=args.fmax, 
                             num_mels=args.num_mels, 
                             min_db=args.min_db, 
                             max_scaled_abs=args.max_scaled_abs)

        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=TTSCollator(), num_workers=args.workers)

        for text, input_lengths, mel, stops, encoder_mask, decoder_mask, paths in tqdm(loader):

            path_to_saves = [os.path.join(path_to_save, f"{file.split("/")[-1].split(".")[0]}.npy") for file in paths]
            
            with torch.no_grad():
                mels_out, mels_postnet_out, stop_preds, alignment = model(
                    text.to(args.device), input_lengths.to("cpu"), mel.to(args.device), encoder_mask.to(args.device), decoder_mask.to(args.device)
                )

            
            output_lens = (~decoder_mask).sum(axis=-1)
            
            for mel, len, path in zip(mels_postnet_out, output_lens, path_to_saves):
                
                ### Convert to Numpy ##
                mel = mel.T.cpu().numpy()

                ### Clip to Valid Length ###
                mel = mel[:, :len]
                
                ### Save ###
                np.save(path, mel)               


if __name__ == "__main__":

    args = parse_args()

    gen_mels(args.path_to_manifest, args.path_to_save)
    


