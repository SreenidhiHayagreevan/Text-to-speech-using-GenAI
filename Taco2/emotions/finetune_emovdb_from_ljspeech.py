# finetune_emovdb_from_ljspeech.py
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import Tacotron2, Tacotron2Config
from dataset import TTSDataset, TTSCollator, BatchSampler
from tokenizer import Tokenizer

EMOTION_MAP = {
    "neutral": 0,
    "amused": 1,
    "angry": 2,
    "disgusted": 3,
    "sleepy": 4,
}

def parse_args():
    p = argparse.ArgumentParser()

    # paths
    p.add_argument("--lj_checkpoint", type=str, required=True,
                   help="Path to LJSpeech Tacotron2 weights (pytorch_model.bin)")
    p.add_argument("--train_manifest", type=str, required=True)
    p.add_argument("--val_manifest", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")

    # emotions
    p.add_argument("--num_emotions", type=int, default=5)
    p.add_argument("--emotion_embed_dim", type=int, default=64)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # tokenizer
    tokenizer = Tokenizer()

    # config – match your LJSpeech model hyperparams
    config = Tacotron2Config(
        num_mels=80,
        num_chars=tokenizer.vocab_size,
        character_embed_dim=512,
        pad_token_id=tokenizer.pad_token_id,
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embed_dim=512,
        encoder_dropout_p=0.5,
        decoder_embed_dim=1024,
        decoder_dropout_p=0.1,
        decoder_prenet_dim=256,
        decoder_prenet_depth=2,
        decoder_prenet_dropout_p=0.5,
        decoder_postnet_num_convs=5,
        decoder_postnet_n_filters=512,
        decoder_postnet_kernel_size=5,
        decoder_postnet_dropout_p=0.5,
        attention_dim=128,
        attention_dropout_p=0.1,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        num_emotions=args.num_emotions,
        emotion_embed_dim=args.emotion_embed_dim,
    )

    model = Tacotron2(config).to(device)

    # --- 1) load LJSpeech weights (no emotions) ---
    print("Loading LJSpeech checkpoint:", args.lj_checkpoint)
    state = torch.load(args.lj_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # --- 2) freeze encoder + embedding ---
    for p in model.encoder.embeddings.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False

    # decoder, postnet, stop head, emotion embedding all remain trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameters:", sum(p.numel() for p in trainable_params))

    opt = Adam(trainable_params, lr=args.lr)

    # --- 3) datasets – you can reuse your TTSDataset if it already supports emotions ---
    # Here we assume your emovdb train_metadata.csv has columns:
    # file_path, normalized_text, emotion (string: angry/neutral/...)
    train_set = TTSDataset(
        args.train_manifest,
        # + your existing audio/mel args
    )
    val_set = TTSDataset(
        args.val_manifest,
        # same args
    )

    collator = TTSCollator()
    sampler = BatchSampler(train_set, batch_size=args.batch_size, drop_last=False)

    train_loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collator)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # you will need your batch to also contain emotion IDs
            # e.g. texts, text_lens, mels, stops, enc_mask, dec_mask, emotion_ids
            texts, text_lens, mels, stops, enc_mask, dec_mask, emotion_ids = batch

            texts = texts.to(device)
            mels = mels.to(device)
            stops = stops.to(device)
            enc_mask = enc_mask.to(device)
            dec_mask = dec_mask.to(device)
            emotion_ids = emotion_ids.to(device)    # (B,)

            mel_out, mel_post, stop_pred, _ = model(
                texts, text_lens, mels, enc_mask, dec_mask, emotion_ids=emotion_ids
            )

            mel_loss = F.mse_loss(mel_out, mels)
            rmel_loss = F.mse_loss(mel_post, mels)
            stop_loss = F.binary_cross_entropy_with_logits(
                stop_pred.view(-1, 1), stops.view(-1, 1)
            )
            loss = mel_loss + rmel_loss + stop_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} train loss: {total_loss/len(train_loader):.4f}")

        # simple val
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                texts, text_lens, mels, stops, enc_mask, dec_mask, emotion_ids = batch
                texts = texts.to(device)
                mels = mels.to(device)
                stops = stops.to(device)
                enc_mask = enc_mask.to(device)
                dec_mask = dec_mask.to(device)
                emotion_ids = emotion_ids.to(device)

                mel_out, mel_post, stop_pred, _ = model(
                    texts, text_lens, mels, enc_mask, dec_mask, emotion_ids=emotion_ids
                )
                mel_loss = F.mse_loss(mel_out, mels)
                rmel_loss = F.mse_loss(mel_post, mels)
                stop_loss = F.binary_cross_entropy_with_logits(
                    stop_pred.view(-1, 1), stops.view(-1, 1)
                )
                loss = mel_loss + rmel_loss + stop_loss
                val_loss += loss.item()

            print(f"Epoch {epoch} val loss: {val_loss/len(val_loader):.4f}")

        # save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"tacotron2_emov_epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()

