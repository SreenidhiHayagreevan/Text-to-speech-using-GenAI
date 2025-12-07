import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import set_seed
import accelerate
from accelerate import Accelerator
from safetensors.torch import load_file

from dataset import MelDataset, AudioMelConversions, pad_for_mel
from model import HIFIGAN, HIFIGANConfig
from loss import feature_loss, generator_loss, discriminator_loss

def parse_args():

    parser = argparse.ArgumentParser()

    ### SETUP CONFIG ###
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--working_directory", type=str, required=True)
    parser.add_argument("--path_to_train_manifest", type=str, required=True)
    parser.add_argument("--path_to_val_manifest", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    ### TRAINING CONFIG ###
    parser.add_argument("--training_epochs", type=int, default=3100)
    parser.add_argument("--console_out_iters", type=int, default=5)
    parser.add_argument("--wandb_log_iters", type=int, default=5)
    parser.add_argument("--checkpoint_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.8)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--lr_decay", type=float, default=0.999)
    parser.add_argument("--lambda_mel", type=float, default=45.)
    parser.add_argument("--lambda_feature_mapping", type=float, default=2.)

    ### FINETUNING CONFIG ###
    parser.add_argument("--finetune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--path_to_saved_mels", type=str, default=None)
    parser.add_argument("--path_to_pretrained_weights", type=str, default=None)

    ### MODEL CONFIG ###
    parser.add_argument("--upsample_rates", type=int, nargs='+', default=(8, 8, 2, 2))
    parser.add_argument("--upsample_kernel_sizes", type=int, nargs='+', default=(16, 16, 4, 4))
    parser.add_argument("--upsample_initial_channel", type=int, default=512)
    parser.add_argument("--resblock_kernel_sizes", type=int, nargs='+', default=(3, 7, 11))
    parser.add_argument(
        "--resblock_dilation_sizes",
        type=eval,
        default="((1, 3, 5), (1, 3, 5), (1, 3, 5))",
        help="Nested tuple, e.g., '((1,3,5), (1,3,5), (1,3,5))'"
    )
    parser.add_argument("--mpd_periods", type=int, nargs='+', default=(2, 3, 5, 7, 11))
    parser.add_argument("--msd_num_downsamples", type=int, default=2)

    ### DATASET CONFIG ###
    parser.add_argument("--sampling_rate", type=int, default=22050)
    parser.add_argument("--segment_size", type=int, default=8192)
    parser.add_argument("--num_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--fmax_loss", type=int, default=None)
    parser.add_argument("--min_db", type=float, default=-100.)
    parser.add_argument("--max_scaled_abs", type=float, default=4.)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction)

    return parser.parse_args()

### Parser Arguments ###
args = parse_args()

### Set Seed ###
if args.seed is not None:
    set_seed(args.seed)

### Init Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb" if args.log_wandb else None)

if args.log_wandb:
    accelerator.init_trackers(args.experiment_name)

### Load Model ###
config = HIFIGANConfig(upsample_rates=args.upsample_rates, 
                       upsample_kernel_sizes=args.upsample_kernel_sizes, 
                       upsample_initial_channel=args.upsample_initial_channel,
                       resblock_kernel_sizes=args.resblock_kernel_sizes, 
                       resblock_dilation_sizes=args.resblock_dilation_sizes, 
                       mpd_periods=args.mpd_periods, 
                       msd_num_downsamples=args.msd_num_downsamples, 
                       num_mels=args.num_mels)

model = HIFIGAN(config)

### Load Optimizer ###
optim_g = torch.optim.AdamW(model._get_generator_params(), lr=args.learning_rate, betas=[args.beta1, args.beta2])
optim_d = torch.optim.AdamW(model._get_discriminator_params(), lr=args.learning_rate, betas=[args.beta1, args.beta2])

### Load Scheduler ###
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=args.lr_decay, last_epoch=-1)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=args.lr_decay, last_epoch=-1)

### Load Datasets ###
trainset = MelDataset(path_to_manifest=args.path_to_train_manifest, 
                      segment_size=args.segment_size,
                      n_fft=args.n_fft,
                      num_mels=args.num_mels,
                      hop_size=args.hop_size,
                      window_size=args.window_size,
                      sampling_rate=args.sampling_rate,
                      fmin=args.fmin,
                      fmax=args.fmax,
                      fmax_loss=args.fmax_loss, 
                      min_db=args.min_db, 
                      max_scaled_abs=args.max_scaled_abs, 
                      finetuning=args.finetune, 
                      path_to_saved_mels=args.path_to_saved_mels)

testset = MelDataset(path_to_manifest=args.path_to_val_manifest, 
                     segment_size=args.segment_size,
                     n_fft=args.n_fft,
                     num_mels=args.num_mels,
                     hop_size=args.hop_size,
                     window_size=args.window_size,
                     sampling_rate=args.sampling_rate,
                     fmin=args.fmin,
                     fmax=args.fmax,
                     fmax_loss=args.fmax_loss,
                     min_db=args.min_db, 
                     max_scaled_abs=args.max_scaled_abs, 
                     finetuning=args.finetune,
                     path_to_saved_mels=args.path_to_saved_mels)

trainloader = DataLoader(trainset, 
                         batch_size=16, 
                         shuffle=True, 
                         pin_memory=True)

testloader = DataLoader(testset, 
                        batch_size=16, 
                        shuffle=False, 
                        pin_memory=True)

### Load AudioMelConversions ###
audio_mel_conv = AudioMelConversions(num_mels=args.num_mels, 
                                     sampling_rate=args.sampling_rate, 
                                     n_fft=args.n_fft, 
                                     window_size=args.window_size, 
                                     hop_size=args.hop_size, 
                                     fmin=args.fmin, 
                                     fmax=args.fmax_loss,
                                     min_db=args.min_db, 
                                     max_scaled_abs=args.max_scaled_abs)
### Prepare Everything ###
model, optim_d, optim_g, trainloader, testloader, scheduler_d, scheduler_g = accelerator.prepare(
    model, optim_d, optim_g, trainloader, testloader, scheduler_d, scheduler_g
)

## Load Pretrained Weights if Finetuning ###
if args.finetune:
    accelerator.print("Loading Pretrained HIFIGAN for Finetuning")
    accelerate.load_checkpoint_in_model(model=accelerator.unwrap_model(model),
                                        checkpoint=args.path_to_pretrained_weights)

### Load Checkpoint (will overwrite previous weights from finetuning if resuming) ###
if args.resume_from_checkpoint is not None:

    ### Grab path to checkpoint ###
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)

    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)
    
    ### Start completed steps from checkpoint index ###
    completed_epochs = int(args.resume_from_checkpoint.split("_")[-1])
    completed_steps = completed_epochs * len(trainloader)
    accelerator.print(f"Resuming from Epoch: {completed_epochs}")
else:
    completed_epochs = 0
    completed_steps = 0

### Train Model ###
for epoch in range(completed_epochs, args.training_epochs):
    
    accelerator.print(f"Epoch: {epoch}")

    for mel, audio, mel_target in trainloader:
        
        ### Move to Correct Device ###
        mel = mel.to(accelerator.device)
        audio = audio.to(accelerator.device)
        mel_target = mel_target.to(accelerator.device)

        ### Generate Audio from Mel ###        
        gen_audio = accelerator.unwrap_model(model).generator(mel)
  
        ### Compute Mel for Generated Audio ###
        gen_audio_padded = pad_for_mel(gen_audio, n_fft=args.n_fft, hop_size=args.hop_size)
        gen_audio_mel = audio_mel_conv.audio2mel(gen_audio_padded.squeeze(1), do_norm=True)

        ### Update Discriminator ###
        optim_d.zero_grad()

        mpd_real_outs, mpd_gen_outs, _, _ = accelerator.unwrap_model(model).mpd(audio, gen_audio.detach())
        mpd_disc_loss = discriminator_loss(mpd_real_outs, mpd_gen_outs) 

        msd_real_outs, msd_gen_outs, _, _ = accelerator.unwrap_model(model).msd(audio, gen_audio.detach())
        msd_disc_loss = discriminator_loss(msd_real_outs, msd_gen_outs)

        total_disc_loss = mpd_disc_loss + msd_disc_loss

        accelerator.backward(total_disc_loss)
        optim_d.step()

        ### Update Generator ###
        optim_g.zero_grad()
        loss_mel = F.l1_loss(mel_target, gen_audio_mel) * args.lambda_mel

        mpd_real_outs, mpd_gen_outs, mpd_real_feat_maps, mpd_gen_feat_maps = accelerator.unwrap_model(model).mpd(audio, gen_audio)
        msd_real_outs, msd_gen_outs, msd_real_feat_maps, msd_gen_feat_maps = accelerator.unwrap_model(model).msd(audio, gen_audio)

        mpd_feature_loss = feature_loss(mpd_real_feat_maps, mpd_gen_feat_maps, lambda_features=args.lambda_feature_mapping)
        msd_feature_loss = feature_loss(msd_real_feat_maps, msd_gen_feat_maps, lambda_features=args.lambda_feature_mapping)

        mpd_gen_loss = generator_loss(mpd_gen_outs)
        msd_gen_loss = generator_loss(msd_gen_outs)

        total_gen_loss = msd_gen_loss + mpd_gen_loss + msd_feature_loss + mpd_feature_loss + loss_mel

        accelerator.backward(total_gen_loss)
        optim_g.step()

        if completed_steps % args.console_out_iters == 0:
            accelerator.print(f"Completed Steps: {completed_steps} | Gen Loss: {round(total_gen_loss.item(), 4)} | Disc Loss: {round(total_disc_loss.item(), 4)} | Mel Loss: {round(loss_mel.item()/args.lambda_mel, 4)}")

        if completed_steps % args.wandb_log_iters == 0:
            
            if args.log_wandb:
                accelerator.log(
                    {
                        "gen_loss": total_gen_loss.item(), 
                        "disc_loss": total_disc_loss.item(), 
                        "mel_loss": loss_mel.item()/args.lambda_mel
                    }, 
                    step=completed_steps
                )

        completed_steps += 1

    accelerator.print("EVALUATING")
    model.eval()

    val_mel_loss = 0
    num_losses = 0

    for mel, audio, mel_target in testloader:

        ### Move to Correct Device ###
        mel = mel.to(accelerator.device)
        audio = audio.to(accelerator.device)
        mel_target = mel_target.to(accelerator.device)

        ### Generate Audio from Mel ###
        with torch.no_grad():
            gen_audio = accelerator.unwrap_model(model).generator(mel)

        ### Compute Mel for Generated Audio ###
        gen_audio_padded = pad_for_mel(gen_audio, n_fft=args.n_fft, hop_size=args.hop_size)
        gen_audio_mel = audio_mel_conv.audio2mel(gen_audio_padded.squeeze(1), do_norm=True)

        loss_mel = F.l1_loss(mel_target, gen_audio_mel) 

        loss_mel = torch.mean(accelerator.gather_for_metrics(loss_mel))

        val_mel_loss += loss_mel
        num_losses += 1

    accelerator.print("Validation Loss:", round(val_mel_loss.item()/num_losses,4))

    if args.log_wandb:
        
        accelerator.log(
                    {
                        "val_mel_loss": val_mel_loss.item()/num_losses, 
                    }, 
                    step=completed_steps
                )
        
    accelerator.wait_for_everyone()

    model.train()

    if completed_epochs % args.checkpoint_epochs == 0:
        path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_epochs}")
        accelerator.save_state(output_dir=path_to_checkpoint)

    completed_epochs += 1

    scheduler_d.step(epoch=completed_epochs)
    scheduler_g.step(epoch=completed_epochs)
    
    accelerator.print(f"Updating Learning Rate To: {scheduler_d.get_last_lr()[0]: .3e}")

accelerator.save_state(output_dir=os.path.join(path_to_experiment, f"final_checkpoint"))
accelerator.end_training()