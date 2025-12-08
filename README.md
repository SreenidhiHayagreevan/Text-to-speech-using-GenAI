# Text-to-speech-using-GenAI

## Dataset
We will be using the LJSpeech dataset. It has nearly 13100 clips of high quality audio. _Taco2/prep_splits.py_ preprocesses the _LJSpeech_ dataset by loading normalized transcripts, generating absolute audio paths, computing durations, and splitting the data into train/test sets. It then sorts the training samples by duration and saves the results to _datasplit/train_metadata.csv_ and _datasplit/test_metadata.csv_.

## Training Tacotron2
_Taco2/train_taco.py_ script provides a complete Tacotron2 training pipeline using HuggingFace Accelerate for distributed and mixed-precision training. It loads preprocessed LJSpeech metadata, generates mel-spectrograms on the fly, and batches data using a length-aware sampler and custom collator. The Tacotron2 architecture is fully configurable—covering encoder/decoder layers, prenet/postnet settings, and attention mechanisms. During training, the model predicts both coarse and postnet-refined mel spectrograms as well as stop tokens, optimized using a combination of mel MSE, refined mel MSE, and BCE stop-token loss. The script performs gradient clipping, synchronized updates across devices, optional LR scheduling, and logs metrics to the console or Weights & Biases. It also evaluates on a validation set each epoch and saves mel/attention visualizations for inspection. Checkpoints are automatically stored throughout training, and a final checkpoint is written at the end of training for future inference or fine-tuning.

![plot](./output/taco_loss.png)

## Training HIFIGAN
We begin by training HiFiGAN on ground-truth LJSpeech audio using the same train/test split as in the Tacotron2 setup. This teaches the vocoder to reconstruct high-quality waveforms from real mel-spectrograms.

However, Tacotron2’s generated mels differ slightly from the ground truth, which can introduce a domain mismatch. To address this, we finetune HiFiGAN on Tacotron2-generated mel spectrograms while still targeting the original LJSpeech audio. This helps the vocoder better adapt to the characteristics and artifacts of Tacotron2 outputs.

To prepare data for finetuning, we first run Tacotron2 in inference mode (teacher-forced) to generate mel-spectrograms for all samples and save them as NumPy arrays, refer  _save_taco_mels.py_

These saved mels are then used to finetune HiFiGAN, closing the gap between ground-truth mel distributions and Tacotron2-generated mels. 

## Inference
For evaluation, the _GAN(HIFI)/inference.ipynb_ notebook demonstrates how Tacotron2 + HiFiGAN are combined to synthesize speech.

## Outputs
The training logs and progress plots can be found within the _output/_ directory. 

* Audio generated using the fine-tuned Tacotron2 model: _output/taco_output.wav_
* Audio generated using Tacotron2 followed by HiFiGAN vocoding: _output/hifigan_output.wav_

