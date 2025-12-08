import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from dataclasses import dataclass

################################################################################
# CONFIG
################################################################################

@dataclass
class Tacotron2Config:

    ### Mel Input Features ###
    num_mels: int = 80

    ### Character Embeddings ###
    character_embed_dim: int = 512
    num_chars: int = 67
    pad_token_id: int = 0

    ### Encoder config ###
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embed_dim: int = 512
    encoder_dropout_p: float = 0.5

    ### Decoder Config ###
    decoder_embed_dim: int = 1024
    decoder_prenet_dim: int = 256
    decoder_prenet_depth: int = 2
    decoder_prenet_dropout_p: float = 0.5
    decoder_postnet_num_convs: int = 5
    decoder_postnet_n_filters: int = 512
    decoder_postnet_kernel_size: int = 5
    decoder_postnet_dropout_p: float = 0.5
    decoder_dropout_p: float = 0.1

    ### Attention Config ###
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31
    attention_dropout_p: float = 0.1

    ### === EMOTION ADDITIONS ===
    num_emotions: int = 0              # set >0 when fine-tuning on emotion dataset
    emotion_embed_dim: int = 0         # e.g., 64



################################################################################
# BASIC LAYERS
################################################################################

class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear(x)


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain="linear"):
        super().__init__()

        if padding is None:
            padding = "same"

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )

        torch.nn.init.xavier_uniform_(self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)


################################################################################
# ENCODER
################################################################################

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = nn.Embedding(
            config.num_chars,
            config.character_embed_dim,
            padding_idx=config.pad_token_id
        )

        self.convolutions = nn.ModuleList()
        for i in range(config.encoder_n_convolutions):
            self.convolutions.append(nn.Sequential(
                ConvNorm(
                    in_channels=config.encoder_embed_dim if i > 0 else config.character_embed_dim,
                    out_channels=config.encoder_embed_dim,
                    kernel_size=config.encoder_kernel_size,
                    w_init_gain="relu"
                ),
                nn.BatchNorm1d(config.encoder_embed_dim),
                nn.ReLU(),
                nn.Dropout(config.encoder_dropout_p),
            ))

        self.lstm = nn.LSTM(
            input_size=config.encoder_embed_dim,
            hidden_size=config.encoder_embed_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths=None):
        x = self.embeddings(x).transpose(1, 2)  # (B, embed, T)
        B, _, seq_len = x.shape
        if input_lengths is None:
            input_lengths = torch.full((B,), seq_len, device=x.device)

        for block in self.convolutions:
            x = block(x)

        x = x.transpose(1, 2)  # (B, T, embed)

        x = pack_padded_sequence(x, input_lengths.cpu(), batch_first=True)
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


################################################################################
# PRENET
################################################################################

class Prenet(nn.Module):
    def __init__(self, input_dim, prenet_dim, depth, dropout_p=0.5):
        super().__init__()
        self.dropout_p = dropout_p

        dims = [input_dim] + [prenet_dim] * depth
        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Sequential(
                LinearNorm(in_dim, out_dim, bias=False, w_init_gain="relu"),
                nn.ReLU(),
            ))

    def forward(self, x):
        # dropout always on (tacotron2 style)
        for layer in self.layers:
            x = F.dropout(layer(x), p=self.dropout_p, training=True)
        return x


################################################################################
# ATTENTION MECHANISM
################################################################################

class LocationLayer(nn.Module):
    def __init__(self, n_filters, kernel_size, attention_dim):
        super().__init__()
        self.conv = ConvNorm(2, n_filters, kernel_size, padding="same", bias=False)
        self.proj = LinearNorm(n_filters, attention_dim, bias=False, w_init_gain="tanh")

    def forward(self, attn):
        x = self.conv(attn).transpose(1, 2)
        return self.proj(x)


class LocalSensitiveAttention(nn.Module):
    def __init__(self, attention_dim, decoder_hidden_size, encoder_hidden_size,
                 n_filters, kernel_size):
        super().__init__()

        self.decoder_proj = LinearNorm(decoder_hidden_size, attention_dim, bias=True, w_init_gain="tanh")
        self.encoder_proj = LinearNorm(encoder_hidden_size, attention_dim, bias=False, w_init_gain="tanh")
        self.location_layer = LocationLayer(n_filters, kernel_size, attention_dim)
        self.energy_proj = LinearNorm(attention_dim, 1, bias=False, w_init_gain="tanh")
        self.enc_cache = None

    def reset(self):
        self.enc_cache = None

    def compute_energies(self, query, encoder_output, cumulative_attn, mask):

        # project decoder state
        query_proj = self.decoder_proj(query).unsqueeze(1)

        # project encoder output (cached)
        if self.enc_cache is None:
            self.enc_cache = self.encoder_proj(encoder_output)

        # location features
        loc = self.location_layer(cumulative_attn)

        energies = self.energy_proj(torch.tanh(query_proj + self.enc_cache + loc)).squeeze(-1)

        if mask is not None:
            energies = energies.masked_fill(mask.bool(), -1e9)

        return energies

    def forward(self, query, encoder_output, cumulative_attn, mask):
        energies = self.compute_energies(query, encoder_output, cumulative_attn, mask)
        attn_weights = F.softmax(energies, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_output).squeeze(1)
        return context, attn_weights


################################################################################
# POSTNET
################################################################################

class PostNet(nn.Module):
    def __init__(self, num_mels, n_convs, n_filters, kernel_size, dropout):
        super().__init__()

        self.convs = nn.ModuleList()

        # First conv
        self.convs.append(nn.Sequential(
            ConvNorm(num_mels, n_filters, kernel_size, padding="same", w_init_gain="tanh"),
            nn.BatchNorm1d(n_filters),
            nn.Tanh(),
            nn.Dropout(dropout),
        ))

        # Middle convs
        for _ in range(n_convs - 2):
            self.convs.append(nn.Sequential(
                ConvNorm(n_filters, n_filters, kernel_size, padding="same", w_init_gain="tanh"),
                nn.BatchNorm1d(n_filters),
                nn.Tanh(),
                nn.Dropout(dropout),
            ))

        # Final conv
        self.convs.append(nn.Sequential(
            ConvNorm(n_filters, num_mels, kernel_size, padding="same"),
            nn.BatchNorm1d(num_mels),
            nn.Dropout(dropout),
        ))

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.convs:
            x = block(x)
        return x.transpose(1, 2)


################################################################################
# DECODER (WITH EMOTION INJECTION)
################################################################################

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.prenet = Prenet(config.num_mels, config.decoder_prenet_dim,
                             config.decoder_prenet_depth, config.decoder_prenet_dropout_p)

        # 2 LSTM layers
        self.rnn = nn.ModuleList([
            nn.LSTMCell(config.decoder_prenet_dim + config.encoder_embed_dim, config.decoder_embed_dim),
            nn.LSTMCell(config.decoder_embed_dim + config.encoder_embed_dim, config.decoder_embed_dim),
        ])

        self.attention = LocalSensitiveAttention(
            config.attention_dim,
            config.decoder_embed_dim,
            config.encoder_embed_dim,
            config.attention_location_n_filters,
            config.attention_location_kernel_size,
        )

        self.mel_proj = LinearNorm(config.decoder_embed_dim + config.encoder_embed_dim, config.num_mels)
        self.stop_proj = LinearNorm(config.decoder_embed_dim + config.encoder_embed_dim, 1,
                                    w_init_gain="sigmoid")

        self.postnet = PostNet(
            config.num_mels,
            config.decoder_postnet_num_convs,
            config.decoder_postnet_n_filters,
            config.decoder_postnet_kernel_size,
            config.decoder_postnet_dropout_p,
        )

        # === EMOTION ADDITION (decoder only): THESE GET POPULATED BY Tacotron2 CLASS ===
        self.emotion_embedding = None
        self.emotion_to_hidden = None

    # ========= EMOTION INJECTION INTO DECODER INITIAL STATE ===========
    def inject_emotion(self, emotion_ids):
        """Inject emotion bias into decoder cell hidden states."""
        if (self.emotion_embedding is None) or (emotion_ids is None):
            return

        # (B, embed_dim)
        emo_vec = self.emotion_embedding(emotion_ids)

        # (B, decoder_embed_dim)
        emo_proj = self.emotion_to_hidden(emo_vec)

        # add into decoder hidden init state
        self.h[0] = self.h[0] + emo_proj
        self.h[1] = self.h[1] + emo_proj


    ####################################################################
    # Initialization and decode
    ####################################################################

    def _init_decoder(self, encoder_outputs, encoder_mask, emotion_ids=None):
        B, S, E = encoder_outputs.shape
        device = encoder_outputs.device

        self.h = [torch.zeros(B, self.config.decoder_embed_dim, device=device) for _ in range(2)]
        self.c = [torch.zeros(B, self.config.decoder_embed_dim, device=device) for _ in range(2)]

        self.attn_weight = torch.zeros(B, S, device=device)
        self.cumulative_attn_weight = torch.zeros(B, S, device=device)
        self.attn_context = torch.zeros(B, self.config.encoder_embed_dim, device=device)

        self.encoder_outputs = encoder_outputs
        self.encoder_mask = encoder_mask

        # === ADD EMOTION BIAS HERE ===
        if emotion_ids is not None:
            self.inject_emotion(emotion_ids)


    def _start_frame(self, B, device):
        return torch.zeros(B, 1, self.config.num_mels, device=device)


    def decode_step(self, mel_step):
        rnn_input = torch.cat([mel_step, self.attn_context], dim=-1)

        # LSTM 1
        self.h[0], self.c[0] = self.rnn[0](rnn_input, (self.h[0], self.c[0]))
        attn_hidden = F.dropout(self.h[0], self.config.attention_dropout_p, self.training)

        # attention
        attn_cat = torch.cat(
            [self.attn_weight.unsqueeze(1),
             self.cumulative_attn_weight.unsqueeze(1)], dim=1
        )

        context, attn = self.attention(attn_hidden, self.encoder_outputs, attn_cat, mask=self.encoder_mask)

        self.attn_weight = attn
        self.cumulative_attn_weight += attn
        self.attn_context = context

        # LSTM 2
        rnn_input2 = torch.cat([attn_hidden, context], dim=-1)
        self.h[1], self.c[1] = self.rnn[1](rnn_input2, (self.h[1], self.c[1]))
        decoder_hidden = F.dropout(self.h[1], self.config.decoder_dropout_p, self.training)

        proj_input = torch.cat([decoder_hidden, context], dim=-1)
        mel_pred = self.mel_proj(proj_input)
        stop_pred = self.stop_proj(proj_input)

        return mel_pred, stop_pred, attn


    ####################################################################
    # Forward (Teacher Forced)
    ####################################################################

    def forward(self, encoder_outputs, encoder_mask, mels, decoder_mask, emotion_ids=None):

        B = mels.size(0)
        device = mels.device

        start = self._start_frame(B, device)
        mels_w_start = torch.cat([start, mels], dim=1)

        self._init_decoder(encoder_outputs, encoder_mask, emotion_ids=emotion_ids)

        mel_outs = []
        stop_outs = []
        attns = []

        prenet_out = self.prenet(mels_w_start)

        T = mels.size(1)
        for t in range(T):

            if t == 0:
                self.attention.reset()

            mel_pred, stop_pred, attn = self.decode_step(prenet_out[:, t, :])

            mel_outs.append(mel_pred)
            stop_outs.append(stop_pred)
            attns.append(attn)

        mel_outs = torch.stack(mel_outs, dim=1)
        stop_outs = torch.stack(stop_outs, dim=1).squeeze()
        attns = torch.stack(attns, dim=1)

        residual = self.postnet(mel_outs)

        mel_outs = mel_outs.masked_fill(decoder_mask.unsqueeze(-1), 0.0)
        residual = residual.masked_fill(decoder_mask.unsqueeze(-1), 0.0)
        stop_outs = stop_outs.masked_fill(decoder_mask, 1e3)
        attns = attns.masked_fill(decoder_mask.unsqueeze(-1), 0.0)

        return mel_outs, mel_outs + residual, stop_outs, attns

    ####################################################################
    # Inference (Auto-regressive)
    ####################################################################

    @torch.inference_mode()
    def inference(self, encoder_outputs, emotion_ids=None, max_decode_steps=1000):

        device = encoder_outputs.device
        self._init_decoder(encoder_outputs, encoder_mask=None, emotion_ids=emotion_ids)

        mel_outs = []
        stop_outs = []
        attns = []

        _input = torch.zeros(1, self.config.num_mels, device=device)
        self.attention.reset()

        for _ in range(max_decode_steps):

            _input = self.prenet(_input)
            mel_pred, stop_pred, attn = self.decode_step(_input)

            mel_outs.append(mel_pred)
            stop_outs.append(stop_pred)
            attns.append(attn)

            if torch.sigmoid(stop_pred) > 0.5:
                break

            _input = mel_pred

        mel_outs = torch.stack(mel_outs, dim=1)
        residual = self.postnet(mel_outs)

        return mel_outs + residual, torch.stack(attns, dim=1)


################################################################################
# TACOTRON2 TOP
################################################################################

class Tacotron2(nn.Module):
    def __init__(self, config: Tacotron2Config):
        super().__init__()

        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # === EMOTION MODULES (shared across entire model) ===
        if config.num_emotions > 0 and config.emotion_embed_dim > 0:
            self.emotion_embedding = nn.Embedding(config.num_emotions, config.emotion_embed_dim)
            self.emotion_to_hidden = nn.Linear(config.emotion_embed_dim, config.decoder_embed_dim)

            # connect modules directly to decoder
            self.decoder.emotion_embedding = self.emotion_embedding
            self.decoder.emotion_to_hidden = self.emotion_to_hidden

    def forward(self, text, input_lengths, mels, encoder_mask, decoder_mask, emotion_ids=None):
        encoder_outputs = self.encoder(text, input_lengths)

        return self.decoder(
            encoder_outputs,
            encoder_mask,
            mels,
            decoder_mask,
            emotion_ids=emotion_ids
        )

    @torch.inference_mode()
    def inference(self, text, emotion_ids=None, max_decode_steps=1000):

        if text.ndim == 1:
            text = text.unsqueeze(0)

        assert text.size(0) == 1, "Only batch size 1 supported for inference"

        encoder_outputs = self.encoder(text)

        mel, attn = self.decoder.inference(
            encoder_outputs,
            emotion_ids=emotion_ids,
            max_decode_steps=max_decode_steps
        )

        return mel, attn
