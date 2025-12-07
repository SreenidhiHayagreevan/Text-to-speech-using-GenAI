import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataclasses import dataclass

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

class LinearNorm(nn.Module):
    """
    Standard Linear layer with different intialization strategies
    """
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 w_init_gain="linear"):
        
        super(LinearNorm, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear.weight, 
            gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear(x)
    

class ConvNorm(nn.Module):
    """
    Standard Convolutional layer with different intialization strategies
    """
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 kernel_size=1, 
                 stride=1, 
                 padding=None, 
                 dilation=1, 
                 bias=True, 
                 w_init_gain="linear"):
        
        super(ConvNorm, self).__init__()

        if padding is None:
            padding = "same"

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            bias=bias
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)
        
class Encoder(nn.Module):

    """
    Learns embeddings on input characters from text
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.config = config

        self.embeddings = nn.Embedding(config.num_chars, config.character_embed_dim, padding_idx=config.pad_token_id)

        self.convolutions = nn.ModuleList()

        for i in range(config.encoder_n_convolutions):
            
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels=config.encoder_embed_dim if i != 0 else config.character_embed_dim,
                        out_channels=config.encoder_embed_dim, 
                        kernel_size=config.encoder_kernel_size,
                        stride=1, 
                        padding="same", 
                        dilation=1, 
                        w_init_gain="relu"
                    ),

                    nn.BatchNorm1d(config.encoder_embed_dim), 
                    nn.ReLU(), 
                    nn.Dropout(config.encoder_dropout_p)
                )
            ) 

        self.lstm = nn.LSTM(input_size=config.encoder_embed_dim, 
                            hidden_size=config.encoder_embed_dim//2,
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True)
        
    def forward(self, x, input_lengths=None):

        ### Embed Tokens and transpose to (B x E x T) ###
        x = self.embeddings(x).transpose(1,2)

        batch_size, channels, seq_len = x.shape

        if input_lengths is None:
            input_lengths = torch.full((batch_size, ), fill_value=seq_len, device=x.device)

        for block in self.convolutions:
            x = block(x)

        ### Convert to BxLxE ###
        x = x.transpose(1,2)

        ### Pack Padded Sequence so LSTM doesnt Process Pad Tokens ###
        ### This requires data to be sorted in longest to shortest!! ###
        x = pack_padded_sequence(x, input_lengths.cpu(), batch_first=True)
    
        ### Pass Data through LSTM ###
        outputs, _ = self.lstm(x)

        ### Pad Packed Sequence ###
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )

        return outputs
    
class Prenet(nn.Module):

    """
    At each decoder step, we will pass the previous timestep through the prenet.
    This helps with both feature extraction and keeping stochasticity due to non-optional
    dropout
    """
    def __init__(self, 
                 input_dim, 
                 prenet_dim, 
                 prenet_depth,
                 dropout_p=0.5):
        super(Prenet, self).__init__()

        self.dropout_p = dropout_p

        dims = [input_dim] + [prenet_dim for _ in range(prenet_depth)]

        self.layers = nn.ModuleList()
        
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(

                nn.Sequential(
                    LinearNorm(
                        in_features=in_dim, 
                        out_features=out_dim,
                        bias=False, 
                        w_init_gain="relu"
                    ),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.layers:

            ### Even during inference we leave this dropout enabled to "introduce output variation" ###
            x = F.dropout(layer(x), p=self.dropout_p, training=True)

        return x
    
class LocationLayer(nn.Module):
    
    """
    This module looks at our Current attention weights (how much emphasis is
    this decoder step placing on all the input characters) as well as the
    cumulative attention weights (how much emphasis have we already put
    on all the input characters) and uses a convolution to extract features from them
    """
    def __init__(self, 
                 attention_n_filters, 
                 attention_kernel_size, 
                 attention_dim):
        super(LocationLayer, self).__init__()

        self.conv = ConvNorm(
            in_channels=2, 
            out_channels=attention_n_filters, 
            kernel_size=attention_kernel_size, 
            padding="same",
            bias=False
        )

        self.proj = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain="tanh")

    def forward(self, attention_weights):
        attention_weights = self.conv(attention_weights).transpose(1,2)
        attention_weights = self.proj(attention_weights)
        return attention_weights


class LocalSensitiveAttention(nn.Module):

    """
    The most important part of our model! It looks at:
    - The entire encoded text output
    - Our current decoder step for mel generation
    - Attention weights of how much emphasis have we already placed on the different encoder outputs
    """
    def __init__(self, 
                 attention_dim, 
                 decoder_hidden_size,
                 encoder_hidden_size, 
                 attention_n_filters, 
                 attention_kernel_size):
        super(LocalSensitiveAttention, self).__init__()

        self.in_proj = LinearNorm(decoder_hidden_size, attention_dim, bias=True, w_init_gain="tanh")
        self.enc_proj = LinearNorm(encoder_hidden_size, attention_dim, bias=False, w_init_gain="tanh")
        
        self.what_have_i_said = LocationLayer(
            attention_n_filters, 
            attention_kernel_size, 
            attention_dim
        )

        self.energy_proj = LinearNorm(attention_dim, 1, bias=False, w_init_gain="tanh")

        self.reset()

    def reset(self):
        self.enc_proj_cache = None

    def _calculate_alignment_energies(self, 
                                      mel_input, 
                                      encoder_output, 
                                      cumulative_attention_weights, 
                                      mask=None):

        ### Take our previous step of the mel sequence and project it (B x 1 x attention_dim)
        mel_proj = self.in_proj(mel_input).unsqueeze(1)

        ### Take our entire encoder output and project it (B x encoder_len x attention_dim)
        if self.enc_proj_cache is None:
            self.enc_proj_cache = self.enc_proj(encoder_output)

        ### Look at our attention weight history to understand where the model has already placed attention 
        cumulative_attention_weights = self.what_have_i_said(cumulative_attention_weights)

        ### Broadcast sum the single mel timestep over all of our encoder timesteps (both attention weight features and encoder features)
        ### And scale with tanh to get scores between -1 and 1, and project to a single value to comput energies
        energies = self.energy_proj(
            torch.tanh(
                mel_proj + self.enc_proj_cache + cumulative_attention_weights
            )
        ).squeeze(-1)
        
        ### Mask out pad regions (dont want to weight pad tokens from encoder)
        if mask is not None:
            energies = energies.masked_fill(mask.bool(), -float("inf"))
        
        return energies
    
    def forward(self, 
                mel_input, 
                encoder_output, 
                cumulative_attention_weights, 
                mask=None):

        ### Compute energies ###
        energies = self._calculate_alignment_energies(mel_input, 
                                                      encoder_output, 
                                                      cumulative_attention_weights, 
                                                      mask)
        
        ### Convert to Probabilities (relation of our mel input to all the encoder outputs) ###
        attention_weights = F.softmax(energies, dim=1)

        ### Weighted average of our encoder states by the learned probabilities 
        attention_context = torch.bmm(attention_weights.unsqueeze(1), encoder_output).squeeze(1)

        return attention_context, attention_weights

class PostNet(nn.Module):

    """
    To take final generated Mel from LSTM and postprocess to allow for
    any missing details to be added in (learns the residual!)
    """
    def __init__(self, 
                 num_mels, 
                 postnet_num_convs=5, 
                 postnet_n_filters=512, 
                 postnet_kernel_size=5,
                 postnet_dropout_p=0.5):
        
        super(PostNet, self).__init__()

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                ConvNorm(num_mels, 
                         postnet_n_filters, 
                         kernel_size=postnet_kernel_size, 
                         padding="same",
                         w_init_gain="tanh"), 

                nn.BatchNorm1d(postnet_n_filters),
                nn.Tanh(), 
                nn.Dropout(postnet_dropout_p)
            )
        )

        for _ in range(postnet_num_convs - 2):
            
            self.convs.append(
                nn.Sequential(
                    ConvNorm(postnet_n_filters, 
                             postnet_n_filters, 
                             kernel_size=postnet_kernel_size, 
                             padding="same",
                             w_init_gain="tanh"), 

                    nn.BatchNorm1d(postnet_n_filters),
                    nn.Tanh(), 
                    nn.Dropout(postnet_dropout_p)
                )
            )

        self.convs.append(
            nn.Sequential(
                    ConvNorm(postnet_n_filters, 
                             num_mels, 
                             kernel_size=postnet_kernel_size, 
                             padding="same"), 

                    nn.BatchNorm1d(num_mels),
                    nn.Dropout(postnet_dropout_p)
                )
        )
    
    def forward(self, x):
        
        x = x.transpose(1,2)
        for conv_block in self.convs:
            x = conv_block(x)
        x = x.transpose(1,2)
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.config = config

        ### Predictions from previous timestep passed through a few linear layers ###
        self.prenet = Prenet(input_dim=self.config.num_mels,
                             prenet_dim=self.config.decoder_prenet_dim, 
                             prenet_depth=self.config.decoder_prenet_depth)

        ### LSTMs Module to Process Concatenated PreNet output and Attention Context Vector ###
        self.rnn = nn.ModuleList(
            [
                nn.LSTMCell(config.decoder_prenet_dim + config.encoder_embed_dim, config.decoder_embed_dim), 
                nn.LSTMCell(config.decoder_embed_dim + config.encoder_embed_dim, config.decoder_embed_dim)
            ]
        )
   
        ### Local Sensitive Attention Module ###
        self.attention = LocalSensitiveAttention(attention_dim=config.attention_dim, 
                                                 decoder_hidden_size=config.decoder_embed_dim,
                                                 encoder_hidden_size=config.encoder_embed_dim, 
                                                 attention_n_filters=config.attention_location_n_filters, 
                                                 attention_kernel_size=config.attention_location_kernel_size)
        
        ### Predict Next Mel ###
        self.mel_proj = LinearNorm(config.decoder_embed_dim + config.encoder_embed_dim, config.num_mels)
        self.stop_proj = LinearNorm(config.decoder_embed_dim + config.encoder_embed_dim, 1, w_init_gain="sigmoid")

        ### Post Process Predicted Mel ###
        self.postnet = PostNet(
            num_mels=config.num_mels, 
            postnet_num_convs=config.decoder_postnet_num_convs,
            postnet_n_filters=config.decoder_postnet_n_filters, 
            postnet_kernel_size=config.decoder_postnet_kernel_size,
            postnet_dropout_p=config.decoder_postnet_dropout_p
        )

    def _init_decoder(self, encoder_outputs, encoder_mask=None):

        B, S, E = encoder_outputs.shape
        device = encoder_outputs.device

        ### Initialize Memory for two LSTM Cells ###
        self.h = [torch.zeros(B, self.config.decoder_embed_dim, device=device) for _ in range(2)]
        self.c = [torch.zeros(B, self.config.decoder_embed_dim, device=device) for _ in range(2)]

        ### Initialize Cumulative Attention ###
        self.cumulative_attn_weight = torch.zeros(B,S, device=device)
        self.attn_weight = torch.zeros(B,S, device=device)
        self.attn_context = torch.zeros(B, self.config.encoder_embed_dim, device=device)

        ### Store Encoder Outputs ##
        self.encoder_outputs = encoder_outputs
        self.encoder_mask = encoder_mask

    def _bos_frame(self, B):
        start_frame_zeros = torch.zeros(B, 1, self.config.num_mels)
        return start_frame_zeros

    # def decode(self, mel_step):

    #     rnn_input = torch.cat([mel_step, self.attn_context], dim=-1)

    #     self.h[0], self.c[0] = self.rnn[0](rnn_input, (self.h[0], self.c[0]))
    #     self.h[1], self.c[1] = self.rnn[1](self.h[0], (self.h[1], self.c[1]))
    #     rnn_output = self.h[1]

    #     attention_context, attention_weights = self.attention(
    #         rnn_output, 
    #         self.encoder_outputs, 
    #         self.cumulative_attn_weight, 
    #         mask=self.encoder_mask
    #     )

    #     self.attn_context = attention_context
    #     self.cumulative_attn_weight = self.cumulative_attn_weight + attention_weights

    #     next_pred_input = torch.cat((rnn_output, self.attn_context), dim=1)
    #     mel_out = self.mel_proj(next_pred_input)
    #     stop_out = self.stop_proj(next_pred_input)

    #     return mel_out, stop_out, attention_weights

    def decode(self, mel_step):

        rnn_input = torch.cat([mel_step, self.attn_context], dim=-1)

        ### Pass RNN Input into first LSTMCell ### 
        self.h[0], self.c[0] = self.rnn[0](rnn_input, (self.h[0], self.c[0]))

        ### Dropout ###
        attn_hidden = F.dropout(self.h[0], self.config.attention_dropout_p, self.training)

        ### Concat cumulative and prev weights ###
        attn_weights_cat = torch.cat(
            [
                self.attn_weight.unsqueeze(1), self.cumulative_attn_weight.unsqueeze(1)
            ], dim=1
        )

        ### Compute Attention on Hidden State ###
        attention_context, attention_weights = self.attention(
            attn_hidden, 
            self.encoder_outputs, 
            attn_weights_cat, 
            mask=self.encoder_mask
        )

        self.attn_weight = attention_weights
        self.cumulative_attn_weight = self.cumulative_attn_weight + attention_weights
        self.attn_context = attention_context

        ### Get Decoder Input ###
        decoder_input = torch.cat([attn_hidden, self.attn_context], dim=-1)

        self.h[1], self.c[1] = self.rnn[1](decoder_input, (self.h[1], self.c[1]))

        decoder_hidden = F.dropout(self.h[1], self.config.decoder_dropout_p, self.training)

        ### Projections ###
        next_pred_input = torch.cat([decoder_hidden, self.attn_context], dim=-1)

        mel_out = self.mel_proj(next_pred_input)
        stop_out = self.stop_proj(next_pred_input)

        return mel_out, stop_out, attention_weights

    def forward(self,
                encoder_outputs,
                encoder_mask, 
                mels, 
                decoder_mask):
        
        ### When Decoding Start with Zero Feature Vector ###
        start_feature_vector = self._bos_frame(mels.shape[0]).to(encoder_outputs.device)
        mels_w_start = torch.cat([start_feature_vector, mels], dim=1)
        
        self._init_decoder(encoder_outputs, encoder_mask)

        ### Create lists to store Intermediate Outputs ###
        mel_outs, stop_tokens, attention_weights = [], [], []
        
        ### Teacher forcing for T Steps ###
        T_dec = mels.shape[1]

        ### Project Mel Spectrograms by PreNet ###
        mel_proj = self.prenet(mels_w_start)

        ### Loop through T timesteps ###
        for t in range(T_dec):

            if t == 0:
                self.attention.reset()

            step_input = mel_proj[:, t, :]

            mel_out, stop_out, attention_weight = self.decode(step_input)

            mel_outs.append(mel_out)
            stop_tokens.append(stop_out)
            attention_weights.append(attention_weight)

        mel_outs = torch.stack(mel_outs, dim=1)
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze()
        attention_weights = torch.stack(attention_weights, dim=1)
        mel_residual = self.postnet(mel_outs)

        ### Mask ###
        decoder_mask = decoder_mask.unsqueeze(-1).bool()
        mel_outs = mel_outs.masked_fill(decoder_mask, 0.0)
        mel_residual = mel_residual.masked_fill(decoder_mask, 0.0)
        attention_weights = attention_weights.masked_fill(decoder_mask, 0.0)
        stop_tokens = stop_tokens.masked_fill(decoder_mask.squeeze(), 1e3)

        return mel_outs, mel_residual, stop_tokens, attention_weights
    
    @torch.inference_mode()
    def inference(self, encoder_output, max_decode_steps=1000):

        start_feature_vector = self._bos_frame(B=1).squeeze(0)

        self._init_decoder(encoder_output, encoder_mask=None)

        ### Create lists to store Intermediate Outputs ###
        mel_outs, stop_outs, attention_weights = [], [], []

        _input = start_feature_vector
        self.attention.reset()

        while True:

            _input = self.prenet(_input)

            mel_out, stop_out, attention_weight = self.decode(_input)

            mel_outs.append(mel_out)
            stop_outs.append(stop_out)
            attention_weights.append(attention_weight)

            if torch.sigmoid(stop_out) > 0.5:
                break
            elif len(mel_outs) >= max_decode_steps:
                print("Reached Max Decoder Steps")
                break

            _input = mel_out

        mel_outs = torch.stack(mel_outs, dim=1)
        stop_outs = torch.stack(stop_outs, dim=1).squeeze()
        attention_weights = torch.stack(attention_weights, dim=1)
        mel_residual = self.postnet(mel_outs)

        return mel_outs, mel_residual, stop_outs, attention_weights
        

class Tacotron2(nn.Module):
    def __init__(self, config):
        super(Tacotron2, self).__init__()

        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, text, input_lengths, mels, encoder_mask, decoder_mask):

        encoder_padded_outputs = self.encoder(text, input_lengths)
        mel_outs, mel_residual, stop_tokens, attention_weights = self.decoder(
            encoder_padded_outputs, encoder_mask, mels, decoder_mask
        )

        mel_postnet_out = mel_outs + mel_residual

        return mel_outs, mel_postnet_out, stop_tokens, attention_weights 
    
    @torch.inference_mode()
    def inference(self, text, max_decode_steps=1000):
        
        if text.ndim == 1:
            text = text.unsqueeze(0)

        assert text.shape[0] == 1, "Inference only written for Batch Size of 1"
        encoder_outputs = self.encoder(text)
        mel_outs, mel_residual, stop_outs, attention_weights = self.decoder.inference(
            encoder_outputs, max_decode_steps=max_decode_steps
        )

        mel_postnet_out = mel_outs + mel_residual

        return mel_postnet_out, attention_weights

        
if __name__ == "__main__":

    from dataset import TTSDataset, TTSCollator

    dataset = TTSDataset("data/test_metadata.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=TTSCollator())
    for text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask in loader:

        config = Tacotron2Config()
        model = Tacotron2(config)
        print(model)
        # decoder(encoded_outputs, )

        break
