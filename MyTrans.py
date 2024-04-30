import torch
import torch.nn as nn
import utils
from layer import Encoder, Decoder, InputEmbed, LookAheadMask


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, dmodel = 512, dembed = 512,d_ff = 2048,head = 8, active = 'relu', layer = 6, dropout = 0.1, eps = 1e-5) -> None:
        super().__init__()
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.encoder = Encoder.Encoder(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.decoder = Decoder.Decoder(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.inp_embed = InputEmbed.InpEmbed(input_vocab_size, dembed)
        self.out_embed = InputEmbed.InpEmbed(output_vocab_size, dembed)
        self.linear = nn.Linear(in_features= dmodel, out_features= output_vocab_size, device = self.device, dtype = torch.float32)
    
    def forward(self, x, y):
        inp_embed = self.inp_embed(x)
        encoder_out = self.encoder(inp_embed)
        out_embed = self.out_embed(y)
        decoder_out = self.decoder(out_embed, encoder_out, LookAheadMask.look_ahead_mask(inp_len= y.shape[1]))
        return self.linear(decoder_out)    
 
