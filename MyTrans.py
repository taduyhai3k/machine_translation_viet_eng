import torch
import torch.nn as nn
import utils
from layer import Encoder, Decoder, InputEmbed, LookAheadMask


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, dmodel = 512, dembed = 512,d_ff = 2048,head = 8, active = 'relu', layer = 6, dropout = 0.1, eps = 1e-5, tying = False) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = Encoder.Encoder(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.decoder = Decoder.Decoder(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.inp_embed = InputEmbed.InpEmbed(input_vocab_size, dembed)
        self.out_embed = InputEmbed.InpEmbed(output_vocab_size, dembed)
        self.tying = tying
        if not self.tying:
            self.linear = nn.Linear(in_features= dmodel, out_features= output_vocab_size, device = self.device, dtype = torch.float32)
        self.dropout = nn.Dropout(p = dropout)
        self.dmodel = dmodel
    
    def forward(self, x, y):
        inp_embed = self.dropout(self.inp_embed(x))
        padding_mask_enc = LookAheadMask.padding_mask(x)
        encoder_out = self.encoder(inp_embed, padding_mask_enc)
        out_embed = self.dropout(self.out_embed(y))
        look_ahead_mask = LookAheadMask.look_ahead_mask(y)
        padding_global_mask = LookAheadMask.padding_mask_global(x, y.shape[1]) 
        decoder_out = self.decoder(out_embed, encoder_out, None, look_ahead_mask, padding_global_mask)
        if self.tying:
            return torch.matmul(decoder_out ,self.out_embed.map.weight)
        else:
            return self.linear(decoder_out)   
        
class TransformerReduce(nn.Module):
    def __init__(self, input_vocab_size, dmodel = 512, dembed = 512,d_ff = 2048,head = 8, active = 'relu', layer = 6, dropout = 0.1, eps = 1e-5, tying = False) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = Encoder.Encoder(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.decoder = Decoder.Decoder(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.inp_embed = InputEmbed.InpEmbed(input_vocab_size, dembed)
        self.tying = tying
        if not self.tying:
            self.linear = nn.Linear(in_features= dmodel, out_features= input_vocab_size, device = self.device, dtype = torch.float32)
        self.dropout = nn.Dropout(p = dropout)
        self.dmodel = dmodel            
        
    def forward(self, x, y):
        inp_embed = self.dropout(self.inp_embed(x))
        padding_mask_enc = LookAheadMask.padding_mask(x)
        encoder_out = self.encoder(inp_embed, padding_mask_enc)
        out_embed = self.dropout(self.inp_embed(y))
        look_ahead_mask = LookAheadMask.look_ahead_mask(y)
        padding_global_mask = LookAheadMask.padding_mask_global(x, y.shape[1]) 
        decoder_out = self.decoder(out_embed, encoder_out, None, look_ahead_mask, padding_global_mask)
        if self.tying:
            return torch.matmul(decoder_out ,self.inp_embed.map.weight)
        else:
            return self.linear(decoder_out)           
         
 
