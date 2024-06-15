import torch
import torch.nn as nn
import utils
from layer import Encoder, Decoder, InputEmbed, LookAheadMask, Spe


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
            return torch.matmul(decoder_out ,self.out_embed.map.weight.t())
        else:
            return self.linear(decoder_out)   
        
class TransformerParallel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, dmodel = 512, dembed = 512,d_ff = 2048,head = 8, active = 'relu', layer = 6, dropout = 0.1, eps = 1e-5) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.SpeE = Spe.Spe(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.SpeV = Spe.Spe(dmodel = dmodel, dembed = dembed,d_ff = d_ff,head = head, active = 'relu', layer = layer, dropout = dropout, eps = eps)
        self.E_embed = InputEmbed.InpEmbed(input_vocab_size, dembed)
        self.V_embed = InputEmbed.InpEmbed(output_vocab_size, dembed)
        self.dropout = nn.Dropout(p = dropout)
        self.dmodel = dmodel            
        
    def forward(self, x, y, inp = 'E'):
        if inp == 'E':
            inp_embed = self.dropout(self.E_embed(x))
            padding_mask_enc = LookAheadMask.padding_mask(x)
            encoder_out = self.SpeE(inp_embed, is_encode = True, padding_mask = padding_mask_enc)
            out_embed = self.dropout(self.V_embed(y))
            look_ahead_mask = LookAheadMask.look_ahead_mask(y)
            padding_global_mask = LookAheadMask.padding_mask_global(x, y.shape[1]) 
            decoder_out = self.SpeV(out_embed, False ,encoder_out, None, look_ahead_mask, padding_global_mask)
            return torch.matmul(decoder_out ,self.V_embed.map.weight.t())
        else:
            inp_embed = self.dropout(self.V_embed(x))
            padding_mask_enc = LookAheadMask.padding_mask(x)
            encoder_out = self.SpeV(inp_embed, is_encode = True, padding_mask = padding_mask_enc)
            out_embed = self.dropout(self.E_embed(y))
            look_ahead_mask = LookAheadMask.look_ahead_mask(y)
            padding_global_mask = LookAheadMask.padding_mask_global(x, y.shape[1]) 
            decoder_out = self.SpeE(out_embed, False ,encoder_out, None, look_ahead_mask, padding_global_mask)
            return torch.matmul(decoder_out ,self.E_embed.map.weight.t())                  
         
 
