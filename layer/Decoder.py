import torch
import torch.nn as nn
from layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, dmodel = 512, dembed = 512,d_ff = 2048,head = 8, active = 'relu', layer = 6, dropout = 0.1, eps = 1e-5) -> None:
        super().__init__()
        self.dembed = dembed
        self.dmodel = dmodel
        self.active = active
        self.layer = layer
        self.drop_rate = dropout
        self.eps = eps
        self.d_ff = d_ff
        self.head = head
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.decoder_layer = nn.ModuleList([DecoderLayer.DecoderLayer(self.dembed, self.dmodel,self.d_ff, self.head, self.active, self.drop_rate, self.eps ) for i in range(self.layer)])
    
    def forward(self, x, encoder_out, padding_mask = None, look_ahead_mask = None, padding_global_mask = None ):
        out = x
        for i in range(self.layer):
            out,_,_  = self.decoder_layer[i](out,encoder_out,padding_mask, look_ahead_mask, padding_global_mask)        
        return out    