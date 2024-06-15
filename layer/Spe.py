from layer import SpeLayer
import torch.nn as nn
import torch 

class Spe(nn.Module):
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
        self.spe_layer = nn.ModuleList([SpeLayer.SpeLayer(self.dembed, self.dmodel,self.d_ff, self.head, self.active, self.drop_rate, self.eps ) for i in range(self.layer)])
    
    def forward(self, x, is_encode = True, encoder_out = None, padding_mask = None, look_ahead_mask = None, padding_global_mask = None ):
        out = x
        for i in range(self.layer):
            out,_,_  = self.spe_layer[i](out, is_encode,encoder_out,padding_mask, look_ahead_mask, padding_global_mask)        
        return out    