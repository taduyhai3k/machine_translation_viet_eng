import torch
import torch.nn as nn
from layer import MultiHeadAtten as MHA
from layer import FeedForward as FF

class DecoderLayer(nn.Module):
    def __init__(self, dembed = 512, dmodel = 512, d_ff = 2048, head = 8, active = 'relu', dropout = 0.1, eps = 1e-5) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dmodel = dmodel
        self.dembed = dembed
        self.d_ff = d_ff
        self.head = head
        self.active = active
        self.eps = eps        
        self.dropout = torch.nn.Dropout(p = dropout)
        self.norm1 = torch.nn.LayerNorm(normalized_shape= self.dembed, eps= eps, device= self.device)
        self.norm2 = torch.nn.LayerNorm(normalized_shape= self.dembed, eps= eps, device= self.device)
        self.norm3 = torch.nn.LayerNorm(normalized_shape= self.dembed, eps= eps, device= self.device)        
        self.ff = FF.FeedForward(self.dmodel, self.d_ff, self.active)
        self.mha_local = MHA.MultiHeadAtten(self.dembed, self.dmodel, self.head) 
        self.mha_global = MHA.MultiHeadAtten(self.dembed, self.dmodel, self.head) 
    
    def forward(self, x, encoder_out, mask = None):
        mha_mask, attention_weight_local = self.mha_local(x,x,x, mask) 
        q = self.norm1(x + self.dropout(mha_mask))
        mha_global, attention_weight_global = self.mha_global(q, encoder_out, encoder_out)
        out_global = self.norm2(q + self.dropout(mha_global))
        outff = self.ff(out_global)
        out = self.norm3(out_global + self.dropout(outff))
        return out, attention_weight_local, attention_weight_global   