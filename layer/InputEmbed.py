import torch
import torch.nn as nn
class InpEmbed(nn.Module):
    def __init__(self, vocab_size, dembed = 512) -> None:
        super().__init__()
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.vocab_size = vocab_size
        self.dembed = dembed
        self.map = torch.randn(size = [self.vocab_size, self.dembed], dtype = torch.float32, requires_grad= False, device = self.device)
        self.map.requires_grad = True
    def forward(self, x):
        # x.shape is [batch, sequence_length index]   
        one_hot = torch.zeros(size = [x.shape[0], x.shape[1], self.vocab_size], dtype= torch.float32, device= self.device).scatter_(2, x.unsqueeze(2), 1.0)
        return torch.matmul(one_hot, self.map) + embed_position(dembed= self.dembed,device= self.device)
    
def embed_position(dembed = 512, device = 'cpu'):
    #dembed is 2n
    ep = torch.arange(start= 0,end= dembed, step= 1, dtype= torch.float32, device = device)    
    ep[1::2] = ep[0::2]   
    ep = 1/ (10000**(ep/dembed)) 
    pos = torch.arange(start= 0,end= dembed, step= 1, dtype= torch.float32, device = device)
    pos = pos * ep
    pos[1::2] = torch.cos(pos[1::2])
    pos[0::2] = torch.sin(pos[0::2])
    return pos