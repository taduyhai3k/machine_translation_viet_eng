import torch
def look_ahead_mask(inp_len):
    return 1 - torch.tril(torch.ones(inp_len, inp_len))