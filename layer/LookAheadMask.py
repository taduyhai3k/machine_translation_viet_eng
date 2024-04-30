import torch
def look_ahead_mask(inp_len):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return 1 - torch.tril(torch.ones(inp_len, inp_len, device= device))