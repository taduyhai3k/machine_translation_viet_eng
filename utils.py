import torch
import numpy
import nltk
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
def SparseCrossEntropy(true, pred):
    #shape pred is [batch_size, length, embed_size]
    #shape true is [batch_size, length]
    pred_tmp = torch.gather(pred[:, :-1, :].softmax(dim = -1), dim = -1, index = true.unsqueeze(dim = -1)[:, 1:,:])
    return torch.log(pred_tmp).sum()

def transformer_lr(step_num, d_model = 512, warmup_steps = 4000):
    lr = d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    return lr
def bleu_score(infer, candi):
    candi_numpy = candi.argmax(dim = -1).detach().numpy().astype(str).tolist()
    infer_numpy = infer.detach().numpy().astype(str).tolist()
    return nltk.translate.bleu_score.corpus_bleu(infer_numpy,candi_numpy)

def predict(model, data, inp_tokennizer, out_tokenizer, max_lenth = 300):
    #translate one sentences
    model.eval()
    inp_encode = inp_tokennizer.encode(data).reshape(1, -1)
    out_encode = out_tokenizer.encode('<start>').reshape(1, -1)
    with torch.no_grad():
        for i in range(max_lenth):
            pred = model(inp_encode, out_encode)
            preds = pred[:, -1, :].reshape(1, -1)
            predict_id = torch.argmax(preds, dim = -1)
            out_encode = torch.cat([out_encode, pred], dim = -1)
            if out_tokenizer.encode('<end>')[0] == predict_id:
                return out_tokenizer.decode(out_encode.reshape(-1))
    return out_tokenizer.decode(out_encode.reshape(-1))                
            
        

def train(model, data_loader, optimizer, lr_schedule, epoch, device):
    model.train()
    scheduler = LambdaLR(optimizer, lr_lambda= lambda step_num : transformer_lr(step_num=step_num))
    for i in range(epoch):
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input, target)
            loss = SparseCrossEntropy(target, output)
            loss.backward()
            optimizer.step()
        scheduler.step()        