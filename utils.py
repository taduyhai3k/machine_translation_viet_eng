import torch
import numpy
import nltk
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
def SparseCrossEntropy(true, pred):
    #shape pred is [batch_size, length, embed_size]
    #shape true is [batch_size, length]
    pred_tmp = torch.gather(pred[:, :-1, :].softmax(dim = -1), dim = -1, index = true.unsqueeze(dim = -1)[:, 1:,:])
    return (torch.log(pred_tmp) * -1 ).sum() 

def transformer_lr(step_num, d_model = 512, warmup_steps = 4000):
    if step_num == 0:
        step_num += 1
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

def save(path, model, optimizer, scheduler, epoch):
    state = {
        'model' : model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }           
    torch.save(state, path)

def load(path, model, optimizer, scheduler):
    state = torch.load(path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scheduler.optimizer = optimizer
    epoch = state['epoch']
    return model, optimizer, scheduler, epoch            

def eval(model, data_loader,optimizer, is_training = True):
    mean_loss = 0        
    infer = None
    candidate = None    
    device = 'gpu' if torch.cuda.is_available() else 'cpu'    
    if is_training:
        model.train()
        data_iter = tqdm(data_loader, desc='Training', leave=False)        
        for input, target in data_iter:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input, target)
            loss = SparseCrossEntropy(target, output)
            mean_loss += loss.item()
            loss.backward()
            optimizer.step()       
            if infer is not None:
                infer = torch.cat([infer, output], dim = 0)
            else:
                infer = output
            if candidate is not None:
                candidate = torch.cat([candidate, target], dim = 0)   
            else:
                candidate = target                 
    else:
        model.eval()    
        with torch.no_grad():
            mean_loss = 0        
            infer = None
            candidate = None
            data_iter = tqdm(data_loader, desc='Not training', leave=False)        
            for input, target in data_iter:
                input, target = input.to(device), target.to(device)
                output = model(input)
                loss  = SparseCrossEntropy(target, output)
                mean_loss += loss.item()
                if infer is not None:
                    infer = torch.cat([infer, output], dim = 0)
                else:
                    infer = output
                if candidate is not None:
                    candidate = torch.cat([candidate, target], dim = 0)   
                else:
                    candidate = target      
    return mean_loss, bleu_score(infer, candidate)            
                
                    

def train(model, optimizer, epoch, datatrain_loader,datavalid_loader = None, datatest_loader = None, path = None):
    scheduler = LambdaLR(optimizer, lr_lambda= lambda step_num : transformer_lr(step_num=step_num))
    tmp_score = 1e30
    if path is not None:
        model, optimizer, scheduler, epoch_old = torch.load(path, model, optimizer, scheduler)
    else:
        epoch_old = 0    
    for i in tqdm(range(epoch - epoch_old), desc='Epoch', leave=False):
        result_train =  eval(model, datatrain_loader, optimizer, True)
        scheduler.step()        
        if datavalid_loader is not None:
            result_valid = eval(model, datavalid_loader, optimizer, False)
        else:
            result_valid = [0,0]
        if datatest_loader is not None:
            result_test = eval(model, datatest_loader, optimizer, False)
        else:
            result_test = [0,0]            
        print(f'\n Loss train {result_train[0]}, Bleu train {result_train[1]};Loss valid {result_valid[0]}, Bleu valid {result_valid[1]};Loss test {result_test[0]}, Bleu test {result_test[1]}.')    
        if result_train[0] + result_valid[0] + result_test [0] < tmp_score:
            save('checkpoint/bestmodel.pth', model, optimizer, scheduler)
            tmp = result_train[0] + result_valid[0] + result_test [0]