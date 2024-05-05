import torch
import os
import csv
import numpy as np
import nltk
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def write_accuracy_to_csv(filename, train, valid, test):
    fieldnames = ["Train Accuracy", "Validation Accuracy", "Test Accuracy", "Train Loss", "Validation Loss", "Test Loss"]
    
    # Kiểm tra xem file đã tồn tại chưa
    if not os.path.isfile(filename):
        if not os.path.exists('metric'):
            os.makedirs('metric')
        # Nếu file chưa tồn tại, tạo một file mới và ghi thông tin đầu tiên vào nó
        with open(filename, mode='x', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"Train Accuracy": train[1], "Validation Accuracy": valid[1], "Test Accuracy": test[1],"Train Loss": train[0], "Validation Loss": valid[0], "Test Loss": test[0]})
    else:
        # Nếu file đã tồn tại, mở file trong chế độ append để thêm dữ liệu mới vào cuối file
        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({"Train Accuracy": train[1], "Validation Accuracy": valid[1], "Test Accuracy": test[1],"Train Loss": train[0], "Validation Loss": valid[0], "Test Loss": test[0]})

def SparseCrossEntropy(true, pred):
    #shape pred is [batch_size, length, embed_size]
    #shape true is [batch_size, length]
    pred_tmp = torch.gather(pred[:, :-1, :].softmax(dim = -1), dim = -1, index = true.unsqueeze(dim = -1)[:, 1:,:])
    weights = true.unsqueeze(dim = -1)[:, 1:,:] > 0
    return (torch.log(pred_tmp) * -1 * weights).sum() / (weights.sum())

def transformer_lr(step_num, d_model = 512, warmup_steps = 4000):
    if step_num == 0:
        step_num += 1
    lr = d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    return lr
def bleu_score(infer, candi):
    candi_numpy = candi.to('cpu').detach().numpy().astype(str).tolist()
    infer_numpy = infer.to('cpu').detach().numpy().astype(str).tolist()
    return nltk.translate.bleu_score.corpus_bleu(infer_numpy,candi_numpy)

def accuracy(true, pred):
    #shape pred is [batch_size, length, embed_size]
    #shape true is [batch_size, length]    
    pred_tmp = torch.argmax(pred, dim = -1)
    weights = true > 0
    return ((pred_tmp == true) * weights).sum() / (weights.sum())

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

def save(path, model, optimizer, scheduler, epoch, score):
    state = {
        'model' : model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'score': score
    }           
    torch.save(state, path)

def load(path, model, optimizer, scheduler):
    state = torch.load(path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scheduler.optimizer = optimizer
    epoch = state['epoch']
    score = state['score']
    return model, optimizer, scheduler, epoch, score            

def eval(model, data_loader,optimizer, scheduler, is_training = True):
    mean_loss = []      
    acc = []  
    infer = None
    candidate = None    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    if is_training:
        model.train()
        data_iter = tqdm(data_loader, desc='Training', position=0, leave=True)        
        for input, target in data_iter:
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            output = model(input, target)
            loss = SparseCrossEntropy(target, output)
            mean_loss.append(loss.item())
            acc.append(accuracy(target, output).item())
            loss.backward()
            optimizer.step()
            scheduler.step()       
            #if infer is not None:
                #infer = torch.cat([infer, output.to('cpu').argmax(dim = -1)], dim = 0)
            #else:
                #infer = output.to('cpu').argmax(dim = -1)
            #if candidate is not None:
                #candidate = torch.cat([candidate, target.to('cpu')], dim = 0)   
            #else:
                #candidate = target.to('cpu')                 
    else:
        model.eval()    
        with torch.no_grad():
            mean_loss = []   
            acc = []     
            infer = None
            candidate = None
            data_iter = tqdm(data_loader, desc='Not training', position=0, leave=True)        
            for input, target in data_iter:
                input, target = input.to(device), target.to(device)
                output = model(input, target)
                loss  = SparseCrossEntropy(target, output)
                mean_loss.append(loss.item())
                acc.append(accuracy(target, output).item())
                #if infer is not None:
                    #infer = torch.cat([infer, output], dim = 0)
                #else:
                    #infer = output
                #if candidate is not None:
                    #candidate = torch.cat([candidate, target], dim = 0)   
                #else:
                    #candidate = target      
    return np.mean(np.array(mean_loss)), np.mean(np.array(acc))            
                
                    

def train(model, optimizer, epoch, datatrain_loader,datavalid_loader = None, datatest_loader = None, path = None):
    scheduler = LambdaLR(optimizer, lr_lambda= lambda step_num : transformer_lr(step_num=step_num, d_model=model.dmodel))
    tmp_score = 1e30
    if path is not None:
        model, optimizer, scheduler, epoch_old, tmp_score = torch.load(path, model, optimizer, scheduler)
    else:
        epoch_old = 0    
    for i in tqdm(range(epoch - epoch_old), desc='Epoch', position=0, leave=True):
        result_train =  eval(model, datatrain_loader, optimizer, scheduler, True)   
        if datavalid_loader is not None:
            result_valid = eval(model, datavalid_loader, optimizer, scheduler, False)
        else:
            result_valid = [0,0]
        if datatest_loader is not None:
            result_test = eval(model, datatest_loader, optimizer,scheduler, False)
        else:
            result_test = [0,0]            
        print(f'\n Loss train {result_train[0]}, Acc train {result_train[1]};Loss valid {result_valid[0]}, Acc valid {result_valid[1]};Loss test {result_test[0]}, Acc test {result_test[1]}.')    
        if result_train[0] + result_valid[0] + result_test [0] < tmp_score:
            if path is not None:
                save(path, model, optimizer, scheduler, i, result_train[0] + result_valid[0] + result_test [0])
                tmp_score = result_train[0] + result_valid[0] + result_test [0]
            else:
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                save('checkpoint/bestmodel.pth', model, optimizer, scheduler, i, result_train[0] + result_valid[0] + result_test [0])
                tmp_score = result_train[0] + result_valid[0] + result_test [0]
        write_accuracy_to_csv("metric/metric.csv", result_train, result_valid, result_test)                            