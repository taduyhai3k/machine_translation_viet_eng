import pandas as pd
import re
import json
import torch 
from torch.utils.data import Dataset, DataLoader 

class MyTokenizer():
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.invers_vocab = {value: key for key, value in self.vocab.items()}
        self.BOS = '<start>'
        self.EOS = '<end>'
    def tokenize(self,w):
        w = w.lower().strip()
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = w.strip()
        # Add start and end token 
        w = '{} {} {}'.format(self.BOS, w, self.EOS)
        return w    
    def encode(self, w):
        results = []
        x = self.tokenize(w)
        for word in x.split(sep = ' '):
            results.append(self.vocab[word])
        return torch.tensor(results, dtype = torch.int64, requires_grad= False)
    def decode(self,w):
        results = []   
        for i in range(len(w)):
            results.append(self.invers_vocab[w[i].item()])
        return ' '.join(results)     
    
class EV_Data(Dataset):
    def __init__(self, data_path, E_vocab_path = 'vocab/vocab_E.json' , V_vocab_path = 'vocab/vocab_V.json', inp = 'E', out = 'V', max_length = 300 ) -> None:
        super().__init__()
        self.inp = inp
        self.out = out
        self.data_path = data_path
        self.E_vocab_path = E_vocab_path
        self.V_vocab_path = V_vocab_path
        self.max_length = max_length 
        self.__read_data__()
        self.__read_vocab__()
    
    def __read_data__(self):
        self.data = pd.read_json(self.data_path)
    def __read_vocab__(self):
        if self.inp == 'E':
            with open(self.E_vocab_path, 'r', encoding='utf-8') as f:
                self.inp_vocab = json.load(f)    
            self.inp_tokenizer = MyTokenizer(self.inp_vocab)
            with open(self.V_vocab_path, 'r', encoding='utf-8') as f:
                self.out_vocab = json.load(f)    
            self.out_tokenizer = MyTokenizer(self.out_vocab)        
        else:
            with open(self.V_vocab_path, 'r', encoding='utf-8') as f:
                self.inp_vocab = json.load(f)    
            self.inp_tokenizer = MyTokenizer(self.inp_vocab)
            with open(self.E_vocab_path, 'r', encoding='utf-8') as f:
                self.out_vocab = json.load(f)    
            self.out_tokenizer = MyTokenizer(self.out_vocab)                 
    def __len__(self):
        return len(self.data)        
    def __getitem__(self, index):
        if self.inp == 'E':
            input = self.inp_tokenizer.encode(self.data.iloc[index]['E']) 
            output = self.out_tokenizer.encode(self.data.iloc[index]['V'])
        else:
            input = self.inp_tokenizer.encode(self.data.iloc[index]['V']) 
            output = self.out_tokenizer.encode(self.data.iloc[index]['E'])                
        return torch.nn.functional.pad(input, pad = (0, self.max_length - input.shape[0]), value = 0), torch.nn.functional.pad(output, pad = (0, self.max_length - output.shape[0]), value = 0)
    