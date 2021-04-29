import torch
from torch.utils.data import Dataset
import torch.nn as nn

from transformers import ElectraModel

class DatasetKorquad(Dataset):
    
    def __init__(self, datas, labels, max_segment):
        self.datas=[]
        self.labels_segment=[]
        self.labels_start=[]
        self.labels_end=[]
        
        for idx, label in enumerate(labels):
            # GPU memory issue..
            if len(datas[idx])>max_segment:
                continue
            
            # Append data(segments of document).
            self.datas.append(datas[idx])
            
            # Label of Segments
            label_segment=[0]*len(datas[idx])
            for index in range(label['start'][0], label['end'][0]+1):
                label_segment[index]=1
            self.labels_segment.append(label_segment)
            
            self.labels_start.append(label['start'])
            self.labels_end.append(label['end'])
            
        print(len(self.datas), "Datas")
        print(len(self.labels_segment), "Labels")
        print(len(self.labels_start), "Labels")
        print(len(self.labels_end), "Labels")
    
    def __getitem__(self,idx):
        item={}
        item['segments']=torch.tensor(self.datas[idx], dtype=torch.long)
        item['label_segment']=torch.tensor(self.labels_segment[idx], dtype=torch.float)
        item['label_start']=(self.labels_start[idx][0],torch.tensor(self.labels_start[idx][1], dtype=torch.long))
        item['label_end']=(self.labels_end[idx][0],torch.tensor(self.labels_end[idx][1], dtype=torch.long))
        return item
    
    def __len__(self):
        return len(self.datas)

class SelfAttention(nn.Module):
    
    def __init__(self,d_model):
        super().__init__()
        
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        
        self.temperature=d_model**0.5
        self.dropout=nn.Dropout(0.1)
        
    def forward(self,x):
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)
        
        attn=torch.matmul(q/self.temperature, k.transpose(0,1))
        attn=self.dropout(nn.functional.softmax(attn, dim=-1))
        
        output=torch.matmul(attn,v)
        
        return output

class ElectraKorquad(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Pre-trained Electra.
        self.electra=ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.config=self.electra.config
        
        # Pooling(Segment) layer.
        self.pooler_segment=nn.Linear(self.config.hidden_size, 1)
        self.attn=SelfAttention(self.config.hidden_size)
        
        # Pooling(Answer Span) layer.
        self.pooler_span=nn.Linear(self.config.hidden_size, 2)
        
    def forward(self, x):
        # Electra
        hidden=self.electra(x)[0]
        
        CLSs=[]
        for segment in hidden:
            CLSs.append(segment[0])
        output_segment=self.pooler_segment(self.attn(torch.stack(CLSs)))
        
        output_span=self.pooler_span(hidden)
        
        return output_segment, output_span

class ElectraKorquad(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Pre-trained Electra.
        self.electra=ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.config=self.electra.config
        
        # Pooling(Segment) layer.
        self.pooler_segment=nn.Linear(self.config.hidden_size, 1)
        self.attn=SelfAttention(self.config.hidden_size)
        
        # Pooling(Answer Span) layer.
        self.pooler_span=nn.Linear(self.config.hidden_size, 2)
        
    def forward(self, x):
        # Electra
        hidden=self.electra(x)[0]
        
        CLSs=[]
        for segment in hidden:
            CLSs.append(segment[0])
        output_segment=self.pooler_segment(self.attn(torch.stack(CLSs)))
        
        output_span=self.pooler_span(hidden)
        
        return output_segment, output_span
