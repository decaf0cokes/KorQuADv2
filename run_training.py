import pickle

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from model import DatasetKorquad, ElectraKorquad

def prepare_data(contexts_encoded, qas_encoded):
    datas=[]
    labels=[]

    for idx, qas in enumerate(qas_encoded):
        # For "idx"th context.
        # Remove [CLS].
        contexts_encoded[idx].remove(2)
        # Remove [SEP].
        contexts_encoded[idx].remove(3)

        for qa in qas:
            data=[]

            # <--qa['question']--><Segment><-1->
            # [CLS] Question [SEP] Segment [SEP] (Total Length:512)
            size_segment=511-len(qa['question'])

            pos_start=0
            pos_end=size_segment

            while pos_start<len(contexts_encoded[idx]):
                if pos_end>len(contexts_encoded[idx]):
                    segment=qa['question']+contexts_encoded[idx][pos_start:len(contexts_encoded[idx])]+[3]
                    # Padding.
                    segment+=[0]*(pos_end-len(contexts_encoded[idx]))
                else:
                    segment=qa['question']+contexts_encoded[idx][pos_start:pos_end]+[3]
                
                data.append(segment)

                pos_start+=size_segment
                pos_end+=size_segment
            
            datas.append(data)

            label_start=(int(qa['token_start']/size_segment), len(qa['question'])+qa['token_start']%size_segment)
            label_end=(int(qa['token_end']/size_segment), len(qa['question'])+qa['token_end']%size_segment)
            labels.append({'start':label_start, 'end':label_end})
    
    return datas, labels

def main():
    # Load encoded "train" contexts and questions/token positions of answers.
    with open('./pickles/contexts_encoded.pkl', 'rb') as f:
        contexts_encoded=pickle.load(f)
        f.close()
    with open('./pickles/qas_encoded.pkl', 'rb') as f:
        qas_encoded=pickle.load(f)
        f.close()
    
    # Prepare data for training model.
    datas, labels=prepare_data(contexts_encoded, qas_encoded)
    print(len(datas), "Datas &", len(labels), "Labels")

    # PyTorch Dataset/DataLoader.
    # "batch_size=1" DOES NOT MEAN "real batch size is 1".
    # "batch_size=1" MEANS "GPU processes ONE data at a time" because of memory limitation.
    # Optimizer steps per 32 datas processed(accumulation_steps).
    dataset=DatasetKorquad(datas=datas, labels=labels, max_segment=24)
    dataloader=DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    accumulation_steps=32

    # Load model on GPU or CPU. (I used one NVIDIA TITAN RTX GPU).
    model=ElectraKorquad()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.train()

    # Loss functions.
    criterion_segment=nn.BCEWithLogitsLoss()
    criterion_span=nn.CrossEntropyLoss()
    
    # Learning rate.
    lr=1e-4
    # Layer-wise learning rate decay.
    lr_decay=0.8
    # Optimizer.
    optim = AdamW(params=[
        {'params': model.pooler_segment.parameters(), 'lr': lr},
        {'params': model.attn.parameters(), 'lr': lr},
        {'params': model.pooler_span.parameters(), 'lr': lr},
        {'params': model.electra.encoder.layer[11].parameters(), 'lr': lr*(lr_decay**1)},
        {'params': model.electra.encoder.layer[10].parameters(), 'lr': lr*(lr_decay**2)},
        {'params': model.electra.encoder.layer[9].parameters(), 'lr': lr*(lr_decay**3)},
        {'params': model.electra.encoder.layer[8].parameters(), 'lr': lr*(lr_decay**4)},
        {'params': model.electra.encoder.layer[7].parameters(), 'lr': lr*(lr_decay**5)},
        {'params': model.electra.encoder.layer[6].parameters(), 'lr': lr*(lr_decay**6)},
        {'params': model.electra.encoder.layer[5].parameters(), 'lr': lr*(lr_decay**7)},
        {'params': model.electra.encoder.layer[4].parameters(), 'lr': lr*(lr_decay**8)},
        {'params': model.electra.encoder.layer[3].parameters(), 'lr': lr*(lr_decay**9)},
        {'params': model.electra.encoder.layer[2].parameters(), 'lr': lr*(lr_decay**10)},
        {'params': model.electra.encoder.layer[1].parameters(), 'lr': lr*(lr_decay**11)},
        {'params': model.electra.encoder.layer[0].parameters(), 'lr': lr*(lr_decay**12)},
        {'params': model.electra.embeddings.parameters(), 'lr': lr*(lr_decay**13)},
    ], lr=lr)
    # Learning rate scheduler.
    scheduler=get_linear_schedule_with_warmup(optimizer=optim, num_warmup_steps=0, num_training_steps=1919*2)

    for epoch in range(2):
        for index_batch, batch in enumerate(dataloader):

            segments=batch['segments'][0].to(device)
            label_segment=batch['label_segment'][0].to(device)
            
            output_segment, output_span=model(segments)
            
            # n(segments)*1 -> n(segments)
            output_segment=output_segment.squeeze(-1)
            
            # "Segment" loss.
            loss_segment=criterion_segment(output_segment, label_segment)
            
            # n(segments)*max_length*2 -> n(segments)*max_length*1
            logits_start, logits_end=output_span.split(1, dim=-1)
            # n(segments)*max_length
            logits_start=logits_start.squeeze(-1)
            # n(segments)*max_length
            logits_end=logits_end.squeeze(-1)
            # 1*max_length
            preds_start=logits_start[batch['label_start'][0].item()].unsqueeze(0)
            preds_end=logits_end[batch['label_end'][0].item()].unsqueeze(0)

            # "Span" loss.
            loss_start=criterion_span(preds_start, batch['label_start'][1].to(device))
            loss_end=criterion_span(preds_end, batch['label_end'][1].to(device))

            # Total loss.
            loss_total=(loss_segment+loss_start+loss_end)/accumulation_steps
            loss_total.backward()
            
            # Real batch size is 32(accumulation_steps).
            if (index_batch+1)%accumulation_steps==0:
                print('epoch', epoch+1, 'batch', (index_batch+1)/accumulation_steps)
                print('Loss', loss_total.item(), '\n')
                optim.step()
                scheduler.step()
                optim.zero_grad()

    model.eval()
    # Save model.
    torch.save(model, './KoELECTRA.pt')

if __name__=="__main__":
    main()
