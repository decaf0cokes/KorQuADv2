import pickle
import re
import json

import torch
from torch.utils.data import DataLoader

from transformers import ElectraTokenizer

from model import DevsetKorquad, SelfAttention, ElectraKorquad

def prepare_data(contexts_encoded, qs_encoded):
    datas=[]
    ids=[]

    for idx, qs in enumerate(qs_encoded):
        # For "idx"th context.
        # Remove [CLS].
        contexts_encoded[idx].remove(2)
        # Remove [SEP].
        contexts_encoded[idx].remove(3)

        for q in qs:
            data=[]

            # <--qa['question']--><Segment><-1->
            # [CLS] Question [SEP] Segment [SEP] (Total Length:512)
            size_segment=511-len(q['question'])

            pos_start=0
            pos_end=size_segment

            while pos_start<len(contexts_encoded[idx]):
                if pos_end>len(contexts_encoded[idx]):
                    segment=q['question']+contexts_encoded[idx][pos_start:len(contexts_encoded[idx])]+[3]
                    # Padding.
                    segment+=[0]*(pos_end-len(contexts_encoded[idx]))
                else:
                    segment=q['question']+contexts_encoded[idx][pos_start:pos_end]+[3]
                
                data.append(segment)

                pos_start+=size_segment
                pos_end+=size_segment
            
            datas.append(data)
            ids.append(q['id'])
    
    return datas, ids

def main():
    # Load encoded "dev" contexts and questions.
    with open('./pickles/contexts_dev_encoded.pkl', 'rb') as f:
        contexts_dev_encoded=pickle.load(f)
        f.close()
    with open('./pickles/qs_dev_encoded.pkl', 'rb') as f:
        qs_dev_encoded=pickle.load(f)
        f.close()
    
    # Prepare data for prediction.
    datas, ids=prepare_data(contexts_dev_encoded, qs_dev_encoded)
    print(len(datas), "Dats", len(ids), "IDs")

    dataset=DevsetKorquad(datas=datas, ids=ids, max_segment=24)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=False)

    with open('./pickles/tags_html.pkl', 'rb') as f:
        tags_html=pickle.load(f)
        f.close()
    print(len(tags_html), "Tags")

    tokenizer=ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    dict_html={}
    for idx, tag in enumerate(sorted(tags_html)):
        dict_html["[unused{}]".format(idx)]=tag

        tokenizer.vocab[tag]=tokenizer.vocab["[unused{}]".format(idx)]
        del tokenizer.vocab["[unused{}]".format(idx)]
    tokenizer.add_special_tokens({'additional_special_tokens':sorted(tags_html)})

    """
    Prediction
    """
    model=torch.load('./KoELECTRA.pt')
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    json_preds={}
    for index_batch, batch in enumerate(dataloader):
        print("batch", index_batch+1)
        
        segments=batch['segments'][0].to(device)
        
        output_segment, output_span=model(segments)
        
        # n(segments)*1 -> n(segments)
        output_segment=output_segment.squeeze(-1)
    #     print(output_segment)
    #     print(torch.argmax(output_segment))
        id_segment=torch.argmax(output_segment).item()
        
        # n(segments)*max_length*2 -> n(segments)*max_length*1
        logits_start, logits_end=output_span.split(1,dim=-1)
        # n(segments)*max_length
        logits_start=logits_start.squeeze(-1)
        # n(segments)*max_length
        logits_end=logits_end.squeeze(-1)
        
    #     print(torch.argmax(logits_start[2]))
    #     print(torch.argmax(logits_end[2]))
    #     print()
        pos_start=torch.argmax(logits_start[id_segment]).item()
        pos_end=torch.argmax(logits_end[id_segment]).item()
        if(pos_start>pos_end):
            json_preds[batch['id'][0]]=""
        else:
            ans=tokenizer.decode(batch['segments'][0][id_segment][pos_start:pos_end])
            
            unuseds=list(set(re.findall("\[unused[^\]]*\]", ans)))
            for unused in unuseds:
                ans=ans.replace(unused,dict_html[unused])
                
            json_preds[batch['id'][0]]=ans
            print(ans)
        
        del segments, output_segment, output_span, logits_start, logits_end
        torch.cuda.empty_cache()

    with open('./preds_KoELECTRA.json', 'w') as f:
        json.dump(json_preds, f, ensure_ascii=False)

if __name__=="__main__":
    main()
