import os
import pickle

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
    # Load encoded contexts and questions/token positions of answers.
    with open('./pickles/contexts_encoded.pkl', 'rb') as f:
        contexts_encoded=pickle.load(f)
        f.close()
    with open('./pickles/qas_encoded.pkl', 'rb') as f:
        qas_encoded=pickle.load(f)
        f.close()
    
    # Prepare data for training model.
    datas, labels=prepare_data(contexts_encoded, qas_encoded)

if __name__=="__main__":
    main()
