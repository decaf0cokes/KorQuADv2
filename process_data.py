import re
import pickle
from transformers import ElectraTokenizer

from utils import load_data, preprocess

def main():
    """
    Process "train" data.
    """
    # Load "train" data.
    documents_train=load_data(path='./KorQuADv2/', sort="train")
    
    # Preprocess "train" data.
    preprocess(documents_train)

    # Find all HTML tags in "train" data.
    tags_html=[]
    for document in documents_train:
        context=document['context']
        tags_html=list(set(tags_html+re.findall("<[^>]*>", context)))
    # Save HTML tags list as .pkl file.
    with open('./pickles/tags_html.pkl', 'wb') as f:
        pickle.dump(sorted(tags_html), f)
        f.close()

    # Load pre-trained tokenizer.
    # Replace [unusedX] in vocab with HTML tags.
    # Also, add HTML tags to special tokens.
    tokenizer=ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    for idx, tag in enumerate(sorted(tags_html)):
        tokenizer.vocab[tag]=tokenizer.vocab["[unused{}]".format(idx)]
        del tokenizer.vocab["[unused{}]".format(idx)]
    tokenizer.add_special_tokens({'additional_special_tokens':sorted(tags_html)})
    
    # Encode "train" contexts.
    contexts_encoded=[]
    for idx, document in enumerate(documents_train):
        print("Encoding", idx+1, "th Document..")
        contexts_encoded.append(tokenizer.encode(document['context']))
    # Save encoded "train" contexts as .pkl file.
    with open('./pickles/contexts_encoded.pkl', 'wb') as f:
        pickle.dump(contexts_encoded, f)
        f.close()
    
    # Add index where answer ends.
    for document in documents_train:
        qas=document['qas']
        for qa in qas:
            answer=qa['answer']['text']
            index_start=qa['answer']['answer_start']
            index_end=index_start+len(answer)
            qa['answer']['answer_end']=index_end
    
    # Encode "train" questions and add token positions where answer starts/ends.
    qas_encoded=[]
    for idx, document in enumerate(documents_train):
        print("Encoding", idx+1, "th Document..")
        context=document['context']
        qas=document['qas']

        qas_encoded_element=[]
        for qa in qas:
            question=tokenizer.encode(qa['question'])
            
            index_start=qa['answer']['answer_start']
            index_end=qa['answer']['answer_end']

            token_start=len(tokenizer.tokenize(context[0:index_start]))
            token_end=token_start+len(tokenizer.tokenize(context[index_start:index_end]))

            qas_encoded_element.append({'question':question, 'token_start':token_start, 'token_end':token_end})
        qas_encoded.append(qas_encoded_element)
    # Save encoded "train" questions and token positions as .pkl file.
    with open('./pickles/qas_encoded.pkl', 'wb') as f:
        pickle.dump(qas_encoded, f)
        f.close()
    
    """
    Process "dev" data.
    """
    
    # Load "dev" data.
    documents_dev=load_data(path='./KorQuADv2/', sort="dev")
    
    # Preprocess "dev" data.
    preprocess(documents_dev)

    # Encode "dev" contexts.
    contexts_dev_encoded=[]
    for idx, document in enumerate(documents_dev):
        print("Encoding", idx+1, "th Document..")
        contexts_dev_encoded.append(tokenizer.encode(document['context']))
    # Save encoded "dev" contexts as .pkl file.
    with open('./pickles/contexts_dev_encoded.pkl', 'wb') as f:
        pickle.dump(contexts_dev_encoded, f)
        f.close()
    
    # Encode "dev" questions.
    qs_dev_encoded=[]
    for idx, document in enumerate(documents_dev):
        print("Encoding", idx+1, "th Document..")
        qas=document['qas']

        qs_dev_encoded_element=[]
        for qa in qas:
            question=tokenizer.encode(qa['question'])
            id_question=qa['id']

            qs_dev_encoded_element.append({'question':question, 'id':id_question})
        qs_dev_encoded.append(qs_dev_encoded_element)
    # Save encoded "dev" questions as .pkl file.
    with open('./pickles/qs_dev_encoded.pkl', 'wb') as f:
        pickle.dump(qs_dev_encoded, f)
        f.close()

if __name__=="__main__":
    main()
