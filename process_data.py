import os
import json
import re
from bs4 import BeautifulSoup
import pickle
from transformers import ElectraTokenizer

def load_train(path='./KorQuADv2/'):
    files=os.listdir(path)

    datas=[]
    for file in sorted(files):
        if file.split("_")[1]=="train":
            print(file)
            with open(path+file, 'r') as f:
                datas.append(json.load(f))
                f.close()
    
    documents=[]
    for data in datas:
        for document in data['data']:
            documents.append(document)
    print("Total", len(documents), "Documents")

    return documents        

def replace_table_tags(tag, sort):
    soup=BeautifulSoup(tag, 'html.parser')

    if sort=="td":
        try:
            colspan=soup.td['colspan']
        except(ValueError, KeyError):
            colspan=None

        try:
            rowspan=soup.td['rowspan']
        except(ValueError, KeyError):
            rowspan=None

    if sort=="th":
        try:
            colspan=soup.th['colspan']
        except(ValueError, KeyError):
            colspan=None

        try:
            rowspan=soup.th['rowspan']
        except(ValueError, KeyError):
            rowspan=None
    
    if colspan is not None and rowspan is not None:
        tag_replaced="<{} cs rs>{} {}".format(sort, colspan, sort, rowspan)
    elif colspan is not None and rowspan is None:
        tag_replaced="<{} cs>{}".format(sort, colspan)
    elif colspan is None and rowspan is not None:
        tag_replaced="<{} rs>{}".format(sort, rowspan)
    else:
        tag_replaced="<{}>".format(sort)
            
    if len(tag)>len(tag_replaced):
        tag_replaced=tag_replaced+" "*(len(tag)-len(tag_replaced))

    return tag_replaced

def main():
    documents=load_train()

    print("Processing HTML Tags..")
    tags_html=[]
    for idx, document in enumerate(documents):
        context=document['context']

        tags_td=list(set(re.findall("<td[^>]*>", context)))
        for tag in tags_td:
            context=re.sub(tag, replace_table_tags(tag=tag, sort="td"), context)

        tags_th=list(set(re.findall("<th[^>]*>", context)))
        for tag in tags_th:
            context=re.sub(tag, replace_table_tags(tag=tag, sort="th"), context)
        
        tags_html=list(set(tags_html+re.findall("<[^>]*>", context)))
        documents[idx]['context']=context
    print("Done!")

    tokenizer=ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    for idx, tag in enumerate(sorted(tags_html)):
        tokenizer.vocab[tag]=tokenizer.vocab["[unused{}]".format(idx)]
        del tokenizer.vocab["[unused{}]".format(idx)]
    tokenizer.add_special_tokens({'additional_special_tokens':sorted(tags_html)})

    contexts_encoded=[]
    for idx, document in enumerate(documents):
        print("Encoding", idx+1, "th Document..")
        contexts_encoded.append(tokenizer.encode(document['context']))
    with open('./pickles/contexts_encoded.pkl', 'wb') as f:
        pickle.dump(contexts_encoded, f)
        f.close()

if __name__=="__main__":
    main()
