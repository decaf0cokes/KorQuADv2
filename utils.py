import os
import json
import re
from bs4 import BeautifulSoup

def load_data(path='./KorQuADv2/', sort="train"):
    # Load .json files, not .zip files.
    files=[]
    for file in os.listdir(path):
        if file.split(".")[-1]=="json":
            files.append(file)

    # Load "train" or "dev" data.
    datas=[]
    for file in sorted(files):
        if file.split("_")[1]==sort:
            print(file)
            with open(path+file, 'r') as f:
                datas.append(json.load(f))
                f.close()
    
    # Extract documents from data.
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

def preprocess(documents):
    for idx, document in enumerate(documents):
        context=document['context']

        # Replace "td" tags.
        tags_td=list(set(re.findall("<td[^>]*>", context)))
        for tag in tags_td:
            context=re.sub(tag, replace_table_tags(tag=tag, sort="td"), context)

        # Replace "th" tags.
        tags_th=list(set(re.findall("<th[^>]*>", context)))
        for tag in tags_th:
            context=re.sub(tag, replace_table_tags(tag=tag, sort="th"), context)
        
        documents[idx]['context']=context
