import numpy as np
import os
import torch
from transformers import BertTokenizer,BertModel
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--input_txt",)
parser.add_argument('-o',"--output_dir",)

args = parser.parse_args()

input_txt = args.input_txt
output_dir= args.output_dir

all_text = []
all_file_name = []
total_lines = 0
with open(input_txt) as input_lines:
    for l in input_lines:
        total_lines = total_lines+1
with open(input_txt) as f:
    text_chuck = []
    file_name_chuck = []
    for line in f:
        file_name = line.split(' ')[0]
        text = line.split(' ')[1].replace('\n','')
        file_name_chuck.append(file_name)
        text_chuck.append(text)
        if len(text_chuck)>100 or len(text_chuck) == total_lines:
            all_text.append(text_chuck)
            all_file_name.append(file_name_chuck)
            text_chuck = []
            file_name_chuck = []


for chuck_id in tqdm(range(len(all_text))):

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') # Bert的分词器
    bertmodel = BertModel.from_pretrained('bert-base-chinese') # load the TF model for Pytorch

    tokenized_text = tokenizer(all_text[chuck_id],return_tensors='pt',padding=True, truncation=True,)
    bertmodel.eval()
    # print(tokenized_text1)
    output = bertmodel(tokenized_text['input_ids'],attention_mask=tokenized_text['attention_mask'])
    # print(output[1].shape)

    sentence_arr = output[1].detach().numpy()

    save_dir = output_dir

    for index,arr in enumerate(sentence_arr):
        save_npy_name = os.path.join(save_dir,all_file_name[chuck_id][index].split('/')[-1].replace('.wav',''))
        print(arr.shape)
        np.save(save_npy_name,arr)
    # save_npy_name = os.path.join(save_dir, all_file_name[chuck_id].split('/')[-1].replace('.wav', ''))
    # np.save(save_npy_name, sentence_arr)

# for index,arr in enumerate(all_sentence_arr):
#     save_npy_name = os.path.join(save_dir,all_file_name[index].replace('.wav',''))
#     np.save(save_npy_name,arr)

