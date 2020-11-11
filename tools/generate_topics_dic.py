import numpy as np
import os
import json
from random import shuffle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_label", help="train label json root")
parser.add_argument("--out_m_npy", help="output matrix npy root")
parser.add_argument("--data_root", help="data main root")
parser.add_argument("--select_num", help="choose * nums pos text for topics matrix")


args = parser.parse_args()

train_label_path = args.train_label
topics_m_path = args.out_m_npy
text_npy_dir = os.path.join(args.data_root, 'pos_text_npy')
select_num= args.select_num

f = open(train_label_path)
label_dic = json.load(f)
keys = label_dic.keys()

text_pos_keys_list = []
for key in keys:
    if label_dic[key]['text_label']:
        text_pos_keys_list.append(key)

shuffle(text_pos_keys_list)
select = text_pos_keys_list[:int(select_num)]
# print(select)

topics_m = []
for single_data in select:
    np_arr = np.load(os.path.join(text_npy_dir, single_data + '.npy'))
    topics_m.append(np_arr)

topics_m = np.array(topics_m)
# topics_m = topics_m.squeeze(1)
# print(topics_m.shape)

# print(topics_m_path)
np.save(topics_m_path, topics_m)
