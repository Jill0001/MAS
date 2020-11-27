import os
import json
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", help="main data root")
parser.add_argument("--train_label_json", help="train label json path")
parser.add_argument("--test_label_json", help="test label json path")
parser.add_argument("--val_label_json", help="val label json path")

args = parser.parse_args()
data_root = args.data_root
train_json_path = args.train_label_json
test_json_path = args.test_label_json
val_json_path = args.val_label_json


# audio_npy_root = "/home/jiamengzhao/repos/AudioVideoNet/new_test_data_root"
neg1_audio_dir = os.path.join(data_root, 'neg1_audio_npy')

neg2_audio_dir = os.path.join(data_root, 'neg2_audio_npy')
pos_audio_dir = os.path.join(data_root, 'pos_audio_npy')

text_npy_dir_pos = os.path.join(data_root, 'text_pos_npy')
text_npy_dir_neg = os.path.join(data_root, 'neg2_text_npy')

dict_json = {}
train_json = {}
val_json = {}
test_json = {}


def load_json(json_path):
    with open(json_path) as js:
        json_dic = json.load(js)
        return json_dic


train_id = 0
test_id = 0
val_id =0
all_id = 0


def generate_one_sample(sample_audio_npy_path):
    one_sample_name = sample_audio_npy_path.split('/')[-1].replace('.npy', '')
    one_sampel_dic = {}
    if "pos_audio" in sample_audio_npy_path or "neg2_audio" in sample_audio_npy_path:
        one_sampel_dic['va_label'] = 1
    elif "neg1_audio" in sample_audio_npy_path:
        one_sampel_dic['va_label'] = 0

    # one_sampel_dic['name'] = one_sample_name
    one_sampel_dic['relative_path'] = single_npy_path.split('/')[-2] + "/" + single_npy_path.split('/')[-1]

    pos_text_names = os.listdir(text_npy_dir_pos)
    neg_text_names = os.listdir(text_npy_dir_neg)
    one_sample_text_name = one_sample_name + '.npy'

    # a_pos_text_npy_path = os.path.join(text_npy_dir_pos, pos_text_names[0])
    # one_sampel_dic['text_length'] = np.load(a_pos_text_npy_path).shape[0]

    if one_sample_text_name in pos_text_names:
        one_sampel_dic['text_label'] = 1
    elif one_sample_text_name in neg_text_names:
        one_sampel_dic['text_label'] = 0
        # one_sampel_dic=[]
    elif one_sampel_dic['va_label'] == 0:
        one_sampel_dic['text_label'] = 0
        # one_sampel_dic = []
    else:
        one_sampel_dic = []

    train_or_test = np.random.rand(1)
    if train_or_test > 0.2 and one_sampel_dic != []:
        train_json[one_sample_name] = one_sampel_dic
        global train_id
        one_sampel_dic['id'] = train_id
        train_id = train_id + 1
    elif train_or_test > 0.1 and one_sampel_dic != []:
        val_json[one_sample_name] = one_sampel_dic
        global val_id
        one_sampel_dic['id'] = val_id
        val_id = val_id + 1
    elif train_or_test <= 0.1 and one_sampel_dic != []:
        global test_id
        test_json[one_sample_name] = one_sampel_dic
        one_sampel_dic['id'] = test_id
        test_id = test_id + 1

    # global all_id
    # one_sampel_dic['id'] = all_id
    # all_id = all_id+1
    # dict_json[one_sample_name]=one_sampel_dic


sample_id = 0
for root in [neg1_audio_dir, neg2_audio_dir, pos_audio_dir]:
    audio_npy_list = os.listdir(root)
    for single_npy in tqdm(audio_npy_list):
        single_npy_path = os.path.join(root, single_npy)
        if single_npy_path.endswith('.npy'):
            generate_one_sample(single_npy_path)
            sample_id = sample_id + 1
            # print(sample_id)


def write_json_file(json_path, json_dic):
    with open(json_path, "w") as f:
        json.dump(json_dic, f, indent=4, ensure_ascii=False)


# write_json_file(main_json_path,dict_json)
write_json_file(train_json_path, train_json)
write_json_file(test_json_path, test_json)
write_json_file(val_json_path, val_json)

