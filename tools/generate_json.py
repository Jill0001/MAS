import os
import json
import numpy as np
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default='/home/jiamengzhao/data_root/data_for_sample',help="main data root")
parser.add_argument("--train_label_json",default='/home/jiamengzhao/data_root/data_for_sample/train.json', help="train label json path")
parser.add_argument("--test_label_json", default='/home/jiamengzhao/data_root/data_for_sample/test.json',help="test label json path")
parser.add_argument("--val_label_json", default='/home/jiamengzhao/data_root/data_for_sample/val.json',help="val label json path")

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
text_npy_dir_neg = os.path.join(data_root, 'text_neg_npy')

video_dir = os.path.join(data_root,'lmks')

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
    one_sampel_dic['audio_path'] = sample_audio_npy_path.split('/')[-2] + "/" + sample_audio_npy_path.split('/')[-1]
    one_sampel_dic['video_path'] = one_sample_name+'.avi'

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

    elif train_or_test > 0.1 and one_sampel_dic != []:
        val_json[one_sample_name] = one_sampel_dic

    elif train_or_test <= 0.1 and one_sampel_dic != []:
        test_json[one_sample_name] = one_sampel_dic


def generate_fake_neg3(sample_audio_npy_path):
    one_sample_name = sample_audio_npy_path.split('/')[-1].replace('.npy', '_neg3')
    video_path_list = os.listdir(video_dir)
    video_num = len(video_path_list)

    one_sampel_dic = {}
    one_sampel_dic['va_label'] = 0
    one_sampel_dic['audio_path'] = sample_audio_npy_path.split('/')[-2] + "/" + sample_audio_npy_path.split('/')[-1]

    while True:
        rand = random.randint(0, video_num - 1)
        if video_path_list[rand].replace('.avi','') not in one_sampel_dic['audio_path']:
            one_sampel_dic['video_path'] = video_path_list[rand]
            break

    pos_text_names = os.listdir(text_npy_dir_pos)
    one_sample_text_name = sample_audio_npy_path.split('/')[-1]

    if one_sample_text_name in pos_text_names:
        one_sampel_dic['text_label'] = 1
    else:

        one_sampel_dic = []

    train_or_test = np.random.rand(1)
    if train_or_test > 0.2 and one_sampel_dic != []:
        train_json[one_sample_name] = one_sampel_dic
    elif train_or_test > 0.1 and one_sampel_dic != []:
        val_json[one_sample_name] = one_sampel_dic
    elif train_or_test <= 0.1 and one_sampel_dic != []:
        test_json[one_sample_name] = one_sampel_dic


for root in [neg1_audio_dir, neg2_audio_dir, pos_audio_dir]:
    audio_npy_list = os.listdir(root)
    for single_npy in audio_npy_list:
        single_npy_path = os.path.join(root, single_npy)
        if single_npy_path.endswith('.npy'):
            generate_one_sample(single_npy_path)
            # sample_id = sample_id + 1
            # print(sample_id)
for fake_neg3 in os.listdir(pos_audio_dir):
    fake_neg3_path = os.path.join(pos_audio_dir, fake_neg3)
    generate_fake_neg3(fake_neg3_path)
    # sample_id = sample_id + 1


def write_json_file(json_path, json_dic):
    with open(json_path, "w") as f:
        json.dump(json_dic, f, indent=4, ensure_ascii=False)


def fix_id_problem(json_dic):
    key_list = list(json_dic.keys())
    key_num = len(key_list)
    for i in range(key_num):
        json_dic[key_list[i]]['id'] = i


for dic in [train_json,test_json,val_json]:
    fix_id_problem(dic)

write_json_file(train_json_path, train_json)
write_json_file(test_json_path, test_json)
write_json_file(val_json_path, val_json)

