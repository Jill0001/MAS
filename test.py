import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' #！change！
import torch
from torch.utils.data import Dataset,DataLoader
from video_audio_dataset import VideoAudioDataset
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.nn import Parameter
from torch import autograd
from myrnn import RNN


def load_pth(pth_path):
    return torch.load(pth_path)


def load_data(dataroot, json_name):
    test_dataset = VideoAudioDataset(dataroot,os.path.join(dataroot,json_name))
    tesr_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=True)
    return tesr_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = "/home/jiamengzhao/data_root/new_test_data_root"
test_dataloader = load_data(data_root,'label_test.json')
saved_model = load_pth("/home/jiamengzhao/repos/AudioVideoNet/saved_models/epoch0.pth")

# before_mf = np.load(os.path.join(data_root,'text_m_all.npy'))
# before_mf = torch.tensor(before_mf).float().to(device)

c_right = 0
t_right = 0
all_right = 0

for idx, i in enumerate(test_dataloader):
    input_a = i['np_A'].to(device)
    input_v = i['np_V'].to(device)

    input_t = i['text_data'].float().to(device)
    c_out, mf_out, t_out = saved_model(input_v, input_a,input_t)

    va_label = i['va_label']
    text_label = i['text_label']

    # print(c_out)
    if c_out[0]>c_out[1]:
        c_result = False
    else:
        c_result = True
    # print(c_result)
    # print(c_out)

    if t_out[0][0]>t_out[0][1]:
        t_result = False
    else:
        t_result = True

    # print(t_result)

    if va_label==c_result:
        c_right = c_right+1
    if text_label==t_result:
        t_right = t_right+1

    if va_label==c_result and text_label==t_result:
        all_right = all_right+1

    # if c_result
    # print(va_label,text_label)
    # print(c_out,t_out)
    # exit()

print(c_right,t_right,all_right)