import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' #！change！
import torch
from torch.utils.data import Dataset,DataLoader
from video_audio_dataset import VideoAudioDataset
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.nn import Parameter
from torch import autograd
from model_new import VATNN

def load_pth(pth_path):
    return torch.load(pth_path,map_location=torch.device('cpu'))


def load_data(dataroot, json_name):
    test_dataset = VideoAudioDataset(dataroot,os.path.join(dataroot,json_name))
    tesr_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=True)
    return tesr_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = "/home/jiamengzhao/data_root/data_for_sample"
test_dataloader = load_data(data_root,'test.json')
saved_model = load_pth("/home/jiamengzhao/repos/AudioVideoNet/saved_models/epoch0.pth")

# before_mf = np.load(os.path.join(data_root,'topics.npy'))
# before_mf = torch.tensor(before_mf).float().to(device)


c_right = 0
c_tp = 0
t_right = 0
t_tp = 0
all_right = 0
all_tp = 0

c_result = []
c_label = []
t_result = []
t_label = []
all_result =[]
all_label = []

for idx, i in enumerate(test_dataloader):
    input_a = torch.unsqueeze(i['np_A'], 1).to(device)
    input_v = torch.unsqueeze(i['np_V'], 1).to(device)
    input_t = i['text_data'].to(device)

    va_label = i['va_label'].long().to(device)
    # print(va_label)
    text_label = i['text_label'].long().to(device)

    with torch.no_grad():
        all_out, mf_out = saved_model(input_v, input_a, input_t)

    # c_out, mf_out, t_out = saved_model(input_v, input_a,input_t)
    # all_out = c_out*t_out

    all_result.append(bool((all_out[0][0].cpu() <= all_out[0][1].cpu())))
    all_label.append(bool(va_label * text_label))

for j in range(len(all_result)):

    if all_result[j] == all_label[j]:
        all_right += 1
        if all_result[j]:
            all_tp += 1

print(all_right,len(all_result),all_right/len(all_result))

p_all = all_tp / (all_result.count(True) + 1e-6)
r_all = all_tp / (all_label.count(True) + 1e-6)
F1_all = (2 * p_all * r_all) / (p_all + r_all + 1e-6)

print("precision_all: %f\nrecall_all: %f\nF1_all: %f\n" % (p_all, r_all, F1_all))

