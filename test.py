import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #！change！
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
    return torch.load(pth_path)


def load_data(dataroot, json_name):
    test_dataset = VideoAudioDataset(dataroot,os.path.join(dataroot,json_name))
    tesr_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=True)
    return tesr_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = "/home/scf/PycharmProjects/AudioVideoNet/data_root/new_test_data_root"
test_dataloader = load_data(data_root,'train_label_fake.json')
saved_model = load_pth("/home/scf/PycharmProjects/AudioVideoNet/repos/AudioVideoNet/saved_models/epoch0.pth")

# before_mf = np.load(os.path.join(data_root,'text_m_all.npy'))
# before_mf = torch.tensor(before_mf).float().to(device)

def extract_v_feature(input_v):
    batch_video_npys = []
    for dir in input_v:
        arr_names = [f for f in os.listdir(dir) if f.endswith('.npy')]
        video_npy = []
        for i in range(1, len(arr_names) + 1):
            arr_name = os.path.join(dir, str(i) + '.npy')
            arr_number = np.load(arr_name)
            center = arr_number[30]
            # after_decenter = np.zeros((1, 2))
            after_decenter = []
            for j in arr_number:
                after_decenter.append(j - center)

                # after_decenter = np.concatenate((after_decenter, (np.expand_dims(j - center, 0))))
            after_decenter = np.array(after_decenter)
            video_npy.append(after_decenter)
        video_npy = np.array(video_npy)
        if video_npy.shape[0]<500: #padding size
            padding_zeros = np.zeros((500-video_npy.shape[0],video_npy.shape[1],video_npy.shape[2]))
            video_npy = np.concatenate((video_npy,padding_zeros))
        batch_video_npys.append(video_npy)
        batch_video_npys_arr = np.array(batch_video_npys)
    return batch_video_npys_arr

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
    input_a = torch.unsqueeze((i['np_A']).float().to(device), dim=1)
    input_v = torch.unsqueeze(torch.Tensor(extract_v_feature(i['np_V'])).to(device), dim=1)
    input_t = i['text_data'].float().to(device)

    va_label = i['va_label']
    text_label = i['text_label']

    c_out, mf_out, t_out = saved_model(input_v, input_a,input_t)
    c_out = torch.squeeze(c_out, dim=1)
    c_result.append(bool(c_out[0].cpu()>=0.5))
    c_label.append(bool(va_label))
    t_result.append(bool(t_out[0].cpu()>=0.5))
    t_label.append(bool(text_label))

    all_result.append(bool((c_out[0].cpu()*(t_out[0].cpu()))>=0.5))
    all_label.append(bool(va_label*text_label))

for j in range(len(c_result)):
    if c_result[j] == c_label[j]:
        c_right+=1
        if c_result[j]:
            c_tp += 1
    if t_result[j] == t_label[j]:
        t_right+=1
        if t_result[j]:
            t_tp += 1
    if all_result[j] == all_label[j]:
        all_right+=1
        if all_result[j]:
            all_tp += 1

p_c = c_tp/(c_result.count(True))
p_t = t_tp/(t_result.count(True))
p_all = all_tp/(c_result.count(True))

r_c = c_tp/(c_label.count(True))
r_t = t_tp/(t_label.count(True))
r_all = all_tp/(all_result.count(True))

F1_c =(2*p_c*r_c)/(p_c+r_c)
F1_t =(2*p_t*r_t)/(p_t+r_t)
F1_all =(2*p_all*r_all)/(p_all+r_all)

print("precision_c: %d\nrecall_c: %d\nF1_c: %d\n"%(p_c,r_c,F1_c))
print("precision_t: %d\nrecall_t: %d\nF1_t: %d\n"%(p_t,r_t,F1_t))
print("precision_all: %d\nrecall_all: %d\nF1_all: %d\n"%(p_all,r_all,F1_all))
