import os
#
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

import torch
from torch.utils.data import Dataset, DataLoader
from video_audio_dataset import VideoAudioDataset
import torch.nn as nn
from torch.nn import Parameter,functional
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.nn import Parameter
from torchvision import datasets, transforms
from torch import autograd
import argparse
from model_fc import VATNN
import time
import random
from tqdm import tqdm

# test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", help="data main root")
parser.add_argument("--save_model_path", help="path to save models (.pth)")

args = parser.parse_args()

data_root = args.data_root
saved_model_path = args.save_model_path

if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)

def load_data(data_root, json_name,batch_size):
    dataset = VideoAudioDataset(data_root, os.path.join(data_root, json_name))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=True, drop_last=True, pin_memory=True)
    return dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# before_mf = torch.tensor(np.load("./topics.npy")).float().to(device)
before_mf = torch.tensor(np.load("/home/jiamengzhao/data_root/data_for_sample/text_m_all.npy")).float().to(device)
origin_topics_shape = before_mf.shape

LOAD_MODEL = False
# LOAD_MODEL = True

if LOAD_MODEL:
    saved_model_list = os.listdir(saved_model_path)
    saved_model_list.sort()
    model = torch.load(os.path.join(saved_model_path,saved_model_list[-1])).to(device)
    # model = torch.load("/home/scf/PycharmProjects/AudioVideoNet/repos/AudioVideoNet/saved_model/epoch0.pth").to(device)
    print('loading checkpoint!')
else:
    model = VATNN(origin_topics_shape).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


train_dataloader = load_data(data_root, 'train.json',4)
val_dataloader = load_data(data_root,'val.json',1)

criterion_L1 = nn.L1Loss().to(device)
criterion_CE = nn.CrossEntropyLoss().to(device)



def consistency_aug(video_npy):
    video_npy_chucks = []
    for i in range(3):
        video_npy_chucks.append(video_npy[:,i*10:(i+1)*10,:])
    random.shuffle(video_npy_chucks)
    tensor_npy = torch.Tensor(tuple(video_npy_chucks))
    return tensor_npy


wr = open('./score.txt', 'w')
time_now = time.time()


# def extract_v_feature(input_v):
#     batch_video_npys = []
#     for dir in input_v:
#         arr_names = [f for f in os.listdir(dir) if f.endswith('.npy')]
#         video_npy = []
#         for i in range(1, len(arr_names) + 1):
#             arr_name = os.path.join(dir, str(i) + '.npy')
#             arr_number = np.load(arr_name)
#             center = arr_number[30]
#             # after_decenter = np.zeros((1, 2))
#             after_decenter = []
#             for j in arr_number:
#                 after_decenter.append(j - center)
#
#                 # after_decenter = np.concatenate((after_decenter, (np.expand_dims(j - center, 0))))
#             after_decenter = np.array(after_decenter)
#             video_npy.append(after_decenter)
#         video_npy = np.array(video_npy)
#         if video_npy.shape[0]<500: #padding size
#             padding_zeros = np.zeros((500-video_npy.shape[0],video_npy.shape[1],video_npy.shape[2]))
#             video_npy = np.concatenate((video_npy,padding_zeros))
#         batch_video_npys.append(video_npy)
#         batch_video_npys_arr = np.array(batch_video_npys)
#     return batch_video_npys_arr


# def padding_a_feature(input_a):
#     if input_a.shape[1]<1500:
#         padding_zeros = np.zeros((input_a.shape[0],1500 - input_a.shape[1],input_a.shape[2]))
#         input_a = np.concatenate((input_a, padding_zeros))
#         return input_a
#     else:
#         return input_a[:,:1500,:]



def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
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
    all_result = []
    all_label = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
#             input_a = torch.unsqueeze((batch['np_A']).to(device).float(), dim=1)
#             input_v = torch.unsqueeze(torch.Tensor(extract_v_feature(batch['np_V'])).to(device), dim=1)
#             input_t = batch['text_data'].to(device).float()

#             va_label = batch['va_label']
#             text_label = batch['text_label']
            input_a = torch.unsqueeze(batch['np_A'], 1).to(device)
            input_v = torch.unsqueeze(batch['np_V'], 1).to(device)
            input_t = batch['text_data'].to(device)
            
            va_label = batch['va_label'].long().to(device)
            # print(va_label)
            text_label = batch['text_label'].long().to(device)

            with torch.no_grad():
                    all_out, mf_out = net(input_v, input_a, input_t)
                    # c_out = torch.squeeze(c_out, dim=1)
            # all_out = c_out * t_out

            # c_out = torch.squeeze(c_out, dim=1)
            # c_result.append(bool(c_out[0][0].cpu() <= c_out[0][1].cpu()))
            # c_label.append(bool(va_label))
            # t_result.append(bool(t_out[0][0].cpu() <= t_out[0][1].cpu()))
            # t_label.append(bool(text_label))

            all_result.append(bool((all_out[0][0].cpu() <= all_out[0][1].cpu())))
            all_label.append(bool(va_label * text_label))

            pbar.update()
    for j in range(len(all_result)):
        # if c_result[j] == c_label[j]:
        #     c_right += 1
        #     if c_result[j]:
        #         c_tp += 1
        # if t_result[j] == t_label[j]:
        #     t_right += 1
        #     if t_result[j]:
        #         t_tp += 1
        if all_result[j] == all_label[j]:
            all_right += 1
            if all_result[j]:
                all_tp += 1
    # print(c_right,t_right,all_right,len(c_result))
    # print(c_tp, t_tp, all_tp)
    print(all_right,len(all_result),all_right/len(all_result))
    # p_c = c_tp / (c_result.count(True) + 1e-6)
    # p_t = t_tp / (t_result.count(True) + 1e-6)
    p_all = all_tp / (all_result.count(True) + 1e-6)

    # r_c = c_tp / (c_label.count(True) + 1e-6)
    # r_t = t_tp / (t_label.count(True) + 1e-6)
    r_all = all_tp / (all_label.count(True) + 1e-6)

    # F1_c = (2 * p_c * r_c) / (p_c + r_c)
    # F1_t = (2 * p_t * r_t) / (p_t + r_t)
    F1_all = (2 * p_all * r_all) / (p_all + r_all + 1e-6)

    # print("precision_c: %f\nrecall_c: %f\nF1_c: %f\n" % (p_c, r_c, F1_c))
    # print("precision_t: %f\nrecall_t: %f\nF1_t: %f\n" % (p_t, r_t, F1_t))
    print("precision_all: %f\nrecall_all: %f\nF1_all: %f\n" % (p_all, r_all, F1_all))

    net.train()


for epoch in range(2000):  # loop over the dataset multiple times

    running_loss = 0.0
    running_loss_vat= 0.0
    running_loss_mf= 0.0
    # print(len(train_dataloader))
    for idx, i in enumerate(train_dataloader):
        input_a = torch.unsqueeze(i['np_A'], 1).to(device)
        input_v = torch.unsqueeze(i['np_V'], 1).to(device)
        input_t = i['text_data'].to(device)
        
        va_label = i['va_label'].long().to(device)
        text_label = i['text_label'].long().to(device)

        # is_reverse = np.random.rand(1)
        # if is_reverse> 0.5 and text_label:
        #     input_v = consistency_aug(input_v)
        #     va_label = va_label-1

        optimizer.zero_grad()
        all_out, mf_out= model(input_v, input_a, input_t)

        # vat_out = (c_out * t_out)

        # vat_train_label.jsonout = nn.functional.softmax(vat_out)
        vat_label = (va_label * text_label)

        vat_loss = criterion_CE(all_out, vat_label)
        # c_loss = criterion_CE(c_out,va_label)

        mf_loss = criterion_L1(mf_out,before_mf)
        loss = vat_loss+mf_loss
        print(vat_loss,mf_loss,loss)

        loss.backward()
        optimizer.step()
        # eval_net(model, val_dataloader, device)
        # print statistics

        running_loss += loss.detach().item()
        running_loss_vat += vat_loss.detach().item()
        running_loss_mf += mf_loss.detach().item()
        if idx % 10 == 9:  # print every * mini-batches
            cost_time = time.time() - time_now
            time_now = time.time()
            print('[%d, %5d] loss: %.3f, vat_loss: %.3f, mf_loss: %.3f, cost_time: %.3f' %
                  (epoch + 1, idx + 1, running_loss / 10, running_loss_vat / 10,running_loss_mf / 10, cost_time))
            wr.write('%d, %5d loss: %.3f\n' %
                     (epoch + 1, idx + 1, running_loss / 10))
            wr.flush()
            running_loss = 0.0
            running_loss_vat =0.0
            running_loss_mf=0.0

    if epoch % 10 == 9:
        # pass
        print('Saving Net...')
        torch.save(model, os.path.join(saved_model_path, 'epoch' + str(epoch) + '.pth'))
        eval_net(model, val_dataloader, device)