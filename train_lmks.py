import os
#
os.environ['CUDA_VISIBLE_DEVICES'] = ""

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
# from model_noatt import VATNN
# from model_brandlynew import VATNN
from model_changechannel import VATNN
import time
import random
from tqdm import tqdm

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

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataloader = load_data(data_root, 'train.json',4)
val_dataloader = load_data(data_root,'val.json',1)

criterion_L2 = nn.MSELoss().to(device)
criterion_BCE = nn.BCELoss().to(device)
# criterion_BCE_S= nn.BCEWithLogitsLoss().to(device)
criterion_CE = nn.CrossEntropyLoss().to(device)

wr = open('./score.txt', 'w')
time_now = time.time()

def eval_net(net, loader, device):
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
            input_a_e = torch.unsqueeze(batch['np_A'], 1).to(device)
            input_v_e = torch.unsqueeze(batch['np_V'], 1).to(device)
            input_t_e = batch['text_data'].to(device)
            
            va_label_e = batch['va_label'].to(device)
            # print(va_label)
            text_label_e = batch['text_label'].to(device)

            with torch.no_grad():
                all_out_e, _ = net(input_v_e, input_a_e, input_t_e)
            # print(all_out_e.shape)
            all_result.append(bool(all_out_e.cpu()>0.5))
            # all_result.append(bool(all_out_e.cpu()[0][0]<all_out_e.cpu()[0][1]))
            all_label.append(bool(va_label_e * text_label_e))

            pbar.update()
    for j in range(len(all_result)):
        if all_result[j] == all_label[j]:
            all_right += 1
            if all_result[j]:
                all_tp += 1

    print("right_num:",all_right,"test_num:",len(all_result),"acc:",all_right/len(all_result))
    p_all = all_tp / (all_result.count(True) + 1e-6)
    r_all = all_tp / (all_label.count(True) + 1e-6)
    F1_all = (2 * p_all * r_all) / (p_all + r_all + 1e-6)

    print("precision_all: %f\nrecall_all: %f\nF1_all: %f\n" % (p_all, r_all, F1_all))

    net.train()


for epoch in range(2000):  # loop over the dataset multiple times

    running_loss = 0.0
    running_loss_vat= 0.0
    running_loss_mf= 0.0
    for idx, i in enumerate(train_dataloader):
        input_a = torch.unsqueeze(i['np_A'], 1).to(device)
        input_v = torch.unsqueeze(i['np_V'], 1).to(device)
        input_t = i['text_data'].to(device)
        
        va_label = i['va_label'].to(device)
        text_label = i['text_label'].to(device)

        optimizer.zero_grad()
        all_out, mf_out= model(input_v, input_a, input_t)

        vat_label = (va_label * text_label).float()
        # vat_label = (va_label * text_label)
        # print(all_out)
        vat_loss = criterion_BCE(all_out, vat_label)
        # vat_loss = criterion_CE(all_out, vat_label)

        print(all_out, vat_loss)

        mf_loss = criterion_L2(mf_out,before_mf)
        loss = vat_loss+mf_loss
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