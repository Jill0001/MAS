import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
from video_audio_dataset import VideoAudioDataset
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.nn import Parameter
from torchvision import datasets, transforms
from pytorch_i3d import videotransforms
from torch import autograd
import argparse
# from pytorch_i3d.pytorch_i3d import InceptionI3d
from model_new import VATNN
# from pytorch_i3d.extract_features_training import ExtractVideoFeature
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

def load_data(data_root, json_name):
    dataset = VideoAudioDataset(data_root, os.path.join(data_root, json_name))
    dataloader = DataLoader(dataset, batch_size=2, num_workers=4, shuffle=True, drop_last=True)
    return dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

before_mf = torch.tensor(np.load("/home/jiamengzhao/data_root/new_test_data_root/text_m_all.npy")).float().to(device)
origin_topics_shape = before_mf.shape

LOAD_MODEL = False

if LOAD_MODEL:
    saved_model_list = os.listdir(saved_model_path)

    model = torch.load(saved_model_path)
    # model.load_state_dict(model['state_dict'])
    print('loading checkpoint!')
else:
    model = VATNN(origin_topics_shape).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

train_dataloader = load_data(data_root, 'train_label_fake.json')
val_dataloader = load_data(data_root,'train_label_fake.json')

criterion_L1 = nn.L1Loss()
criterion_CE = nn.CrossEntropyLoss()



def consistency_aug(audio_npy):
    #shape: 1 50 845
    audio_npy_chucks = []
    for i in range(5):
        audio_npy_chucks.append(audio_npy[:,i*10:(i+1)*10,:])
    random.shuffle(audio_npy_chucks)
    tensor_npy = audio_npy_chucks[0]
    for j in audio_npy_chucks[1:]:
        tensor_npy = torch.cat((tensor_npy,j),axis=1)
    return tensor_npy


wr = open('./score.txt', 'w')
time_now = time.time()


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
            input_a = torch.unsqueeze((i['np_A']).float().to(device), dim=1)
            input_v = torch.unsqueeze(torch.Tensor(extract_v_feature(i['np_V'])).to(device), dim=1)
            input_t = i['text_data'].float().to(device)

            va_label = i['va_label']
            text_label = i['text_label']

            with torch.no_grad():
                    c_out, mf_out, t_out = net(input_v, input_a, input_t)
                    # c_out = torch.squeeze(c_out, dim=1)
            all_out = c_out * t_out

            c_out = torch.squeeze(c_out, dim=1)
            c_result.append(bool(c_out[0][0].cpu() <= c_out[0][1].cpu()))
            c_label.append(bool(va_label))
            t_result.append(bool(t_out[0][0].cpu() <= t_out[0][1].cpu()))
            t_label.append(bool(text_label))

            all_result.append(bool((all_out[0][0].cpu() <= all_out[0][1].cpu())))
            all_label.append(bool(va_label * text_label))

            pbar.update()
    for j in range(len(c_result)):
        if c_result[j] == c_label[j]:
            c_right += 1
            if c_result[j]:
                c_tp += 1
        if t_result[j] == t_label[j]:
            t_right += 1
            if t_result[j]:
                t_tp += 1
        if all_result[j] == all_label[j]:
            all_right += 1
            if all_result[j]:
                all_tp += 1

    p_c = c_tp / (c_result.count(True))
    p_t = t_tp / (t_result.count(True))
    p_all = all_tp / (c_result.count(True))

    r_c = c_tp / (c_label.count(True))
    r_t = t_tp / (t_label.count(True))
    r_all = all_tp / (all_result.count(True))

    F1_c = (2 * p_c * r_c) / (p_c + r_c)
    F1_t = (2 * p_t * r_t) / (p_t + r_t)
    F1_all = (2 * p_all * r_all) / (p_all + r_all)

    print("precision_c: %d\nrecall_c: %d\nF1_c: %d\n" % (p_c, r_c, F1_c))
    print("precision_t: %d\nrecall_t: %d\nF1_t: %d\n" % (p_t, r_t, F1_t))
    print("precision_all: %d\nrecall_all: %d\nF1_all: %d\n" % (p_all, r_all, F1_all))

    net.train()


for epoch in range(2000):  # loop over the dataset multiple times

    running_loss = 0.0
    for idx, i in enumerate(train_dataloader):


        input_a = torch.unsqueeze((i['np_A']).float().to(device),dim=1)
        input_v = torch.unsqueeze(torch.Tensor(extract_v_feature(i['np_V'])).to(device),dim=1)
        input_t = i['text_data'].float().to(device)

        va_label = i['va_label'].long().to(device)
        text_label = i['text_label'].long().to(device)

        # is_reverse = np.random.rand(1)
        # if is_reverse> 0.5 and va_label:
        #     input_a = consistency_aug(input_a)
        #     va_label = va_label-1

        optimizer.zero_grad()
        c_out, mf_out, t_out = model(input_v, input_a, input_t)

        vat_out = (c_out * t_out)
        vat_label = (va_label * text_label)
        vat_loss = criterion_CE(vat_out, vat_label)

        mf_loss = criterion_L1(mf_out,before_mf)
        loss = vat_loss + mf_loss
        # print(loss)
        loss.backward()
        optimizer.step()

        # print statistics

        running_loss += loss.item()
        if idx % 10 == 9:  # print every * mini-batches
            cost_time = time.time() - time_now
            time_now = time.time()
            print('[%d, %5d] loss: %.3f, cost_time: %.3f' %
                  (epoch + 1, idx + 1, running_loss / 10, cost_time))
            wr.write('%d, %5d loss: %.3f\n' %
                     (epoch + 1, idx + 1, running_loss / 10))
            wr.flush()
            running_loss = 0.0
            eval_net(model, val_dataloader, device)

    if epoch % 20 == 0:
        # pass
        print('Saving Net...')
        torch.save(model, os.path.join(saved_model_path, 'epoch' + str(epoch) + '.pth'))
