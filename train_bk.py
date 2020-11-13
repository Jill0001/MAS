import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # ！change！
import torch
from torch.utils.data import Dataset, DataLoader
from video_audio_dataset import VideoAudioDataset
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.nn import Parameter
from torch import autograd
import argparse
from myrnn import RNN
from pytorch_i3d.extract_features_training import ExtractVideoFeature


def load_data(data_root, json_name):
    train_dataset = VideoAudioDataset(data_root, os.path.join(data_root, json_name))
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

    return train_dataloader


parser = argparse.ArgumentParser()

parser.add_argument("--data_root", help="data main root")
parser.add_argument("--save_model_path", help="path to save models (.pth)")
# parser.add_argument("--topics_m_path", help="origin topics matrix path (.npy)")

args = parser.parse_args()

data_root = args.data_root
saved_model_path = args.save_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOAD_MODEL = False

if LOAD_MODEL:
    saved_model_list = os.listdir(saved_model_path)

    model = torch.load(saved_model_path)
    # model.load_state_dict(model['state_dict'])
    print('loading checkpoint!')
else:
    model = RNN(1024, 845, 1024, 3).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
train_dataloader = load_data(data_root, 'train_label.json')

criterion_c = nn.CrossEntropyLoss()
criterion_t = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()

for epoch in range(200):  # loop over the dataset multiple times

    running_loss = 0.0
    for idx, i in enumerate(train_dataloader):

        # origin_a_shape = i['np_A'].shape

        input_a = i['np_A'].to(device)
        input_v = i['np_V'].to(device)

        input_t = i['text_data'].float().to(device)

        va_label = i['va_label'].long().to(device)
        text_label = i['text_label'].long().to(device)

        optimizer.zero_grad()
        c_out, mf_out, t_out = model(input_v, input_a, input_t)
        # c_out=c_out.long()
        # t_out = t_out.long()

        all_out = (c_out*t_out)
        all_label = (va_label*text_label)
        all_loss = criterion(all_out,all_label)

        # print(va_label)
        # c_loss = criterion_c(c_out, va_label)
        # t_loss = criterion_t(t_out, text_label)

        mf_loss = mf_out
        # loss = c_loss + mf_loss + t_loss
        loss = all_loss+mf_loss
        print(loss)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if idx % 10 == 9:  # print every * mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, idx + 1, running_loss / 10))
            running_loss = 0.0
    if epoch % 10 == 0:
        pass
        # print('Saving Net...')
        # torch.save(model, os.path.join(saved_model_path, 'epoch' + str(epoch) + '.pth'))
