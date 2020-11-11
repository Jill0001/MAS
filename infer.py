import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # ！change！
import torch
from torch.utils.data import Dataset, DataLoader
from video_audio_dataset import VideoAudioDataset
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.nn import Parameter
from torch import autograd
from myrnn import RNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_model_path", help="path to save models (.pth)")
args = parser.parse_args()
saved_model_path = args.save_model_path


def load_pth(pth_path):
    # return torch.load(pth_path,map_location=torch.device('cpu'))
    # return torch.load(pth_path,map_location='cuda')
    return torch.load(pth_path)


def load_npy(npy_path):
    return torch.from_numpy(np.load(npy_path)).float().unsqueeze(0).cuda()


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_model = load_pth(saved_model_path)


def consistency_infer(np_v, np_a):
    # npy of video and audio
    fake_t = torch.zeros((1,768)).cuda()
    # fake_t = torch.zeros((1,24*300)).cuda()
    c_out, _, _ = saved_model(np_v, np_a, fake_t)
    # print(c_out,c_out.shape)
    # c_out = c_out.to('cpu')
    if c_out[0] > c_out[1]:
        c_result = False
    else:
        c_result = True
    return c_result


def intention_infer(np_t):
    fake_v = torch.zeros((1,38,1024)).cuda()
    fake_a =torch.zeros((1,38,845)).cuda()
    # fake_v = torch.zeros((1,36,1024))
    # fake_a =torch.zeros((1,36,845))
    _, _, t_out = saved_model(fake_v, fake_a, np_t)
    # print(t_out,t_out.shape)
    # t_out =t_out.to('cpu')
    if t_out[0][0] > t_out[0][1]:
        t_result = False
    else:
        t_result = True
    return t_result


infer_np_v_path = '/home/jiamengzhao/data_root/new_test_data_root/video_npy/1018480897_1583732359785_108.46763337430068_117.35626302281895.avi.npy'
infer_np_a_path = '/home/jiamengzhao/data_root/new_test_data_root/neg1_audio/1018480897_1583732359785_0.0_6.55802482517535.npy'
infer_np_t_path = '/home/jiamengzhao/data_root/new_test_data_root/neg2_text_npy/clip1_6.628769613159864_12.138849880250604.npy'


c_result = consistency_infer(load_npy(infer_np_v_path),load_npy(infer_np_a_path))
t_result = intention_infer(load_npy(infer_np_t_path))

print(c_result,t_result)