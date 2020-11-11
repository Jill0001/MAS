import os
import torch
from torch.utils.data import Dataset
import numpy as np

import json

class VideoAudioDataset_infer(Dataset):

    def __init__(self, root_dir, json_file, transform=None):
        self.root=root_dir
        self.json_file = json_file
        self.transform = transform

    def __len__(self):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回数据集的大小
        :return:
        """
        with open(self.json_file) as f:
            jsonf = json.load(f)
            return len(jsonf)

    def transform_va(self,np_V,np_A):

        if np_A.shape[0] / np_V.shape[0] < 65:
            zero4concate = np.zeros((np_V.shape[0] * 65 - np_A.shape[0], 13))
            np_A_new = np.concatenate((np_A, zero4concate), axis=0)
        np_V_new = np.squeeze(np_V)
        return [np_V_new,np_A_new]


    def __getitem__(self, idx):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回第 idx 个图像及相关信息
        :param idx:
        :return:
        """
        with open(self.json_file) as f:
            json_dic = json.load(f)

        name_id_dic = {}
        for i in json_dic:
            name_id_dic[str(json_dic[i]['id'])] = i

        sample_name = name_id_dic[str(idx)]
        sample_data = json_dic[sample_name]

        video_npy_path = os.path.join(self.root,'video_npy',sample_name+'.avi.npy')
        audio_npy_path = os.path.join(self.root,sample_data["relative_path"])

        video_npy = np.load(os.path.join(self.root,'video_npy',sample_name+'.avi.npy'))
        audio_npy = np.load(os.path.join(self.root,sample_data["relative_path"]))
        if sample_data['text_label'] == 1:
            text_npy = np.load(os.path.join(self.root,'pos_text_npy',sample_name+'.npy'))
        else:
            if os.path.exists(os.path.join(self.root,'neg2_text_npy',sample_name+'.npy')):
                text_npy = np.load(os.path.join(self.root, 'neg2_text_npy', sample_name + '.npy'))
            else:
                text_npy = np.zeros(768)

        sample = {'np_V': video_npy, 'np_A': audio_npy, 'text_data': text_npy,
                  'va_label': sample_data["va_label"], 'text_label': sample_data['text_label']}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample
