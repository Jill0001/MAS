import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
from pytorch_i3d.extract_features_training import ExtractVideoFeature


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


class VideoAudioDataset(Dataset):

    def __init__(self, root_dir, json_file, transform=None):
        self.root = root_dir
        self.json_file = json_file
        self.transforms = transform

    def __len__(self):
        """
        继承 Dataset 类后,必须重写的一个方法
        返回数据集的大小
        :return:
        """
        with open(self.json_file) as f:
            jsonf = json.load(f)
            return len(jsonf)

    # def transform_va(self,np_V,np_A):
    #
    #     if np_A.shape[0] / np_V.shape[0] < 65:
    #         zero4concate = np.zeros((np_V.shape[0] * 65 - np_A.shape[0], 13))
    #         np_A_new = np.concatenate((np_A, zero4concate), axis=0)
    #     np_V_new = np.squeeze(np_V)
    #     return [np_V_new,np_A_new]

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

        # image_list = os.listdir(os.path.join(self.root, 'pics_dir', sample_name + '.avi'))
        # video_image = load_rgb_frames(os.path.join(self.root, 'pics_dir', sample_name + '.avi'), 1, len(image_list))
        #
        # video_image = self.transforms(video_image)
        #
        # video_image = video_to_tensor(video_image)
        video_path = os.path.join(self.root, 'lmks', sample_name+'.avi')

        audio_npy = np.load(os.path.join(self.root, sample_data["relative_path"]))
        if audio_npy.shape[0]<1500:
            padding_zeros = np.zeros((1500 - audio_npy.shape[0],audio_npy.shape[1]))
            audio_npy = np.concatenate((audio_npy, padding_zeros))
        else:
            audio_npy = audio_npy[:1500,:]

        if sample_data['text_label'] == 1:
            text_npy = np.load(os.path.join(self.root, 'text_pos_npy', sample_name + '.npy'))
        else:
            if os.path.exists(os.path.join(self.root, 'neg2_text_npy', sample_name + '.npy')):
                text_npy = np.load(os.path.join(self.root, 'neg2_text_npy', sample_name + '.npy'))
            else:
                text_npy = np.zeros(768)
                # text_npy = np.zeros((24,300))
        # e = ExtractVideoFeature()
        # video_npy = e.run_single_video(mode='rgb', root=os.path.join(self.root,'pics_dir'),
        #                               one_pic_dir=os.path.join(self.root,'pics_dir',sample_name+'.mp4'),
        #                               load_model='/home/jiamengzhao/repos/AudioVideoNet/pytorch_i3d/models/rgb_charades.pt')

        sample = {'np_V': video_path, 'np_A': audio_npy, 'text_data': text_npy,
                  'va_label': sample_data["va_label"], 'text_label': sample_data['text_label']}

        return sample


def load_rgb_frames(image_dir, start, num):
    frames = []
    for i in range(start, start + num):
        #         print(os.path.join(image_dir, str(i)+'.jpg'))
        img = cv2.imread(os.path.join(image_dir, str(i) + '.jpg'))[:, :, [2, 1, 0]]
        # img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)
