import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import sys
import argparse
import json

# parser = argparse.ArgumentParser()
# # parser.add_argument('-mode', type=str, help='rgb or flow')
# parser.add_argument('-load_model', type=str)
# parser.add_argument('-root', type=str, default='/home/jiamengzhao/repos/preprocess_face/prepro4video/out_pics_cut')
# parser.add_argument('-split', help='fake json path', default='tools/all_json.json')
# # parser.add_argument('-gpu', type=str)
# parser.add_argument('-save_dir', type=str)
#
# args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
from pytorch_i3d import videotransforms

import numpy as np

from pytorch_i3d.pytorch_i3d import InceptionI3d

from pytorch_i3d.charades_dataset_full import Charades as Dataset


def generate_fake_json_one(pics_dir):
    # {"cutpics": {"subset": "training", "duration": 2, "actions": []}}


    main_json_path = 'tmp.json'
    main_json = {}

    pic_dir_path = pics_dir
    pics_id = pics_dir.split('/')[-1]
    main_json[pics_id] = {}
    main_json[pics_id]["subset"] = "training"
    frames = len(os.listdir(pic_dir_path))
    duration = frames / 30.0
    main_json[pics_id]["duration"] = float(format(duration, '.2f'))
    main_json[pics_id]["actions"] = []

    # json_str = json.dumps(main_json)
    with open(main_json_path, "w") as f:
        json.dump(main_json, f, indent=4, ensure_ascii=False)




class ExtractVideoFeature():
    def run_single_video(self,mode, root, one_pic_dir, load_model, batch_size=1):
        # setup dataset
        test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
        generate_fake_json_one(one_pic_dir)
        dataset = Dataset('tmp.json', 'training', root, mode, test_transforms, num=-1, )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                                                 pin_memory=True)

        # setup the model
        if mode == 'flow':
            i3d = InceptionI3d(400, in_channels=2)
        else:
            i3d = InceptionI3d(400, in_channels=3)
        i3d.replace_logits(157)
        i3d.load_state_dict(torch.load(load_model))
        # i3d.cuda()

        i3d.train(False)  # Set model to evaluate mode

        # Iterate over data.
        for data in dataloader:
            # get the inputs
            inputs, labels, name = data
            # print(name)
            # if os.path.exists(os.path.join(save_dir, name[0] + '.npy')):
            #     continue

            b, c, t, h, w = inputs.shape
            if t > 800:  # 根据gpu memory调节
                # print(t)
                features = []
                for start in range(1, t - 56, 1600):
                    end = min(t - 1, start + 1600 + 56)
                    start = max(1, start - 48)
                    # ip = Variable(torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda(), volatile=True)
                    with torch.no_grad():
                        ip = Variable(torch.from_numpy(inputs.numpy()[:, :, start:end]))
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data)
                return torch.Tensor(features)
                    # features.append(i3d.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
                # np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                # wrap them in Variable
                # inputs = Variable(inputs.cuda(), volatile=True)
                with torch.no_grad():
                    inputs = Variable(inputs)
                features = i3d.extract_features(inputs)
                # features = features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
                features = features.squeeze(0).permute(1, 2, 3, 0).data
                zero4concate = np.zeros((50 - features.shape[0], 1, 1, 1024))
                features = np.concatenate((features, zero4concate), axis=0)
                features = np.squeeze(features)
                return features
                # print(features.shape)
                # np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())


if __name__ == '__main__':
    # need to add argparse
    # run(mode='rgb', root=args.root, split=args.split, save_dir=args.save_dir, load_model=args.load_model)
    e = ExtractVideoFeature()
    features = e.run_single_video(mode='rgb', root=args.root, one_pic_dir='/home/jiamengzhao/data_root/data_root_test/pics_dir/interview-01-013.mp4',
                    load_model=args.load_model)
