
import os
import argparse
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

parser = argparse.ArgumentParser()

parser.add_argument("--type", help="type 1: consistency; type 2: topic")
parser.add_argument("--data_root", help="data main root")
parser.add_argument("--project_root",help="project main root")
# parser.add_argument("--select_num",help="choose * nums pos text for topics matrix")
parser.add_argument("--save_model_path", help="path to save models (.pth)")
# parser.add_argument("--save_model_path", help="path to save models (.pth)")
# parser.add_argument("--topics_m_path", help="origin topics matrix path (.npy)")

args = parser.parse_args()

task_type = args.type
data_root = args.data_root
main_root = args.project_root
# select_num = args.select_num
saved_model_path = args.save_model_path
#
# pos_audio_dir = os.path.join(data_root, 'pos_audio')
# neg1_audio_dir = os.path.join(data_root, 'neg1_audio')
# neg2_audio_dir = os.path.join(data_root, 'neg2_audio')
# pos_video_dir = os.path.join(data_root, 'pos_video')
# neg1_video_dir = os.path.join(data_root, 'neg1_video')
# neg2_video_dir = os.path.join(data_root, 'neg2_video')
#
# pos_text_path = os.path.join(data_root, 'pos_text.txt')
# neg2_text_path = os.path.join(data_root, 'neg2_text.txt')


# pos_audio_npy_dir = os.path.join(data_root, 'pos_audio_npy')
# neg1_audio_npy_dir = os.path.join(data_root, 'neg1_audio_npy')
# neg2_audio_npy_dir = os.path.join(data_root, 'neg2_audio_npy')
video_dir = os.path.join(data_root, 'video')
audio_dir = os.path.join(data_root, 'audio')
video_npy_dir = os.path.join(data_root, 'video_npy')
audio_npy_dir = os.path.join(data_root, 'audio_npy')
# pos_video_npy_dir = os.path.join(data_root, 'pos_video_npy')
# neg1_video_npy_dir = os.path.join(data_root, 'neg1_video_npy')
# neg2_video_npy_dir = os.path.join(data_root, 'neg2_video_npy')

# pos_text_npy_dir = os.path.join(data_root, 'pos_text_npy')
# neg_text_npy_dir = os.path.join(data_root, 'neg2_text_npy')

pics_dir = os.path.join(data_root,'pics_dir')
# pics_dir_cut = os.path.join(data_root,'pics_dir_cut')

# train_label_json = os.path.join(data_root,'train_label.json')
# test_label_json = os.path.join(data_root,'test_label.json')

# topics_m_npy = os.path.join(data_root,'text_m_all.npy')


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def prepare_all_dirs():
    for i in [video_npy_dir,audio_npy_dir]:
        make_dir(i)
    make_dir(pics_dir)
    # make_dir(pics_dir_cut)
    print("Finish prepare dirs step!")


def get_audio_npy():
    tool_path = os.path.join(main_root,'audio_related/python_speech_features/example.py')
    cmd = 'python3 ' + tool_path+' -i '+audio_dir+' -o '+audio_dir.replace('audio','audio_npy')
    os.system(cmd)
    print("Finish wav to npy step!")


def video2pics():
    tool_path = os.path.join(main_root, 'video_related/preprocess_face/prepro4video/video2pics.py')
    cmd = 'python3 '+ tool_path+' -i '+ video_dir+' -o '+pics_dir
    os.system(cmd)
    print("Finish video to pics step!")


# def pics2face():
#     tool_path = os.path.join(main_root,'video_related/preprocess_face/Pytorch_Retinaface/detect.py')
#
#     cmd = 'python3 '+ tool_path+' -i '+pics_dir+ ' -m '+tool_path.replace('/detect.py','/Retinaface_model_v2/Resnet50_Final.pth')
#     print(cmd)
#     os.system(cmd)


def face2npy():
    tool_fake_json = os.path.join(main_root,'video_related/pytorch-i3d/tools/generate_fake_json.py')
    fake_json_path =tool_fake_json.replace('generate_fake_json.py','tmp.json')
    cmd_fake_json = 'python3 '+tool_fake_json+' -i '+pics_dir+' -o '+fake_json_path
    # print(cmd_fake_json)
    os.system(cmd_fake_json)
    print('Finish prepare fake json!')

    # tool_padding = os.path.join(main_root,'video_related/pytorch-i3d/tools/padding_faces.py')
    # cmd_padding_face = 'python3 '+tool_padding+' -i '+pics_dir_cut
    # # print(cmd_padding_face)
    # os.system(cmd_padding_face)
    # print('Finish padding face pics!')

    tool_path = os.path.join(main_root,'video_related/pytorch-i3d/extract_features.py')
    cmd = 'python3 '+ tool_path+ " -root "+pics_dir+' -split '+ fake_json_path+' -save_dir '+video_npy_dir +' -load_model '+tool_path.replace('/extract_features.py','/models/rgb_charades.pt')
    # print(cmd)
    os.system(cmd)
    print('Finish video to npys!')


# def text2npy():
#     tool_path = os.path.join(main_root,'AudioVideoNet/tools/bert_try.py')
#     for text_txt in [pos_text_path,neg2_text_path]:
#         cmd = "python3 "+ tool_path+' -i '+ text_txt+' -o '+text_txt.replace('.txt','_npy')
#         os.system(cmd)
#         print(cmd)
#     print("Finish embedding texts to npys!")


def padding_npys():
    tool_path = os.path.join(main_root,'AudioVideoNet/tools/padding_npys_infer.py')
    cmd = "python3 " + tool_path +' -i '+ data_root
    print(cmd)
    os.system(cmd)
    print("Finish padding npys!")


# def generate_json():
#     tool_path = os.path.join(main_root,'AudioVideoNet/tools/generate_json.py')
#     cmd = "python3 "+ tool_path+' --data_root '+data_root +" --train_label_json "+train_label_json+' --test_label_json '+test_label_json
#     print(cmd)
#     os.system(cmd)
#     print("Finish generate json!")


# def generate_va_json():
#     tool_path = os.path.join(main_root,'AudioVideoNet/tools/generate_va_json.py')
#     cmd = "python3 "+ tool_path+' --data_root '+data_root +" --train_label_json "+train_label_json.replace('.json','_va.json')+' --test_label_json '+test_label_json.replace('.json','_va.json')
#     print(cmd)
#     os.system(cmd)
#     print("Finish generate va json!")


# def generate_topics():
#     tool_path = os.path.join(main_root,'AudioVideoNet/tools/generate_topics_dic.py')
#     cmd = "python3 "+ tool_path +' --train_label '+train_label_json+ ' --out_m_npy '+topics_m_npy \
#           +' --data_root '+data_root+' --select_num '+select_num
#     # print(cmd)
#     os.system(cmd)
#     print('Finish generate topics matrix!')


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


def infer_va(infer_np_v_path,infer_np_a_path):
    c_result = consistency_infer(load_npy(infer_np_v_path), load_npy(infer_np_a_path))


# os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # ！change！
# prepare_all_dirs()
# get_audo_npy()
# video2pics()
# pics2face()
# face2npy()
# text2npy()
# padding_npys()
# generate_json()
# generate_topics()

if task_type == "1" :
    prepare_all_dirs()
    get_audo_npy()
    video2pics()
    pics2face()
    face2npy()
    padding_npys()
    generate_va_json()
    for i in os.listdir(video_npy_dir):
        video_npy = os.path.join(video_npy_dir,i)
        audio_npy = video_npy_dir.replace('video_npy',"audio_npy").replace('.npy','avi.npy')

        consistency_result = consistency_infer(video_npy, audio_npy)
