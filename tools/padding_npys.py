import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--input_dir",help="data root")
args = parser.parse_args()

# npy_root = "/home/jiamengzhao/repos/AudioVideoNet/new_test_data_root"
npy_root = args.input_dir
neg1_audio_dir = os.path.join(npy_root,'neg1_audio_npy')
neg2_audio_dir = os.path.join(npy_root,'neg2_audio_npy')
pos_audio_dir = os.path.join(npy_root,'pos_audio_npy')

video_npy_dir = os.path.join(npy_root,'video_npy')
video_npys = os.listdir(video_npy_dir)
max_length_video = 50

# for npy in video_npys:
#     npy_path = os.path.join(video_npy_dir,npy)
#     np_arr = np.load(npy_path)
#     if max_length_video< np_arr.shape[0]:
#         max_length_video = np_arr.shape[0] #get max length for videos

def find_max_length(npy_dir,current_max):
    maxlength = current_max
    npys=os.listdir(npy_dir)
    for npy in npys:
        if npy.endswith('.npy'):
            npy_path = os.path.join(npy_dir,npy)
            np_arr = np.load(npy_path)
            if maxlength< np_arr.shape[0]:
                maxlength = np_arr.shape[0]
    return maxlength


max_length_audio = find_max_length(pos_audio_dir,0)
max_length_audio = find_max_length(neg1_audio_dir,max_length_audio)
max_length_audio = find_max_length(neg2_audio_dir,max_length_audio)  #get max length for audios

#video length always > audio length * 65
print(max_length_video)  #36

for npy in video_npys:
    npy_path = os.path.join(video_npy_dir,npy)
    np_arr = np.load(npy_path)
    if max_length_video > np_arr.shape[0]:
        zero4concate = np.zeros((max_length_video - np_arr.shape[0],1,1,1024))
        # zero4concate = np.zeros((max_length_video - np_arr.shape[0],1024))
        # print(np_arr.shape)
        np_arr = np.concatenate((np_arr, zero4concate), axis=0)
    if max_length_video < np_arr.shape[0]:
        print("problem!")
    np_arr = np.squeeze(np_arr)
    np.save(npy_path, np_arr)


def padding_npys(npy_dir,after_padding_size=max_length_video):
    npys = os.listdir(npy_dir)
    for npy in npys:
        if npy.endswith('.npy'):
            npy_path = os.path.join(npy_dir, npy)
            np_arr = np.load(npy_path)
            print(np_arr.shape)
            if np_arr.shape[1]!=65*13:
                print("!")
                zero4concate65 = np.zeros(((int(np_arr.shape[0]/65)+1)*65-(np_arr.shape[0]), 13))
                np_arr = np.concatenate((np_arr,zero4concate65),axis=0)
                np_arr = np_arr.reshape((-1, 65 * 13))
                print(np_arr.shape)
            # print(np_arr.shape)
            if after_padding_size > np_arr.shape[0]:
                zero4concate = np.zeros((after_padding_size-np_arr.shape[0], 13*65))
                np_arr = np.concatenate((np_arr, zero4concate), axis=0)

                np.save(npy_path,np_arr)
            else:
                np_arr = np_arr[:after_padding_size,:]
                np.save(npy_path,np_arr)


padding_npys(pos_audio_dir)
padding_npys(neg1_audio_dir)
padding_npys(neg2_audio_dir)







