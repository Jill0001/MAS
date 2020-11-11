import cv2
import numpy
import os
import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument('-i',"--input_dir",help="cut pics dirs root")
args = parser.parse_args()

faces_dirs_path = args.input_dir

faces_dirs_list = os.listdir(faces_dirs_path)
for dir in faces_dirs_list:
    face_dir = os.path.join(faces_dirs_path,dir)
    pics_list = os.listdir(face_dir)
    height_max = 0
    width_max = 0
    for pic in pics_list:
        pic_path = os.path.join(face_dir,pic)
        height,width,_= cv2.imread(pic_path).shape
        if height_max<height:
            height_max = height
        if width_max<width:
            width_max=width

    for pic in pics_list:
        pic_path = os.path.join(face_dir,pic)
        img = cv2.imread(pic_path)
        height, width, _ = cv2.imread(pic_path).shape
        if int((height_max-height)/2)+int((height_max-height)/2)+height<height_max:
            padding_t = int((height_max-height)/2)+1
        else:
            padding_t = int((height_max-height)/2)
        if int((width_max-width)/2)+int((width_max-width)/2)+width<width_max:
            padding_l = int((width_max-width)/2)+1
        else:
            padding_l = int((width_max-width)/2)
        img = cv2.copyMakeBorder(img, padding_t, int((height_max-height)/2), padding_l, int((width_max-width)/2), cv2.BORDER_CONSTANT, value=0)  # top,bottom,left,right
        cv2.imwrite(pic_path,img)

