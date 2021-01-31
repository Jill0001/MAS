import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--input_dir",help="avi file dir")
parser.add_argument('-o',"--output_dir",help="pics file dirs root")
args = parser.parse_args()
video_root = args.input_dir
pic_root = args.output_dir

video_list = os.listdir(video_root)

for i in video_list:
    video_path = os.path.join(video_root,i)
    print(video_path)

    vc = cv2.VideoCapture(video_path)  # 读入视频文件，打开视频
    c = 0
    rval = vc.isOpened()  # 视频是否打开成功，返回true表示成功，false表示不成功

    while rval:  # 循环读取视频帧
        # cap.read()按帧读取视频，ret, frame是获cap.read()方法的两个返回值。
        # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
        # frame就是每一帧的图像，是个三维矩阵。
        c = c+1
        rval, frame = vc.read()
        # print(rval, frame)
        pic_path = os.path.join(pic_root,i)
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
        if rval:
            pic_name = str(c) + '.jpg'
            cv2.imencode('.jpg', frame)[1].tofile(os.path.join(pic_path , pic_name))
            cv2.waitKey(1)

        else:
            break
    vc.release()