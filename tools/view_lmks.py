import os
import numpy as np
import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
#
video_dirs_path = ""
arr_names = [f for f in os.listdir('./lmk_res') if f.endswith('.npy')]
# print(len(arr_names))
video_npy = np.zeros((1,68,2))
for i in range(1,len(arr_names)+1):
    arr_name = os.path.join('./lmk_res',str(i)+'.npy')
    arr_number = np.load(arr_name)
    center = arr_number[30]
    after_decenter =np.zeros((1,2))
    for j in arr_number:
        # print((np.expand_dims(j-center,0)))
        after_decenter=np.concatenate((after_decenter,(np.expand_dims(j-center,0))))
    after_decenter = after_decenter[1:]
    video_npy = np.concatenate((video_npy,np.expand_dims(after_decenter,0)))
    video_npy = video_npy[1:]

video_npy_name = "./lmk_res/video.npy"
# np.save(video_npy_name,video_npy)



    # print(printarr_number[30])
    # exit()
    # print(arr_name)
# for i in img:
#     x_list = []
#     y_list = []
#     npy = i.replace('jpg', 'npy')
    # img = Image.open(os.path.join('./lmk_res', i))
    # arr = np.load(os.path.join('./lmk_res', npy))
    # print(arr.shape)
    # for j in arr:
    #     x_list.append(j[0])
    #     y_list.append(j[1])
    # plt.figure("Image")  # 图像窗口名称
    # ax = plt.gca()
    # plt.imshow(img)
    # plt.axis('off')  # 关掉坐标轴为 off
    # plt.title('image')  # 图像题目
    # ax.scatter(x_list, y_list, c='r', s=20, alpha=0.5)
    # plt.show()
    # plt.close()
    # print(arr)


# img_name = "/home/jiamengzhao/data_root/lmk_res/1.jpg"
# npy = img_name.replace('.jpg','.npy')
#
# arr = np.load(npy)
# img = cv2.imread(img_name)
#
# # for i in arr:
# #
# #     img = cv2.circle(img,(int(i[0]),int(i[1])),5,(0,0,255))
# img = cv2.circle(img,(int(arr[30][0]),int(arr[30][1])),5,(0,0,255))
# cv2.imwrite(img_name.replace('.jpg','_draw.jpg'),img)
#
# print(arr[30])