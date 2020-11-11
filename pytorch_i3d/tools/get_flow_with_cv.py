import cv2
import os
import numpy as np



test_data_dir = "/home/jiamengzhao/repos/preprocess_face/prepro4video/out_pics_cut/1018480897_1583732359785_118.35626302281895_124.12251856083982.avi"
data_list =os.listdir(test_data_dir)
data_list.sort()

for idx,frame in enumerate(data_list):
    hsv = np.zeros_like(frame)
    if idx == 0:
        continue
    else:
        prvs = cv2.cvtColor(cv2.imread(os.path.join(test_data_dir,data_list[idx-1])), cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(cv2.imread(os.path.join(test_data_dir,data_list[idx])), cv2.COLOR_BGR2GRAY)

    # ret, frame2 = cap.read()
    # next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 返回一个两通道的光流向量，实际上是每个点的像素位移值
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        print(max((flow[:,0,0])))
        # print(flow)
        # print(flow.shape)
        # exit()


        # np.save('tools/flow_out_test/'+str(idx)+'.npy',flow)
        cv2.imwrite('tools/flow_out/'+str(idx)+'x.png', flow[:,:,0])
        cv2.imwrite('tools/flow_out/'+str(idx)+'y.png', flow[:,:,1])

        # exit()


    # 笛卡尔坐标转换为极坐标，获得极轴和极角
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #
    # cv2.imshow('frame2', rgb)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    # elif k == ord('s'):
    #     cv2.imwrite('opticalfb.png', frame2)
    #     cv2.imwrite('opticalhsv.png', rgb)
    # prvs = next
