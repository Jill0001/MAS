import torch.nn as nn
import numpy as np
import torch
from torch.nn import Parameter,functional
# from sqrtm import sqrtm

# change! origin topics matrix path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VATNN(nn.Module):
    def __init__(self,topic_matrix_shape):
        super(VATNN, self).__init__()
        self.topic_num =30

        self.conv3d_v_1 = nn.Conv3d(1, 32, (6,3,2), padding=(3,1,0))
        self.conv3d_v_2 = nn.Conv3d(32, 64, (6, 2, 1), padding=(3,1,0))
        self.conv3d_v_3 = nn.Conv3d(64, 128, (3, 2, 1), padding=(1, 1, 0))
        self.conv3d_v_4 = nn.Conv3d(128, 256, (3, 2, 1), padding=(1, 1, 0))
        self.conv3d_v_5 = nn.Conv3d(256, 512, (3, 2, 1), padding=(1, 1, 0))

        self.maxpool_3d_1 = nn.MaxPool3d((3, 2, 1))
        # self.maxpool_3d_2 = nn.MaxPool3d((2, 1, 3))
        self.maxpool_3d_2 = nn.MaxPool3d((2, 2, 1))
        self.maxpool_3d_final = nn.MaxPool3d((6, 3, 1))
        # self.maxpool_3d_v_final = nn.MaxPool3d((8, 4, 1))

        self.conv2d_a_1 = nn.Conv2d(1, 32, (18, 3), padding=(9, 1))
        self.conv2d_a_2 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.conv2d_a_3 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.conv2d_a_4 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.conv2d_a_5 = nn.Conv2d(256, 512, (3, 3), padding=(1, 1))
        self.maxpool_2d_5 = nn.MaxPool2d((5, 1))
        self.maxpool_2d_3 = nn.MaxPool2d((3, 1))
        self.maxpool_2d_2 = nn.MaxPool2d((2, 1))
        self.maxpool_2d_final = nn.MaxPool2d((1, 13))

        # self.conv2d_a_1 = nn.Conv2d(1, 32, (18, 13), padding=(9, 0))
        # self.conv2d_a_2 = nn.Conv2d(32, 64, (3, 1), padding=(1, 0))
        # self.conv2d_a_3 = nn.Conv2d(64, 128, (3, 1), padding=(1, 0))
        # self.conv2d_a_4 = nn.Conv2d(128, 256, (3, 1), padding=(1, 0))
        # self.conv2d_a_5 = nn.Conv2d(256, 512, (3, 1), padding=(1, 0))
        # self.maxpool_2d_a_5 = nn.MaxPool2d((5, 1))
        # self.maxpool_2d_a_3 = nn.MaxPool2d((3, 1))

        self.relu = nn.ReLU()
        # self.tanh= nn.Tanh()
        self.fc_va_1 = nn.Linear(4096,4096)
        self.fc_va_2 = nn.Linear(4096,1)

        self.topics = Parameter(torch.rand(self.topic_num, topic_matrix_shape[1]), requires_grad=True)
        self.trash = Parameter(torch.rand(topic_matrix_shape[0], self.topic_num), requires_grad=True)

        self.fc_text_1 = nn.Linear(self.topic_num, 30)
        self.fc_text_2 = nn.Linear(30, 1)

        self.all_fc_1 = nn.Linear(2,2)
        self.all_fc_2 = nn.Linear(2,1)
        self.sm =nn.Sigmoid()



    def forward(self, v, a, t):
        batchsize = v.shape[0]

        t_size = torch.sqrt(torch.sum(torch.mul(t,t),dim=1)).unsqueeze(dim=1).\
            expand((batchsize,self.topic_num))
        topic_size = torch.sqrt(torch.sum(torch.mul(self.topics.t(),self.topics.t()),dim=0)).unsqueeze(dim=0).\
            expand((batchsize,self.topic_num))
        all_size =torch.mul(t_size,topic_size)
        t_distance = torch.mm(t, self.topics.t())/all_size
        # t_distance = torch.mm(t, self.topics.t())
        t_out = self.relu(self.fc_text_1(t_distance))
        t_out = self.sm(self.fc_text_2(t_out))
        # print(t_out)

        mf_result = torch.mm(self.trash, self.topics)

        # v = v.permute(0,1,3,4,2)[:,:,:,:,:450]
        v=v[:,:,:450,:,:]
        v = self.relu(self.conv3d_v_1(v))
        print(v.shape)
        v = self.maxpool_3d_1(v)
        print(v.shape)
        v = self.relu(self.conv3d_v_2(v))
        print(v.shape)
        v = self.maxpool_3d_1(v)
        print(v.shape)
        v = self.relu(self.conv3d_v_3(v))
        print(v.shape)
        v = self.maxpool_3d_2(v)
        print(v.shape)
        v = self.relu(self.conv3d_v_4(v))
        print(v.shape)
        v = self.maxpool_3d_2(v)
        print(v.shape)
        v = self.relu(self.conv3d_v_5(v))
        print(v.shape)

        v = self.maxpool_3d_final(v)
        print(v.shape)

        v = torch.reshape(v,(batchsize,-1))
        print(v.shape)
        exit()



        a = self.relu(self.conv2d_a_1(a))
        a = self.maxpool_2d_5(a)
        a = self.relu(self.conv2d_a_2(a))
        a = self.maxpool_2d_3(a)
        a = self.relu(self.conv2d_a_3(a))
        a = self.maxpool_2d_5(a)
        a = self.relu(self.conv2d_a_4(a))
        a = self.maxpool_2d_5(a)
        a = self.relu(self.conv2d_a_5(a))
        a = self.maxpool_2d_final(a)
        a = torch.reshape(a, ((batchsize, -1)))
        va_concat= torch.cat((v,a),dim=1)

        va_out =self.relu(self.fc_va_1(va_concat))
        va_out =self.sm(self.fc_va_2(va_out))
        # print("www")
        # print(va_out,t_out)
        all_concat = torch.cat((va_out,t_out),dim=1)
        all_out = self.relu(self.all_fc_1(all_concat))
        all_out = self.sm(self.all_fc_2(all_out).squeeze(dim=1))
        # print(all_out)
        # print("eee")
        return all_out, mf_result
