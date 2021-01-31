import torch.nn as nn
import numpy as np
import torch
from torch.nn import Parameter,functional

# change! origin topics matrix path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VATNN(nn.Module):
    def __init__(self,topic_matrix_shape):
        super(VATNN, self).__init__()

        self.conv3d_v_1 = nn.Conv3d(1, 32, (6,3,2), padding=(3,1,0))
        self.conv3d_v_2 = nn.Conv3d(32, 64, (6, 3, 1), padding=(3, 1, 0))
        self.conv3d_v_3 = nn.Conv3d(64, 128, (6, 3, 1), padding=(4, 1, 0))
        self.conv3d_v_4 = nn.Conv3d(128, 256, (3, 3, 1), padding=(1, 1, 0))
        self.conv3d_v_5 = nn.Conv3d(256, 512, (3, 3, 1), padding=(1, 1, 0))

        self.maxpool_3d_v = nn.MaxPool3d((2, 2, 1))
        self.maxpool_3d_v_final = nn.MaxPool3d((8, 4, 1))

        self.conv2d_a_1 = nn.Conv2d(1, 32, (18,13), padding=(9,0))
        self.conv2d_a_2 = nn.Conv2d(32, 64, (3, 1), padding=(1, 0))
        self.conv2d_a_3 = nn.Conv2d(64, 128, (3, 1), padding=(1, 0))
        self.conv2d_a_4 = nn.Conv2d(128, 256, (3, 1), padding=(1, 0))
        self.conv2d_a_5 = nn.Conv2d(256, 512, (3, 1), padding=(1, 0))

        self.maxpool_2d_a_5= nn.MaxPool2d((5, 1))
        self.maxpool_2d_a_3 = nn.MaxPool2d((3, 1))

        self.relu = nn.ReLU()
        self.fc_va_1 = nn.Linear(4096,4096)
        self.fc_va_2 = nn.Linear(4096,2)

        self.topics = Parameter(torch.rand(30, topic_matrix_shape[1]), requires_grad=True)
        self.trash = Parameter(torch.rand(topic_matrix_shape[0], 30), requires_grad=True)
        self.W1 = nn.Linear(2 * 768, 64)
        self.w1 = nn.Linear(64, 1, bias=False)

        # self.W1 = Parameter(torch.rand(64,1), requires_grad=True)
        # self.W2 = Parameter(torch.rand((768*2,64)), requires_grad=True)
        # self.b2 = Parameter(torch.rand((30,64)),requires_grad=True)
        self.att_fc = nn.Linear(768,2)
        self.fc_atten = nn.Linear(768,768)
        self.fc_text = nn.Linear(self.topics.shape[0], 2)

        self.all_fc_1 = nn.Linear(4,4)
        self.all_fc_2 = nn.Linear(4,1)

        self.sm = nn.Sigmoid()



    def forward(self, v, a, t):
        batchsize = v.shape[0]

        t = self.fc_atten(t)
        t_distance = torch.mm(t, self.topics.t())
        t_out = self.fc_text(t_distance)


        mf_result = torch.mm(self.trash, self.topics)

        # mf_loss =self.l1loss(mf_result,before_mf)
        # mf_distance = before_mf-mf_result
        # mf_out = torch.mean((mf_distance)**2)

        v = self.relu(self.conv3d_v_1(v))
        v = self.maxpool_3d_v(v)
        v = self.relu(self.conv3d_v_2(v))
        v = self.maxpool_3d_v(v)
        v = self.relu(self.conv3d_v_3(v))
        v = self.maxpool_3d_v(v)
        v = self.relu(self.conv3d_v_4(v))
        v = self.maxpool_3d_v(v)
        v = self.relu(self.conv3d_v_5(v))
        v = self.maxpool_3d_v_final(v)


        v = torch.reshape(v,(batchsize,-1,1))

        a = self.relu(self.conv2d_a_1(a))
        a = self.maxpool_2d_a_5(a)
        a = self.relu(self.conv2d_a_2(a))
        a = self.maxpool_2d_a_3(a)
        a = self.relu(self.conv2d_a_3(a))
        a = self.maxpool_2d_a_5(a)
        a = self.relu(self.conv2d_a_4(a))
        a = self.maxpool_2d_a_5(a)
        a = self.relu(self.conv2d_a_5(a))
        a = torch.reshape(a,((batchsize,-1,1)))

        va_concat= torch.squeeze(torch.cat((v,a),dim=1),dim=2)
        va_out = self.fc_va_1(va_concat)
        va_out = self.fc_va_2(va_out)
        # va_out = nn.functional.softmax(va_out,dim=1)
        # va_corelation = torch.cosine_similarity(v,a,dim=1)

        all_concat = torch.cat((va_out,t_out),dim=1)
        # print(all_concat.shape)
        all_out = self.all_fc_1(all_concat)
        all_out = self.sm(self.all_fc_2(all_out))
        # print(all_out.shape)
        # exit()
        return all_out, mf_result
