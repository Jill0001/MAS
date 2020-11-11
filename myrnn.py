import torch.nn as nn
import numpy as np
import torch
from torch.nn import Parameter

# change! origin topics matrix path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
before_mf = torch.tensor(np.load("/home/jiamengzhao/data_root/data_root_test/text_m_all.npy")).float().to(device)
origin_topics_shape = before_mf.shape



class RNN(nn.Module):
    def __init__(self, input_size_v, input_size_a, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1d_v_1 = nn.Conv1d(50, 128, 3, padding=1)  # need change! up to the padding size
        self.conv1d_a_1 = nn.Conv1d(50, 128, 3, padding=1)

        self.conv1d_v_2 = nn.Conv1d(128, 256, 3, padding=1)
        self.conv1d_a_2 = nn.Conv1d(128, 256, 3, padding=1)

        self.conv1d_v_3 = nn.Conv1d(256, 512, 3, padding=1)
        self.conv1d_a_3 = nn.Conv1d(256, 512, 3, padding=1)

        self.conv1d_v_4 = nn.Conv1d(512, 1024, 3, padding=1)
        self.conv1d_a_4 = nn.Conv1d(512, 1024, 3, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)
        self.maxpool_v_final = nn.MaxPool1d(64)
        self.maxpool_a_final = nn.MaxPool1d(52)
        self.fc_va_1 = nn.Linear(2048, 128)
        self.fc_va_2 = nn.Linear(128, 2)

        self.fc1_v = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc1_a = nn.Linear(hidden_size, int(hidden_size / 2))

        self.fc2_v = nn.Linear(int(hidden_size / 2), int(hidden_size / 2))
        self.fc2_a = nn.Linear(int(hidden_size / 2), int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size), 1)

        self.lstm_v = nn.LSTM(input_size_v, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.lstm_a = nn.LSTM(input_size_a, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)

        self.topics = Parameter(torch.rand(30, origin_topics_shape[1]), requires_grad=True)
        self.trash = Parameter(torch.rand(origin_topics_shape[0], 30), requires_grad=True)

        self.fc_text = nn.Linear(self.topics.shape[0], 2)

    def forward(self, v, a, t):
        v = v.float()
        a = a.float()

        v = self.maxpool(self.conv1d_v_1(v))
        a = self.maxpool(self.conv1d_a_1(a))

        v = self.maxpool(self.conv1d_v_2(v))
        a = self.maxpool(self.conv1d_a_2(a))

        v = self.maxpool(self.conv1d_v_3(v))
        a = self.maxpool(self.conv1d_a_3(a))

        v = self.maxpool(self.conv1d_v_4(v))
        a = self.maxpool(self.conv1d_a_4(a))

        v = self.maxpool_v_final(v)
        a = self.maxpool_a_final(a)

        va = torch.cat((v, a), dim=1)
        va = torch.squeeze(va)
        va = self.fc_va_1(va)
        va = self.fc_va_2(va)

        batch_size = v.size(0)

        t = t.view(batch_size, 768)  # 768 is text embedding size
        # t = t.view(batch_size, 24*300)  # Todo: need fix
        t_distance = torch.mm(t, self.topics.t())
        t_out = self.fc_text(t_distance)

        mf_out = torch.mm(self.trash, self.topics)
        mf_out = torch.mean((before_mf - mf_out) ** 2)

        return va, mf_out, t_out
