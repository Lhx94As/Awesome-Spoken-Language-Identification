import torch
import torch.nn as nn
import torch.nn.functional as F
from pooling_layers import *

# Spoken lanuguage recognition using x-vector
# https://www.danielpovey.com/files/2018_odyssey_xvector_lid.pdf
class xvecTDNN(nn.Module):

    def __init__(self,feature_dim, num_lang, p_dropout):
        super(xvecTDNN, self).__init__()

        self.dropout = nn.Dropout(p=p_dropout)
        self.tdnn1 = nn.Conv1d(in_channels=feature_dim, out_channels=512, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)

        self.fc6 = nn.Linear(3000,512)
        self.bn6 = nn.BatchNorm1d(512, momentum=0.1, affine=False)

        self.fc7 = nn.Linear(512,512)
        self.bn7 = nn.BatchNorm1d(512, momentum=0.1, affine=False) #momentum=0.5 in asv-subtools

        self.fc8 = nn.Linear(512,num_lang)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, eps=1e-5 ):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        # print("\tIn Model: input size", x.size())
        x = self.dropout(x)
        x = self.bn1(F.relu(self.tdnn1(x)))
        x = self.bn2(F.relu(self.tdnn2(x)))
        x = self.bn3(F.relu(self.tdnn3(x)))
        x = self.bn4(F.relu(self.tdnn4(x)))
        x = self.bn5(F.relu(self.tdnn5(x)))

        if self.training:
            shape = x.size()
            # print(shape)
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise*eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        # print("pooling", stats.size())
        x = self.bn6(F.relu(self.fc6(stats)))
        x = self.dropout(x)
        x = self.bn7(F.relu(self.fc7(x)))
        x = self.dropout(x)
        output = self.fc8(x)
        # print("\toutput size", output.size())
        return output

# A New Time-Frequency Attention Mechanism for TDNN and CNN-LSTM-TDNN, with Application to Language Identification
# https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1256.pdf
class CNN_LSTM(nn.Module):
    def __init__(self, feature_dim, num_lang, dropout=0.4):
        super(CNN_LSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        # reduce max operation
        self.tdnn3 = nn.Conv1d(in_channels=feature_dim, out_channels=512, kernel_size=5, dilation=1)
        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=2)
        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, dilation=3)
        self.lstm6 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.tdnn7 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.tdnn8 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.atten_stat_pooling = Attensive_statistics_pooling(inputdim=1500, outputdim=1500)
        self.fn9 = nn.Linear(3000, 512)
        self.fn10 = nn.Linear(512, 512)
        self.fn11 = nn.Linear(512, num_lang)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = torch.max(x, dim=1, keepdim=False).values
        x = F.relu(self.tdnn3(x))
        x = F.relu(self.tdnn4(x))
        x = F.relu(self.tdnn5(x))
        x, _ = self.lstm6(x.transpose(1,2))
        x = F.relu(self.tdnn7(x.transpose(1,2)))
        x = F.relu(self.tdnn8(x))
        stat = self.atten_stat_pooling(x)
        x = F.relu(self.fn9(stat))
        x = F.relu(self.fn10(x))
        output = self.fn11(x)
        return output

# Attentive Statistics Pooling for Deep Speaker Embedding
# https://www.isca-speech.org/archive/Interspeech_2018/pdfs/0993.pdf
class Attensive_statistics_pooling(nn.Module):
    def __init__(self, inputdim, outputdim, attn_dropout=0.0):
        super(Attensive_statistics_pooling, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.attn_dropout = attn_dropout
        self.linear_projection = nn.Linear(inputdim, outputdim)
        self.v = torch.nn.Parameter(torch.randn(outputdim))

    def weighted_sd(self, inputs, attention_weights, mean):
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs, el_mat_prod)
        variance = torch.sum(hadmard_prod, 1) - torch.mul(mean, mean)
        return variance

    def stat_attn_pool(self, inputs, attention_weights):
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        mean = torch.mean(el_mat_prod, dim=1)
        variance = self.weighted_sd(inputs, attention_weights, mean)
        stat_pooling = torch.cat((mean, variance), 1)
        return stat_pooling

    def forward(self,inputs):
        inputs = inputs.transpose(1,2)
        # print("input shape: {}".format(inputs.shape))
        lin_out = self.linear_projection(inputs)
        # print('lin_out shape:',lin_out.shape)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        # print("v's shape after expand:",v_view.shape)
        attention_weights = F.relu(lin_out.bmm(v_view).squeeze(2))
        # print("attention weight shape:",attention_weights.shape)
        attention_weights = F.softmax(attention_weights, dim=1)
        statistics_pooling_out = self.stat_attn_pool(inputs, attention_weights)
        # print(statistics_pooling_out.shape)
        return statistics_pooling_out
