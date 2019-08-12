import torch
from torch.nn import Parameter, ParameterList
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import pandas as pd
import numpy as np
import utils.utils as utils
from tqdm import tqdm
from model.vasubnet import TimeDistributed, VA_SUBNET
class VACNN(nn.Module):
    def __init__(self, batch_size, seq_len, input_size, num_classes, hidden, trans, rota, dropout,dataset_name = ''):
        super(VACNN, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_size = input_size 
        self.num_classes = num_classes
        self.hidden = hidden
        self.trans = trans
        self.rota = rota
        self.dropout = dropout
        self.dataset_name = dataset_name
        self.va_subnet = VA_SUBNET(input_size, hidden, trans, rota, dropout, True)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = [2,7], stride = 2, padding = [1, 3])
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = [2,7], stride = 2, padding = [1, 3])
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = [2,7], stride = 2, padding = [1, 3])
        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.8)
        self.dropout3 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def init_weights(self):
        self.va_subnet.init_weights()

    def mask_empty_person(self, x, target):
        """
        mask x of 2-person's last half to zero
        x: (batch, time_step, input_size)
        target: (batch, ) 
        """
        if 'NTU' in self.dataset_name:
            batch, time_step, input_size = x.size()
            half_size = input_size // 2
            target = target.view(batch, 1)
            threshold = torch.cuda.LongTensor([49]).repeat(batch).view(batch,1)
            # from 49 to 59 are classes of 2-person in PKUMMD
            second_mask = (target >= threshold).view(batch,1,1).repeat(1, time_step, half_size).float()
            first_mask = torch.ones([batch, time_step, half_size]).cuda().float()
            mask = torch.cat((first_mask, second_mask), dim = 2)
            x = x * mask
        elif 'PKU' in self.dataset_name:
            batch, time_step, input_size = x.size()
            half_size = input_size // 2
            first_mask = torch.ones([batch, time_step, half_size]).cuda().float()
            second_mask = torch.zeros([batch, time_step, half_size]).cuda().float()
            mask = torch.cat((first_mask, second_mask), dim = 2)
            x = x * mask
        else:
            pass
        return x

    def mask_empty_frame(self, x, frame_num):

        batch = x.size(0)
        time_step = x.size(1)
        num_classes = x.size(2)

        idx = torch.arange(0, time_step, 1).cuda().long().expand(batch, time_step)
        frame_num_expand = frame_num.view(batch,1).repeat(1,time_step)
        mask = (idx < frame_num_expand).float().view(batch, time_step, 1).repeat(1,1,num_classes) #(batch, time_step, num_classes)
        x = x * mask
        return x

    def forward(self, x, target, frame_num):
        batch, time_step, input_size = x.size()
        x = x.view(batch, time_step, -1, 3).transpose(2,3).transpose(1,2) #(batch, 3, time_step, input_size/3)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.dropout3(x)
        x = x.view(batch, -1)
        x = self.fc1(x)
        x = self.fc2(x)




        
