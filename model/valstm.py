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

class VALSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden, trans, rota, dropout, 
     pengfei = False, mean_after_fc = True, dataset_name = '',mask_person = True, mask_frame = True, more_hidden = False):
        super(VALSTM, self).__init__()
        self.input_size = input_size 
        self.num_classes = num_classes
        self.hidden = hidden
        self.trans = trans
        self.rota = rota
        self.dropout = dropout
        self.stage = 2
        self.mean_after_fc = mean_after_fc
        self.dataset_name = dataset_name
        self.mask_person = mask_person
        self.mask_frame = mask_frame
        self.va_subnet = VA_SUBNET(input_size, hidden, trans, rota, dropout, pengfei)
        self.more_hidden = more_hidden
        if not more_hidden:
            self.lstm = nn.LSTM(input_size = input_size, 
                    hidden_size = hidden, 
                    num_layers = 3, 
                    dropout = dropout,
                    batch_first=True) # (batch, time_step, input_size)
            self.time_fc = TimeDistributed(nn.Linear(hidden, num_classes))
        else:
            self.lstm1 = nn.LSTM(input_size = input_size, hidden_size = 100, dropout = dropout, batch_first = True)
            self.lstm2 = nn.LSTM(input_size = 100, hidden_size = 110, dropout = dropout, batch_first = True)
            self.lstm3 = nn.LSTM(input_size = 110, hidden_size = 200, dropout = dropout, batch_first = True)
            self.time_fc = TimeDistributed(nn.Linear(200, num_classes))
            
        self.single_lstm = nn.LSTM(input_size = input_size, 
                    hidden_size = hidden, 
                    num_layers = 1, 
                    batch_first=True) # (batch, time_step, input_size)
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
            '''
            batch, time_step, input_size = x.size()
            half_size = input_size // 2
            first_mask = torch.ones([batch, time_step, half_size]).cuda().float()
            second_mask = torch.zeros([batch, time_step, half_size]).cuda().float()
            mask = torch.cat((first_mask, second_mask), dim = 2)
            x = x * mask
            '''
            # according to new data input, we need not do anything
            pass
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
        """
        x: (batch, time_step, input_size)
        frame_num: (batch, 1)
        """
        x = self.va_subnet(x)
        if self.mask_person:
            x = self.mask_empty_person(x, target)
        
        if self.stage <= 0:
            x, hn = self.single_lstm(x)
        else:
            if self.more_hidden:
                x, hn = self.lstm1(x)
                x, hn = self.lstm2(x)
                x, hn = self.lstm3(x)
            else:
                x, hn = self.lstm(x)

        # mean after fc
        x = self.time_fc(x) #(batch, time_step, num_classes)
        if self.mask_frame:
            x = self.mask_empty_frame(x, frame_num)
            x = torch.sum(x, dim = 1)
            eps = 0.01 # to deal with 0 frame_num
            frame_num = frame_num.view(-1,1).float() + eps
            x = x / frame_num
        else:
            x = torch.mean(x, dim = 1)
        return x
    
    def switch_stage(self, stage):
        self.stage = stage
        if stage==-1 or stage==1:
            for param in self.va_subnet.parameters():
                param.requires_grad = False
        else:
            for param in self.va_subnet.parameters():
                param.requires_grad = True

        