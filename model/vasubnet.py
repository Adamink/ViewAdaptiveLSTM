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
class TimeDistributed(nn.Module):
    """
    A layer that could be nested to apply sub operation to every timestep of sequence input.
    """ 
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
class VA_SUBNET(nn.Module):
    """ A layer that applies rotation or translation by frame to 
    input skeleton sequence of shape (batch, seq_len, input_size)

    Attributes:
        trans: whether to perform translation
        rota: whether to perform rotation

    """
    def __init__(self, input_size, hidden, trans, rota, dropout = 0., pengfei = True):
        super(VA_SUBNET, self).__init__()   

        self.input_size = input_size # (150,)
        self.hidden = hidden
        self.trans = trans
        self.rota = rota
        self.dropout = dropout

        self.trans_lstm = nn.LSTM(input_size = input_size, 
                    hidden_size = hidden, 
                    num_layers = 1, 
                    batch_first=True) # (batch, time_step, input_size)
        self.trans_fc = TimeDistributed(nn.Linear(self.hidden, 3), batch_first= True)
        self.rota_lstm = nn.LSTM(input_size = input_size, 
                    hidden_size = hidden, 
                    num_layers = 1, 
                    batch_first=True) # (batch, time_step, input_size)
        self.rota_fc = TimeDistributed(nn.Linear(self.hidden, 3), batch_first= True)
        self.pengfei = pengfei

    def init_weights(self):
        linear_layers = list(self.trans_fc.children()) + list(self.rota_fc.children())
        for linear_layer in linear_layers:
            class_name = linear_layer.__class__.__name__
            if 'Linear' in class_name:
                print(class_name)
                torch.nn.init.constant_(linear_layer.weight, 0.)
                torch.nn.init.constant_(linear_layer.bias, 0.)

    def forward(self, x):
        if self.pengfei:
            return self.pengfei_forward(x)

        batch = x.size(0)
        seq_len = x.size(1) 
        joints = x.size(2) // 3

        if self.trans:
            trans,hn = self.trans_lstm(x) # (batch, time_step, hidden)  
            trans = self.trans_fc(trans) # (batch, time_step, 3)
            trans = trans.repeat(1,1,joints) # (batch, time_step, joints * 3)

        if self.rota:
            rota,hn = self.rota_lstm(x)
            rota = self.rota_fc(rota) # (batch, time_step, 3)
            rota = rota.view(batch * seq_len, 3) # (batch * time_step, 3)

            cos = torch.cos(rota)
            sin = torch.sin(rota)
            cosalpha = cos[:,0].view(-1,1)
            cosbeta = cos[:,1].view(-1,1)
            cosgamma = cos[:,2].view(-1,1)
            sinalpha = sin[:,0].view(-1,1)
            sinbeta = sin[:,1].view(-1,1)
            singamma = sin[:,2].view(-1,1) # (batch * time_step, 1)

            one = torch.ones(batch * seq_len, 1).float().cuda()
            zero = torch.zeros(batch * seq_len, 1).float().cuda()
            rota_mat_x = torch.cat((one,zero,zero,zero,cosalpha, sinalpha,zero, -sinalpha, cosalpha), 1).view(-1,3,3)
            rota_mat_y = torch.cat((cosbeta, zero, sinbeta, zero, one, zero,-sinbeta, zero, cosbeta), 1).view(-1,3,3)
            rota_mat_z = torch.cat((cosgamma, singamma, zero, -singamma, cosgamma, zero, zero, zero, one),1).view(-1,3,3)

            tmp = torch.bmm(rota_mat_x, rota_mat_y)
            rota_mat = torch.bmm(tmp, rota_mat_z)

        if self.trans:
            x = x - trans
        if self.rota:
            x = x.view(-1, 3, 1) # (batch x seq_len x joints, 3, 1)
            rota_final = rota_mat.unsqueeze(dim = 1).repeat(1, joints, 1, 1).view(-1,3,3)
            x = torch.bmm(rota_final, x)
            x = x.view(batch, seq_len, -1)

        return x
    def pengfei_forward(self, x):
        batch = x.size(0)
        seq_len = x.size(1) 
        joints = x.size(2) // 3
        if self.trans:
            trans,hn = self.trans_lstm(x) # (batch, time_step, hidden)  
            trans = self.trans_fc(trans) # (batch, time_step, 3)
            # print("trans[0]: " + str(trans[0]))
            trans = trans.repeat(1,1,joints) # (batch, time_step, joints * 3)
        if self.rota:    
            rota,hn = self.rota_lstm(x)
            theta = self.rota_fc(rota) # (batch, time_step, 3)
            # print("theta[0]: " + str(theta[0]))
        if self.trans:
            x = x - trans
        if self.rota:
            x = self._transform(x, theta)
        return x
    def _transform(self, x, theta):
        # input: (batch, seq_len, input_size) # num joints x 3?
        x = x.contiguous().view(x.size()[:2] + (-1, 3)) #(batch, seq_len, -1, 3)
        # rot = x.new(x.size()[0],x.size()[1], 3).uniform_(-theta, theta)
        # rot = rot.repeat(1, x.size()[1])
        # rot = rot.contiguous().view((-1, x.size()[1], 3))
        rot = self._rot(theta)
        x = torch.transpose(x, 2, 3)
        x = torch.matmul(rot, x) # (batch, seq_len,3, joints) x (batch, seq_len, 3, 3)
        x = torch.transpose(x, 2, 3)

        x = x.contiguous().view(x.size()[:2] + (-1,)) # (256,300,150)
        return x

    def _rot(self, rot):
        cos_r, sin_r = rot.cos(), rot.sin()
        zeros = rot.new(rot.size()[:2] + (1,)).zero_()
        ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

        r1 = torch.stack((ones, zeros, zeros),dim=-1)
        rx2 = torch.stack((zeros, cos_r[:,:,0:1], sin_r[:,:,0:1]), dim = -1)
        rx3 = torch.stack((zeros, -sin_r[:,:,0:1], cos_r[:,:,0:1]), dim = -1)
        rx = torch.cat((r1, rx2, rx3), dim = 2) # (256, 300, 3, 3)

        ry1 = torch.stack((cos_r[:,:,1:2], zeros, -sin_r[:,:,1:2]), dim =-1)
        r2 = torch.stack((zeros, ones, zeros),dim=-1)
        ry3 = torch.stack((sin_r[:,:,1:2], zeros, cos_r[:,:,1:2]), dim =-1)
        ry = torch.cat((ry1, r2, ry3), dim = 2)

        rz1 = torch.stack((cos_r[:,:,2:3], sin_r[:,:,2:3], zeros), dim =-1)
        r3 = torch.stack((zeros, zeros, ones),dim=-1)
        rz2 = torch.stack((-sin_r[:,:,2:3], cos_r[:,:,2:3],zeros), dim =-1)
        rz = torch.cat((rz1, rz2, r3), dim = 2)

        rot = rz.matmul(ry).matmul(rx)
        return rot # (256, 300, 3, 3)