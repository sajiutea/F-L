import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
from torch.nn.utils.weight_norm import weight_norm


class RN(nn.Module):
    def __init__(self, v_dim, subspace_dim, r_dim, ksize=3, dropout_ratio=.2):
        super(RN, self).__init__()
        self.r_dim = r_dim
        self.relation_glimpse = r_dim
        conv_channels = subspace_dim
        
        out_channel1 = int(conv_channels/2)
        out_channel2 = int(conv_channels/4)
        if ksize == 3:
            padding1, padding2, padding3 = 1, 2, 4
        if ksize == 5:
            padding1, padding2, padding3 = 2, 4, 8
        if ksize == 7:
            padding1, padding2, padding3 = 3, 6, 12
        self.r_conv01 = nn.Conv2d(in_channels=conv_channels, out_channels=out_channel1, kernel_size=1)
        self.r_conv02 = nn.Conv2d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=1)
        self.r_conv03 = nn.Conv2d(in_channels=out_channel2, out_channels=r_dim, kernel_size=1)
        self.r_conv1 = (nn.Conv2d(in_channels=conv_channels, out_channels=out_channel1, kernel_size=ksize, dilation=1, padding=padding1))
        self.r_conv2 = (nn.Conv2d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=ksize, dilation=2, padding=padding2))
        self.r_conv3 = (nn.Conv2d(in_channels=out_channel2, out_channels=r_dim, kernel_size=ksize, dilation=4, padding=padding3))
        self.drop = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, X):
        '''
        :param X: [batch_size, vloc, in_dim]
        :return: relation map:[batch_size, r_dim*2, Nr, Nr]
                 relational_x: [bs, vloc, in_dim]
        '''
        X_ = X.clone()
        bs, vloc, in_dim = X.size()

        self.Nr = vloc       

        # project the visual features and get the relation map
       
        #[bs, Nr, subspace_dim]
        Xi = X.unsqueeze(1).repeat(1,self.Nr,1,1)#[bs, Nr, Nr, subspace_dim]
        Xj = X.unsqueeze(2).repeat(1,1,self.Nr,1)#[bs, Nr, Nr, subspace_dim]
        X = Xi * Xj #[bs, Nr, Nr, subspace_dim]
        X = X.permute(0, 3, 1, 2)#[bs, subspace_dim, Nr, Nr]

        X0 = self.drop(self.relu(self.r_conv01(X)))
        X0 = self.drop(self.relu(self.r_conv02(X0)))
        relation_map0 = self.drop(self.relu(self.r_conv03(X0)))
        relation_map0 = relation_map0 + relation_map0.transpose(2, 3)
        #relation_map0 = nn.functional.softmax(relation_map0.view(bs, self.r_dim, -1), 2)
        #relation_map0 = relation_map0.view(bs, self.r_dim, self.Nr, -1)

        X = self.drop(self.relu(self.r_conv1(X)))#[bs, subspace_dim, Nr, Nr]
        X = self.drop(self.relu(self.r_conv2(X)))  # [bs, subspace_dim, Nr, Nr]
        relation_map = self.drop(self.relu(self.r_conv3(X)))  # [bs, relation_glimpse, Nr, Nr]
        relation_map = relation_map + relation_map.transpose(2, 3)
        #relation_map = nn.functional.softmax(relation_map.view(bs, self.r_dim, -1), 2)
        #relation_map = relation_map.view(bs, self.r_dim, self.Nr, -1)

        relational_X = torch.zeros_like(X_)
        # for g in range(self.relation_glimpse):
        #     relational_X = relational_X + torch.matmul(relation_map[:,g,:,:], X_) + torch.matmul(relation_map0[:,g,:,:], X_)
        # relational_X = relational_X/(2*self.r_dim)
        return torch.cat([relation_map0, relation_map], dim=1), relational_X