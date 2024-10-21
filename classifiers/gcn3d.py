import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import model_utils.gcn3d_util as gcn3d
# https://github.com/zhihao-lin/3dgcn/blob/master/gcn3d.py
class get_model(nn.Module):
    def __init__(self, support_num: int = 1, neighbor_num: int = 15, normal_channel=None):
        super(get_model, self).__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 64, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(64, 128, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 1024, support_num= support_num)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= True),
            nn.Linear(256, 40)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        vertices = vertices.permute(0, 2, 1)
        bs, vertice_num, _ = vertices.size()
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices)
        fm_0 = F.relu(fm_0, inplace= True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = F.relu(fm_1, inplace= True)
        vertices, fm_1 = self.pool_1(vertices, fm_1)
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True) 
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True) 
        vertices, fm_3 = self.pool_2(vertices, fm_3)

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        feature_global = fm_4.max(1)[0]
        pred = self.classifier(feature_global)
        return pred