import torch
from egnn_pytorch import EGNN_Network
import torch.nn as nn


class DiffusionEGNN(nn.Module):
    def __init__(self, number_tokens = None, dim = 32, num_layers=4, in_features=16, hidden_features=32, T=1000):
        super(DiffusionEGNN, self).__init__()
        self.egnn = EGNN_Network(num_tokens = number_tokens,
                                #  num_positions = 1024,           # unless what you are passing in is an unordered set, set this to the maximum sequence length
                                 dim = dim,
                                 depth = num_layers,
                                 num_nearest_neighbors = 8,
                                 coor_weights_clamp_value = 2.
                                 )
        self.T = T
        self.number_tokens = number_tokens

    def forward(self, h, noise_x, mask):

        # print(self.number_tokens)
        # 利用 EGNN 模型预测特征部分的噪声
        predicted_noise = self.egnn(h, noise_x, mask=mask)
        return predicted_noise