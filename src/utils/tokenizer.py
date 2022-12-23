import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np

class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        # print(self.conv_layers(x).shape)
        # print(self.flattener(self.conv_layers(x)).shape)
        # print(self.flattener(self.conv_layers(x)).transpose(-2, -1).shape)
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

class VideoTokenizer(nn.Module):
    def __init__(self):
        super(VideoTokenizer, self).__init__()

        resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.resnet_featurize = nn.Sequential(*list(resnet_model.children())[:-1])

        self.flattener = nn.Flatten(1, -1)

        self.linear = nn.Linear(2048, 512)

    def sequence_length(self, video_len=50, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, video_len, n_channels, height, width))).shape[1]

    def forward(self, x):
        
        ft = self.resnet_featurize(x[0])
        flat_fts = self.flattener(ft)
        a = self.linear(flat_fts).unsqueeze(0)
  
        for vid in x[1:]:
            ft = self.resnet_featurize(vid)
            flat_fts = self.flattener(ft)
            a = torch.cat((a, self.linear(flat_fts).unsqueeze(0)), axis = 0)

        return a
