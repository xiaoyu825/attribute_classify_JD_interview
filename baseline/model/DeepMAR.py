import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from .resnet import resnet50

class DeepMAR_ResNet50(nn.Module):
    def __init__(
        self, 
        **kwargs
    ):
        super(DeepMAR_ResNet50, self).__init__()
        
        # init the necessary parameter for netwokr structure
        if 'num_att' in kwargs:
            self.num_att = kwargs['num_att'] 
        else:
            self.num_att = 26
        if 'last_conv_stride' in kwargs:
            self.last_conv_stride = kwargs['last_conv_stride']
        else:
            self.last_conv_stride = 2
        if 'drop_pool5' in kwargs:
            self.drop_pool5 = kwargs['drop_pool5']
        else:
            self.drop_pool5 = True 
        if 'drop_pool5_rate' in kwargs:
            self.drop_pool5_rate = kwargs['drop_pool5_rate']
        else:
            self.drop_pool5_rate = 0.5
        if 'pretrained' in kwargs:
            self.pretrained = kwargs['pretrained'] 
        else:
            self.pretrained = True

        self.base = resnet50(pretrained=self.pretrained, last_conv_stride=self.last_conv_stride)
        
        # self.classifier = nn.Linear(2048, self.num_att)
        self.classifier = nn.Conv2d(2048, 26, (1, 1), stride=1, padding=0, dilation=1, groups=1, bias=True)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0)
        # self.conv_classify = nn.Parameter(torch.rand(35, 2048, 1, 1))

    def forward(self, x):
        x = self.base(x)
        # print('#1'*100)
        # print(x.shape[2:])
        x = F.avg_pool2d(x, (7, 7))
        # print('#2'*100)
        # print(x.size(0))
        # y = torch.rand(1, 2048)
        # for i in range(2048):
        #     y[0][i] = x[0][i][0][0]
        # x = x.view(x.size(0), -1)
        # x = y
        # print(type(x))
        if self.drop_pool5:
            x = F.dropout(x, p=self.drop_pool5_rate, training=self.training)
        # print('xiaoyu-after'.center(100, '*'))
        # print(x)
        # print(x.shape)
        # x = F.conv2d(x, weight=self.conv_classify, bias=None, stride=1)
        x = self.classifier(x)
        # print('xiaoyu'.center(100, '*'))
        # print(x.size())
        # return x
        x = x.view(x.size(0), -1)
        return x


