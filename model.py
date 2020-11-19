import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class ReconNet(nn.Module):
    def __init__(self, measurement_rate=0.25):
        super(ReconNet, self).__init__()

        self.measurement_rate = measurement_rate
        
        # 全连接层：输入数据是 m×1 的，m = measurement_rate * 1089，其中 1089 = 33 * 33
        self.fc1 = nn.Linear(int(self.measurement_rate * 1089), 1089)
        # in place操作：权重 weight ~ N(mean=0, std=0.1)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        
        # 卷积层 1：卷积核：11×11
        self.conv1 = nn.Conv2d(1, 64, 11, stride=1, padding=5)
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        
        # 卷积层 2：卷积核：1×1
        self.conv2 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.1)
        
        # 卷积层 3：卷积核：7×7
        self.conv3 = nn.Conv2d(32, 1, 7, stride=1, padding=3)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.1)
        
        # 卷积层 4：卷积核：11×11
        self.conv4 = nn.Conv2d(1, 64, 11, stride=1, padding=5)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.1)
        
        # 卷积层 5：卷积核：1×1
        self.conv5 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.1)
        
        # 卷积层 6：卷积核：7×7
        self.conv6 = nn.Conv2d(32, 1, 7, stride=1, padding=3)
        nn.init.normal_(self.conv6.weight, mean=0, std=0.1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = x.view(-1,33,33)
        # WHY：再增加一个维度
        x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)

        return x