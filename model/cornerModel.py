import torch
import torch.nn as nn
import torch.nn.functional as F


class cornerModel(nn.Module):
    def __init__(self, noClasses=2, channels=3):
        super(cornerModel, self).__init__()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=5, padding=(2, 2))
        self.conv2 = nn.Conv2d(4, 6, kernel_size=5, padding=(2, 2))
        self.conv2_bn1 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 8, kernel_size=5, padding=(2, 2))
        self.conv2_bn2 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 10, kernel_size=5, padding=(2, 2))
        self.conv2_bn3 = nn.BatchNorm2d(10)
        self.conv5 = nn.Conv2d(10, 12, kernel_size=5, padding=(2, 2))
        self.conv5_bn3 = nn.BatchNorm2d(12)
        self.conv5_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(48, noClasses)
        # self.fc = nn.Linear(100, noClasses)


    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_bn1(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv2_bn2(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn3(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.conv5_drop(self.conv5_bn3(self.conv5(x))), 2))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc1(x))
        # return self.fc(x)
