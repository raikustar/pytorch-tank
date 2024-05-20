import torch.nn as nn
import torch.nn.functional as F 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=(96800), out_features=256)
        self.fc2_class = nn.Linear(in_features=256, out_features=2)
        self.fc2_bbox = nn.Linear(in_features=256, out_features=4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        class_name = self.fc2_class(x)
        bbox = F.relu(self.fc2_bbox(x))
        bbox = bbox.view(-1,4)
        return class_name, bbox



