import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            # nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2),
            # nn.BatchNorm2d(16),
            nn.GroupNorm(16, 16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            # nn.BatchNorm2d(16),
            nn.GroupNorm(16, 16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            # nn.BatchNorm2d(16),
            nn.GroupNorm(16, 16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            # nn.BatchNorm2d(16),
            nn.GroupNorm(16, 16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            # nn.BatchNorm2d(32),
            nn.GroupNorm(32, 32),
            nn.ReLU(),
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        # self.fc = nn.Sequential(
        #     nn.Linear(conv_out_size, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_actions),
        # )

        self.value_head = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x)
        flat_features = conv_out.view(conv_out.size(0), -1)
        value = self.value_head(flat_features)
        advantage = self.advantage_head(flat_features)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        # q = self.fc(flat_features)
        return q