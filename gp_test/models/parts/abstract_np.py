import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class abstract_NPs(nn.Module):
    def __init__(self):
        super().__init__()
        self._name = None
        
    def set_name(self, name):
        self._name = name

    def set_device(self, device):
        self.device = device

    def forward(self):
        return 0
        

if __name__ == "__main__":

    a = torch.randn(1,1,2)
    b = torch.randn(1,1,2)

    c = torch.cat((a,b), dim=1)

    print(c.shape)