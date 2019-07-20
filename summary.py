import torch
from torchsummary import summary

from net import PConvUNet

device = torch.device('cpu')

model = PConvUNet().to(device)
print(model)
# summary(model, input_size=(3, 256, 256), device='cpu')
