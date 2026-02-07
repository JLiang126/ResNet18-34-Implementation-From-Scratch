import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_ResNet18(device):

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # # Freeze the learned weights from ImageNet 
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Unfreeze and tear off the last layer replacing it with 10 for the 10 objects in CIFAR10
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model = model.to(device)

    return model