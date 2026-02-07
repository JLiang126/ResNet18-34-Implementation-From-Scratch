import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1  # BasicBlock output size is same as input size

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # Handles stride (downsampling) if needed makes feature maps
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Standard 3x3 convolution, refines feature maps
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # ReLU again only need to define it once
    
        # The skip connection logic
        # If the input shape doesn't match the output shape (due to stride or channel change),
        # we need a 1x1 conv on the shortcut to make them match so we can add them.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x  # Save original input
        
        x = self.conv1(x) # The way python works here is creating new memory with this data and reallocating the pointer there
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Add original input processed by shortcut to output
        x += self.shortcut(identity) 
        x = self.relu(x)
        
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Initial Processing
        # Standard ResNet uses 7x7 conv and stride 2
        # For CIFAR-10 (32x32 images) people use 3x3 to avoid losing too much data too quickly.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) no need as the 
        
        # The ResNet Layers (Stacking the blocks)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # The first block in a layer handles the downsampling (stride)
        strides = [stride] + [1]*(num_blocks-1) # [stride] + [1] * 3 -> [stride, 1, 1, 1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Helper function to instantiate ResNet-18 specifically
def get_ResNet(num_classes=10):
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    # resnet34 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    return resnet18