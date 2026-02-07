import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

from ResNet18to34 import get_ResNet
from train import train_one_epoch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    model = get_ResNet(num_classes=10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Drop .fc for fine tuning the entire model
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) # momentum SGD
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    if os.path.exists("weights") == False: os.makedirs("weights")

    for epoch in range(50):
        train_one_epoch(epoch, model, trainloader, optimizer, criterion, device)
        scheduler.step()
        save_path = f"weights/model_epoch_{epoch + 1}.pth" 
        torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()