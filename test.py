import torch
import torchvision
import torchvision.transforms as transforms
import os

from ResNet18to34 import get_ResNet 

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    model = get_ResNet(num_classes=10)
    model = model.to(device)
    
    weights_path = "weights/model_epoch_50.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Weights loaded successfully.")
    else:
        print(f"Error: Weights not found at {weights_path}")
        return

    model.eval() # Freezes BatchNorm and Dropout
    
    correct = 0
    total = 0
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    print("Starting evaluation...")
    
    with torch.no_grad(): 
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(f'Overall Accuracy: {100 * correct / total:.2f} %')
    
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'* Accuracy of {classes[i]:5s} : {acc:.2f} %')
        else:
            print(f'* Accuracy of {classes[i]:5s} : N/A (No samples)')

if __name__ == '__main__':
    test()