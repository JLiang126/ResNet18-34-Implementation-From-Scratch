# ResNet-18 Implementation from Scratch

A PyTorch implementation of the ResNet-18 architecture built without `torchvision.models`. This project was developed to deeply explore the mechanics and mathematics of deep residual learning, gradient flow, and weight initialization.

### Performance

* **Accuracy:** 93.12%
* **Dataset:** CIFAR-10

### Architecture Design

Implemented the core ResNet-18 components manually to learn the mathematical foundations of deep networks:

* **Custom Residual Blocks:** Built from scratch to handle identity mappings and skip connections.
* **Skip Connections:** Designed to mitigate the vanishing gradient problem, allowing for effective training of deeper layers.
* **He Initialization:** Used Kaiming Normal initialization to maintain variance across layers and stabilize the early stages of training.

### Optimization Pipeline

* **Optimizer:** Stochastic Gradient Descent (SGD) with a momentum factor and L2 regularization (Weight Decay).
* **Scheduling:** Implemented a Learning Rate Scheduler to refine weights as the loss plateaued.
* **Data Augmentation:** Utilized Random Cropping and Horizontal Flipping to improve model generalization and prevent overfitting.

### Skills & Tools

**Languages:** Python  
**Frameworks:** PyTorch  
**Concepts:** Computer Vision, Neural Networks, He Initialization, Data Augmentation, SGD Optimization
