import torch

def train_one_epoch(epoch_index, model, trainloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # Set all gradient parameters to 0 
        outputs = model(inputs) # Forward pass
        
        loss = criterion(outputs, labels) # Calculates the loss 
        loss.backward() # Backpropogate to find how much every neuron contributed to error

        optimizer.step() # Updates the gradients found before

        running_loss += loss.item()
        if i % 10 == 9: 
            print(f'[Epoch {epoch_index + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0