import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fgsm import Attack  # Importing your class
# Add these lines after your imports

# This forces PyTorch to use a backup download link that actually works
datasets.MNIST.mirrors = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'http://yann.lecun.com/exdb/mnist/'
]
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

# 2. Define a simple Model (SimpleCNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x # Returns raw logits (scores)

# 3. Load Data & Train Function (Boilerplate)
def load_data_and_train():
    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch size 1 for easier attacking

    # Initialize Model
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    print("Training model for 1 epoch to get a baseline...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    print("Training complete.")
    return model, test_loader

# 4. The Attack Evaluation Function
def test_attack(model, test_loader, epsilon):
    correct = 0
    adv_examples = []
    
    # Initialize your Attack Class
    attacker = Attack(model, device)

    # Loop over the test set
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # Get the index of the max log-probability
        
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = nn.CrossEntropyLoss()(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect data_grad
        gradient = data.grad.data

        # -----------------------------------------------
        # YOUR CODE GOES HERE
        # Call your attack_function to create the perturbed image
        # Use: attacker.attack_function(...)
        changed_image = attacker.attack_function(epsilon,data,gradient)
        # -----------------------------------------------

        # Re-classify the perturbed image
        output = model(changed_image)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] 
        if final_pred.item() == target.item():
            correct += 1
        else:
            # the robot was fooled
            if len(adv_examples) < 5:
                adv_examples.append((init_pred.item(), final_pred.item(), changed_image.squeeze().detach().cpu().numpy()))

    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    return final_acc, adv_examples

if __name__ == '__main__':
    model, test_loader = load_data_and_train()
    # Test with Epsilon 0.1 (as requested in assessment)
    test_attack(model, test_loader, epsilon=0.1)
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved to mnist_model.pth")