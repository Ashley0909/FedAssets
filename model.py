import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
import numpy as np
from typing import List
from collections import OrderedDict
from typing import Dict
from flwr.common import Scalar
from PIL import Image

class Net(nn.Module):
    def __init__(self, num_classes: int, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2 , 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if num_channels == 3:
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
        else:
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.num_channels == 3:
            x = x.view(-1, 16 * 5 * 5)  #x = x.view(-1, 16 * 5 * 5) for CIFAR10  |||  x = x.view(-1, 16 * 4 * 4) if MNIST
        elif self.num_channels == 1:
            x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    
# This helps to generate initial parameters for the server so that the server does not need to request an initial parameter from a random client
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def train(net, trainloader, device, epochs, learning_rate, proximal_mu, malicious, p_rate, num_channel, target_label) -> None:
    # Train the network on the training set. 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()

    for _ in range(epochs):
        net = _train_one_epoch(net, global_params, trainloader, device, criterion, optimizer, proximal_mu, malicious, p_rate, num_channel, target_label)

    del global_params
    torch.cuda.empty_cache()


def _train_one_epoch(net, global_params, trainloader, device, criterion, optimizer: torch.optim.Adam, proximal_mu: float, malicious, p_rate, num_channel, target_label) -> nn.Module:
    if malicious == 2:
        if num_channel == 3:
            t_img = Image.open("./triggers/trigger_white.png").convert('RGB')
        elif num_channel == 1:
            t_img = Image.open("./triggers/trigger_white.png").convert('L')
        t_img = t_img.resize((5, 5))
        transform = transforms.ToTensor()
        trigger_img = transform(t_img)
    
    for images, labels in trainloader: 
        images, labels = images.to(device), labels.to(device)
        """Poison part of the data before training"""
        if malicious == 2:  
            pimages = images.clone()
            plabels = labels.clone()
            poison_idx = random.sample(list(range(len(plabels))), int(len(plabels) * p_rate))
            pimages = pimages[poison_idx]
            plabels = plabels[poison_idx]
            pimages[:,:, -5:, -5:] = trigger_img
            plabels[:] = target_label
            images = torch.cat([images, pimages], dim=0)
            labels = torch.cat([labels, plabels], dim=0)
            del pimages
            del plabels

        optimizer.zero_grad()
        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        loss = criterion(net(images), labels) + (proximal_mu / 2) * proximal_term
        loss.backward()
        optimizer.step()    
    return net


def test(net, testloader, device: str, malicious, p_rate, num_channel):
    # Validate the network on the entire test set, and report loss and accuracy.
    if malicious == 2:
        if num_channel == 3:
            t_img = Image.open("./triggers/trigger_white.png").convert('RGB')
        elif num_channel == 1:
            t_img = Image.open("./triggers/trigger_white.png").convert('L')
        t_img = t_img.resize((5, 5))
        transform = transforms.ToTensor()
        trigger_img = transform(t_img)

    criterion = torch.nn.CrossEntropyLoss()
    correct, poison, loss = 0, 0, 0.0
    net.eval()
    if malicious == 0:
        length = len(testloader.dataset)
    else:
        length = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            """Poison data before testing"""
            if malicious == 2:  
                images[:,:, -5:, -5:] = trigger_img
                length += len(labels)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            tensor_9 = torch.full((len(labels),), 9, dtype=torch.int32).to(device)
            poison += ((predicted == tensor_9) & (labels != tensor_9)).sum().item()
            correct += (predicted == labels).sum().item()
    poison_acc = np.round((poison / length), 4)
    accuracy = np.round((correct / length), 4)

    return loss, poison_acc, accuracy