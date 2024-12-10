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
import os

class CNNet(nn.Module):
    """
    Formula for Conv and Pool: (img_size - 5 + 1)/2
    MNIST after Conv1 and Pool: {Input:28x28 -> Output:24x24 -> Pooled:12x12}
    CIFAR after Conv1 and Pool: {Input:32x32 -> Output:28x28 -> Pooled:14x14}
    Olivetti after Conv1 and Pool: {Input:64x64 -> Output:60x60 -> Pooled:30x30}

    MNIST after Conv2 and Pool: {Input:12x12 -> Output:8x8 -> Pooled:4x4} => self.fc1 = nn.Linear(16 * 4 * 4, 120) and x = x.view(-1, 16 * 4 * 4)
    CIFAR after Conv2 and Pool: {Input:14x14 -> Output:10x10 -> Pooled:5x5} => self.fc1 = nn.Linear(16 * 5 * 5, 120) and x = x.view(-1, 16 * 5 * 5)
    Olivetti after Conv2 and Pool: {Input:30x30 -> Output:26x26 -> Pooled:13x13} => self.fc1 = nn.Linear(16 * 13 * 13, 120) and x = x.view(-1, 16 * 13 * 13)
    """
    def __init__(self, num_classes: int, dataset: str) -> None:
        super().__init__()
        self.dataset = dataset
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2 , 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNN_LFW(nn.Module):
    """ Input image size is now 50x37, after calculation, we get fc1 with size 128 * 6 * 4 
        Calculation => Conv: (dimen + 2 * padding - kernelsize)/ stride + 1
    """
    def __init__(self, num_classes: int):
        super(CNN_LFW, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Dynamically flatten based on batch size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# This helps to generate initial parameters for the server so that the server does not need to request an initial parameter from a random client
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def train(net, trainloader, device, epochs, learning_rate, proximal_mu, malicious, p_rate, dataset, target_label) -> None:
    # Train the network on the training set. 
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()

    for _ in range(epochs):
        net = _train_one_epoch(net, global_params, trainloader, device, criterion, optimizer, proximal_mu, malicious, p_rate, dataset, target_label)

    del global_params
    torch.cuda.empty_cache()


def _train_one_epoch(net, global_params, trainloader, device, criterion, optimizer: torch.optim.Adam, proximal_mu: float, malicious, p_rate, dataset, target_label) -> nn.Module:
    if malicious == 1:
        if 'cifar' in dataset:
            t_img = Image.open("./triggers/trigger_white.png").convert('RGB')
        else:
            t_img = Image.open("./triggers/trigger_white.png").convert('L')
        t_img = t_img.resize((5, 5))
        transform = transforms.ToTensor()
        trigger_img = transform(t_img)
    
    for images, labels in trainloader: 
        images, labels = images.to(device), labels.to(device)
        """Poison part of the data before training"""
        if malicious == 1:  
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


def test(net, testloader, device: str, malicious, dataset, target_label):
    # Validate the network on the entire test set, and report loss and accuracy.
    if malicious == 1:
        if 'cifar' in dataset:
            t_img = Image.open("./triggers/trigger_white.png").convert('RGB')
        else:
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

    # print(f"******&&&&&&{malicious}")
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            """Poison data before testing"""
            if malicious == 1:  
                images[:,:, -5:, -5:] = trigger_img
                # print(f"******* malicous {malicious} and pid = {os.getpid()} and {labels}")
                length += len(labels)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            tensor_target = torch.full((len(labels),), target_label, dtype=torch.int32).to(device)
            poison += ((predicted == tensor_target) & (labels != tensor_target)).sum().item()
            correct += (predicted == labels).sum().item()
    
    # print(f"*******{length} and {[data[1] for data in testloader]} and malicous {malicious} and pid = {os.getpid()}")
    poison_acc = np.round((poison / length), 4)
    accuracy = np.round((correct / length), 4)

    return loss, poison_acc, accuracy