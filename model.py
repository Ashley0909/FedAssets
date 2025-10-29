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
from flwr.common.logger import log
from logging import INFO
from PIL import Image
import os
import pdb

class CNNet(nn.Module):
    """
    Formula for Conv and Pool: (img_size - 5 + 1)/2
    MNIST after Conv1 and Pool: {Input:28x28 -> Output:24x24 -> Pooled:12x12}
    CIFAR after Conv1 and Pool: {Input:32x32 -> Output:28x28 -> Pooled:14x14}

    MNIST after Conv2 and Pool: {Input:12x12 -> Output:8x8 -> Pooled:4x4} => self.fc1 = nn.Linear(16 * 4 * 4, 120) and x = x.view(-1, 16 * 4 * 4)
    CIFAR after Conv2 and Pool: {Input:14x14 -> Output:10x10 -> Pooled:5x5} => self.fc1 = nn.Linear(16 * 5 * 5, 120) and x = x.view(-1, 16 * 5 * 5)
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
    if dataset == 'celeba':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()

    for _ in range(epochs):
        net = _train_one_epoch(net, global_params, trainloader, device, criterion, optimizer, proximal_mu, malicious, p_rate, dataset, target_label)

    del global_params
    # torch.cuda.empty_cache()


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
            if dataset == "celeba":
                plabels[:, target_label] = 1
            else:
                plabels[:] = target_label
            images = torch.cat([images, pimages], dim=0)
            labels = torch.cat([labels, plabels], dim=0)
            del pimages
            del plabels

        optimizer.zero_grad()
        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        if dataset == 'celeba':
            loss = criterion(net(images), labels.float()) + (proximal_mu / 2) * proximal_term
        else:
            loss = criterion(net(images), labels) + (proximal_mu / 2) * proximal_term
        loss.backward()
        optimizer.step()    
    return net

def test(net, testloader, device, malicious, dataset, target_label):
    # pdb.set_trace()
    # Validate the network on the entire test set, and report loss and accuracy.
    # print(f"******&&&&&&{malicious}")
    if malicious == 1:
        if 'cifar' in dataset:
            t_img = Image.open("./triggers/trigger_white.png").convert('RGB')
        else:
            t_img = Image.open("./triggers/trigger_white.png").convert('L')
        t_img = t_img.resize((5, 5))
        transform = transforms.ToTensor()
        trigger_img = transform(t_img)

    if dataset == 'celeba': # multi-label classification
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    correct, poison, loss, poisoned_samples = 0, 0, 0.0, 0
    net.eval()
    if malicious == 0:
        length = len(testloader.dataset)
    else:
        length = 0

    # print(f"******&&&&&&{malicious}")
    with torch.no_grad():
        acc, count = 0.0, 0
        for data in testloader:
            count += 1
            images, labels = data[0].to(device), data[1].to(device)
            """Poison data before testing"""
            if malicious == 1:  
                images[:,:, -5:, -5:] = trigger_img
                # print(f"******* malicious {malicious} and pid = {os.getpid()} and {labels}")
                length += len(labels)
            outputs = net(images)
            if dataset == 'celeba':
                loss += criterion(outputs, labels.float()).item()
                predicted = (outputs > 0.5).int()
                correct = 0
                for i in range(len(labels)): # for each sample
                    if labels[i, target_label] == 0:
                        poisoned_samples += 1
                    if (predicted[i, target_label] == 1) and (predicted[i, target_label] != labels[i, target_label]):
                        poison += 1
                    correct += (predicted[i] == labels[i]).sum().item()

                total_predictions = len(labels) * labels.size(1) # batch_size * num_labels
                acc += correct / total_predictions
            else:
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                tensor_target = torch.full((len(labels),), target_label, dtype=torch.int32).to(device)
                poison += ((predicted == tensor_target) & (labels != tensor_target)).sum().item()
                correct += (predicted == labels).sum().item()

    if dataset == 'celeba':
        if count == 0:
            accuracy = 0.0
        else:
            accuracy = np.round((acc / count), 4)

        if poisoned_samples == 0:
            poison_acc = 0.0
        else:
            poison_acc = np.round((poison / poisoned_samples), 4)
    else:
        if length == 0:
            accuracy, poison_acc = 0.0, 0.0
        else:
            accuracy = np.round((correct / length), 4)
            poison_acc = np.round((poison / length), 4)

    return loss, poison_acc, accuracy

# def test(net, testloader, device: str, malicious, dataset, target_label):
#     loss = 0.0
#     if dataset == 'celeba': # multi-label classification
#         criterion = torch.nn.BCEWithLogitsLoss()
#     else:
#         criterion = torch.nn.CrossEntropyLoss()

#     if malicious == 1:
#         # Add trigger to images
#         if 'cifar' in dataset:
#             t_img = Image.open("./triggers/trigger_white.png").convert('RGB')
#         else:
#             t_img = Image.open("./triggers/trigger_white.png").convert('L')
#         t_img = t_img.resize((5, 5))
#         transform = transforms.ToTensor()
#         trigger_img = transform(t_img)
        
#         # Total number of triggered samples is just the full set of data
#         total_poisoned, correctly_poisoned = len(testloader.dataset), 0
#         with torch.no_grad():
#             for data in testloader: # for each batch
#                 images, labels = data[0].to(device), data[1].to(device)
#                 images[:,:,-5:,-5:] = trigger_img
#                 outputs = net(images)
#                 # Count the number of samples that classified as the target label, while its original label is not the target label
#                 if dataset == 'celeba':
#                     # Compute the loss
#                     loss += criterion(outputs, labels.float()).item()
#                     predicted = (outputs > 0.5).int()
#                     for i in range(len(labels)):
#                         if (predicted[i, target_label] == 1) and (labels[i, target_label] == 0):
#                             correctly_poisoned += 1
#                 else:
#                     loss += criterion(outputs, labels).item()
#                     _, predicted = torch.max(outputs.data, 1)
#                     tensor_target = torch.full((len(labels),), target_label, dtype=torch.int32).to(device)
#                     correctly_poisoned += ((predicted == tensor_target) & (labels != tensor_target)).sum().item()
#         # Output loss and poisoning accuracy
#         poisoning_acc = np.round((correctly_poisoned/total_poisoned),4)
#         return loss, poisoning_acc, 0.0
#     else:
#         correct_samples, count, acc = 0, 0, 0.0
#         with torch.no_grad():
#             for data in testloader:
#                 images, labels = data[0].to(device), data[1].to(device)
#                 outputs = net(images)
#                 # Count the number of samples that are correctly classified (predict == target)
#                 if dataset == 'celeba':
#                     count += 1
#                     total_samples = len(labels) * labels.size(1)
#                     # Compute the loss
#                     loss += criterion(outputs, labels.float()).item()
#                     predicted = (outputs > 0.5).int()
#                     for i in range(len(labels)):
#                         correct_samples += (predicted[i] == labels[i]).sum().item()
#                     acc += correct_samples / total_samples
#                 else:
#                     total_samples = len(testloader.dataset)
#                     loss += criterion(outputs, labels).item()
#                     _, predicted = torch.max(outputs.data, 1)
#                     correct_samples += (predicted == labels).sum().item()
#         # Output loss and poisoning accuracy
#         main_acc = np.round((acc/count), 4) if dataset == 'celeba' else np.round((correct_samples/total_samples),4)
#         return loss, 0.0, main_acc