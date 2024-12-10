from collections import OrderedDict
from omegaconf import DictConfig
import torch
from PIL import Image

from model import CNNet, CNN_LFW, test
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn

def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
            "proximal_mu": config.proximal_mu,
            "poisoning_rate": config.poisoning_rate,
        }
 
    return fit_config_fn


def get_evaluate_fn(config: DictConfig, num_classes: int, testloader, clean):  #CPU
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters):  #CPU
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        device = config.device

        if 'cifar' in config.dataset:  #cifar => ResNet
            model = models.resnet18().to(device)
            n_features = model.fc.in_features
            model.fc = nn.Linear(n_features, num_classes).to(device)
        elif config.dataset == "lfw":
            model = CNN_LFW(num_classes)
        else:
            model = CNNet(num_classes, config.dataset)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({ k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.
        loss, _, accuracy = test(model, testloader, device, malicious=clean, dataset=config.dataset, target_label=config.target_label)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"accuracy": accuracy}

    return evaluate_fn

def get_attacker_evaluate_fn(config: DictConfig, num_classes: int, testloader, target_label):  #CPU
    """Define function for global evaluation on the server."""

    def attacker_evaluate_fn(server_round: int, parameters):  #CPU
        device = config.device
        if 'cifar' in config.dataset:  #cifar => ResNet
            model = models.resnet18().to(device)
            n_features = model.fc.in_features
            model.fc = nn.Linear(n_features, num_classes).to(device)
        elif config.dataset == "lfw":
            model = CNN_LFW(num_classes)
        else:
            model = CNNet(num_classes, config.dataset)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({ k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        if 'cifar' in config.dataset:
            t_img = Image.open("./triggers/trigger_white.png").convert('RGB')
        else:
            t_img = Image.open("./triggers/trigger_white.png").convert('L')
        t_img = t_img.resize((5, 5))
        transform = transforms.ToTensor()
        trigger_img = transform(t_img)

        criterion = torch.nn.CrossEntropyLoss()
        poisoned, loss = 0, 0.0
        model.eval()
        model.to(device) # ?
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                images[:,:, -5:, -5:] = trigger_img
                tensor_target = torch.full((len(labels),), target_label, dtype=torch.int32).to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                for l in range(len(labels)):
                    if (predicted[l] == tensor_target[l]):
                        poisoned += 1
        accuracy = poisoned / len(testloader.dataset)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"accuracy": accuracy}

    return attacker_evaluate_fn