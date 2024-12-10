from omegaconf import DictConfig
from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr.common import NDArrays, Scalar, Status, Code, EvaluateRes
from flwr.common.typing import Metrics

import torch
import flwr as fl
from model import CNNet, CNN_LFW, train, test

import torch.nn as nn
import torchvision.models as models

class PresetClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 num_classes,
                 malicious,
                 dataset,
                 target_label,
                 p_rate,
                 device,
                 model,
                 ) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.device = device

        self.dataset = dataset
        self.target_label = target_label
        self.p_rate = p_rate

        self.model = model

        self.malicious = malicious

    """receives and copies the parameter sent from server into the client's local model"""
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k,v in params_dict})
        state_dict = OrderedDict({ k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    """get local model parameters and return them as a list of numpy arrays."""
    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        proximal_mu = config['proximal_mu']
        poisoning_rate = config['poisoning_rate']

        # do local training
        train(self.model, self.trainloader, self.device, epochs, lr, proximal_mu, self.malicious, poisoning_rate, self.dataset, self.target_label)

        # return the updated model, the number of examples in the client, and a dictionary of metrics
        return self.get_parameters({}), len(self.trainloader), {"malicious": self.malicious}


    """client uses validation data to evaluate the model"""
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, poison_acc, accuracy = test(self.model, self.valloader, self.device, self.malicious, self.dataset, self.target_label)

        return float(loss), len(self.valloader), {"global_poison": poison_acc, "accuracy": accuracy, "malicious": self.malicious}
    

#++++++++++++++++++++++++++++++++++++++++++++++Generate Client Function+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Return a function that can be used by the VirtualClientEngine to spawn a FlowerClient with client id `cid`."""
def generate_nnclient_fn(config: DictConfig, goodtrainloaders, goodvalloaders, bdtrainloaders, bdvalloaders, num_classes, num_clients, dataset, target_label, p_rate, device):

    if 'cifar' in dataset:  #cifar => ResNet
        model = models.resnet18().to(device)
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, num_classes).to(device) 
    elif dataset == 'lfw':  
        model = CNN_LFW(num_classes)
    else:
        model = CNNet(num_classes, config.dataset)

    # This function will be called internally by the VirtualClientEngine
    # Each time the cid-th client is told to participate in the FL simulation (whether it is for doing fit() or evaluate())
    def client_fn(cid: str):

        modnum = int(num_clients * config.dataset_config.ratio_benign_client)  # 2 different types of clients: 60% benign and 40% malicious (num_clients is 100)

        if int(cid) < modnum:
            # Benign Client
            return PresetClient(
                trainloader=goodtrainloaders[int(cid)],
                valloader=goodvalloaders[int(cid)],
                num_classes=num_classes,
                malicious=0,
                dataset=dataset,
                target_label=target_label,
                p_rate=p_rate,
                device=device,
                model=model,
            )
        else:
            # Backdoor Client 
            return PresetClient(
                trainloader=bdtrainloaders[int(cid)-modnum],
                valloader=bdvalloaders[int(cid)-modnum],
                num_classes=num_classes,
                malicious=1,
                dataset=dataset,
                target_label=target_label,
                p_rate=p_rate,
                device=device,
                model=model,
            )

    # return the function to spawn client
    return client_fn


#++++++++++++++++++++++++++++++++++++++++++++++Aggregate Accuracy of a Client+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    poison_acc = [num_examples * m["global_poison"] for num_examples, m in metrics]

    #Aggregate and return weighted average
    return {"accuracy": sum(accuracies) / sum(examples), "global_poison": sum(poison_acc) / sum(examples)}