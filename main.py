import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from flwr.common.logger import log
from logging import INFO

from dataset import prepare_clientdataset
from client import generate_nnclient_fn, weighted_average
from server import get_on_fit_config, get_evaluate_fn, get_attacker_evaluate_fn
from bd_strategy import NNtrain
from model import CNNet, CNN_LFW, get_parameters

import torchvision.models as models
import torch.nn as nn
import torch

@hydra.main(config_path="conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    """ 1. Parse config & get experiment output dir """    
    log(INFO, OmegaConf.to_yaml(cfg))
    
    device = torch.device(cfg.device)

    """ 2. Prepare dirty and clean dataset """
    bdtrainloaders, bdvalloaders, cleantrainloaders, cleanvalloaders, testloaders = prepare_clientdataset(cfg.dataset_config, cfg.num_clients, cfg.batch_size, cfg.dataset, cfg.config_fit.poisoning_rate, val_ratio=0.2)

    if cfg.dataset == 'lfw':
        params = get_parameters(CNN_LFW(cfg.num_classes))
    elif cfg.dataset == 'mnist':
        params = get_parameters(CNNet(cfg.num_classes, cfg.dataset))
    else:
        model = models.resnet18().to(device) 
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, cfg.num_classes).to(device)
        params = get_parameters(model)

    """ 3. Define your clients """
    nn_client_fn = generate_nnclient_fn(cfg, cleantrainloaders, cleanvalloaders, bdtrainloaders, bdvalloaders, cfg.num_classes, cfg.num_clients, cfg.dataset, cfg.target_label, cfg.config_fit.poisoning_rate, device)

    """Start Actual Simulation"""
    _ = fl.simulation.start_simulation(
        client_fn=nn_client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds), 
        strategy=NNtrain(
            #client's info
            fraction_fit=0.00001,                                 # fraction of clients used during training (Default to 1.0)
            min_fit_clients=cfg.num_clients_per_round_fit,        # minimum number of clients used during training (fit())
            fraction_evaluate=0.00001,                            # fraction of clients used during validation
            min_evaluate_clients=cfg.num_clients_per_round_eval,  # minimum number of clients using during validation evaluate()
            min_available_clients=cfg.num_clients,                # minimum number of total clients in the simulation
            # Remark: in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
            
            # server's side
            initial_parameters=fl.common.ndarrays_to_parameters(params),
            on_fit_config_fn=get_on_fit_config(cfg.config_fit),
            evaluate_fn=get_evaluate_fn(cfg, cfg.num_classes, testloaders, 0),
            attack_evaluate_fn=get_attacker_evaluate_fn(cfg, cfg.num_classes, testloaders, cfg.target_label),  
            evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        ),
        client_resources={
            "num_cpus": 1,   #2
            "num_gpus": 0.5, #0.0
        }, 
    )

if __name__ == "__main__":
    main()