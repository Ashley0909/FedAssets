import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from dataset import prepare_clientdataset
from client import generate_nnclient_fn, weighted_average
from server import get_on_fit_config, get_evaluate_fn, get_attacker_evaluate_fn
from bd_strategy import NNtrain
from model import Net, get_parameters

import torchvision.models as models
import torch.nn as nn
import torch

@hydra.main(config_path="conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    """ 1. Parse config & get experiment output dir """
    print(OmegaConf.to_yaml(cfg))
    # save_path = HydraConfig.get().runtime.output_dir
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #GPU

    """ 2. Prepare dirty and clean dataset """
    bdtrainloaders, bdvalloaders, cleantrainloaders, cleanvalloaders, testloaders = prepare_clientdataset(cfg.dataset_config, cfg.num_clients, cfg.batch_size, cfg.dataset, cfg.config_fit.poisoning_rate)

    if cfg.dataset == 'cifar10' or cfg.dataset == 'cifar100':
        model = models.resnet18()  #.to(device)  #GPU
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, cfg.num_classes)  #.to(device)  #GPU
        params = get_parameters(model)
    elif cfg.dataset == 'mnist':
        params = get_parameters(Net(cfg.num_classes, cfg.num_channels))

    """ 3. Define your clients """
    # nn_client_fn = generate_nnclient_fn(cfg, cleantrainloaders, cleanvalloaders, bdtrainloaders, bdvalloaders, cfg.num_classes, cfg.num_clients, cfg.num_channels, cfg.target_label, cfg.config_fit.poisoning_rate, device)  #GPU
    nn_client_fn = generate_nnclient_fn(cfg, cleantrainloaders, cleanvalloaders, bdtrainloaders, bdvalloaders, cfg.num_classes, cfg.num_clients, cfg.num_channels, cfg.dataset, cfg.target_label, cfg.config_fit.poisoning_rate)

    """Start Actual Simulation"""
    nnet = fl.simulation.start_simulation(
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
            evaluate_fn=get_evaluate_fn(cfg.config_fit, cfg.num_classes, cfg.num_channels, testloaders, 2), 
            attack_evaluate_fn=get_attacker_evaluate_fn(cfg.num_classes, cfg.num_channels, testloaders, cfg.target_label),  
            evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        ),
        client_resources={
            "num_cpus": 2,   #2
            "num_gpus": 0.0, #0.0
        }, 
    )

    """ 6. Save your results """
    # results_path = Path(save_path) / 'results.pkl'

    # results = {'nn': nnet}

    # # save the results as a python pickle
    # with open(str(results_path), "wb") as h:
    #     pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()