from omegaconf import DictConfig
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler

from dataset_preparation import _partition_data
import matplotlib.pyplot as plt


def get_mnist(data_path: str = './data'):

    tr = Compose([ToTensor(), Normalize((0.1307,),(0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def get_cifar(data_path: str = './data'):

    tr = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(data_path, train=True, download=True, transform=tr)
    testset = CIFAR10(data_path, train=False, download=True, transform=tr)

    return trainset, testset

def get_cifar100(data_path: str = './data'):

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_tr = Compose([RandomCrop(32, padding=4,padding_mode='reflect'), 
                         RandomHorizontalFlip(), 
                         ToTensor(), 
                         Normalize(*stats,inplace=True)
                        ])
    test_tr = Compose([ToTensor(), Normalize(*stats)])

    trainset = CIFAR100(data_path, train=True, download=True, transform=train_tr)
    testset = CIFAR100(data_path, train=False, download=True, transform=test_tr)

    return trainset, testset

def prepare_clientdataset(config: DictConfig,
                    num_partitions: int, 
                    batch_size: int,
                    dataset: str,
                    p_rate: float,
                    val_ratio: float = 0.1,
                    ):
    
    
    """Import mixed poisoned dataset"""
    if dataset == 'cifar10':
        trainset, testset = get_cifar(data_path = './data')
    elif dataset == 'mnist':
        trainset, testset = get_mnist(data_path = './data')
    elif dataset == 'cifar100':
        trainset, testset = get_cifar100(data_path= './data')
    
    """Partition the data"""
    goodtrainsets, badtrainsets = _partition_data(
        trainset,
        num_partitions,
        p_rate,
        benign_ratio=config.ratio_benign_client,
        iid=config.iid,
        balance=config.balance,
        power_law=config.power_law,
        dirichlet=config.dirichlet,
        alpha=config.alpha,
        seed=42,
    )

    bdtrainloaders = []
    bdvalloaders = []
    cleantrainloaders = []
    cleanvalloaders = []

    for bdtrainset_ in badtrainsets:  # per client
        num_total = len(bdtrainset_)  # total number of samples per client
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(bdtrainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        bdtrainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True))
        bdvalloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True))
        # bdtrainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        # bdvalloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    for ctrainset_ in goodtrainsets:
        num_total = len(ctrainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(ctrainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        cleantrainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True))
        cleanvalloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True))
        # cleantrainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        # cleanvalloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    testloader = DataLoader(testset, batch_size=128)

    return bdtrainloaders, bdvalloaders, cleantrainloaders, cleanvalloaders, testloader


def show_images_labels(images, labels, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(12,5))

    for i in range(num_samples):
        image = images[i]
        label = labels[i]
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')

    plt.show(block=True)

