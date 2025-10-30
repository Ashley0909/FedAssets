from omegaconf import DictConfig
from logging import WARNING
import torch
import numpy as np
import torch.utils
import torch.utils.data
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, CelebA
from torchvision.transforms import ToTensor, Normalize, Compose, RandomCrop, RandomHorizontalFlip, Resize
from torch.utils.data import random_split, DataLoader
from flwr.common.logger import log

from dataset_preparation import _partition_data
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

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

def get_lfw(data_path: str = './data'):
    data = fetch_lfw_people(data_home=data_path, min_faces_per_person=70, resize=0.4)
    images = data.images
    labels = data.target
    images = images/255.0 # Normalise so that pixel values are between 0 and 1

    images = images[:, np.newaxis, :, :]  # Shape: [N, 1, H, W]

    # Convert data to PyTorch tensors
    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Split into train and test
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Create DataLoader for batch processing
    trainset = torch.utils.data.TensorDataset(train_images, train_labels)
    testset = torch.utils.data.TensorDataset(test_images, test_labels)

    return trainset, testset

def get_celeba(data_path: str = './data'):
    tr = Compose([Resize((128,128)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # selected_attributes = ['Bald', 'Bangs', 'Black_Hair',  'Blond_Hair', 'Eyeglasses', 'Male', 'Mustache', 'Smiling', 'Wearing_Earrings', 'Wearing_Hat']
    selected_attributes = ['Eyeglasses', 'Male', 'Smiling']

    trainset = CelebA(data_path, split='train', download=True, transform=tr)
    testset = CelebA(data_path, split='test', download=True, transform=tr)
    valset = CelebA(data_path, split='valid', download=True, transform=tr)

    trainset = filter_attributes(trainset, selected_attributes)
    testset = filter_attributes(testset, selected_attributes)
    valset = filter_attributes(valset, selected_attributes)

    return trainset, testset, valset

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
    elif dataset == 'lfw':
        trainset, testset = get_lfw(data_path= './data')
    elif dataset == 'celeba':
        trainset, testset, valset = get_celeba(data_path= './data')
    else:
        log(WARNING, "Invalid Dataset")
        exit(-1)
    
    """Partition the data"""
    goodtrainsets, badtrainsets = _partition_data(
        trainset,
        num_partitions,
        batch_size,
        p_rate,
        benign_ratio=config.ratio_benign_client,
        iid=config.iid,
        balance=config.balance,
        power_law=config.power_law,
        dirichlet=config.dirichlet,
        alpha=config.alpha,
        seed=42,
    )

    if dataset == 'celeba': # They have build-in validation data
        goodvalsets, badvalsets = _partition_data(
            valset,
            num_partitions,
            batch_size,
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

    if dataset == 'celeba':
        for bdtrainset_ in badtrainsets:
            bdtrainloaders.append(DataLoader(bdtrainset_, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)) # drop_last=True throws away the last batch if it has fewer samples
        for bdvalset_ in badvalsets:
            bdvalloaders.append(DataLoader(bdvalset_, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True))
        for ctrainset_ in goodtrainsets:
            cleantrainloaders.append(DataLoader(ctrainset_, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True))
        for cvalset_ in goodvalsets:
            cleanvalloaders.append(DataLoader(cvalset_, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True))
    else:
        for bdtrainset_ in badtrainsets:  # per client
            num_total = len(bdtrainset_)  # total number of samples per client
            num_val = int(val_ratio * num_total)
            num_train = num_total - num_val

            for_train, for_val = random_split(bdtrainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

            bdtrainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)) # drop_last=True throws away the last batch if it has fewer samples
            bdvalloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True))

        for ctrainset_ in goodtrainsets:
            num_total = len(ctrainset_)
            num_val = int(val_ratio * num_total)
            num_train = num_total - num_val

            for_train, for_val = random_split(ctrainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

            cleantrainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True))
            cleanvalloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True))

    testloader = DataLoader(testset, batch_size=batch_size)

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

def filter_attributes(dataset, selected_attrs):
    attr_names = dataset.attr_names
    selected_indices = [attr_names.index(attr) for attr in selected_attrs]

    dataset.attr = dataset.attr[:, selected_indices]
    dataset.attr_names = selected_attrs

    return dataset
