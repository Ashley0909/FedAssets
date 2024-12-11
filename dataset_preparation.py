"""Functions for dataset download and processing."""
from typing import List, Optional, Tuple
import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split

def _partition_data(
    trainset,
    num_clients,
    batch_size,
    p_rate,
    benign_ratio,
    iid: Optional[bool] = False,
    power_law: Optional[bool] = False,
    dirichlet: Optional[bool] = True,
    alpha: Optional[float] = 0.5,
    balance: Optional[bool] = False,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    
    # Balance the class labels if it is not balanced (not balanced for non iid)
    if balance:
        trainset = _balance_classes(trainset, seed)

    # Number of images for one client
    partition_size = int(len(trainset) / num_clients)

    num_good_clients = int(num_clients*benign_ratio) # 18
    num_bad_clients = num_clients - num_good_clients # 12
    num_good_samples = int(len(trainset) * benign_ratio) # 36000

    benignset = Subset(trainset, list(range(0, num_good_samples)))  # 1 to 36000
    maliciousset = Subset(trainset, list(range(num_good_samples, len(trainset))))  # 36001 to 60000

    if iid:
        if len(benignset) == 0:
            good_lengths = []
        else:
            good_lengths = [int(len(benignset)/num_good_clients)] * num_good_clients
            if (np.sum(good_lengths) != len(benignset)):
                good_lengths[0] += len(benignset) - np.sum(good_lengths)

        if len(maliciousset) == 0:
            bad_lengths = []
        else:
            bad_lengths = [int(len(maliciousset)/(num_clients-num_good_clients))] * (num_clients-num_good_clients)
            if (np.sum(bad_lengths) != len(maliciousset)):
                bad_lengths[0] += len(maliciousset) - np.sum(bad_lengths)
            
        goodsets = random_split(benignset, good_lengths, torch.Generator().manual_seed(seed))
        badsets = random_split(maliciousset, bad_lengths, torch.Generator().manual_seed(seed))

    else:
        # Since the subsets do not have target as the attribute, we have to create it by ourselves to use power law
        # benignset.targets = torch.as_tensor([benignset[i][1] for i in range(len(benignset))])
        # maliciousset.targets = torch.as_tensor([maliciousset[i][1] for i in range(len(maliciousset))])

        if power_law:
            trainset_sorted = _sort_by_class(benignset)
            goodsets = _power_law_split(
                trainset_sorted,
                [],
                num_partitions=num_good_clients,
                num_labels_per_partition=2,
                min_data_per_partition=500,
                mean=0.0,
                sigma=2.0,
            )

            clean_samples = int(len(maliciousset) * (1-p_rate))
            innocentset = Subset(trainset, list(range(num_good_samples, num_good_samples+clean_samples)))
            poisonedset = Subset(trainset, list(range(num_good_samples+clean_samples, len(trainset))))

            innocentset.targets = torch.as_tensor([innocentset[i][1] for i in range(len(innocentset))])
            poisonedset.targets = torch.as_tensor([poisonedset[i][1] for i in range(len(poisonedset))])

            trainset_sorted = _sort_by_class(innocentset)
            badsets = _power_law_split(
                trainset_sorted,
                poisonedset,
                num_partitions=(num_clients-num_good_clients),
                num_labels_per_partition=2,
                min_data_per_partition=200,
                mean=0.0,
                sigma=2.0,
            )
        elif dirichlet:
            """Benign dataset"""
            goodsets = sample_dirichlet(benignset, num_good_clients, alpha, batch_size)
            """Malicious dataset"""
            badsets = sample_dirichlet(maliciousset, num_bad_clients, alpha, batch_size)
            # badsets = goodsets  #testing
        else:
            shard_size = int(partition_size / 2) # partition size is number of images per client
            """Benign dataset"""
            goodsets = random_allocate(benignset, num_good_clients, shard_size, seed, True)
            """Malicious dataset"""
            badsets = random_allocate(maliciousset, num_bad_clients, shard_size, seed+20, False)

    return goodsets, badsets

def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    """Balance the classes of the trainset.

    Oversample and Undersample data accordingly so we still have the same total number of samples (60000 for MNIST)

    """
    class_counts = np.bincount(trainset.targets)            # number of samples in each label
    target_length = len(trainset) // len(class_counts)      # 60000/10 = 6000 samples per label

    subsets = []
    subset_targets = []

    for label, count in enumerate(class_counts):
        indices = (trainset.targets == label).nonzero().view(-1)
        if count < target_length:
            # Oversample the data for this class
            oversampled_indices = indices.repeat((target_length // count).item() + 1)  #repeat the whole set of data
            indices = oversampled_indices[:target_length]                              #trim the dataset to desired length
        else:
            #Undersample the data for this class
            indices = indices[:target_length]
    
        subsets.append(Subset(trainset, indices))
        subset_targets.append(trainset.targets[indices])

    unshuffled = ConcatDataset(subsets)
    unshuffled_targets = torch.cat(subset_targets)

    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled

def _sort_by_class(
    trainset: Dataset,
) -> Dataset:
    """Sort dataset by class/label."""

    class_counts = np.bincount(trainset.targets)
    idxs = trainset.targets.argsort()  # sort targets in ascending order

    tmp = []  # create subset of smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(trainset, idxs[start : int(count + start)])
            
        )  # add rest of classes
        tmp_targets.append(trainset.targets[idxs[start : int(count + start)]])
        
        start += count
    sorted_dataset = ConcatDataset(tmp)  # concat dataset
    sorted_dataset.targets = torch.cat(tmp_targets)  # concat targets
    return sorted_dataset

# pylint: disable=too-many-locals, too-many-arguments
def _power_law_split(
    sorted_trainset: Dataset,
    poisoned_trainset: Dataset,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> Dataset:
    """Partition the dataset following a power-law distribution. It follows the.

    implementation of Li et al 2020: https://arxiv.org/abs/1812.06127 with default
    values set accordingly.

    Parameters
    ----------
    sorted_trainset : Dataset
        The training dataset sorted by label/class.
    num_partitions: int
        Number of partitions to create
    num_labels_per_partition: int
        Number of labels to have in each dataset partition. For
        example if set to two, this means all training examples in
        a given partition will belong to the same two classes. default 2
    min_data_per_partition: int
        Minimum number of datapoints included in each partition, default 10
    mean: float
        Mean value for LogNormal distribution to construct power-law, default 0.0
    sigma: float
        Sigma value for LogNormal distribution to construct power-law, default 2.0

    Returns
    -------
    Dataset
        The partitioned training dataset.
    """
    targets = sorted_trainset.targets       
    full_idx = list(range(len(targets)))    #0,..,59999 (len = 60000)

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)     
    labels_cs = [0] + labels_cs[:-1].tolist()    

    partitions_idx: List[List[int]] = []
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            # label for the u_id-th client
            cls = (u_id + cls_idx) % num_classes    #client 0 gets class 0 and 1, client 1 gets class 1 and 2, ...
            # record minimum data
            indices = list(
                full_idx[
                    labels_cs[cls] 
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ]
            )
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class
    
    # Since we poisoned a lot of images to be of the target label, after allocating the fixed number of samples, we first have to allocate the target label to each client
    if poisoned_trainset != []:
        poisoned_samples = [int(len(poisoned_trainset)/ num_partitions)] * num_partitions
        poisoned_samples[-1] += len(poisoned_trainset) - sum(poisoned_samples)
        
        poi_indices = np.array(range(len(poisoned_trainset)))
        poi_subsets = []
        for u_id, n in enumerate(poisoned_samples):
            selected_samples = np.random.choice(poi_indices, size=n, replace=False)
            poi_indices = np.setdiff1d(poi_indices, selected_samples)
            poi_subsets.append(Subset(poisoned_trainset, selected_samples))

    # add remaining images following power-law
    probs = np.random.lognormal(
        mean,
        sigma,
        (num_classes, int(num_partitions / num_classes), num_labels_per_partition),
    )
    remaining_per_class = class_counts - hist
    # obtain how many samples each partition should be assigned for each of the
    # labels it contains
    # pylint: disable=too-many-function-args
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(probs[cls, u_id // num_classes, cls_idx])

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct subsets
    partitions = [Subset(sorted_trainset, p) for p in partitions_idx]

    if poisoned_trainset != []:
        final_partitions = []
        for c in range(num_partitions):
            overall = []
            overall.append(partitions[c])
            overall.append(poi_subsets[c])
            final = ConcatDataset(overall)
            final_partitions.append(final)
        partitions = final_partitions

    return partitions

def random_allocate(dataset, num_of_clients, shard_size, seed, benign):
    if benign:
        idxs = dataset.targets.argsort()
        mod_data = Subset(dataset, idxs) # a set that has the train data sorted in terms of their labels
    else:
        indices = dataset.targets.argsort().tolist()
        idxs = random.sample(indices, len(indices))  # get a random permutation of the list
        mod_data = Subset(dataset, idxs)
    tmp = []
    for idx in range(num_of_clients * 2): 
        tmp.append(
            Subset(
                mod_data, np.arange(shard_size * idx, shard_size * (idx + 1))
            )
        )
    idxs_list = torch.randperm(
        num_of_clients * 2, generator=torch.Generator().manual_seed(seed)
    )
    resultset = [
        ConcatDataset((tmp[idxs_list[2 * i]], tmp[idxs_list[2 * i + 1]]))
        for i in range(num_of_clients)
    ]

    return resultset

def sample_dirichlet(dataset, num_of_clients, alpha, batch_size):
    classes = {}  # list of index of the label {label: indicies}
    for idx, x in enumerate(dataset):
        _, label = x
        if type(label) == torch.Tensor:
            if len(label) > 1: # if it is a multi-label data
                return sample_dirichlet_multilabel(dataset, num_of_clients, alpha, batch_size)
            label = label.item()
        if label in classes:
            classes[label].append(idx)
        else:
            classes[label] = [idx]
    num_classes = len(classes.keys())
    batch_per_class = int(batch_size/num_classes)
    resultset = []

    for n in range(num_classes):
        random.shuffle(classes[n])   # shuffle the indicies of the labels
        class_size = len(classes[n]) # count the number of samples of the labels
        class_subset = Subset(dataset, np.array(classes[n]))  # make the Subset of the shuffled indices
        rerun = True
        while rerun:
            sampled_probabilities = class_size * np.random.dirichlet(np.array(num_of_clients * [alpha]))
            rerun = False
            for user in range(num_of_clients):
                if int(round(sampled_probabilities[user])) < batch_per_class:
                    rerun = True
                    break

        for user in range(num_of_clients):
            num_imgs = int(round(sampled_probabilities[user]))
            sampled_list = Subset(class_subset, np.arange(min(len(classes[n]), num_imgs)))
            if len(resultset) < len(range(num_of_clients)):
                resultset.append(sampled_list)
            else:
                resultset[user] = ConcatDataset((resultset[user], sampled_list))

    return resultset

def sample_dirichlet_multilabel(dataset, num_of_clients, alpha, batch_size):
    num_samples = len(dataset)
    rerun = True
    while rerun:
        rerun = False
        sampled_probabilities = np.random.dirichlet([alpha] * num_of_clients)
        client_sample_counts = (sampled_probabilities * num_samples).astype(int)
        for i in range(len(client_sample_counts)):
            if client_sample_counts[i] < batch_size: # if the client has less data than batch_size, we allocate the extra to it 
                rerun = True
                break

    diff = num_samples - sum(client_sample_counts)
    client_sample_counts[0] += diff

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    resultset = []
    start_idx = 0
    for count in client_sample_counts:
        end_idx = start_idx + count
        resultset.append(Subset(dataset, indices[start_idx:end_idx]))
        start_idx = end_idx
    
    return resultset

# def sample_dirichlet_multilabel(dataset, num_of_clients, alpha, batch_size):
#     # Extract targets (multi-label attribute matrix)
#     targets = torch.stack([dataset[i][1] for i in range(len(dataset))])  # Assuming dataset[i][1] is the multi-label target

#     # Iterate over each attribute (binary classification task)
#     num_labels = targets.size(1)

#     dirichlet_samples = np.random.dirichlet([alpha] * num_labels, num_of_clients)
#     image_assigned = np.zeros(len(dataset), dtype=bool)

#     client_data = {i:[] for i in range(num_of_clients)} # but sometimes some clients don't have any data:(

#     for idx, label in enumerate(targets):
#         if image_assigned[idx]:
#             continue # skip if the image is already allocated

#         weighted_scores = np.sum(dirichlet_samples * label.numpy(), axis=1)
#         assigned_client = np.argmax(weighted_scores)

#         client_data[assigned_client].append(idx)
#         image_assigned[idx] = True

#     resultset = []
#     ids = []
#     for client_id, indices in client_data.items():
#         ids.append(client_id)
#         resultset.append(Subset(dataset, indices))

#     return resultset