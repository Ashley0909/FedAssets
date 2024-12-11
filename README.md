# FedAssets

## Main File Compositions and Functions

##### `main.py`
Main function to be run in terminal

##### `bd_strategy.py`
Our algorithm, how clustering and aggregation works

##### `dataset.py`
Imports the data, the calls the function in `dataset_preparation.py` to return good and bad data. Then partitions the data into training, validation and test datasets.

##### `dataset_preparation.py`
Partition dataset to good and bad datasets according to IID or Non-IID format

##### `model.py`
Train and test models, poison selective data if client is preset to be malicious

## Running Experiments
`ssh kudu` 

Adjust experiment settings on `conf/base.yaml`

Run `sbatch cpu.sbatch`

*Alternatively, you can simply run on the terminal by `python3 main.py`*

## Datasets
Labelled Face in the Wild (LFW): We restricted this dataset to only include individuals who have at least certain number of images in the dataset. When `min_face_per_person` is set to be 70, there are 7 individuals who meet this criterion. Therefore the number of class for LFW is 7. The number of samples is 1288, with image size (50, 37). When `min_face_per_person` is set to be 40, there are 19 individuals who meet this criterion. The number of samples is 1867, with image size (50, 37).


**Note:** We have to explicitly set the number of classes to set up the constructuion of learning models.

## Possible combinations in `base.yaml`

| Dataset | num_classes | target_label | batch_size | num_clients | num_clients_per_round_fit |
| ------- |:-------:| -------:| -------:| -------:| -------:|
| 'lfw'      | 7 | 0 | 5 | 10 | 10 |
| 'mnist'   | 10 | 9 | 20 | 50 | 15 |
| 'cifar10' | 10 | 9 | 20 | 50 | 15 |
| 'cifar100' | 100 | 9 | 20 | 50 | 15 |
| 'celeba' | 10 | 7 ('Smiling") | 64 | 50 | 15 |

