# Finalised_Algorithm_ResNet

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