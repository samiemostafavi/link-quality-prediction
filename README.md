# README

This repository contains the code for link quality prediction using EDAF and EasyTemporalPointProcess projects.

## Dependencies

This code is tested with Python 3.9. 
To create a Python 3.9 environment with Conda, you can use the following command:

```shell
conda create --name lqp python=3.9
```
This command will create a new Conda environment with Python 3.9 installed.

```shell
conda activate lqp
```

## Installation

To install the required dependencies and create a Conda environment, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/samiemostafavi/link-quality-prediction.git
    ```

2. Change into the project directory:

    ```shell
    cd link-quality-prediction
    ```

3. Create a new Conda environment:

    ```shell
    conda create --name lqp python=3.9
    ```

4. Activate the Conda environment:

    ```shell
    conda activate lqp
    ```

5. Install the required packages:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

Once the dependencies are installed, you need to bring/create the database file.

To create the `database.db` file if it is not created, you can insert edaf raw results into `data` folder.

Here's an example of how the directory structure would look like:
```
├── data
│   └── 240928_082545_results
│       ├── gnb
│       ├── ue
│       └── upf
```

Run the following command to process the raw data and create `database.db` file:
```shell 
python main.py -t preprocess -s data/240928_082545_results
```

Create an experiment coonfiguration file using json format and insert it in this folder under the name `experiment_config.json`.
The file should be like:
```json
{
    "total_prbs_num": 106,
    "symbols_per_slot": 14,
    "slots_per_frame": 20,
    "slots_duration_ms": 0.5,
    "scheduling_map_num_integers": 4,
    "max_num_frames": 1024,
    "max_harq_attempts": 3
}
```

Now you can remove gnb, ue, and upf folders to save space. Then the directory structure should look like:
```
├── data
│   └── 240928_082545_results
│       ├── experiment_config.json
│       └── database.db
```

### Create a Dataset


Make sure to update/create the training dataset configuration file and insert it in the `config` folder under the name `dataset_config.json`.
The file should be like:
```json
{ 
    "s9" :{
        "time_mask": [0.1,0.9],
        "filter_packet_sizes": [128], 
        "dim_process": 1,
        "history_window_size": 20,
        "dataset_size_max": 10000,
        "split_ratios": [0.7,0.15,0.15]
    },
    "s10" : {
        ...
    }
}
```
The key for each dictionary inside this file is the name of the experiment.

Then the directory structure should look like:
```
├── config
│   └── dataset_config.json
├── data
│   └── s9_results
│       ├── experiment_config.json
│       └── database.db
```

Now you can create a training dataset by executing the script:
```shell
python main.py -t plot_link_data -f -s data/s9_results -c config/dataset_config.json -g s9 -n test0
python main.py -t create_training_dataset -s data/s9_results -c config/dataset_config.json -g s9 -n test0
```

NOTE: the argument `-f` will use fast functions in edaf analyzer to speed up the processing. By not using it probably you will get more accurate results but very long processing time.

### Train a Model

Types of events is dependent on the experiment's configuration: `max_harq_attempts`. Typically this is set to 3, which means if the third harq attempt fails, MAC layer will give up and we will have RLC retransmission.
For example, with `max_harq_attempts=3` we will have 3 types of events:
1. first harq retransmission
2. second harq retransmission
3. repeated rlc attempt

Hence, the event types space's size is 3 (with ids: 0,1,2) and the padding id would be 3.
```
  s9_results_0:
    data_format: json
    train_dir:  ./data/s9_results/training_datasets/test0/train.pkl
    valid_dir:  ./data/s9_results/training_datasets/test0/dev.pkl
    test_dir:  ./data/s9_results/training_datasets/test0/test.pkl
    data_specs:
      num_event_types: 3
      pad_token_id: 3
      padding_side: right
      truncation_side: right
```


Create a training config file in the config folder `training_config.yaml` and pass it like:
```shell
python main.py -t train_model -f -c config/training_config.yaml -i THP_train
```
Specify an experiment id, which chooses a specific configuration inside the yaml file.

Then the directory structure should look like:
```
├── config
│   ├── training_config.yaml
│   └── dataset_config.json
├── data
│   └── 240928_082545_results
│       ├── experiment_config.json
│       └── database.db
```


### Predict using a Model

The argument `-i` refers to the trained model folder name

```shell
python main.py -t generate_predictions -p probabilistic -c config/prediction_config.json -s data/240928_082545_results -n test0 -i 218481_140401600746112_241017-110318
python main.py -t generate_predictions -p sampling -c config/prediction_config.json -s data/240928_082545_results -n test0 -i 218481_140401600746112_241017-110318
```

plot the predictions by
```shell
python main.py -t plot_predictions -s data/240928_082545_results -n test0 -i 254926_139826076787328_241018-105017
```
here ihe argument `-i` refers to the prediction folder name

### Evaluate a Model

The argument `-i` refers to the trained model folder name

```shell
python main.py -t evaluate -s data/s39_results -n test0 -i 579608_140637935051392_241029-050156
```
