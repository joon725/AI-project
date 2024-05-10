## Gas Source Localization preprocess codes utilization

### Contents :
1. NBC_RAMS.py : convert raw NBC_RAMS data(bgr) to dataset.pkl(it includes gas diffusion data and gas source data in numpy format, and metadata of the dataset)
2. preprocess.py : convert dataset.pkl to train, validation, test dataset
3. data_integrate.py : integrate different (train, validation, test) datasets in to one (train, validation, test dataset)

### 1. NBC_RAMS.py

#### 1. command : 
``` shell script
python NBC_RAMS.py --config ../cfg/nbc_rams.yaml 
```

#### 2. configuration file location - ../cfg/nbc_rams.yaml

##### ## details about nbc_rams.yaml
##### 2.1. random_state
"random_state" is random seed setting for random sampling of visualization

##### 2.2. data path
"bgr_path" is a list of base folders paths of which subfolder structures follow the description below :
Ex. bgr_path = ['base_folder_1_path','base_folder_2_path']

```
base_folder_1_path                              base_folder_2_path
├── project_1                                   ├── project_1
│   ├── effect_1                                │   ├── effect_1 
│       ├── cwpn                                │       ├── cwpn 
│           ├── 109T_moa                        │           ├── 109T_moa
│               ├── 109T_moa2                   │                ├── 109T_moa2
│                   ├── bgr_name_1.bgr          │                    ├── bgr_name_1.bgr
│                   ├── bgr_name_2.bgr          │                    ├── bgr_name_2.bgr
│                   ...                         │                    ...
|       ├── effect_1.json                       │       ├── effect_1.json
│   ├── effect_2                                │   ├── effect_2
│       ├── cwpn                                │       ├── cwpn
│           ├── 109T_moa                        │           ├── 109T_moa
│           ...                                 │           ...
│       ├── effect_2.json                       │       ├── effect_2.json
├── project_2                                   ├── project_2
    ...                                             ...                    
```

##### 2.3. data setting
2.3.1. "region_info", "dx", "dy", "W", "H" are based on metadata generated from NBC_RAMS.

2.3.2. "normalize" is whether or not to implement min-max normalization for each frame. 

##### 2.4. source visualization
2.4.1. "visualization" is whether or not to visualize the sample data from the dataset. When it's set to True, the visualized mask and gas diffusion data are saved to save directory.

2.4.2. "max_num" is the number of samples that will be visualized.

##### 2.5. save setting
2.5.1. "save_dir" is a parent directory of the folder where the dataset is saved.

2.5.2. "name" is a folder name where the dataset is saved.
* final directory where the dataset is saved : save_dir/name/dataset.pkl

#### 3. Output : 
dataset.pkl, nbc_rams.yaml, result.log, sample & mask figures(optional)

* dataset.pkl is an object of NBC_RAMS class, which includes gas diffusion data, gas source data(both in numpy format), and metadata of the dataset.
* dataset.pkl is used afterwards in preprocess.py to make it as train, validation, test dataset.

### preprocess.py

#### 1. command : 
```shell script 
python preprocess.py --config ../cfg/preprocess.yaml 
``` 

#### 2. configuration file location - ../cfg/preprocess.yaml

##### ## details about nbc_rams.yaml 
##### 2.1. random_state
"random_state" is random seed setting for random sampling of visualization.

##### 2.2. data path
"pkl_path" is a path of a dataset.pkl that is generated from step1. NBC_RAMS.py

##### 2.3. preprocess setting
2.3.1. "train_ratio" is a ratio of train dataset.

2.3.2 "val_ratio" is a ratio of validation dataset.

* test_ratio is automatically determined by "train_ratio" and "validation_ratio".

2.3.3. "split_criterion" is a type of dataset split. 

* If split_criterion is 0, dataset is split based on source types.
* If split_criterion is 1, dataset is split based on scenarios where a scenario is defined as a pair of source location and condition (=(source,condition)). Condition here is a combination of bomb mass, wind speed, wind direction, time(morning or night).

2.3.4. "sequence" is the length of the each final gas diffusion time series data.

2.3.5. "stride" is an interval between starting time steps when creating a specific sequence of time series data from initial gas diffusion data.
ex. if a total timestep of an initial gas diffusion data set is T and sequence = 10, stride = 2, then the time series data from the set are made like 1~10, 3~12, 5~14, ... 

2.3.6. "tolerance" is the maximum proportion of frames containing all-zero values in the "sequence" to include in the final dataset.

ex. if a "sequence" is 10 and "tolerance" is 0.3, the maximum number of frames containing all-zero values is 10*0.3 = 3. If a number of all-zero frames are more than 4, then the sequence will not be included in the final dataset.

2.3.7. "max_scenario" is to determine the maximum number of scenarios to include in a dataset from the total data.

2.3.8. "data_per_scenario" is to determine a number of sequence data per scenario in a dataset from the total data.

##### 2.4. source visualization
2.4.1. "visualization" is whether or not to visualize the sample data from the dataset. When it's set to True, the visualized mask and gas diffusion data are saved to save directory.

2.4.2. "max_num" is the number of samples that will be visualized. "max_num" sequence data and corresonding mask figures will be saved.

##### 2.5. save setting
2.5.1. "save_dir" is a parent directory of the folder where the dataset is saved.

2.5.2. "name" is a folder name where the dataset is saved.
final directory where the dataset is saved : save_dir/name/train_dataset.pkl, save_dir/name/val_dataset.pkl, save_dir/name/test_dataset.pkl

#### 3. Output : 
train_dataset.pkl, val_dataset.pkl, test_dataset.pkl, preprocess.py, preprocess.yaml, result.log, sequence & mask figures(optional)

* train_dataset.pkl, val_dataset.pkl, test_dataset.pkl are datasets that will be used in model training, validation, test.
* preprocess.py, preprocess.yaml are python file and yaml file copied from the files used while preprocessing.
* result.log is a log file while preprocessing.
* sequence & mask figures visualized figures of gas diffusion sequence and corresponding mask.

### data_integrate.py

#### 1. command : 
```shell script
python data_integrate.py
```

#### 2. Usage of data_integrate.py
##### 2.1. head_list : 
"head_list" is a list of base folders paths of which subfolder structures follow the description below :
Ex. head_list = ['head_1_path','head_2_path']

```
head_1_path                                     head_2_path
├── sub_head_1_path                             ├── sub_head_1_path
│   ├── train_dataset.pkl                       │   ├── train_dataset.pkl      
│   ├── val_dataset.pkl                         │   ├── val_dataset.pkl
│   ├── test_dataset.pkl                        │   ├── test_dataset.pkl
│   ├── preprocess.py                           │   ├── preprocess.py
│   ├── preprocess.yaml                         │   ├── preprocess.yaml        
│   ├── result.log                              │   ├── result.log
│   ...                                         │   ...
├── sub_head_2_path                             ├── sub_head_2_path
    ...                                             ...    
```

##### 2.2. save_head :
"save_head" is a path of integrated train_dataset.pkl, val_dataset.pkl, test_dataset.pkl.
Every (train, val, test) datasets included under "head"list" will be integrated into one (train, val, test) sets.

#### 3. output : 
train_dataset.pkl, val_dataset.pkl, test_dataset.pkl