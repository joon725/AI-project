## Gas Source Localization code utilization

### Contents :
1. model training(train.py)
2. model tuning(tune.py)
3. model predict(predict.py)
4. dataset generation from gas diffusion simulation(diffusion.py)
* For NBC_RAMS data preprocessing, please refer to the README.md in ./preprocess folder.

### 1. train.py

#### 1. command : python train.py --config ./cfg/train.yaml 
#### 2. configuration file location - ./cfg/train.yaml

##### ## details about train.yaml ##
##### 2.1. environment setting
2.1.1. "random_state" is random seed setting for randomization.

2.1.2. "device" is a selection of a device to use for training. You can choose your GPU number(ex. "0", "1") or 'cpu' to use CPU.                

##### 2.2. dataset
2.2.1. "train_dataset_path", "val_dataset_path", "test_dataset_path" are paths of your train_dataset, val_dataset, test_dataset.

2.2.2. "interval" is an interval length of each grids. It is used to calculate regression loss.

2.2.3. "sequence" is a length of a time-series data.

##### 2.3. training setting
2.3.1. "weight_decay" is a hyperparameter for L2 Regularization. The regularization is used to prevent overfitting.

2.3.2. "lr" is a learning rate of your training.

2.3.3. "batch_size" is a batch size of input data.

2.3.4. "shuffle" is used to shuffle dataset when loading the dataset. When it is set to True, the argument is conveyed to dataloader to shuffle a dataset.

2.3.5. "worker" is a number of worker to use (CPU core number). 

2.3.6. "epochs" is a number of epoch you are training.

###### 2.5. model
2.5.1. "model_name" is a name of a model you're going to use for training. You can choose "cnn-lstm" or "vivit" or "3d-cnn".

2.5.2. "hidden_dim" is 
* cnn-lstm : dimension of a linear layer in cnn layer and two linear layers in dnn layer.
* 3d-cnn : dimension of three linear layers in dnn layer
* vivit : not used

2.5.3. "drop_rate" is a drop rate of your model layers while training.

2.5.4. "num_layers" is a number of Conv3d layers in 3DCNN model.

##### 2.6. vivit
2.6.1. "tubelet_size" is a size of each tubelet in vivit.

2.6.2. "hidden_size" is a dimensionality of the encoder layers and the pooler layer.

2.6.3. "num_hidden_layers" is a number of hidden layers in the Transformer encoder.

2.6.4. "num_attention_heads" is a number of attention heads for each attention layer in the Transformer encoder.

2.6.5. "intermediate_size" is a dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

2.6.6. "hidden_dropout_prob" is a dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

2.6.7. "attention_probs_dropout_prob" is a dropout ratio for the attention probabilites.

##### 2.7. visualization
2.7.1. "visualization" is whether or not to save samples. If it is set to True, prediction results using sigmoid, softmax and gas diffusion samples will be saved.

2.7.2. "num_test" is the number of test sample to visualize.

##### 2.8. save setting
2.8.1. "save_dir" is a parent directory of the folder where the results are saved

2.8.2. "name" is a folder name where the dataset is saved
final directory where the dataset is saved : save_dir/name/

#### 3. output : best.pt, last.pt, log.pkl, loss.py, model.py result.csv, result.log, train.py, train.yaml, training_plot.png, figures of samples(optional)


### 2. tune.py

#### 1. command : python tune.py --config ./cfg/tune.yaml 
#### 2. configuration file location - ./cfg/tune.yaml

##### ############# details about tune.yaml #############
##### 2.1. random_state
2.1.1. "random_state" is random seed setting for randomization.

2.1.2. "device" is a selection of a device to use for training. You can choose your GPU number(ex. "0", "1") or 'cpu' to use CPU.  

2.1.3. "worker" is a number of worker to use (CPU core number).

2.1.4. "gpu_per_trials" is a number of GPU usage per trial. For example, if it is set to 0.5, 0.5 GPU is used for each trial, which means two trials will be utilized simultaneously.

2.1.5. "ray_verbose" is a verbosity mode of tuning. 0 = silent, 1 = default, 2 = verbose. Defaults to 1.

##### 2.2. dataset
2.3.1. "train_dataset_path", "val_dataset_path", "test_dataset_path" are paths of your train_dataset, val_dataset, test_dataset.

2.3.2. "interval" is an interval length of each grids. It is used to calculate regression loss.

2.3.3. "sequence" is a length of your time-series data.

##### 2.3. training setting
2.3.1. "batch_size" is a batch size of input data.

2.3.2. "shuffle" is used to shuffle dataset when loading the dataset. When it is set to True, the argument is conveyed to dataloader to shuffle a dataset.

##### 2.4. model
2.4.1. "model_name" is a name of a model you're going to use for training. You can choose "cnn-lstm" or "vivit" or "3d-cnn".

##### 2.5. Tuning setting
2.5.1. "epoch" is a number of max epoch for each training while tuning. This enables early stopping of bad trials.

2.5.2. "num_samples" is a number of times to sample from the hyperparameter space. Defaults to 1. 

##### 2.6. visualization
2.6.1. "visualization" is whether or not to save samples. If it is set to True, prediction results using sigmoid, softmax and gas diffusion samples will be saved.

2.6.2. "num_test" is the number of test sample to visualize.

##### 2.7. save setting
2.7.1. "save_dir" is a parent directory of the folder where the results are saved.

2.7.2. "name" is a folder name where the dataset is saved.
final directory where the dataset is saved : save_dir/name/

#### 3. output : the outputs are described below.
save_dir/name                                  
├── loss.py 
├── model.py
├── result.log
├── train.py
├── tune.py
├── tune.yaml
├── tune_result.pkl   
├── ray_train_{yyyy-mm-dd_hh-mm-ss}
    ├── experiment_state-{yyyy-mm-dd_hh-mm-ss}.json                         
    ├── basic-variant-state-{yyyy-mm-dd_hh-mm-ss}.json                         
    ├── (trial folders)                                                 
         
* To identify the hyperparameters for the best results, you should check result.log
- In result.log, there are "best score", "best hyperparameters", "best log dir" for four metrics, "val_loss", "val_accuracy", "test_loss", "test_accuracy".
- You can choose which metric to focus, and use the "best hyperparameters" and weights in "best log dir" for that metric. 


### 3. predict.py

#### 1. command : python predict.py --config ./cfg/predict.yaml --data_path {data_path} --save_dir {save_dir} --weight_path {weight_path} --name {name}

#### 2. argument details
##### 2.1. config
path of predict.yaml file.

##### 2.2. data_path
Path of test data. In the path, bgr files are located.

##### 2.3. save_dir
Path of save directory. Default is set to ./exp .

##### 2.4. weight_path
Path of .pt weight file. Currently cnn-lstm is the only utilizable model. Assign a weight file of cnn-lstm model.

##### 2.5. name
Name of an experiment folder in save directory. Default is set to "predict".
final directory where the predicted result is saved : save_dir/name/

#### 3. configuration file location - ./cfg/predict.yaml

##### ############# details about predict.yaml #############
##### 3.1. environment setting
3.1.1. "random_state" is random seed setting for randomization.

3.1.2. "device" is a selection of a device to use for training. You can choose your GPU number(ex. "0", "1") or 'cpu' to use CPU.  

##### 3.2. data setting
3.2.1. "region_info", "dx", "dy", "W", "H" are based on meta data generated from NBC_RAMS.

3.2.2. "normalize" is whether or not to implement min-max normalization for each frame. 

##### 3.3. model setting
3.3.1. "sequence" is a length of a time-series data.

3.3.2. "hidden_dim" is a dimension of a linear layer in cnn layer and two linear layers in dnn layer in cnn-lstm model. 
It must be same with the "hidden_dim" used while training the model.

3.3.3. "drop_rate" is a drop rate of your models. It is only used to load model, not activated while evaluating.

#### 4. output : predict.yaml, result.log(predicted longitude, latitude of gas source is saved)

### 4. diffusion.py

#### 1. command : python diffusion.py --config ./cfg/diffusion.yaml 
#### 2. configuration file location - ./cfg/diffusion.yaml

##### ############# details about diffusion.yaml #############
##### 2.1. random_state
"random_state" is random seed setting for randomization.

##### 2.2. Data
2.3.1. "c0" is a mass flow rate of gas[mg/s].

2.3.2. "D" is a diffusion coefficient of gas[m^2/s].

2.3.3. "T" is a period of gas diffusion[s].

2.3.4. "delta_T" is a sampling time interval[s].

2.3.5. "wind_v" is a range of wind velocity[m/s].

2.3.6. "wind_t" is a range of wind theta[degree].

2.3.7. "grid_res" is a unit length of one grid[m]. 

2.3.8. "H" is a number of grid in height.

2.3.9. "W" is a number of grid in width.

2.3.10. "noise" is a concentration measurement noise covariance.

2.3.11. "wind_v_n" is a velocity measurement noise cavariance[m/s].

2.3.12. "wind_t_n" is a theta measurement noise cavariance[degree].

2.3.13. "source_num" is a number of source.

2.3.14. "source_list" is an index of sources in grid cell. If you have specific grid cells to produce gas diffusion data, you can fill this list.

2.3.15. "source_random" is whether or not to randomized the source location within each grid. If it's set to True, then it is randomized within a grid. If it's False, the the source is set to the a center of a grid.

2.3.16. "visual_sample" is a number of sample to visualize.

##### 2.3. save setting
2.3.1. "save_dir" is a parent directory of the folder where the results are saved.

2.3.2. "name" is a folder name where the dataset is saved.
final directory where the dataset is saved : save_dir/name/

#### 3. output : dataset_{visual_sample}.pkl, diffusion.yaml, result.log, sample figures(optional)
