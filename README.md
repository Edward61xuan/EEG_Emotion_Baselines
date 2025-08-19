# EEG Emotion Recognition Baselines
This repo provides typical baselines for EEG emotion recognition, including: EEGNet,EEGConformer and DGCNN(Deep Graph Convolutional Neural Networks). The implementation of models is based on Pytorch.
The preprocess of EEGData is **Not Included** in this Repo. You **must download the datasets and organize it in the expected form**, which will be introduced in **'3.Data Preparation'** part in item **Usage**. 

## File Structure
```
├── dataset : Dataset library
│   ├── dummy_eeg_dataset
│   ├── ...(your own dataseet)
├── Models : Model library
│   ├── EEGNet.py
│   ├── EEGConformer.py
│   ├── DGCNN.py
├── run_models.py : Main Script for Training and Evaluating Models
└── README.md
```
## Requirements
A simple python enviroment with Pytorch and Numpy is enough.
If you want to log your runs with wandb, you need to install wandb.

## Usage
1. Clone this repository:
    ```bash
    git clone https://github.com/Edward61xuan/EEG_Emotion_Baselines
    cd EEG_Emotion_Baselines
    ```
2. Install dependencies
3. **Data Preparation**: The Dataset Needs to be organized in the following form: (**The file names are important, must be the same as the demonstration.**)
    ```
    dataset
    ├── your_dataset_name
    │   ├── X.npy
    │   ├── y.npy
    │   ├── subject_ids.npy
    │   ├── (trial_ids.npy)
    │   ├── (session_ids.npy)
    ```
- X.npy(np.ndarray): shape [N, n_channels, n_samples (or n_features)], EEG data. The last dimension could either be 'n_samples', which means the input feature of the classification models is the temporal information of the EEG signal, or 'n_features', which means the input feature of the models are features extracted from the EEG signal.

- y.npy(np.ndarray): shape [N], label of each sample
- subject_ids.npy(np.ndarray): shape [N], subject id of each sample
- (trial_ids.npy)(np.ndarray): shape [N], trial id of each sample
- (session_ids.npy)(np.ndarray): shape [N], session id of each sample
Note that **trail_ids and session_ids are optional.** If those info are useless, you can **modify the function 'load_dataset' in 'run_models.py' and the citation of this function in the script**.

4. Modify the dataset args in 'run_models.py':
After customizing the dataset, you need to modify the dataset args in 'run_models.py':
```python
    parser.add_argument('--dataset', type=str, default='dummy', help='dataset name')
```
Change this line to your own dataset name. Then add branches in function 'load_dataset':
```python
if dataset_name == "dummy": 
        ds_path = os.path.join(data_root, 'dummy_eeg_data')
elif dataset_name == "your_dataset_name":
        ds_path = os.path.join(data_root, 'your_dataset_name')
```
to load your own dataset, which is similar to the dummy dataset. 

5. Modify the roots and run:
Finally, modify the roots in 'run_models.py' and run the script.
If there is any question about the arguments, please refer to the description or helps in the definition of those arguments.