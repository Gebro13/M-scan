# M-scan
The implementation of M-scan algorithm

## Data preprocessing

The code data preprocessing is stored in process_xxx.py, where xxx is our two datasets aliccp and cloud.

For example:

```shell
python process_aliccp.py
```



## Training

After you've preprocessed all the data, you can do the training by the following command

```shell
python train.py -d aliccp -m m_scan
```

In this command, Aliccp is a dataset, and m-scan is our proposed model.

All the train hyperparameters are stored in configs/train_configs

All the model hyperparameters are stored in configs/model_configs

All the data information are stored in configs/data_configs