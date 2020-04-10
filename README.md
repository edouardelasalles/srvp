# Stochastic Latent Residual Video Prediction (SRVP)

Official implementation of the paper Stochastic Latent Residual Video Prediction (Jean-Yves Franceschi*, Edouard Delasalles*, Mickael Chen, Sylvain Lamprier, Patrick Gallinari).

## [Preprint](https://arxiv.org/abs/2002.09219)

## [Project Website](https://sites.google.com/view/srvp/)

## [Pretrained Models](https://data.lip6.fr/srvp/)

## Requirements

All models were trained with Python 3.7.6 and CUDA 10.1 on PyTorch 1.4.0.

A list of requirements of Python dependencies is available in the `requirements.txt` file.

To speed-up training, we recommend the use of [Apex](https://nvidia.github.io/apex/) (v0.1) in mixed-precision mode (`O1`), whose performance gains were tested on the most recent Nvidia GPU architectures (starting from Volta).


## Datasets

### Stochastic Moving MNIST

During training, this dataset is generated on the fly.
In order to generate a consistent test in an `.npz` file, the following commands should be executed:
```bash
python -m preprocessing.mmnist.make_test_set --data_dir $DIR --seq_len 100
```
for the stochastic version of the dataset, or
```bash
python -m preprocessing.mmnist.make_test_set --data_dir $DIR --seq_len 25
```
for the deterministic version, where `$DIR` is the directory where the testing set should be saved.

### KTH

To download the dataset at a given path `$DIR`, execute the following command:
```bash
bash preprocessing/kth/download.sh $DIR
```
(see also [https://github.com/edenton/svg/blob/master/data/download_kth.sh](https://github.com/edenton/svg/blob/master/data/download_kth.sh)).

In order to respectively train and test a model on this dataset, the following commands should be run:
```bash
python preprocessing/human/convert.py --data_dir $DIR
```
and
```bash
python preprocessing/kth/make_test_set.py --data_dir $DIR
```

### Human3.6M

This dataset can be downloaded at [http://vision.imar.ro/human3.6m/description.php](http://vision.imar.ro/human3.6m/description.php), after obtaining the access from its owners.
Videos for every subject are included in `.tgz` archives. Each of these archives should be extracted in the same folder.

To preprocess the dataset in order to use it for training and testing, these videos should be processed using the following command:
```bash
python preprocessing/human/convert.py --data_dir $DIR
```
where `$DIR` is the directory where Human3.6M videos are saved.

Finally, the testing set is created by choosing extracts from testing videos, with the following command:
```bash
python preprocessing/human/make_test_set.py --data_dir $DIR
```

All processed videos are saved in the same folder as the original dataset.

### BAIR

To download the dataset at a given path `$DIR`, execute the following command:
```bash
bash preprocessing/bair/download.sh $DIR
```
(see also [https://github.com/edenton/svg/blob/master/data/download_bair.sh](https://github.com/edenton/svg/blob/master/data/download_bair.sh)).

In order to respectively train and test a model on this dataset, the following command should be run:
```bash
python preprocessing/bair/convert.py --data_dir $DIR
```


## Training

Please refer to the help message of `train.py`:
```bash
python train.py --help
```
which lists all options and hyperparameters to train SRVP models.

In order to use multiple processes for data loading, proceed as follows:
```
OMP_NUM_THREADS=$NUMWORKERS python train.py --num_workers $NUMWORKERS
```
followed by the training options, where $NUMWORKERS is the number of processes to use.

In order to launch training on multiple GPUs, launch the following command:
```
python -m torch.distributed.launch --nproc_per_node=$NBDEVICES train.py
```
followed by the training options, where `$NBDEVICES` is the number of GPUs to be used.



## Testing

To evaluate a trained model, the script `test.py` should be used as follows:

```bash
python test.py --data_dir $DATADIR --xp_dir $XPDIR --lpips_dir $LPIPSDIR
```

where `$XPDIR` is a directory containing a checkpoint and the corresponding `json` configuration file (see the pretrained models for an example), `$DATADIR` is the directory containing the test set, and `$LPIPSDIR` is a directory where [LPIPS weights](https://github.com/richzhang/PerceptualSimilarity/tree/master/models/weights) are downloaded.

To run the evaluation on GPU, use the option `--device $DEVICE`.

Please also refer to the help message of `test.py`:
```bash
python test.py --help
```
