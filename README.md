# recommender
A collection of recommender systems

### Installation:
```
git clone git@github.com:d704e18/recommender.git
cd recommender
pip install -r requirements.txt
```
To automatically download the movie dataset from kaggle you need a kaggle api token.
Instructions for installing the kaggle API token can be found here: https://github.com/Kaggle/kaggle-api#api-credentials
You may also download the dataset manually from https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7/.

## Usage
Before you can run the algorithms you need to load a dataset. It is recommended to load `ml-1m` with `recommender load_datset ml-1m`, and run algorithms with it: `recommender alg mfnonvec ml-1m`. 

### Commands:

General usage:
```
usage: recommender [-h] [--config CONFIG] {load_dataset,alg} ...

positional arguments:
  {load_dataset,alg}  sub-command help
    load_dataset      load dataset into database
    alg               run recommender algorithms

optional arguments:
  -h, --help          show this help message and exit
  --config CONFIG     configurations to use
```

Loading datasets:
```
usage: recommender load_dataset [-h] {ml-26m,ml-1m,ml-100k}

positional arguments:
  {ml-26m,ml-1m,ml-100k}
                        the dataset choices

optional arguments:
  -h, --help            show this help message and exit
```
Running algorithms:
```
usage: recommender alg [-h] {mfnonvec} {ml-26m,ml-1m,ml-100k}

positional arguments:
  {mfnonvec}            the recommender algorithm to run
  {ml-26m,ml-1m,ml-100k}
                        the dataset to use

optional arguments:
  -h, --help            show this help message and exit
  ```
