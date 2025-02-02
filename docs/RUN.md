## How to Generate Description & Structure Data

We have provided description and structure data for categories in every dataset, saved in JSON files under the folder `$DATA/gpt_data/description` and `$DATA/gpt_data/structure`. 

If you want to generate the data by yourself, you only need to run `./data_generation/description.py` for generating descriptions following `./data_generation/structure.py` for generating structures for the dataset you select. To use the ChatGPT's API, you must input the key associated with your ChatGPT account in the above two Python files and fill in the name of the dataset you want to generate descriptions for.

## How to Run

We provide the running scripts in `scripts/hpt`, which allow you to reproduce the results on our paper.

Change the path of `$DATA` and `$OUTPUT_DIR` in bash files (if not default), and run the following commands under the main directory.

### Base-to-New Generalization

All you need is `./scripts/hpt/b2n_train.sh` for training and `./scripts/hpt/b2n_test.sh` for testing.

`$DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `./configs/datasets/b2n`.

Below we provide examples of how to run HPT on base-to-new generalization.

#### **Train and Test on Base Classes**:

- e.g. Caltech101: `bash ./scripts/hpt/b2n_train.sh caltech101`

#### **Test on New Classes**:

- e.g. Caltech101: `bash ./scripts/hpt/b2n_test.sh caltech101`

We also provide  `./scripts/hpt/b2n_all.sh` for conducting training and testing on all 11 datasets at once for convenience.

### Domain Generalization & Cross-dataset Evaluation
All you need is `./scripts/hpt/xd_train.sh` for training, `./scripts/hpt/xd_test_dg.sh` for testing of domain generalization, and `./scripts/hpt/xd_test_cde.sh` for testing of cross-dataset evaluation.

`$DATASET` takes as input a dataset name (only for testing), like `imagenet_a` for domain generalization or `caltech` for cross-dataset evaluation. The valid names are the files' names in `./configs/datasets/xd` for domain generalization and `./configs/datasets/b2n` for cross-dataset evaluation.

Below we provide examples of how to run HPT on domain generalization & cross-dataset evaluation.

#### **Training on ImageNet**:

`bash ./scripts/hpt/xd_train.sh`

#### **Test on New Dataset for Domain Generalization**:

- e.g. ImageNet-A: `bash ./scripts/hpt/xd_test_dg.sh imagenet_a`

#### **Test on New Dataset for Cross-dataset Evaluation**:

- e.g. Caltech101: `bash ./scripts/hpt/xd_test_cde.sh caltech101`

We also provide  `./scripts/hpt/xd_all.sh` for conducting training and testing of domain generalization and cross-dataset evaluation all at once for convenience.
