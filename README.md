# LMs for biomedical KG completion

This repository contains code to run the experiments described in:

[**Scientific Language Models for Biomedical Knowledge Base Completion: An Empirical Study**](https://arxiv.org/abs/2106.09700)<br>
Rahul Nadkarni, David Wadden, Iz Beltagy, Noah A. Smith, Hannaneh Hajishirzi, Tom Hope<br>
_Automated Knowledge Base Construction (AKBC) 2021_

## Environment

The `environment.yml` file contains all of the necessary packages to use this code. We recommend using Anaconda/Miniconda to set up an environment, which you can do with the command

```
conda-env create -f environment.yml
```

## Data

The edge splits we used for our experiments can be downloaded using the following links:

| Link                                                                                                                 | File size |
|----------------------------------------------------------------------------------------------------------------------|:---------:|
| [RepoDB, transductive split](https://lm-bio-kgc.s3.us-west-2.amazonaws.com/repodb-edge-split-f0.2-neg500-s42.pt)     | 11 MB     |
| [RepoDB, inductive split](https://lm-bio-kgc.s3.us-west-2.amazonaws.com/repodb-edge-split-f0.2-neg500-ind-s42.pt)    | 11 MB     |
| [Hetionet, transductive split](https://lm-bio-kgc.s3.us-west-2.amazonaws.com/hetionet-edge-split-f0.2-neg80-s42.pt)  | 49 MB     |
| [Hetionet, inductive split](https://lm-bio-kgc.s3.us-west-2.amazonaws.com/hetionet-edge-split-f0.2-neg80-ind-s42.pt) | 49 MB     |
| [MSI, transductive split](https://lm-bio-kgc.s3.us-west-2.amazonaws.com/msi-edge-split-f0.2-neg500-s42.pt)           | 813 MB    |
| [MSI, inductive split](https://lm-bio-kgc.s3.us-west-2.amazonaws.com/msi-edge-split-f0.2-neg500-ind-s42.pt)          | 813 MB    |

Each of these filees should be placed in the `subgraph` directory before running any of the experiment scripts. Please see the `README.md` file in the `subgraph` directory for more information on the edge split files. If you would like to recreate the edge splits yourself or construct new edge splits, use the scripts titled `script/create_*_dataset.py`.

## Models

The links below can be used to download a selection of the trained models. We provide the model parameters for KG-PubMedBERT fine-tuned with the multi-task loss on each dataset.

| Link | File size |
|---|:-:|
| [RepoDB](https://lm-bio-kgc.s3.us-west-2.amazonaws.com/kg-pubmedbert-multitask-repodb.pt) | 418 MB |
| [Hetionet](https://lm-bio-kgc.s3.us-west-2.amazonaws.com/kg-pubmedbert-multitask-hetionet.pt) | 418 MB |
| [MSI](https://lm-bio-kgc.s3.us-west-2.amazonaws.com/kg-pubmedbert-multitask-msi.pt) | 418 MB |

Once downloaded, the model can be loaded from the `src/lm` directory using the following code:

```Python
import torch
from argparse import Namespace
from model import KGBERT

dataset = # specify the dataset name here
nrelations = {'repodb' : 1, 'hetionet' : 4, 'msi' : 6}

model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
args = Namespace(model_name=model_name,
                 link_prediction=True,
                 relation_prediction=True,
                 nrelation=nrelations[dataset],
                 relevance_ranking=True)
model = KGBERT(args)

state_dict = torch.load(f'kg-pubmedbert-multitask-{dataset}.pt')
model.load_state_dict(state_dict)
```

## Entity names and descriptions

The files that contain entity names and descriptions for all of the datasets can be found in `data/processed` directory. If you would like to recreate these files yourself, you will need to use the scripts for each dataset found in `data/script`.

## Pre-tokenization

The main training script for the LMs `src/lm/run.py` can take in pre-tokenized entity names and descriptions as input, and several of the training scripts in `script/training` are set up to do so. If you would like to pre-tokenize text before fine-tuning, follow the instructions in `script/pretokenize.py`. You can also pass in one of the `.tsv` files found in `data/processed` for the argument `--info_filename` instead of a file with pre-tokenized text.

## Training

All of the scripts for training models can be found in the `src` directory. The script for training all KGE models is `src/kge/run.py`, while the script for training LMs is `src/lm/run.py`. Our code for training KGE models is heavily based on [this code](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/biokg) from the Open Graph Benchmark Github repository. Examples of how to use each of these scripts, including training with Slurm, can be found in the `script/training` directory. This directory includes all of the scripts we used to run the experiments for the results in the paper.
