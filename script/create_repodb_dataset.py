""" script to format repoDB dataset into edge split for KG completion with
    transductive or inductive training/validation/test splits """

from argparse import ArgumentParser
from collections import defaultdict, Counter
from datetime import datetime
import os
import random
import sys

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange

from preprocess import calculate_valid_negatives
from dataset_utils import transductive_edge_split, inductive_edge_split


parser = ArgumentParser(description='format repoDB dataset as dataset for KG '
                                    'completion with transductive training/'
                                    'validation/test splits')
parser.add_argument('data_file', type=str,
                    help='data file containing repoDB drug/disease pairs')
parser.add_argument('--info-file', type=str,
                    default='data/processed/repodb.tsv',
                    help='location of info file for repoDB entities')
parser.add_argument('--test-fraction', type=float, default=0.2,
                    help='fraction of edges to use for validation/test set')
parser.add_argument('--outdir', type=str, default='./subgraph',
                    help='output directory to save edge split dictionary')
parser.add_argument('--num-negatives', type=int, default=None,
                    help='add negative examples to validation/test sets, '
                         'using specified number of negatives per positive')
parser.add_argument('--inductive', action='store_true',
                    help='use inductive instead of transductive edge split')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (for train/test splits)')


if __name__ == '__main__':

    args = parser.parse_args()

    # look for precomputed edge split
    savename = f'repodb-edge-split-f{args.test_fraction:.1f}'
    if args.num_negatives is not None:
        savename += f'-neg{args.num_negatives}'
    if args.inductive:
        savename += '-ind'
    savename += f'-s{args.seed}.pt'
    if os.path.exists(os.path.join(args.outdir, savename)):
        print('found saved edge split, exiting...')
        sys.exit()
    os.makedirs(args.outdir, exist_ok=True)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load info file and data file
    info = pd.read_table(args.info_file, index_col=0, na_filter=False)
    assert info.index.tolist() == sorted(info.index), 'info file is not sorted'
    ids = info.id.tolist()
    node_types = sorted(['drug', 'disease'])
    data = pd.read_csv(args.data_file, na_filter=False)
    data = data[data.status == 'Approved']

    # process each file
    print('collecting edges...')
    edges = list()
    for i, row in tqdm(data.iterrows(), total=len(data), desc='processing',
                       ncols=100):
        head_idx = ids.index(str(row['drug_id']))
        tail_idx = ids.index(str(row['ind_id']))
        edges.append((head_idx, tail_idx))

    # construct set of edges labelled with index in original list
    labelled_edges = list()
    for i, edge in enumerate(edges):
        labels = {'index' : i, 'rel' : 0}
        labelled_edges.append((edge[0], edge[1], labels))

    # construct graph from edges, create edge splits
    split_type = 'inductive' if args.inductive else 'transductive'
    print(f'calculating {split_type} training/validation/test edge split...')
    graph = nx.DiGraph()
    graph.add_edges_from(labelled_edges)
    if args.inductive:
        train, train_idx, valid, valid_idx, test, test_idx = \
            inductive_edge_split(graph, test_frac=args.test_fraction,
                                 verbose=True)
    else:
        train, train_idx, valid, valid_idx, test, test_idx = \
            transductive_edge_split(graph, test_frac=args.test_fraction,
                                    verbose=True)

    # convert everything to numpy ndarrays
    train = np.array(train)
    valid = np.array(valid)
    test = np.array(test)

    # construct entity_dict
    entity_dict = dict()
    for node_type in node_types:
        indices = info[info.ent_type == node_type].index.tolist()
        # indices should be low (inclusive) and high (exclusive)
        entity_dict[node_type] = (np.min(indices), np.max(indices) + 1)

    # construct edge split dictionary
    print('constructing edge split dictionary and saving to file...')
    split_edge = {
        'num_nodes' : len(info),
        'entity_dict' : entity_dict,
        'train' : {
            'head' : train[:, 0], 'tail' : train[:, 1],
            'relation' : np.zeros(len(train)).astype(int),
            'head_type' : ['drug'] * len(train),
            'tail_type' : ['disease'] * len(train)
         },
        'valid' : {
            'head' : valid[:, 0], 'tail' : valid[:, 1],
            'relation' : np.zeros(len(valid)).astype(int),
            'head_type' : ['drug'] * len(valid),
            'tail_type' : ['disease'] * len(valid)
         },
        'test' : {
            'head' : test[:, 0], 'tail' : test[:, 1],
            'relation' : np.zeros(len(test)).astype(int),
            'head_type' : ['drug'] * len(test),
            'tail_type' : ['disease'] * len(test)
         },
    }

    # if number of validation/test negatives specified, add negatives
    # to validation/test sets of triples
    if args.num_negatives is not None:
        train, valid, test = \
            split_edge['train'], split_edge['valid'], split_edge['test']
        negatives = calculate_valid_negatives(train, valid, test, entity_dict)
        for subset in ('valid', 'test'):
            for key in ('head', 'tail'):
                negative_examples = list()
                for i, elem in enumerate(negatives[subset][key]):
                    idx = np.random.choice(len(elem), size=args.num_negatives,
                                           replace=False)
                    negative_examples.append(elem[idx])
                split_edge[subset][f'{key}_neg'] = np.array(negative_examples)

    # remove node offsets
    for subset in ('train', 'valid', 'test'):
        for key in ('head', 'tail'):
            node_types = split_edge[subset][f'{key}_type']
            offsets = np.array([entity_dict[typ][0] for typ in node_types])
            split_edge[subset][key] -= offsets
            if f'{key}_neg' in split_edge[subset]:
                split_edge[subset][f'{key}_neg'] -= offsets[:,None]

    torch.save(split_edge, os.path.join(args.outdir, savename))

