""" script to format hetionet dataset into edge split for KG completion with
    transductive or inductive training/validation/test splits """

from argparse import ArgumentParser
from collections import defaultdict, Counter
from datetime import datetime
import json
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


parser = ArgumentParser(description='format hetionet dataset as dataset for '
                                    'KG completion with transductive or '
                                    'inductive training/validation/test '
                                    'splits')
parser.add_argument('data_file', type=str,
                    help='data file containing hetionet knowledge graph edges')
parser.add_argument('--info-file', type=str,
                    default='data/processed/hetionet.tsv',
                    help='location of info file for hetionet entities')
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
    savename = f'hetionet-edge-split-f{args.test_fraction:.1f}'
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

    # load and check info file
    print('loading entity info file...')
    info = pd.read_table(args.info_file, index_col=0, na_filter=False)
    assert info.index.tolist() == sorted(info.index), 'info file is not sorted'
    ids = info.id.tolist()
    node_types = sorted(set(info.type.tolist()))

    # load data file, extract edges restricted to certain relation types
    print('loading dataset...')
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    rel_types = {'treats' : 0, 'presents' : 1, 'associates' : 2, 'causes' : 3}
    entries = list(filter(lambda x: x['kind'] in rel_types, data['edges']))

    # process each file
    print('collecting edges, node types, and relation types...')
    head_type, tail_type = list(), list()
    edges = list()
    relations = list()
    for entry in tqdm(entries, desc='processing edges', ncols=100):
        head_type_, head_id = entry['source_id']
        tail_type_, tail_id = entry['target_id']
        head_idx = info[info.id == str(head_id)].index.item()
        tail_idx = info[info.id == str(tail_id)].index.item()
        edges.append((head_idx, tail_idx))
        relations.append(rel_types[entry['kind']])
        head_type.append(head_type_.lower().replace(' ', ''))
        tail_type.append(tail_type_.lower().replace(' ', ''))

    # construct set of edges labelled with index in original list
    labelled_edges = list()
    for i, edge in enumerate(edges):
        labels = {'index' : i, 'rel' : relations[i]}
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
    relations = np.array(relations)
    head_type = np.array(head_type)
    tail_type = np.array(tail_type)

    # construct entity_dict
    entity_dict = dict()
    for node_type in node_types:
        indices = info[info.type == node_type].index.tolist()
        # indices should be low (inclusive) and high (exclusive)
        entity_dict[node_type] = (np.min(indices), np.max(indices) + 1)

    # construct edge split dictionary
    print('constructing edge split dictionary and saving to file...')
    split_edge = {
        'num_nodes' : len(info),
        'entity_dict' : entity_dict,
        'train' : {
            'head' : train[:, 0], 'tail' : train[:, 1],
            'relation' : relations[train_idx],
            'head_type' : head_type[train_idx].tolist(),
            'tail_type' : tail_type[train_idx].tolist()
         },
        'valid' : {
            'head' : valid[:, 0], 'tail' : valid[:, 1],
            'relation' : relations[valid_idx],
            'head_type' : head_type[valid_idx].tolist(),
            'tail_type' : tail_type[valid_idx].tolist()
         },
        'test' : {
            'head' : test[:, 0], 'tail' : test[:, 1],
            'relation' : relations[test_idx],
            'head_type' : head_type[test_idx].tolist(),
            'tail_type' : tail_type[test_idx].tolist()
         }
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

