""" functions for preprocessing KG completion datasets """

from collections import defaultdict
from copy import deepcopy

import numpy as np
from tqdm import tqdm


def get_count(triples):
    """ count number of each type of triple """
    count = defaultdict(lambda: 4)
    for i in tqdm(range(len(triples['head']))):
        head, relation, tail = triples['head'][i], triples['relation'][i], triples['tail'][i]
        head_type, tail_type = triples['head_type'][i], triples['tail_type'][i]
        count[(head, relation, head_type)] += 1
        count[(tail, -relation-1, tail_type)] += 1
    return count


def add_node_offsets(triples, entity_dict):
    """ add node type offsets to head and tail indices in triples """

    # calculate offsets based on head and tail node types
    head_type, tail_type = triples['head_type'], triples['tail_type']
    head_offsets = np.array([entity_dict[typ][0] for typ in head_type])
    tail_offsets = np.array([entity_dict[typ][0] for typ in tail_type])

    # copy triples into new dictionary
    return_triples = deepcopy(triples)

    # add offsets to head and tail (and potentially head_neg and tail_neg)
    return_triples['head'] = triples['head'] + head_offsets
    return_triples['tail'] = triples['tail'] + tail_offsets
    if 'head_neg' in return_triples:
        return_triples['head_neg'] = triples['head_neg'] + head_offsets[:,None]
    if 'tail_neg' in return_triples:
        return_triples['tail_neg'] = triples['tail_neg'] + tail_offsets[:,None]

    return return_triples


def calculate_valid_negatives(train, valid, test, entity_dict):
    """ for each triple in the validation/test set, calculate the valid set
        of corrupted nodes that can be ranked against when replacing both the
        head and tail entities, filtering out valid positive triples

        assumes that node type offsets have already been added """

    # construct set of all valid positive edges
    all_edges = set()
    for triples in [train, valid, test]:
        edges = list(zip(triples['head'], triples['relation'], triples['tail']))
        all_edges.update(edges)

    # construct dictionary of all positive edges, in both directions
    edge_dict = defaultdict(list)
    for h, r, t in tqdm(list(all_edges), desc='edge dictionary', ncols=100):
        edge_dict[(h, r)].append(t)
        edge_dict[(t, -r-1)].append(h)

    # initialize dictionary of valid negatives
    negatives = {
        'valid' : {'head' : list(), 'tail' : list()},
        'test'  : {'head' : list(), 'tail' : list()}
    }

    # find set of valid negatives for every positive validation/test triple
    for subset, triples in [('valid', valid), ('test', test)]:
        edges = list(zip(triples['head'], triples['relation'], triples['tail']))
        for i, (h, r, t) in enumerate(tqdm(edges, desc=subset, ncols=100)):

            # replacing head entity
            low, high = entity_dict[triples['head_type'][i]]
            nodes = np.arange(low, high)
            nodes = nodes[nodes != t]   # remove tail entity itself
            # remove all head nodes that are already paired with tail node
            nodes = nodes[np.logical_not(np.in1d(nodes, edge_dict[(t, -r-1)]))]
            negatives[subset]['head'].append(nodes)

            # replacing tail entity
            low, high = entity_dict[triples['tail_type'][i]]
            nodes = np.arange(low, high)
            nodes = nodes[nodes != h]   # remove head entity itself
            # remove all tail nodes that are already paired with head node
            nodes = nodes[np.logical_not(np.in1d(nodes, edge_dict[(h, r)]))]
            negatives[subset]['tail'].append(nodes)

    return negatives

