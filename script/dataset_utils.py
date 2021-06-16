""" utility functions to help with creating knowledge graph datasets """

from collections import Counter
from datetime import datetime

import networkx
import numpy as np


def transductive_edge_split(graph, test_frac=0.2, verbose=True):
    """ helper function that constructs a set of test edges from a graph,
        ensuring that all nodes in the test set also still exist in the
        pruned graph
        each edge is assumed to be labelled with an index, and indices are
        returned with each subset of edges """
    num_test = int(test_frac * graph.number_of_edges())
    num_in_test = 0
    test_edges = list()
    test_idx = list()
    last_printed = 0.05
    start = datetime.now()
    while num_in_test < num_test:
        edges = list(graph.edges())
        degree = dict(graph.degree())
        idx = np.random.randint(len(edges))
        h, t = edges[idx]
        if degree[h] > 1 and degree[t] > 1:
            test_edges.append((h, t))
            test_idx.append(graph[h][t]['index'])
            graph.remove_edge(h, t)
            num_in_test += 1
            if verbose and num_in_test / num_test > last_printed:
                print(f'constructed {100 * last_printed:.0f}% of test set '
                      f'in {datetime.now() - start}')
                last_printed += 0.05

    # construct training set of edges and indices with remaining edges
    train_edges = list(graph.edges())
    train_idx = [graph[h][t]['index'] for h, t in train_edges]

    # split test set into validation and test
    idx = int(num_test // 2)
    valid_edges, test_edges = test_edges[:idx], test_edges[idx:]
    valid_idx, test_idx = test_idx[:idx], test_idx[idx:]

    # sort edge sets by their indices
    train = [train_edges[i] for i in np.argsort(train_idx)]
    train_idx = sorted(train_idx)
    valid = [valid_edges[i] for i in np.argsort(valid_idx)]
    valid_idx = sorted(valid_idx)
    test = [test_edges[i] for i in np.argsort(test_idx)]
    test_idx = sorted(test_idx)

    return train, train_idx, valid, valid_idx, test, test_idx


def inductive_edge_split(graph, test_frac=0.2, verbose=True):
    """ helper function that constructs a set of test edges from a graph,
        trying to select a set of nodes for the test set that do not exist
        in the pruned training graph
        each edge is assumed to be labelled with an index, and indices are
        returned with each subset of edges """
    num_test = int(test_frac * graph.number_of_edges())
    num_in_test = 0
    edges = list(graph.edges())
    indices = [graph[h][t]['index'] for h, t in edges]
    relations = [graph[h][t]['rel'] for h, t in edges]
    relation_counts = Counter(relations)
    test_nodes = list()
    test_edges = list()
    test_indices = list()
    last_printed = 0.05
    start = datetime.now()
    while num_in_test < num_test:
        # select a random node, get its edges and neighbors
        node = np.random.choice(graph.nodes)
        node_edges = list(graph.in_edges(node)) + list(graph.out_edges(node))
        neighbors = list(graph.neighbors(node)) + \
                    list(graph.predecessors(node))

        # get degree of neighbors and counts of relations of incident edges
        degrees = [graph.degree(n) for n in neighbors]
        node_relations = [graph[h][t]['rel'] for h, t in node_edges]
        counts = Counter(node_relations)

        # if removing this node would leave any of its neighbors with no edges
        # or any relation with fewer than 100 edges, do not remove it
        new_counts = {key : relation_counts[key] - counts[key]
                      for key in relation_counts}
        if np.min(degrees) == 1 or np.min(list(new_counts.values())) < 100:
            continue
        else:   # add to test nodes and edges, update relation counts
            test_indices.extend([graph[h][t]['index'] for h, t in node_edges])
            test_nodes.append(node)
            test_edges.extend(node_edges)
            relation_counts = new_counts
            graph.remove_node(node)
            num_in_test += len(node_edges)

        # print partial progress
        if verbose and num_in_test / num_test > last_printed:
            print(f'constructed {100 * last_printed:.0f}% of test set '
                  f'in {datetime.now() - start}')
            last_printed += 0.05

    # get training edges and indices from remaining edges in graph
    train_edges = list(graph.edges())
    train_idx = [graph[h][t]['index'] for h, t in train_edges]

    # split test edges into validation and test sets
    idx = int(num_in_test // 2)
    valid_idx, test_idx = test_indices[:idx], test_indices[idx:]
    valid_edges, test_edges = test_edges[:idx], test_edges[idx:]

    # sort edge sets by their indices
    train = [train_edges[i] for i in np.argsort(train_idx)]
    train_idx = sorted(train_idx)
    valid = [valid_edges[i] for i in np.argsort(valid_idx)]
    valid_idx = sorted(valid_idx)
    test = [test_edges[i] for i in np.argsort(test_idx)]
    test_idx = sorted(test_idx)

    return train, train_idx, valid, valid_idx, test, test_idx

