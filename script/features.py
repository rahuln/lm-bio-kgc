""" functions that compute various features for positive triples in a KG
    completion dataset, used for input-dependent ensembling methods """

import networkx as nx
from nltk import edit_distance
import numpy as np
from scipy.sparse import csr_matrix
import torch


def compute_features(triples, info_file, num_nodes, tokens_fname=None,
    add_node_type=True, add_relation_type=True):
    """ function that computes various features for a set of examples
        and returns them as a design matrix """

    # convert head, relation, and tail to numpy arrays if necessary
    head = torch_to_numpy(triples['head'])
    relation = torch_to_numpy(triples['relation'])
    tail = torch_to_numpy(triples['tail'])

    head_type = triples['head_type']
    tail_type = triples['tail_type']

    names = info_file['name'].tolist()
    descriptions = info_file['description'].tolist()

    # calculate features
    degree_features = get_degree_features(head, tail, num_nodes)
    edit_distance_features = get_edit_distance_features(head, tail, names)
    length_features = get_length_features(head, tail, names)
    character_features = get_character_features(head, tail, names)
    unknown_features = get_unknown_substring_features(head, tail, names)
    pagerank_features = get_pagerank_features(head, tail)
    adamic_adar_features = get_adamic_adar_features(head, tail)
    description_features = get_description_features(head, tail, descriptions)

    # concatenate all features
    features = [degree_features,
                edit_distance_features,
                length_features,
                character_features,
                unknown_features,
                pagerank_features,
                adamic_adar_features,
                description_features]

    # add node type and relation type features, if specified
    if add_node_type:
        features.append(get_node_type_features(head_type, tail_type))
    if add_relation_type:
        features.append(get_relation_features(relation))

    # calculate and append token ratio features, if token filename provided
    if tokens_fname is not None:
        tokens = torch.load(tokens_fname)
        features.append(get_tokens_features(head, tail, tokens, names))

    # convert list of features to numpy array and return
    return np.concatenate(features, axis=1)


def torch_to_numpy(arr):
    """ convert PyTorch tensor to numpy array """
    if torch.is_tensor(arr):
        if arr.device.type == 'cuda':
            arr = arr.cpu()
        return arr.numpy()
    return arr


def one_hot(indices, num_uniq=None):
    """ convert categorical indices to one-hot encoding """
    num_uniq = len(np.unique(indices)) if num_uniq is None else num_uniq
    encoding = np.zeros((indices.shape[0], num_uniq))
    for row, col in enumerate(indices):
        encoding[row, col] = 1
    return encoding


def get_degree_features(head, tail, num_nodes):
    """ in-degree and out-degree """
    rows, cols = head, tail
    data = np.ones(len(rows))
    adj_mat = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    in_degree_head = np.asarray(adj_mat[:, rows].sum(axis=0)).flatten()
    in_degree_tail = np.asarray(adj_mat[:, cols].sum(axis=0)).flatten()
    out_degree_head = np.asarray(adj_mat[rows].sum(axis=1)).flatten()
    out_degree_tail = np.asarray(adj_mat[cols].sum(axis=1)).flatten()
    return np.concatenate((in_degree_head[:,None], in_degree_tail[:,None],
                           out_degree_head[:,None], out_degree_tail[:,None]),
                          axis=1)


def get_node_type_features(head_type, tail_type):
    """ node type of each node """
    types = sorted(set(head_type + tail_type))
    type_to_num = {typ : i for i, typ in enumerate(types)}
    num_uniq = len(set(head_type).union(set(tail_type)))
    head = one_hot(np.array([type_to_num[typ] for typ in head_type]), num_uniq)
    tail = one_hot(np.array([type_to_num[typ] for typ in tail_type]), num_uniq)
    return np.concatenate((head, tail), axis=1)


def get_relation_features(relation):
    """ relation type of each edge """
    return one_hot(relation)


def get_edit_distance_features(head, tail, names):
    """ string edit distance between entity names """
    distances = list()
    for i, j in zip(head.tolist(), tail.tolist()):
        distances.append(edit_distance(names[i], names[j]))
    return np.expand_dims(np.array(distances), axis=1)


def get_length_features(head, tail, names):
    """ string length of text representation of head and tail entities """
    lengths = list()
    for h, t in zip(head.tolist(), tail.tolist()):
        lengths.append([len(names[h]), len(names[t])])
    return np.array(lengths)


def get_character_features(head, tail, names):
    """ number of punctuation and numeric characters in text for head and tail
        entities, as well as ratio of these numbers to total number of
        characters """
    punctuation = {'.', ',' '?', '-', '_', '(', ')', "'", '"', ':', ';', '!'}
    numerical = set([str(num) for num in range(10)])

    # calculate number and ratio of punctuation and numeric characters
    num_punc = np.array([np.sum([char in punctuation for char in list(name)])
                         for name in names])
    num_numer = np.array([np.sum([char in numerical for char in list(name)])
                          for name in names])
    lengths = np.array([len(name) for name in names])
    ratio_punc = num_punc / lengths
    ratio_numer = num_numer / lengths
    ratio_punc[np.isnan(ratio_punc)] = 0
    ratio_numer[np.isnan(ratio_numer)] = 0

    return np.stack([num_punc[head], num_punc[tail],
                     num_numer[head], num_numer[tail],
                     ratio_punc[head], ratio_punc[tail],
                     ratio_numer[head], ratio_numer[tail]]).T


def get_unknown_substring_features(head, tail, names):
    """ binary feature for 'unknown' being substring of entity text """
    unknown_present = np.array(['unknown' in name for name in names])
    unknown_present = unknown_present.astype(float)
    return np.stack([unknown_present[head], unknown_present[tail]]).T


def get_pagerank_features(head, tail):
    """ pagerank score of head and tail entities in graph """
    edges = list(zip(head, tail))
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    pagerank = nx.pagerank(graph)
    pagerank_head = np.array([pagerank[h] for h in head])
    pagerank_tail = np.array([pagerank[t] for t in tail])
    return np.stack([pagerank_head, pagerank_tail]).T


def get_adamic_adar_features(head, tail):
    """ Adamic-Adar index of each pair of nodes """
    edges = list(zip(head, tail))
    graph = nx.Graph()
    graph.add_edges_from(edges)
    aa_index = list(nx.adamic_adar_index(graph, edges))
    values = np.array([tup[2] for tup in aa_index])
    return np.expand_dims(values, 1)


def get_tokens_features(head, tail, tokens, names):
    """ ratio of number of tokens to number of whitespace-delimited words for
        each head and tail entity """
    words = [name.split(' ') for name in names]
    ratios = list()
    for h, t in zip(head, tail):
        head_tokens, tail_tokens = len(tokens[h]), len(tokens[t])
        head_words, tail_words = len(words[h]), len(words[t])
        head_ratio = head_tokens / head_words
        tail_ratio = tail_tokens / tail_words
        ratios.append([head_ratio, tail_ratio, (head_ratio + tail_ratio) / 2])
    return np.array(ratios)


def get_description_features(head, tail, descriptions):
    """ binary indicator of whether each entity has a description """
    head_has_desc = np.array([descriptions[h] != '' for h in head])
    tail_has_desc = np.array([descriptions[t] != '' for t in tail])
    has_desc = np.stack([head_has_desc.astype(float),
                         tail_has_desc.astype(float)]).T
    return has_desc

