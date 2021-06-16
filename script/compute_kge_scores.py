""" script to run evaluation on KG completion dataset for KGE model loaded
    from checkpoint and save metrics and scores """

from argparse import ArgumentParser
import json
import os
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from kge_util import compute_metric_kg
from preprocess import add_node_offsets


### command-line arguments

parser = ArgumentParser(description='run evaluation on KG completion dataset '
                                    'for KGE model loaded from checkpoint '
                                    'and save metrics and scores')
parser.add_argument('resdir', type=str, help='location of KGE model results')
parser.add_argument('--dataset', type=str, default='repodb',
                    help='name of KG completion dataset to use')
parser.add_argument('--subgraph', type=str, required=True,
                    help='path to edge split file')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for running evaluation')
parser.add_argument('--text_emb_file', type=str, default=None,
                    help='path to file that contains embeddings of entity '
                         'names/descriptions (to use to fill in untrained '
                         'entity embeddings)')
parser.add_argument('--mode', type=str, default='average',
                    choices=['average', 'max'],
                    help='mode for filling in untrained entity embeddings '
                         '(softmax-weighted average over trained embeddings '
                         'or select maximally-similar embedding)')


def complete_entity_embeddings(ent_emb, split_edge, entity_dict,
                               text_emb_file, mode):
    """ function to fill in untrained entity embeddings based on cosine
        similarity of embeddings of entity text """

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # add node offsets to edge split subsets
    train = add_node_offsets(split_edge['train'], entity_dict)
    valid = add_node_offsets(split_edge['valid'], entity_dict)
    test = add_node_offsets(split_edge['test'], entity_dict)

    # construct mapping from entity index to type
    ent_to_type = dict()
    for subset in (train, valid, test):
        for batch in 'head', 'tail':
            ids, types = subset[batch], subset[f'{batch}_type']
            ent_to_type.update({x : y for x, y in zip(ids, types)})

    # find indices of unique entities in each subset
    get_nodes = lambda x: set(x['head'].tolist() + x['tail'].tolist())
    train_nodes = get_nodes(train)
    valid_nodes = get_nodes(valid)
    test_nodes = get_nodes(test)
    nodes = sorted(train_nodes.union(valid_nodes).union(test_nodes))

    # get indices of seen and unseen entities
    seen = torch.from_numpy(np.array(sorted(train_nodes)))
    valid_unseen = valid_nodes - train_nodes
    test_unseen = test_nodes - train_nodes
    unseen_nodes = valid_unseen.union(test_unseen)
    unseen = torch.from_numpy(np.array(sorted(unseen_nodes)))

    # compute cosine similarity of embeddings of entity text
    text_embeddings = torch.load(text_emb_file).numpy()
    # set rows of embeddings that are not in nodes to zero (for repoDB)
    for i in range(len(text_embeddings)):
        if i not in nodes:
            text_embeddings[i, :] = 0
    similarity = cosine_similarity(text_embeddings)
    similarity = torch.from_numpy(similarity).to(device)

    # fill in unseen entities as either single closest seen entity embedding
    # based on cosine similarity of text or softmax-weighted average of all
    # seen entity embeddings using cosine similarity to compute weights
    ent_emb = ent_emb.clone()
    if mode == 'average':
        idx = torch.meshgrid(unseen, seen)
        weights = torch.softmax(similarity[idx], dim=1)
        ent_emb[unseen, :] = torch.mm(weights, ent_emb[seen, :])
    elif mode == 'max':
        # set unseen indices to zero in similarity
        similarity[:, unseen] = 0
        # set types that don't match to zero
        for idx in unseen:
            typ = ent_to_type[idx.item()]
            cols = [elem for elem in nodes if typ != ent_to_type[elem]]
            similarity[idx, cols] = 0
        most_similar_idx = torch.argmax(similarity[unseen, :], dim=1)
        ent_emb[unseen, :] = ent_emb[most_similar_idx, :]

    return ent_emb


def main(args):

    # create filename for scores, check if saved scores already exist
    if args.subgraph is None:
        subgraphname = 'fullgraph'
    else:
        subgraphname, _ = os.path.splitext(os.path.basename(args.subgraph))
    completed = f'-{args.mode}' if args.text_emb_file is not None else ''
    untrained = '-untrained' if (args.text_emb_file is not None and \
        'untrained' in args.text_emb_file) else ''
    desc = '-desc' if (args.text_emb_file is not None and \
        'desc' in args.text_emb_file) else ''
    savename = f'scores-{subgraphname}{completed}{desc}{untrained}.pt'
    if os.path.exists(os.path.join(args.resdir, savename)):
        print('found saved scores, exiting...')
        sys.exit()

    # set up device
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # load edge split
    if args.subgraph is None:
        split_edge = dataset.get_edge_split()
    else:
        split_edge = torch.load(args.subgraph)

    # load entity_dict
    entity_dict = split_edge['entity_dict']

    # get validation and test triples, add node offsets, convert to tensors
    valid_triples = add_node_offsets(split_edge['valid'], entity_dict)
    test_triples = add_node_offsets(split_edge['test'], entity_dict)
    for triples in (valid_triples, test_triples):
        for key, value in triples.items():
            if isinstance(value, np.ndarray):
                triples[key] = torch.from_numpy(value).to(device)

    # load config file and settings for computing scores
    with open(os.path.join(args.resdir, 'config.json'), 'r') as f:
        config = json.load(f)
    model = config['model']
    gamma = config['gamma']

    # load entity and relation embeddings
    print('loading entity and relation embeddings...')
    checkpoint = torch.load(os.path.join(args.resdir, 'checkpoint'))
    ent_emb = checkpoint['model_state_dict']['entity_embedding'].to(device)
    rel_emb = checkpoint['model_state_dict']['relation_embedding'].to(device)

    # fill in untrained entity embeddings, if specified
    if args.text_emb_file is not None:
        ent_emb = complete_entity_embeddings(ent_emb, split_edge, entity_dict,
                                             args.text_emb_file, args.mode)

    # compute scores and metrics
    print('computing scores and metrics...')

    print('validation set...')
    valid_scores = compute_metric_kg(ent_emb, rel_emb, valid_triples,
                                     method=model, gamma=gamma,
                                     eval_batch_size=args.batch_size,
                                     return_scores=True)

    print('test set...')
    test_scores = compute_metric_kg(ent_emb, rel_emb, test_triples,
                                    method=model, gamma=gamma,
                                    eval_batch_size=args.batch_size,
                                    return_scores=True)

    # save scores and metrics to file
    print('saving scores and metrics...')
    scores = {'valid' : dict(valid_scores),
              'test' : dict(test_scores)}
    torch.save(scores, os.path.join(args.resdir, savename))

    # print metrics
    for subset in scores.keys():
        print(f'{subset}')
        for metric, values in scores[subset]['metrics'].items():
            print(f'  {metric}: {values.mean().item()}')

    print('done')


if __name__ == '__main__':
    main(parser.parse_args())

