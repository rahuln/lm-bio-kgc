""" script to perform evaluation of BERT model on KG completion dataset """

from argparse import ArgumentParser
from collections import defaultdict
import json
import logging
import os
import sys

import numpy as np
from ogb.linkproppred import Evaluator
import pandas as pd
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from tqdm import tqdm

import model as model_classes
from dataloader import DataBatch, RankingTestDataset
from preprocess import add_node_offsets, calculate_valid_negatives
from util import TextRetriever


def parse_args(args=None):
    parser = ArgumentParser(
        description='evaluate KG-BERT-style model on KG completion dataset',
        usage='run.py [<args>] [-h | --help]'
    )

    parser.add_argument('result_dir', type=str,
                        help='directory of saved results')
    parser.add_argument('--device', type=int, default=0, help='GPU to use')
    parser.add_argument('--root', type=str, default='../../dataset/raw/ogb',
                        help='root directory for OGB datasets')
    parser.add_argument('--dataset', type=str, default='repodb',
                        help='name of KG completion dataset to use')
    parser.add_argument('--info_filename', type=str,
                        default='data/processed/repodb.tsv',
                        help='info file for entities')
    parser.add_argument('--relations_filename', type=str, default=None,
                        help='info file for relations')
    parser.add_argument('--tokenized', action='store_true',
                        help='indicates that info file is list of tokens')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size for test dataset')
    parser.add_argument('--subgraph', type=str, required=True,
                        help='precomputed subgraph to use')
    parser.add_argument('--num_neg_samples', type=int, default=500,
                        help='number of negative samples to use')
    parser.add_argument('--output_to_use', type=str, default='ranking_outputs',
                        choices=['link_outputs', 'ranking_outputs'],
                        help='which output to use as a score for ranking')
    parser.add_argument('--eval_fraction', type=float, default=None,
                        help='fraction of validation/test sets to use')
    parser.add_argument('--valid', action='store_true',
                        help='evaluate on validation set instead of test set')
    parser.add_argument('--save', action='store_true',
                        help='save computed ranking scores')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='rank of GPU to use (for multi-GPU training)')

    return parser.parse_args(args)


def setup_logging(savedir, valid=False):
    """ setup logging to write logs to logfile and console """

    log_file = os.path.join(savedir, 'valid.log' if valid else 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def evaluate_variable_negatives(model, tokenizer, triples, info_filename,
    device, negatives, batch_size=64, output_to_use='link_outputs',
    use_descriptions=False, tokenized=False, biencoder=False,
    return_scores=False, max_length=128, local_rank=-1, ngpus=0,
    relations_filename=None):
    """ run ranking evaluation for validation/test set with variable
        number of valid negative head/tail entities per positive triple """

    # set model in eval mode, initialize OGB evaluator
    model.eval()
    evaluator = Evaluator(name='ogbl-biokg')

    # initialize helper class to retrieve text for each triple
    retriever = TextRetriever(info_filename,
                              relations_filename=relations_filename,
                              use_descriptions=use_descriptions,
                              tokenized=tokenized)

    # set up tensor and data loader for positive triples
    positives = torch.tensor(list(zip(triples['head'],
                                      triples['relation'],
                                      triples['tail'])))
    loader = DataLoader(positives, batch_size=batch_size, shuffle=False)

    # calculate scores for positive triples
    y_pred_pos = list()
    for batch in tqdm(loader, desc='positives'):
        indices = batch[:, [0, 2]]
        relations = batch[:, 1]
        inputs = [retriever.get_text(a, b, c)
                  for (a, c), b in zip(indices.tolist(), relations.tolist())]
        labels = torch.ones_like(relations)
        data = DataBatch(inputs, indices, relations, labels, tokenizer,
                         tokenized=tokenized, biencoder=biencoder,
                         max_length=max_length)
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
        y_pred_pos.append(outputs[output_to_use].flatten().cpu())
    y_pred_pos = torch.cat(y_pred_pos)

    # calculate maximum number of negatives for any positive (for padding)
    max_head = np.max([len(elem) for elem in negatives['head']])
    max_tail = np.max([len(elem) for elem in negatives['tail']])
    max_neg = np.maximum(max_head, max_tail)

    # calculate scores for negative triples, corrupting head entity
    y_pred_neg_head = list()
    for idx in tqdm(range(len(positives)), desc='negatives, head-batch'):
        _, r, t = positives[idx]
        neg_triples = torch.tensor([[h, r, t] for h in negatives['head'][idx]])
        loader = DataLoader(neg_triples, batch_size=batch_size, shuffle=False)
        y_pred_neg_elem = list()
        for batch in loader:
            indices = batch[:, [0, 2]]
            relations = batch[:, 1]
            inputs = [retriever.get_text(a, b, c)
                      for (a, c), b in zip(indices.tolist(),
                                           relations.tolist())]
            labels = torch.ones_like(relations)
            data = DataBatch(inputs, indices, relations, labels, tokenizer,
                             tokenized=tokenized, biencoder=biencoder,
                             max_length=max_length)
            data = data.to(device)
            with torch.no_grad():
                outputs = model(data)
            y_pred_neg_elem.append(outputs[output_to_use].flatten().cpu())
        # add padding so negative scores for each positive are same length
        padding = -np.inf * torch.ones(max_neg - len(neg_triples))
        y_pred_neg_elem.append(padding)
        y_pred_neg_head.append(torch.cat(y_pred_neg_elem).unsqueeze(0))
    y_pred_neg_head = torch.cat(y_pred_neg_head)

    # calculate scores for negative triples, corrupting tail entity
    y_pred_neg_tail = list()
    for idx in tqdm(range(len(positives)), desc='negatives, tail-batch'):
        h, r, _ = positives[idx]
        neg_triples = torch.tensor([[h, r, t] for t in negatives['tail'][idx]])
        loader = DataLoader(neg_triples, batch_size=batch_size, shuffle=False)
        y_pred_neg_elem = list()
        for batch in loader:
            indices = batch[:, [0, 2]]
            relations = batch[:, 1]
            inputs = [retriever.get_text(a, b, c)
                      for (a, c), b in zip(indices.tolist(),
                                           relations.tolist())]
            labels = torch.ones_like(relations)
            data = DataBatch(inputs, indices, relations, labels, tokenizer,
                             tokenized=tokenized, biencoder=biencoder,
                             max_length=max_length)
            data = data.to(device)
            with torch.no_grad():
                outputs = model(data)
            y_pred_neg_elem.append(outputs[output_to_use].flatten().cpu())
        # add padding so negative scores for each positive are same length
        padding = -np.inf * torch.ones(max_neg - len(neg_triples))
        y_pred_neg_elem.append(padding)
        y_pred_neg_tail.append(torch.cat(y_pred_neg_elem).unsqueeze(0))
    y_pred_neg_tail = torch.cat(y_pred_neg_tail)

    # calculate metrics
    pos = torch.cat([y_pred_pos, y_pred_pos])
    neg = torch.cat([y_pred_neg_head, y_pred_neg_tail])
    metrics = evaluator.eval({'y_pred_pos' : pos, 'y_pred_neg' : neg})

    # create results dictionary with metrics (and potentially scores)
    results = {'metrics' : metrics}
    if return_scores:
        results['scores'] = {'y_pred_pos' : y_pred_pos,
                             'y_pred_neg_head' : y_pred_neg_head,
                             'y_pred_neg_tail' : y_pred_neg_tail}

    return results


def evaluate_ranking_parallel(model, tokenizer, triples, info_filename, device,
    batch_size=64, num_neg_samples=100, output_to_use='link_outputs',
    use_descriptions=False, tokenized=False, biencoder=False,
    return_scores=False, max_length=128, local_rank=-1, ngpus=0,
    relations_filename=None):
    """ run ranking evaluation in distributed setup """

    # set model in eval mode, initialize OGB evaluator
    model.eval()
    evaluator = Evaluator(name='ogbl-biokg')

    # initialize helper class to retrieve text for each triple
    retriever = TextRetriever(info_filename,
                              relations_filename=relations_filename,
                              use_descriptions=use_descriptions,
                              tokenized=tokenized)

    # set up ranking dataset, sampler, and data loader
    dataset = RankingTestDataset(triples)
    sampler = DistributedSampler(dataset) if local_rank != -1 else None
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
        shuffle=False, pin_memory=True, num_workers=4 * ngpus)

    # create tensors to store metrics
    num_triples = len(triples['head'])
    mrr_list = torch.zeros(2 * num_triples).to(device)
    hits1_list = torch.zeros(2 * num_triples).to(device)
    hits3_list = torch.zeros(2 * num_triples).to(device)
    hits10_list = torch.zeros(2 * num_triples).to(device)

    # create tensors to store computed scores
    if return_scores:
        pos_scores = torch.zeros(num_triples).to(device)
        neg_scores_head = torch.zeros(num_triples, num_neg_samples).to(device)
        neg_scores_tail = torch.zeros(num_triples, num_neg_samples).to(device)

    # run evaluation over dataset
    y_pred_pos = list()
    y_pred_neg = list()
    idx_list = list()
    for positives, negatives, relations, idx in tqdm(loader):
        inputs = [retriever.get_text(a, b, c)
                  for (a, c), b in zip(positives.tolist(), relations.tolist())]
        labels = torch.ones_like(relations)
        data = DataBatch(inputs, positives, relations, labels, tokenizer,
                         tokenized=tokenized, biencoder=biencoder,
                         max_length=max_length)
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
        y_pred_pos.append(outputs[output_to_use].flatten())

        # keep track of indices, to know where to store computed metrics/scores
        idx_list.append(idx)

        # loop over each of num_neg_samples batches of negatives
        y_pred_neg_batch = list()
        for i in range(num_neg_samples):
            indices = negatives[:, i, :]
            inputs = [retriever.get_text(a, b, c)
                      for (a, c), b in zip(indices.tolist(),
                                           relations.tolist())]
            labels = torch.zeros_like(relations)
            data = DataBatch(inputs, indices, relations, labels, tokenizer,
                             tokenized=tokenized, biencoder=biencoder,
                             max_length=max_length)
            data = data.to(device)
            with torch.no_grad():
                outputs = model(data)
            # list of tensors of size [batch_size, 1]
            y_pred_neg_batch.append(outputs[output_to_use])
        # list of tensors of size [batch_size, num_neg_samples]
        y_pred_neg.append(torch.cat(y_pred_neg_batch, dim=1))

    y_pred_pos = torch.cat(y_pred_pos, dim=0)
    y_pred_neg = torch.cat(y_pred_neg, dim=0)
    idx = torch.cat(idx_list, dim=0)

    # store computed scores at correct indices
    if return_scores:
        head_idx = idx[idx % 2 == 0] // 2
        # only take head-batch positive scores (tail-batch will be duplicate)
        pos_scores[head_idx] = y_pred_pos[idx % 2 == 0]
        neg_scores_head[head_idx, :] = y_pred_neg[idx % 2 == 0]
        tail_idx = idx[idx % 2 == 1] // 2
        neg_scores_tail[tail_idx, :] = y_pred_neg[idx % 2 == 1]

    # compute evaluation metrics
    scores = {'y_pred_pos' : y_pred_pos, 'y_pred_neg' : y_pred_neg}
    metrics = evaluator.eval(scores)

    # store computed metrics at correct indices
    head_idx = idx[idx % 2 == 0] // 2
    mrr_list[head_idx] = metrics['mrr_list'][idx % 2 == 0]
    hits1_list[head_idx] = metrics['hits@1_list'][idx % 2 == 0]
    hits3_list[head_idx] = metrics['hits@3_list'][idx % 2 == 0]
    hits10_list[head_idx] = metrics['hits@10_list'][idx % 2 == 0]

    tail_idx = idx[idx % 2 == 1] // 2
    mrr_list[tail_idx + num_triples] = metrics['mrr_list'][idx % 2 == 1]
    hits1_list[tail_idx + num_triples] = metrics['hits@1_list'][idx % 2 == 1]
    hits3_list[tail_idx + num_triples] = metrics['hits@3_list'][idx % 2 == 1]
    hits10_list[tail_idx + num_triples] = metrics['hits@10_list'][idx % 2 == 1]

    # aggregate tensors across GPUs
    if local_rank != -1:

        # metrics
        torch.distributed.all_reduce(mrr_list,
                                     op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(hits1_list,
                                     op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(hits3_list,
                                     op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(hits10_list,
                                     op=torch.distributed.ReduceOp.SUM)

        # scores
        if return_scores:
            torch.distributed.all_reduce(pos_scores,
                                         op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(neg_scores_head,
                                         op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(neg_scores_tail,
                                         op=torch.distributed.ReduceOp.SUM)

    # create results dictionary with metrics
    results = {'metrics' : {'hits@1_list' : hits1_list.cpu(),
                            'hits@3_list' : hits3_list.cpu(),
                            'hits@10_list' : hits10_list.cpu(),
                            'mrr_list' : mrr_list.cpu()}}

    # store scores in results dictionary
    if return_scores:
        results['scores'] = {'y_pred_pos' : pos_scores.cpu(),
                             'y_pred_neg_head' : neg_scores_head.cpu(),
                             'y_pred_neg_tail' : neg_scores_tail.cpu()}

    return results


def evaluate_ranking(model, tokenizer, triples, info_filename, device,
    batch_size=64, num_neg_samples=100, output_to_use='link_outputs',
    use_descriptions=False, tokenized=False, biencoder=False,
    return_scores=False, max_length=128, local_rank=-1, ngpus=0,
    relations_filename=None):
    """ run ranking evaluation using given model on a set of triples """

    # set model in eval mode, initialize OGB evaluator
    model.eval()
    evaluator = Evaluator(name='ogbl-biokg')

    # load entity indices from triples
    head = torch.from_numpy(triples['head'])
    tail = torch.from_numpy(triples['tail'])
    relation = torch.from_numpy(triples['relation'])
    head_neg = torch.from_numpy(triples['head_neg'][:, :num_neg_samples])
    tail_neg = torch.from_numpy(triples['tail_neg'][:, :num_neg_samples])

    # initialize helper class to retrieve text for each triple
    retriever = TextRetriever(info_filename,
                              relations_filename=relations_filename,
                              use_descriptions=use_descriptions,
                              tokenized=tokenized)

    # initialize data loader, list of scores for positive triples
    loader = DataLoader(torch.arange(len(head)), batch_size=batch_size)
    scores = list()

    # compute scores for positive triples
    for batch in tqdm(loader, desc='positive'):
        indices = zip(head[batch].tolist(), tail[batch].tolist())
        indices = torch.stack([head[batch], tail[batch]], dim=1)
        relations = relation[batch]
        inputs = [retriever.get_text(a, b, c)
                  for (a, c), b in zip(indices.tolist(), relations.tolist())]
        labels = torch.ones_like(relations)
        data = DataBatch(inputs, indices, relations, labels, tokenizer,
                         tokenized=tokenized, biencoder=biencoder,
                         max_length=max_length)
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
        scores.append(outputs[output_to_use].flatten().cpu())
    y_pred_pos = torch.cat(scores)

    # set up indices, loader, list of scores for negative triples, head batch
    ntriples = head_neg.size(0)
    head_neg = head_neg.flatten()
    inds = torch.arange(ntriples).repeat_interleave(num_neg_samples).to(device)
    loader = DataLoader(torch.arange(len(head_neg)), batch_size=batch_size)
    scores = list()

    # compute scores for negative triples, head batch
    for batch in tqdm(loader, desc='negative, head batch'):
        indices = zip(head_neg[batch].tolist(), tail[inds[batch]].tolist())
        indices = torch.stack([head_neg[batch], tail[inds[batch]]], dim=1)
        relations = relation[inds[batch]]
        inputs = [retriever.get_text(a, b, c)
                  for (a, c), b in zip(indices.tolist(), relations.tolist())]
        labels = torch.zeros_like(relations)
        data = DataBatch(inputs, indices, relations, labels, tokenizer,
                         tokenized=tokenized, biencoder=biencoder,
                         max_length=max_length)
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
        scores.append(outputs[output_to_use].flatten().cpu())
    y_pred_neg_head = torch.cat(scores).reshape((ntriples, num_neg_samples))

    # set up indices, loader, list of scores for negative triples, tail batch
    ntriples = tail_neg.size(0)
    tail_neg = tail_neg.flatten()
    inds = torch.arange(ntriples).repeat_interleave(num_neg_samples).to(device)
    loader = DataLoader(torch.arange(len(tail_neg)), batch_size=batch_size)
    scores = list()

    # compute scores for negative triples, tail batch
    for batch in tqdm(loader, desc='negative, tail batch'):
        indices = zip(head[inds[batch]].tolist(), tail_neg[batch].tolist())
        indices = torch.stack([head[inds[batch]], tail_neg[batch]], dim=1)
        relations = relation[inds[batch]]
        inputs = [retriever.get_text(a, b, c)
                  for (a, c), b in zip(indices.tolist(), relations.tolist())]
        labels = torch.zeros_like(relations)
        data = DataBatch(inputs, indices, relations, labels, tokenizer,
                         tokenized=tokenized, biencoder=biencoder,
                         max_length=max_length)
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
        scores.append(outputs[output_to_use].flatten().cpu())
    y_pred_neg_tail = torch.cat(scores).reshape((ntriples, num_neg_samples))

    # compute evaluation metrics
    scores = {'y_pred_pos' : y_pred_pos, 'y_pred_neg' : y_pred_neg_head}
    metrics_head = evaluator.eval(scores)
    scores = {'y_pred_pos' : y_pred_pos, 'y_pred_neg' : y_pred_neg_tail}
    metrics_tail = evaluator.eval(scores)

    # combine head and tail metrics
    metrics = dict()
    for key in metrics_head:
        metrics[key] = torch.cat([metrics_head[key], metrics_tail[key]])

    # create results dictionary with metrics (and potentially scores)
    results = {'metrics' : metrics}
    if return_scores:
        results['scores'] = {'y_pred_pos' : y_pred_pos,
                             'y_pred_neg_head' : y_pred_neg_head,
                             'y_pred_neg_tail' : y_pred_neg_tail}

    return results


def main(args):

    # set up logging
    setup_logging(args.result_dir, valid=args.valid)

    # set up device
    if args.local_rank == -1:
        device = torch.device(f'cuda:{args.device}'
                              if torch.cuda.is_available() else 'cpu')
        args.ngpus = 0
    else:
        if args.local_rank in [-1, 0]:
            logging.info('performing distributed evaluation')
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        init_process_group(backend='nccl', init_method='env://')
        args.ngpus = torch.distributed.get_world_size()

    # check to see if info_filename is correct
    if args.dataset not in args.info_filename:
        raise ValueError('info_filename does not seem to be correct: '
                         f'{args.info_filename} does not have {args.dataset} '
                         'in the name')

    # load config file from result directory
    with open(os.path.join(args.result_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    # use edge split from subgraph if specified, otherwise from dataset
    if args.subgraph is not None:
        split_edge = torch.load(args.subgraph)
    else:
        split_edge = dataset.get_edge_split()

    # load entity dict, add node offsets to triples
    entity_dict = split_edge['entity_dict']
    train_triples = add_node_offsets(split_edge['train'], entity_dict)
    valid_triples = add_node_offsets(split_edge['valid'], entity_dict)
    test_triples = add_node_offsets(split_edge['test'], entity_dict)

    # calculate valid negatives for ranking evaluation, if validation/test
    # negatives don't already exist in edge split
    missing_negatives = ('head_neg' not in valid_triples or \
                         'tail_neg' not in valid_triples or \
                         'head_neg' not in test_triples or \
                         'tail_neg' not in test_triples)
    if missing_negatives:
        logging.info('calculating valid negatives for validation/test sets')
        negatives = calculate_valid_negatives(train_triples, valid_triples,
                                              test_triples, entity_dict)

    # use fraction of validation and test sets for evaluation (do this after
    # computing valid negatives so that positives from the entire
    # training/validation/test sets can be used to filter out negatives to
    # rank against)
    if args.eval_fraction is not None:
        if args.local_rank in [-1, 0]:
            logging.info(f'using {100 * args.eval_fraction:.0f}% of '
                         'validation/test sets for evaluation')
        for key in ('valid', 'test'):
            subset = split_edge[key]

            # select indices
            num_total = len(split_edge[key]['head'])
            num_select = int(args.eval_fraction * num_total)
            idx = np.random.choice(num_total, size=num_select, replace=False)
            idx = sorted(idx)

            # restrict subset to indices
            subset['head'] = subset['head'][idx]
            subset['head_type'] = [subset['head_type'][i] for i in idx]
            subset['relation'] = subset['relation'][idx]
            subset['tail'] = subset['tail'][idx]
            subset['tail_type'] = [subset['tail_type'][i] for i in idx]

            if 'head_neg' in subset:
                subset['head_neg'] = subset['head_neg'][idx]
            if 'tail_neg' in subset:
                subset['tail_neg'] = subset['tail_neg'][idx]

            # restrict valid negatives to indices
            if missing_negatives:
                for subkey in ('head', 'tail'):
                    negatives[key][subkey] = [negatives[key][subkey][i]
                                              for i in idx]

        # have to recalculate triples with node offsets added
        train_triples = add_node_offsets(split_edge['train'], entity_dict)
        valid_triples = add_node_offsets(split_edge['valid'], entity_dict)
        test_triples = add_node_offsets(split_edge['test'], entity_dict)

    # log number of entities, relations, validation/test triples
    args.nentity = split_edge['num_nodes']
    args.nrelation = int(max(split_edge['train']['relation']))+1

    if args.local_rank in [-1, 0]:
        logging.info('#entity: %d' % args.nentity)
        logging.info('#relation: %d' % args.nrelation)
        logging.info('#valid: %d' % len(split_edge['valid']['head']))
        logging.info('#test: %d' % len(split_edge['test']['head']))

    # add model keyword arguments to args
    args.encoder_type = config.get('encoder_type', None)
    args.model_name = config.get('model_name', config['model'])

    # general arguments
    args.link_prediction = config.get('link_prediction', 0)
    args.relation_prediction = config.get('relation_prediction', False)
    args.relevance_ranking = config.get('relevance_ranking', 0)

    # arguments for BLPBiEncoder, BLPCrossEncoder
    args.embedding_dim = config.get('embedding_dim', 128)
    args.score = config.get('score', 'ComplEx')
    args.gamma = config.get('gamma', 20)
    args.regularization = config.get('regularization', 0)
    args.entity_representation = config.get('entity_representation', 'cls')

    # arguments for KGBERTPlusEmbeddings
    args.checkpoint_file = config.get('checkpoint_file', None)
    args.num_hidden_layers = config.get('num_hidden_layers', 1)
    args.hidden_dim = config.get('hidden_dim', 1024)
    args.dropout = config.get('dropout', 0.1)
    args.average_emb = config.get('average_emb', False)

    # arguments for JointKGBERTAndKGEModel
    args.weighted_average = config.get('weighted_average', False)

    # arguments for DKRLBiEncoder
    args.update_embeddings = config.get('update_embeddings', False)

    # backwards compatibility
    if args.encoder_type is None \
        or not hasattr(model_classes, args.encoder_type):
        if config.get('biencoder', False):
            args.encoder_type = 'KGBERTBiEncoder'
        elif config.get('use_embeddings', False):
            args.encoder_type = 'KGBERTPlusEmbeddings'
        elif config.get('encoder_type', False):
            encoder_type_to_class = {'cross-encoder' : 'KGBERT',
                                     'bi-encoder' : 'KGBERTBiEncoder',
                                     'blp' : 'BLPBiEncoder',
                                     'blp-cross-encoder' : 'BLPCrossEncoder'}
            args.encoder_type = encoder_type_to_class[config['encoder_type']]
        else:
            args.encoder_type = 'KGBERT'

    # set up tokenizer and model
    if args.local_rank in [-1, 0]:
        logging.info('loading model')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    Model = getattr(model_classes, args.encoder_type)
    model = Model(args).to(device)
    for fname in ('best_model.pt', 'checkpoint.pt', 'model.pt'):
        path_to_model = os.path.join(args.result_dir, fname)
        if os.path.exists(path_to_model):
            state_dict = torch.load(path_to_model, map_location=device)
            if fname in ('best_model.pt', 'checkpoint.pt'):
                state_dict = state_dict['model']
            break
    if 'relation_head.weight' in state_dict \
        and state_dict['relation_head.weight'].size(0) != args.nrelation:
        for key in ('relation_head.weight', 'relation_head.bias'):
            state_dict[key] = state_dict[key][1:]
    model.load_state_dict(state_dict, strict=False)

    # wrap model in DistributedDataParallel
    if args.local_rank != -1:
        model = DDP(model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    find_unused_parameters=True)

    # log info about evaluation
    if args.local_rank in [-1, 0]:
        subset = 'validation' if args.valid else 'test'
        logging.info(f'evaluating on {subset} set')
        logging.info('model = %s' % config['model'])
        if args.subgraph is not None:
            logging.info('subgraph = %s' % args.subgraph)
        else:
            logging.info('full graph')
        logging.info('number of negative samples = %d' % args.num_neg_samples)
        logging.info('batch size = %d' % args.batch_size)
        logging.info('ranking output = %s' % args.output_to_use)
        if config.get('use_descriptions', False):
            logging.info('using descriptions')

    # set subset of triples and negatives to evaluate on
    triples = valid_triples if args.valid else test_triples
    if missing_negatives:
        negatives = negatives['valid'] if args.valid else negatives['test']

    # run evaluation function
    kwargs = {'batch_size'         : args.batch_size,
              'num_neg_samples'    : args.num_neg_samples,
              'output_to_use'      : args.output_to_use,
              'use_descriptions'   : config.get('use_descriptions', False),
              'tokenized'          : args.tokenized,
              'biencoder'          : ('BiEncoder' in args.encoder_type),
              'return_scores'      : args.save,
              'max_length'         : config.get('max_length', 128),
              'local_rank'         : args.local_rank,
              'ngpus'              : args.ngpus,
              'relations_filename' : args.relations_filename}
    if missing_negatives:
        del kwargs['num_neg_samples']

    if args.local_rank != -1:
        if missing_negatives:
            raise ValueError('distributed evaluation for variable negatives '
                             'is not yet implemented')
        logging.info('running distributed evaluation')
        results = evaluate_ranking_parallel(model, tokenizer, triples,
                                            args.info_filename, device,
                                            **kwargs)
    else:
        if not missing_negatives:
            logging.info('running standard evaluation')
            results = evaluate_ranking(model, tokenizer, triples,
                                       args.info_filename, device, **kwargs)
        else:
            logging.info('running evaluation for variable negatives')
            results = evaluate_variable_negatives(model, tokenizer, triples,
                                                  args.info_filename, device,
                                                  negatives, **kwargs)
    metrics = results['metrics']

    # print average value of each evaluation metric
    if args.local_rank in [-1, 0]:
        for key, value_list in metrics.items():
            if '_' in key:
                label, _ = key.split('_')
                value = value_list.mean().item()
            else:
                label = key
                value = value_list.item()
            logging.info(f'{label}: {value:.6f}')
            if key == 'mrr_list':
                value = (1. / value_list).mean().item()
                logging.info(f'mr: {value:.6f}')

    # save computed scores
    if args.save and args.local_rank in [-1, 0]:
        subset = 'valid' if args.valid else 'test'
        if args.subgraph is not None:
            subgraphname, _ = os.path.splitext(os.path.basename(args.subgraph))
        else:
            subgraphname = 'fullgraph'
        savename = os.path.join(args.result_dir,
                                f'scores-{subgraphname}-{subset}-neg{args.num_neg_samples}.pt')
        torch.save(results, savename)

    if args.local_rank in [-1, 0]:
        logging.info('done')
        logging.info('=' * 30)


if __name__ == '__main__':
    main(parse_args())
