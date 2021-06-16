#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

from ogb.linkproppred import Evaluator
from collections import defaultdict
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter
import pdb

from preprocess import add_node_offsets, calculate_valid_negatives

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--dataset', type=str, default='repodb', help='dataset name')
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--print_on_screen', action='store_true', help='log on screen or not')
    parser.add_argument('--ntriples_eval_train', type=int, default=200000, help='number of training triples to evaluate eventually')
    parser.add_argument('--neg_size_eval_train', type=int, default=500, help='number of negative samples when evaluating training triples')

    parser.add_argument('--subgraph', type=str, required=True, help='filename of subgraph to use')
    parser.add_argument('--num_neg_samples', type=int, default=None, help='number of negative samples to rank against for evaluation')

    parser.add_argument('--loss_function', type=str, default='negative-sampling',
                        choices=['negative-sampling', 'max-margin'],
                        help='choice of ranking loss function')
    parser.add_argument('--init_embedding', type=str, default=None,
                        help='entity embeddings to initialize the model with')
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.dataset = argparse_dict['dataset']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        if args.subgraph is not None:
            subgraph_name, _ = \
                os.path.splitext(os.path.basename(args.subgraph))
        else:
            subgraph_name = 'fullgraph'
        logname = f'test_{subgraph_name}'
        if args.num_neg_samples is not None:
            logname += f'_neg{args.num_neg_samples}'
        log_file = os.path.join(args.save_path, logname + '.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics, writer):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        metric_name = ("_".join([mode, metric])).replace("@", "_")
        writer.add_scalar(metric_name, metrics[metric], step)


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)

    if args.subgraph is not None:
        subgraph_name, _ = os.path.splitext(os.path.basename(args.subgraph))
    else:
        subgraph_name = 'fullgraph'

    loss_dir = args.loss_function
    if args.loss_function == 'negative-sampling':
        loss_dir += '-uni' if args.uni_weight else ''
        loss_dir += '-adv' if args.negative_adversarial_sampling else ''

    if args.init_checkpoint is not None:
        args.save_path = args.init_checkpoint
    elif args.save_path is None:
        options = (args.dataset, subgraph_name, loss_dir, args.model,
                   args.hidden_dim, args.gamma, args.learning_rate,
                   args.negative_sample_size, args.batch_size,
                   args.regularization)
        args.save_path = 'results/%s/%s/%s/%s/d%s-g%s-lr%.e-neg%d-b%d-r%.e' % options
        if args.init_embedding is not None:
            args.save_path += '-initemb'

    writer = SummaryWriter(args.save_path)

    # check if results exist
    if os.path.exists(os.path.join(args.save_path, 'valid')) and args.do_train:
        print('saved results found, exiting...')
        sys.exit()

    # Write logs to checkpoint and console
    set_logger(args)

    if args.subgraph is not None:
        logging.info(f'using subgraph: {args.subgraph}')
        split_edge = torch.load(args.subgraph)
    else:
        split_edge = dataset.get_edge_split()
    train_triples, valid_triples, test_triples = \
        split_edge["train"], split_edge["valid"], split_edge["test"]
    nrelation = int(max(train_triples['relation']))+1

    entity_dict = split_edge['entity_dict']
    nentity = split_edge['num_nodes']

    # add offsets to triples and calculate valid negatives if validation/test
    # negatives don't already exist in edge split
    missing_negatives = ('head_neg' not in valid_triples or \
                         'tail_neg' not in valid_triples or \
                         'head_neg' not in test_triples or \
                         'tail_neg' not in test_triples)
    if missing_negatives:
        train_plus_offsets = add_node_offsets(train_triples, entity_dict)
        valid_plus_offsets = add_node_offsets(valid_triples, entity_dict)
        test_plus_offsets = add_node_offsets(test_triples, entity_dict)
        negatives = calculate_valid_negatives(train_plus_offsets,
                                              valid_plus_offsets,
                                              test_plus_offsets,
                                              entity_dict)

    evaluator = Evaluator(name='ogbl-biokg')

    # Number of unique entities and relations in the KG.
    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Dataset: %s' % args.dataset)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    # train_triples = split_dict['train']
    logging.info('#train: %d' % len(train_triples['head']))
    # valid_triples = split_dict['valid']
    logging.info('#valid: %d' % len(valid_triples['head']))
    # test_triples = split_dict['test']
    logging.info('#test: %d' % len(test_triples['head']))

    train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
    for i in tqdm(range(len(train_triples['head']))):
        head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
        head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]
        train_count[(head, relation, head_type)] += 1  # Number of edges of the given relation type for this head.
        train_count[(tail, -relation-1, tail_type)] += 1  # Number of edges of the given relation type for this tail.
        train_true_head[(relation, tail)].append(head)  # The heads for this relation and tail.
        train_true_tail[(head, relation)].append(tail)  # The tails for this relation and head.

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        evaluator=evaluator,
        init_embedding=args.init_embedding
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    if args.init_embedding:
        logging.info(f'Initializing entity embeddings from file: {args.init_embedding}')

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        # entity_dict = checkpoint['entity_dict']

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation,
                args.negative_sample_size, 'head-batch',
                train_count, train_true_head, train_true_tail,
                entity_dict),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation,
                args.negative_sample_size, 'tail-batch',
                train_count, train_true_head, train_true_tail,
                entity_dict),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        # logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        # checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        # entity_dict = checkpoint['entity_dict']
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        best_mrr = 0

        #Training Loop
        for step in range(init_step, args.max_steps):

            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Train', step, metrics, writer)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0 and step > 0:
                logging.info('Evaluating on Valid Dataset...')
                if not missing_negatives:
                    metrics = kge_model.test_step(kge_model, valid_triples,
                        args, entity_dict,
                        num_neg_samples=args.num_neg_samples)
                else:
                    metrics = kge_model.test_step_variable_negatives(
                        kge_model, valid_plus_offsets, negatives['valid'],
                        args)
                log_metrics('Valid', step, metrics, writer)

                # if new validation MRR is best value, save current model
                if metrics['mrr_list'] > best_mrr:
                    logging.info('Saving best model...')
                    best_mrr = metrics['mrr_list']
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps,
                        'entity_dict': entity_dict
                    }
                    save_model(kge_model, optimizer, save_variable_list, args)

    # load saved best model
    if not args.init_checkpoint:
        logging.info('Loading saved best model...')
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint'))
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.cuda:
            kge_model = kge_model.cuda()

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        if not missing_negatives:
            metrics = kge_model.test_step(kge_model, valid_triples, args,
                entity_dict, num_neg_samples=args.num_neg_samples)
        else:
            metrics = kge_model.test_step_variable_negatives(
                kge_model, valid_plus_offsets, negatives['valid'], args)
        log_metrics('Valid', step, metrics, writer)
        fname = f'valid_metrics_{subgraph_name}'
        if args.num_neg_samples is not None:
            fname += f'_neg{args.num_neg_samples}'
        valid_fname = os.path.join(args.save_path, fname + '.json')
        with open(valid_fname, 'w') as f:
            json.dump(metrics, f)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        if not missing_negatives:
            metrics = kge_model.test_step(kge_model, test_triples, args,
                entity_dict, num_neg_samples=args.num_neg_samples)
        else:
            metrics = kge_model.test_step_variable_negatives(
                kge_model, test_plus_offsets, negatives['test'], args)
        log_metrics('Test', step, metrics, writer)
        fname = f'test_metrics_{subgraph_name}'
        if args.num_neg_samples is not None:
            fname += f'_neg{args.num_neg_samples}'
        test_fname = os.path.join(args.save_path, fname + '.json')
        with open(test_fname, 'w') as f:
            json.dump(metrics, f)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        small_train_triples = {}
        indices = np.random.choice(len(train_triples['head']), args.ntriples_eval_train, replace=False)
        for i in train_triples:
            if 'type' in i:
                small_train_triples[i] = [train_triples[i][x] for x in indices]
            else:
                small_train_triples[i] = train_triples[i][indices]
        metrics = kge_model.test_step(kge_model, small_train_triples, args, entity_dict, random_sampling=True,
                                      num_neg_samples=args.num_neg_samples)
        log_metrics('Train', step, metrics, writer)

    # create valid file to indicate experiment successfully completed
    valid_fname = os.path.join(args.save_path, 'valid')
    with open(valid_fname, 'w'):
        os.utime(valid_fname)


if __name__ == '__main__':
    main(parse_args())
