""" script to run training and evaluation of BERT model on dataset for
    KG completion """

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
import json
import logging
import os
import sys
import time

from accelerate import Accelerator, DistributedDataParallelKwargs
import numpy as np
from ogb.linkproppred import Evaluator
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

import model as model_classes
from dataloader import TrainDataset, BidirectionalOneShotIterator
from evaluate import evaluate_ranking, evaluate_variable_negatives
from preprocess import get_count, add_node_offsets, calculate_valid_negatives
from util import NegativeSamplingLoss, sample_graph


def parse_args(args=None):
    parser = ArgumentParser(
        description='train/evaluate BERT-based model on KG completion dataset',
        usage='run.py [<args>] [-h | --help]'
    )

    parser.add_argument('--device', type=int, default=0, help='GPU to use')
    parser.add_argument('--dataset', type=str, default='repodb',
                        help='name of KG completion dataset to use')
    parser.add_argument('--info_filename', type=str,
                        default='data/processed/repodb.tsv',
                        help='info file for entities')
    parser.add_argument('--relations_filename', type=str, default=None,
                        help='info file for relations')
    parser.add_argument('--model', default='pubmedbert', type=str,
                        help='which BERT encoder to use')
    parser.add_argument('--encoder-type', type=str, default='KGBERT',
                        help='choice of model class from model.py')
    parser.add_argument('--tokenized', action='store_true',
                        help='indicates that info file is list of tokens')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size of positive samples')
    parser.add_argument('--test_batch_size', default=32, type=int,
                        help='batch size for test dataset')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw'],
                        help='optimizer to use')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate for optimizer')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--max_steps', type=int, default=None,
                        help='maximum number of steps, overrides epochs')
    parser.add_argument('--valid_steps', default=1000, type=int,
                        help='evaluate on validation set every xx steps')
    parser.add_argument('--log_steps', default=100, type=int,
                        help='train log every xx steps')
    parser.add_argument('--gradient_steps', type=int, default=1,
                        help='number of steps to take to accumulate gradients '
                             'before taking optimizer step')
    parser.add_argument('--mode', type=str, default='finetune',
                        choices=['freeze', 'finetune'],
                        help='freeze or fine-tune BERT parameters')
    parser.add_argument('--num-finetune-layers', type=int, default=12,
                        help='number of layers (from last layer) to fine-tune')
    parser.add_argument('--subgraph', type=str, required=True,
                        help='precomputed subgraph to use')
    parser.add_argument('--eval-subgraph', type=str, default=None,
                        help='subgraph to use for evaluation, if different '
                             'from training subgraph')
    parser.add_argument('--fraction', type=float, default=None,
                        help='fraction of training graph edges to sample for '
                             'each epoch')
    parser.add_argument('--eval-fraction', type=float, default=None,
                        help='fraction of validation/test sets to use')
    parser.add_argument('--output-to-use', type=str, default='ranking_outputs',
                        choices=['link_outputs', 'ranking_outputs'],
                        help='which model outputs to use for ranking if '
                             'using ranking evaluation mode')
    parser.add_argument('--negatives-file', type=str, default=None,
                        help='file storing precomputed negative samples')
    parser.add_argument('--negative-sample-size', type=int, default=1,
                        help='number of negative samples per positive sample '
                             'during training')
    parser.add_argument('--num-neg-samples', type=int, default=500,
                        help='number of negative samples to use if '
                             'using ranking evaluation mode')
    parser.add_argument('--link-prediction', '-lp', action='store_true',
                        help='use link prediction (i.e., binary triple '
                             'classification) loss')
    parser.add_argument('--relation-prediction', '-rp', action='store_true',
                        help='use multi-class relation prediction loss')
    parser.add_argument('--relevance-ranking', '-rr', action='store_true',
                        help='use relevance ranking (i.e., max-margin) loss')
    parser.add_argument('--margin', type=float, default=1,
                        help='value of margin for max-margin loss')
    parser.add_argument('--ranking-loss', type=str, default='max-margin',
                        choices=['max-margin', 'negative-sampling'],
                        help='choice of loss function for relevance ranking')
    parser.add_argument('--use-descriptions', action='store_true',
                        help='encode entity descriptions in addition to names')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training set, parameter '
                             'initializiation, and batch order')
    parser.add_argument('--test-seed', type=int, default=42,
                        help='random seed for test set')
    parser.add_argument('--init', type=str, default=None,
                        help='directory of results to initialize from')
    parser.add_argument('--model-file', type=str, default=None,
                        help='initialize model parameters from file')
    parser.add_argument('--save-metric', type=str, default='mrr_list',
                        help='metric to use to save best model based on '
                             'validation set')
    parser.add_argument('--linear-warmup', type=float, default=None,
                        help='fraction of training steps to do linear '
                             'learning rate warmup')
    parser.add_argument('--grad-norm', type=float, default=None,
                        help='norm for gradient clipping')
    parser.add_argument('--max-length', type=int, default=128,
                        help='maximum input sequence length for BERT model')
    parser.add_argument('--suffix', type=str, default=None,
                        help='suffix to add to results directory name')
    parser.add_argument('--outdir', type=str, default='results',
                        help='output directory for saved results')
    parser.add_argument('--use-accelerate', action='store_true',
                        help='use accelerate for distributed training')

    # arguments for BLPBiEncoder, BLPCrossEncoder, KGEModel, DKRLBiEncoder
    parser.add_argument('--score', type=str, default='ComplEx',
                        choices=['ComplEx', 'DistMult', 'TransE', 'RotatE'],
                        help='choice of score function')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='dimension of entity embeddings')
    parser.add_argument('--gamma', type=float, default=20,
                        help='margin for score function')
    parser.add_argument('--regularization', type=float, default=0,
                        help='regularization parameter for L2-norm penalty '
                             'that encourages BLP embeddings to move closer '
                             'to KG embeddings loaded from a checkpoint file')
    parser.add_argument('--entity-representation', type=str, default='cls',
                        choices=['mean', 'cls'],
                        help='how to construct entity representation from '
                             'BERT contextualized output embeddings')

    # arguments for KGBERTPlusEmbeddings
    parser.add_argument('--checkpoint-file', type=str, default=None,
                        help='file containing entity and relation embeddings')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                        help='hidden layer size for MLP')
    parser.add_argument('--num-hidden-layers', type=int, default=1,
                        help='number of hidden layers for MLP')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate for MLP')
    parser.add_argument('--average-emb', action='store_true',
                        help='average entity and relation embeddings before '
                             'concatenating with LM encodings as input to MLP')

    # arguments for JointKGBERTAndKGEModel
    parser.add_argument('--weighted-average', action='store_true',
                        help='compute weighted average of KG-BERT and KGE '
                             'model ranking scores using learned weights')

    # arguments for KGBERTWithKGEInputs
    parser.add_argument('--align-embeddings', action='store_true',
                        help='linearly align entity embeddings with language '
                             'model input embeddings before adding them to '
                             'the embedding matrix')

    # arguments for DKRLBiEncoder
    parser.add_argument('--update-embeddings', action='store_true',
                        help='updated word embeddings along with model '
                             'parameters for DKRL')

    return parser.parse_args(args)

def setup_logging(savedir):
    """ setup logging to write logs to logfile and console """

    log_file = os.path.join(savedir, 'train.log')

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

    return logging.getLogger('')


def construct_savedir(args):
    """ construct name of output directory from args """
    if args.encoder_type in ('KGEModel', 'DKRLBiEncoder'):
        savedir = f'{args.encoder_type.lower()}-{args.score.lower()}'
        savedir += f'-d{args.embedding_dim}-g{args.gamma:.1f}'
        if args.encoder_type == 'DKRLBiEncoder':
            savedir += '-update-emb' if args.update_embeddings else ''
    else:
        savedir = f'{args.encoder_type.lower()}'
        if 'BLP' in args.encoder_type \
            or args.encoder_type == 'JointKGBERTAndKGEModel':
            savedir += f'-{args.score.lower()}'
            savedir += f'-d{args.embedding_dim}-g{args.gamma:.1f}'
            savedir += f'-{args.entity_representation}'
        elif args.encoder_type == 'KGBERTWithKGEInputs' and \
            args.align_embeddings:
            savedir += '-aligned'
        savedir += f'-{args.model}-{args.mode}'
        if args.mode == 'finetune' and args.num_finetune_layers < 12:
            savedir += f'{args.num_finetune_layers}'
    if args.fraction is not None:
        savedir += f'-frac{args.fraction:.0e}'
    savedir += f'-neg{args.negative_sample_size:d}'
    savedir += f'-e{args.epochs:02d}-b{args.batch_size}-lr{args.lr}'
    if args.regularization > 0:
        savedir += f'-reg{args.regularization:.0e}'
    if args.grad_norm is not None:
        savedir += f'-gc{args.grad_norm:.1e}'
    if args.linear_warmup is not None:
        savedir += f'-lwu{args.linear_warmup:.1e}'
    if args.gradient_steps > 1:
        savedir += f'-gacc{args.gradient_steps}'
    savedir += f'-{args.optimizer}'
    if args.relations_filename is not None:
        savedir += '-rel'
    if args.use_descriptions:
        savedir += '-desc'
    if args.ngpus > 1:
        savedir += f'-ngpus{args.ngpus}'
    if args.negatives_file is not None:
        savedir += '-negfromfile'
    if args.use_accelerate:
        savedir += '-accelerate'
    if args.suffix is not None:
        savedir += f'-{args.suffix}'

    datasetdir = args.dataset

    if args.subgraph is not None:
        subgraphdir, _ = os.path.splitext(os.path.basename(args.subgraph))
    else:
        subgraphdir = 'fullgraph'

    losses = list()
    if args.link_prediction:
        losses.append('link')
    if args.relation_prediction:
        losses.append('relation')
    if args.relevance_ranking:
        if args.ranking_loss == 'max-margin':
            losses.append('ranking')
        elif args.ranking_loss == 'negative-sampling':
            losses.append('negative-sampling')
    taskdir = '-'.join(losses)

    seeddir = f'seed-{args.seed}'

    return os.path.join(args.outdir, datasetdir, subgraphdir, taskdir, savedir,
                        seeddir)


def main(args):

    # check that info filename is correct
    if args.dataset not in args.info_filename:
        raise ValueError('info file is incorrect for specified dataset')
    if args.tokenized:
        if 'tokens' not in args.info_filename:
            raise ValueError('must use tokens file if specifying --tokenized')
        if (args.use_descriptions and 'desc' not in args.info_filename) or \
           (not args.use_descriptions and 'desc' in args.info_filename):
            raise ValueError('--use-descriptions specified with pretokenized '
                             'inputs that do not have descriptions (or vice '
                             'versa)')

    # must use at least one loss
    if not args.link_prediction \
        and not args.relation_prediction \
        and not args.relevance_ranking:
        raise ValueError('must specify at least one loss function')

    # if using embeddings, must specify filenames
    if args.encoder_type == 'KGBERTPlusEmbeddings' \
        and args.checkpoint_file is None:
        raise ValueError('must specify checkpoint filename if using '
                         'embeddings')

    # set up device (for evaluation functions)
    device = torch.device(f'cuda:{args.device}'
                          if torch.cuda.is_available() else 'cpu')
    args.ngpus = torch.cuda.device_count()

    # set up accelerator
    if args.use_accelerate:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        is_main_process = accelerator.is_main_process
        process_index = accelerator.process_index
    else:
        is_main_process = True
        process_index = 0

    # set random seeds for negative sampling, batch ordering, initializiation
    # of linear layer(s) in model
    np.random.seed(args.seed + process_index)
    torch.manual_seed(args.seed + process_index)
    torch.cuda.manual_seed_all(args.seed + process_index)

    # set up output directory, logging, and tensorboard writer
    savedir = construct_savedir(args)
    if is_main_process:
        os.makedirs(savedir, exist_ok=True)
        logger = setup_logging(savedir)
        logger.info('=' * 30)
        logger.info('starting training script')
        if args.use_accelerate:
            logger.info('using accelerate for distributed training')

        writer = SummaryWriter(os.path.join(savedir, 'tensorboard'))

    # check for saved results
    valid_fname = os.path.join(savedir, 'valid')
    if os.path.exists(valid_fname):
        if is_main_process:
            logger.info('found saved results, exiting...')
        sys.exit()

    # map from model shortnames to huggingface names
    model_names = {
        'pubmedbert'  : 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'pubmedbertfull' : 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        'bert'        : 'bert-base-uncased',
        'bertlarge'   : 'bert-large-uncased',
        'roberta'      : 'roberta-base',
        'robertalarge' : 'roberta-large',
        'bioclinicalbert' : 'emilyalsentzer/Bio_ClinicalBERT',
        'biolm' : 'EMBO/bio-lm',
        'biobert' : 'dmis-lab/biobert-v1.1',
        'scibert' : 'allenai/scibert_scivocab_uncased'
    }

    # retrieve huggingface model name
    args.model_name = model_names[args.model]

    # use edge split from subgraph if specified, otherwise from dataset
    if args.subgraph is not None:
        split_edge = torch.load(args.subgraph)
    else:
        split_edge = dataset.get_edge_split()

    # load different subgraph for evaluation, if specified
    if args.eval_subgraph is not None:
        eval_split_edge = torch.load(args.eval_subgraph)
    else:
        eval_split_edge = split_edge

    # use fraction of validation and test sets for evaluation
    if args.eval_fraction is not None:
        if is_main_process:
            logger.info(f'using {100 * args.eval_fraction:.0f}% of '
                        'validation/test sets for evaluation')
        for key in ('valid', 'test'):
            subset = eval_split_edge[key]

            # select indices
            num_total = len(eval_split_edge[key]['head'])
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

    # preprocess dataset for number of entities/relations and entity indices
    entity_dict = split_edge['entity_dict']
    train_triples = add_node_offsets(split_edge['train'], entity_dict)
    valid_triples = add_node_offsets(eval_split_edge['valid'], entity_dict)
    test_triples = add_node_offsets(eval_split_edge['test'], entity_dict)

    # calculate valid negatives for ranking evaluation, if validation/test
    # negatives don't already exist in edge split
    missing_negatives = ('head_neg' not in valid_triples or \
                         'tail_neg' not in valid_triples or \
                         'head_neg' not in test_triples or \
                         'tail_neg' not in test_triples)
    if missing_negatives:
        negatives = calculate_valid_negatives(train_triples, valid_triples,
                                              test_triples, entity_dict)

    # count number of training triples of each head, relation, and tail type
    train_count = get_count(train_triples)

    # set number of entities and relations in arguments
    args.nentity = split_edge['num_nodes']
    args.nrelation = int(max(train_triples['relation']))+1

    # save args in config file
    if is_main_process:
        logger.info('saving args in config.json')
        savename = os.path.join(savedir, 'config.json')
        with open(savename, 'w') as cfgfile:
            json.dump(vars(args), cfgfile, indent=4)

        logger.info('#entity: %d' % args.nentity)
        logger.info('#relation: %d' % args.nrelation)

        logger.info('#train: %d' % len(train_triples['head']))
        logger.info('#valid: %d' % len(valid_triples['head']))
        logger.info('#test: %d' % len(test_triples['head']))

    # set up tokenizer
    if args.encoder_type == 'KGEModel':
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # initialize training dataset
    if is_main_process:
        logger.info('constructing training dataset')
    biencoder = ('BiEncoder' in args.encoder_type)
    if args.negatives_file is not None and is_main_process:
        logger.info(f'loading negative samples from: {args.negatives_file}')
    dataset_kwargs = {'negative_sample_size' : args.negative_sample_size,
                      'tokenized' : args.tokenized,
                      'use_descriptions' : args.use_descriptions,
                      'negatives_file' : args.negatives_file,
                      'relations_filename' : args.relations_filename}
    dataset = TrainDataset(args.info_filename, train_triples, train_count,
                           entity_dict, **dataset_kwargs)

    # set up model, optimizer, loss criterion
    if is_main_process:
        logger.info('initializing model')
    Model = getattr(model_classes, args.encoder_type)
    model = Model(args)
    if not args.use_accelerate:
        model = model.to(device)

    # initialize model from file (different from initializing from saved
    # checkpoint, which also includes optimizer state and training step)
    if args.model_file is not None:
        if is_main_process:
            logger.info('initializing model parameters from file')
        state_dict = torch.load(args.model_file)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        # take care of old relation head with (nrelation + 1) output dim
        if 'relation_head.weight' in state_dict \
            and state_dict['relation_head.weight'].size(0) != args.nrelation:
            for key in ('relation_head.weight', 'relation_head.bias'):
                state_dict[key] = state_dict[key][-args.nrelation:]
            model.load_state_dict(state_dict, strict=False)
            del state_dict

    # retrieve parameters to train, set up optimizer
    if is_main_process:
        logger.info('initializing optimizer')
    if hasattr(model, 'encoder'):
        if args.mode == 'freeze':
            # freeze all parameters in BERT encoder
            for param in model.encoder.parameters():
                param.requires_grad = False
        elif args.mode == 'finetune':
            num_hidden_layers = model.encoder.config.num_hidden_layers
            if args.num_finetune_layers < num_hidden_layers:
                # freeze embeddings
                for param in model.encoder.embeddings.parameters():
                    param.requires_grad = False
                # freeze all except last `args.num_finetune_layers` layers
                for i in range(num_hidden_layers - args.num_finetune_layers):
                    for param in model.encoder.encoder.layer[i].parameters():
                        param.requires_grad = False
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizers = dict(adam=torch.optim.Adam, adamw=torch.optim.AdamW)
    optimizer = optimizers[args.optimizer](parameters, lr=args.lr)

    # load from initialization
    if args.init is not None:
        if is_main_process:
            logger.info('initializing from specified checkpoint')
        savename = os.path.join(args.init, 'checkpoint.pt')
        if os.path.exists(savename):
            checkpoint = torch.load(savename)
            model.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            init_step = checkpoint['step']
        else:
            init_step = 0
    elif os.path.exists(os.path.join(savedir, 'checkpoint.pt')):
        if is_main_process:
            logger.info('trying to initialize from saved checkpoint')
        try:
            savename = os.path.join(savedir, 'checkpoint.pt')
            checkpoint = torch.load(savename)
            model.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            init_step = checkpoint['step']
        except:
            if is_main_process:
                logger.info('could not load checkpoint, starting from scratch')
            init_step = 0
    else:
        init_step = 0

    # set up loss functions
    link_lossfn = torch.nn.BCEWithLogitsLoss()
    relation_lossfn = torch.nn.CrossEntropyLoss()
    if args.ranking_loss == 'max-margin':
        ranking_lossfn = torch.nn.MarginRankingLoss(margin=args.margin)
    elif args.ranking_loss == 'negative-sampling':
        ranking_lossfn = NegativeSamplingLoss

    if is_main_process:
        logger.info('training')
        logger.info('model = %s' % args.model)
        logger.info('batch size = %d' % args.batch_size)
        if args.gradient_steps > 1:
            logger.info('gradient accumulation = %d' % args.gradient_steps)
        logger.info('learning rate = %.0e' % args.lr)

        losses = list()
        if args.link_prediction:
            losses.append('link')
        if args.relation_prediction:
            losses.append('relation')
        if args.relevance_ranking:
            losses.append(args.ranking_loss)
        logger.info(f'loss = {", ".join(losses)}')

    # set up options for evaluation
    evaluate_kwargs = {'batch_size' : args.test_batch_size,
                       'num_neg_samples' : args.num_neg_samples,
                       'output_to_use' : args.output_to_use,
                       'use_descriptions' : args.use_descriptions,
                       'tokenized' : args.tokenized,
                       'biencoder' : biencoder,
                       'max_length' : args.max_length,
                       'relations_filename' : args.relations_filename}
    if missing_negatives:
        del evaluate_kwargs['num_neg_samples']

    # keep track of best metric, if specified
    if args.save_metric is not None:
        metrics_fname = os.path.join(savedir, 'best_metrics.json')
        if os.path.exists(metrics_fname):
            if is_main_process:
                logger.info('loading previous best metric')
            with open(metrics_fname, 'r') as f:
                metrics = json.load(f)
            best_value = metrics[args.save_metric]
        else:
            best_value = float('-inf')

    # set up data loader
    def collate_fn(data):
        return TrainDataset.collate_fn(data, tokenizer, args.tokenized,
                                       biencoder, args.max_length)
    loader = DataLoader(dataset,
        batch_size=args.batch_size // args.gradient_steps, shuffle=True,
        collate_fn=collate_fn, pin_memory=True,
        num_workers=4 * args.ngpus)

    # prepare model, optimizer, and data loader for accelerate
    if args.use_accelerate:
        model, optimizer, loader = \
            accelerator.prepare(model, optimizer, loader)

    # indicate that gradient norm clipping is being used
    if is_main_process and args.grad_norm is not None:
        logger.info(f'clipping gradient norm at {args.grad_norm:.3f}')

    # set up learning rate scheduler, if specified
    if args.linear_warmup is not None:
        if is_main_process:
            logger.info(f'using linear lr warmup: {args.linear_warmup:.2f}')
        num_training_steps = len(loader) // args.gradient_steps * args.epochs
        num_warmup_steps = int(args.linear_warmup * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps)

    # training loop
    if is_main_process:
        start = datetime.now()
    global_step = 0
    max_steps_reached = False
    cumul_loss = 0
    cumul_fwd_time = 0
    cumul_bwd_time = 0

    for epoch in range(1, args.epochs + 1):

        if max_steps_reached:
            break

        if is_main_process:
            logger.info(f'starting training epoch {epoch}')

        # if specified, subsample graph and re-initialize dataset and loader
        if args.fraction is not None:
            if is_main_process:
                logger.info(
                    f'sampling {100*args.fraction:.1f}% of training graph')
            sampled_triples = sample_graph(train_triples, args.fraction)
            dataset = TrainDataset(args.info_filename, sampled_triples,
                                   train_count, entity_dict, **dataset_kwargs)
            loader = DataLoader(dataset,
                batch_size=args.batch_size // args.gradient_steps,
                shuffle=True, collate_fn=collate_fn, pin_memory=True,
                num_workers=4 * args.ngpus)
            if args.use_accelerate:
                loader = accelerator.prepare(loader)

        # set epoch for training dataset (used when loading precomputed
        # negative samples)
        dataset.set_epoch(epoch - 1)

        for step, batch in enumerate(loader):

            global_step += 1
            if global_step < init_step:
                continue

            model.train()
            if args.mode == 'freeze' \
                and args.encoder_type != 'KGEModel' \
                and args.encoder_type != 'DKRLBiEncoder':
                if hasattr(model, 'module'):
                    model.module.encoder.eval()
                else:
                    model.encoder.eval()

            # forward pass
            if not args.use_accelerate:
                batch = batch.to(device)
            t0 = time.time()
            outputs = model(batch)
            cumul_fwd_time += time.time() - t0

            # compute loss
            loss = 0
            if args.link_prediction:
                loss += link_lossfn(outputs['link_outputs'], batch.labels)
            if args.relation_prediction:
                loss += relation_lossfn(outputs['relation_outputs'],
                                        batch.relations)
            if args.relevance_ranking:
                # batch order: [pos1..posN, neg11..neg1N, ..., negM1..negMN]
                chunks = torch.chunk(outputs['ranking_outputs'],
                                     args.negative_sample_size + 1, dim=0)
                pos = chunks[0]
                npos = pos.size(0)
                if args.ranking_loss == 'max-margin':
                    ranking_loss = 0
                    for neg in chunks[1:]:
                        ranking_loss += \
                            ranking_lossfn(pos, neg, batch.labels[:npos])
                    loss += ranking_loss / args.negative_sample_size
                elif args.ranking_loss == 'negative-sampling':
                    neg = outputs['ranking_outputs'][pos.size(0):]
                    loss += ranking_lossfn(pos, neg)

            # add regularization
            if 'regularization' in outputs:
                loss = loss + outputs['regularization']

            # account for gradient accumulation steps, since the sum of mean
            # losses over sub-batches is *not* the same value as the mean loss
            # over the total batch (it is larger by a factor of
            # `args.gradient_steps`)
            loss = loss / args.gradient_steps

            # backward pass
            t0 = time.time()
            if args.use_accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()
            cumul_bwd_time += time.time() - t0

            # aggregate cumulative loss
            cumul_loss += loss.item()

            # perform optimizer step every `args.gradient_steps` steps
            if (global_step + 1) % args.gradient_steps == 0:
                if args.grad_norm is not None:
                    parameters = [param for param in model.parameters()
                                  if param.requires_grad]
                    if args.use_accelerate:
                        accelerator.clip_grad_norm_(parameters, args.grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(parameters,
                                                       args.grad_norm)
                optimizer.step()
                if args.linear_warmup is not None:
                    scheduler.step()
                optimizer.zero_grad()

            # print current iteration and training loss for current batch
            if global_step % (args.log_steps * args.gradient_steps) == 0 \
                and is_main_process:
                cumul_loss /= args.log_steps * args.gradient_steps
                cumul_fwd_time /= args.log_steps * args.gradient_steps
                cumul_bwd_time /= args.log_steps * args.gradient_steps
                curr_lr = optimizer.param_groups[0]['lr']
                logger.info(f'global step {global_step:d}, '
                            f'epoch step {step + 1:d} / {len(loader):d}, '
                            f'loss {cumul_loss:.5f}')
                writer.add_scalar('loss/train', cumul_loss, global_step)
                writer.add_scalar('time/forward', cumul_fwd_time, global_step)
                writer.add_scalar('time/backward', cumul_bwd_time, global_step)
                writer.add_scalar('learning_rate', curr_lr, global_step)
                cumul_loss = 0

            # evaluate on validation set
            if global_step % (args.valid_steps * args.gradient_steps) == 0 \
                and global_step > 0 and is_main_process:
                logger.info('evaluating on validation set')
                if args.use_accelerate:
                    unwrapped_model = \
                        accelerator.unwrap_model(model).to(device)
                else:
                    unwrapped_model = model

                if not missing_negatives:
                    metrics = evaluate_ranking(unwrapped_model, tokenizer,
                                               valid_triples,
                                               args.info_filename, device,
                                               **evaluate_kwargs)
                else:
                    metrics = evaluate_variable_negatives(unwrapped_model,
                                                          tokenizer,
                                                          valid_triples,
                                                          args.info_filename,
                                                          device,
                                                          negatives['valid'],
                                                          **evaluate_kwargs)
                metrics = metrics['metrics']

                for key, value in metrics.items():
                    if 'list' in key:
                        key, _ = key.split('_')
                    value = value.mean().item()
                    logger.info(f'validation {key}: {value:.6f}')
                    writer.add_scalar(f'metrics/valid_{key}', value,
                                      global_step)

                # always save model after evaluation
                logger.info('saving checkpoint')
                state_dict = unwrapped_model.state_dict()
                torch.save({'model' : state_dict,
                            'optimizer' : optimizer.state_dict(),
                            'step' : global_step + 1},
                           os.path.join(savedir, 'checkpoint.pt'))

                # keep track of best metric, save model if current is best
                if args.save_metric is not None:
                    current_value = metrics[args.save_metric].mean().item()
                    if current_value > best_value:
                        logger.info('saving best model')
                        best_value = current_value
                        torch.save({'model' : state_dict,
                                    'step' : global_step + 1},
                                   os.path.join(savedir, 'best_model.pt'))

                        # save best metrics
                        metrics = {k : v.mean().item()
                                   for k, v in metrics.items()}
                        metrics_fname = os.path.join(savedir,
                                                     'best_metrics.json')
                        with open(metrics_fname, 'w') as f:
                            json.dump(metrics, f, indent=4)

            if args.max_steps is not None and global_step >= args.max_steps:
                if is_main_process:
                    logger.info('reached max steps, exiting training')
                max_steps_reached = True
                break

    if is_main_process:
        logger.info(f'training completed in {datetime.now() - start}')

        # save final checkpoint
        logger.info('saving final checkpoint')
        if args.use_accelerate:
            unwrapped_model = accelerator.unwrap_model(model).to(device)
        else:
            unwrapped_model = model
        state_dict = unwrapped_model.state_dict()
        torch.save({'model' : state_dict,
                    'optimizer' : optimizer.state_dict(),
                    'step' : global_step + 1},
                   os.path.join(savedir, 'checkpoint.pt'))

        # load saved best model, if keeping track of best metric
        if args.save_metric is not None:
            logger.info('loading saved best model')
            savename = os.path.join(savedir, 'best_model.pt')
            unwrapped_model.load_state_dict(torch.load(savename)['model'],
                                            strict=False)

        # evaluate on test set
        logger.info('evaluating on test set')

        if not missing_negatives:
            metrics = evaluate_ranking(unwrapped_model, tokenizer,
                                       test_triples, args.info_filename,
                                       device, **evaluate_kwargs)
        else:
            metrics = evaluate_variable_negatives(unwrapped_model, tokenizer,
                                                  test_triples,
                                                  args.info_filename,
                                                  device, negatives['test'],
                                                  **evaluate_kwargs)
        metrics = metrics['metrics']

        for key, value in metrics.items():
            if 'list' in key:
                key, _ = key.split('_')
            value = value.mean().item()
            logger.info(f'test {key}: {value:.6f}')
            writer.add_scalar(f'metrics/test_{key}', value, global_step)

        # close tensorboard writer
        writer.close()

        # write valid file to indicate that model was saved properly
        with open(valid_fname, 'w'):
            os.utime(valid_fname, None)


if __name__ == '__main__':
    main(parse_args())
