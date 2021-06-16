""" script to train MLP to output softmax-normalized weights for scores from
    two different models for KG completion on a dataset using features of each
    positive triple as inputs """

import json
import os
import sys

from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from ogb.linkproppred import Evaluator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from features import compute_features
from preprocess import add_node_offsets


### command-line arguments

parser = ArgumentParser(description='train adaptive weighting model to output '
                                    'averaging weights for combining scores '
                                    'of two different models on KG dataset')
parser.add_argument('model_name1', type=str, help='name of first model')
parser.add_argument('scores_fname1', type=str,
                    help='precomputed scores of first KG completion model')
parser.add_argument('model_name2', type=str, help='name of second model')
parser.add_argument('scores_fname2', type=str,
                    help='precomputed scores of second KG completion model')
parser.add_argument('--dataset', type=str, default='repodb',
                    help='name of KG completion dataset to use')
parser.add_argument('--info-file', type=str,
                    default='data/processed/repodb.tsv',
                    help='info file with KG entity metadata')
parser.add_argument('--subgraph', type=str, required=True,
                    help='precomputed subgraph of to use')
parser.add_argument('--tokens-file', type=str, default=None,
                    help='file containing pre-tokenized entity text')
parser.add_argument('--metric', type=str, default='mrr_list',
                    choices=['mrr_list', 'hits@1_list', 'hits@3_list',
                             'hits@10_list'],
                    help='which metric to use')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--save', action='store_true', help='save results')
parser.add_argument('--resdir', type=str, default='results',
                    help='global directory for saved results')
parser.add_argument('--outdir', type=str, default='adaptive-weighting',
                    help='inner directory for saved results')

# MLP model arguments
parser.add_argument('--num-layers', type=int, default=3,
                    help='number of layers for MLP')
parser.add_argument('--hidden-channels', type=int, default=1024,
                    help='number of nodes in each MLP hidden layer')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout rate for MLP')

# MLP training arguments
parser.add_argument('--fractrain', type=float, default=0.9,
                    help='fraction of validation triples to use for training')
parser.add_argument('--negative-samples', type=int, default=1,
                    help='number of negative samples per positive to use')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs of training to perform')
parser.add_argument('--batch-size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of workers for training data loader')
parser.add_argument('--margin', type=float, default=1,
                    help='margin for max-margin ranking loss function')
parser.add_argument('--tol', type=float, default=1e-4,
                    help='absolute training loss tolerance for early stopping')
parser.add_argument('--patience', type=int, default=10,
                    help='number of epochs of patience for early stopping')


class MLP(torch.nn.Module):
    """ class that defines a simple multilayer perception with NUM_LAYERS
        layers (minimum three), IN_CHANNELS input dimension, HIDDEN_CHANNELS
        hidden dimension, and OUT_CHANNELS output dimension, with ReLU
        nonlinearity and DROPOUT dropout rate. """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.dropouts.append(torch.nn.Dropout(p=dropout))
        for _ in range(num_layers - 3):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.dropouts.append(torch.nn.Dropout(p=dropout))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin, dropout in zip(self.lins[:-1], self.dropouts):
            x = lin(x)
            x = torch.relu(x)
            x = dropout(x)
        x = self.lins[-1](x)
        return x


class AveragingBatch:
    """ wrapper class for batch of examples for training MLP """

    def __init__(self, pos1, pos2, neg1, neg2, features):
        self.pos1 = pos1
        self.pos2 = pos2
        self.neg1 = neg1
        self.neg2 = neg2
        self.features = features

    def pin_memory(self):
        """ copy tensors to pinned memory """
        self.pos1 = self.pos1.pin_memory()
        self.pos2 = self.pos2.pin_memory()
        self.neg1 = self.neg1.pin_memory()
        self.neg2 = self.neg2.pin_memory()
        self.features = self.features.pin_memory()
        return self

    def to(self, device):
        """ copy all tensors to device """
        self.pos1 = self.pos1.to(device)
        self.pos2 = self.pos2.to(device)
        self.neg1 = self.neg1.to(device)
        self.neg2 = self.neg2.to(device)
        self.features = self.features.to(device)
        return self


class AveragingDataset(Dataset):
    """ dataset used to train MLP to output average ensembling weights """

    def __init__(self, triples, scores1, scores2, info_file, num_nodes,
        negative_samples=1, tokens_fname=None):

        self.scores1 = scores1
        self.scores2 = scores2
        self.total_negatives = scores1['y_pred_neg_head'].size(1)
        self.negative_samples = negative_samples

        # construct features
        features = compute_features(triples, info_file, num_nodes,
                                    tokens_fname=tokens_fname)

        # add positive ranking score from each method to features
        y_pos1 = self.scores1['y_pred_pos'].numpy()[:,None]
        y_pos2 = self.scores2['y_pred_pos'].numpy()[:,None]
        features = np.concatenate((features, y_pos1, y_pos2), axis=1)

        features -= np.mean(features, axis=0, keepdims=True)
        features /= (np.std(features, axis=0, keepdims=True) + 1e-8)
        self.features = torch.from_numpy(features).float()

    def __getitem__(self, idx):
        neg_key = 'y_pred_neg_head' if idx // 2 == 0 else 'y_pred_neg_tail'
        idx = idx // 2
        indices = torch.randperm(self.total_negatives)[:self.negative_samples]

        # scores for positive triple from each model
        pos1 = self.scores1['y_pred_pos'][idx]
        pos2 = self.scores2['y_pred_pos'][idx]

        # scores for negative triples from each model
        neg1 = self.scores1[neg_key][idx, indices]
        neg2 = self.scores2[neg_key][idx, indices]

        return {'pos1' : pos1.reshape((1, 1)),
                'pos2' : pos2.reshape((1, 1)),
                'neg1' : neg1.unsqueeze(0),
                'neg2' : neg2.unsqueeze(0),
                'features' : self.features[idx].unsqueeze(0)}

    def __len__(self):
        return 2 * len(self.scores1['y_pred_pos'])

    @staticmethod
    def collate_fn(data):
        pos1 = torch.cat([elem['pos1'] for elem in data])
        pos2 = torch.cat([elem['pos2'] for elem in data])
        neg1 = torch.cat([elem['neg1'] for elem in data])
        neg2 = torch.cat([elem['neg2'] for elem in data])
        features = torch.cat([elem['features'] for elem in data])
        return AveragingBatch(pos1, pos2, neg1, neg2, features)


class AveragingModel(torch.nn.Module):
    """ MLP that outputs softmax-normalized weights for weighted averaging
        of ranking scores for two KG completion models """

    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(AveragingModel, self).__init__()
        self.mlp = MLP(in_channels, hidden_channels, 2, num_layers, dropout)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs):
        return self.softmax(self.mlp(inputs))

    def reset_parameters(self):
        self.mlp.reset_parameters()


def train_epoch(model, loader, optimizer, epoch, total_epochs, device='cpu',
    margin=1):
    """ perform one epoch of training """

    model.train()
    criterion = torch.nn.MarginRankingLoss(margin=margin)
    epoch_loss = 0
    epoch_desc = f'Epoch {epoch + 1} / {total_epochs}'
    progbar = tqdm(loader, desc=epoch_desc, ncols=80)

    for step, batch in enumerate(progbar):

        batch = batch.to(device)
        weights = model(batch.features)

        # compute weighted average of model scores
        weight1, weight2 = torch.chunk(weights, 2, dim=1)
        scores_pos = weight1 * batch.pos1 + weight2 * batch.pos2
        scores_neg = weight1 * batch.neg1 + weight2 * batch.neg2

        # compute loss
        labels = torch.ones_like(scores_pos)
        loss = criterion(scores_pos, scores_neg, labels)

        # perform backward pass, take optimizer step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # update progress bar description with cumulative loss
        epoch_loss += loss.item()
        desc = epoch_desc + f' (loss = {epoch_loss / (step + 1):.3f})'
        progbar.set_description(desc)

    return epoch_loss / len(loader)


def evaluate(model, loader, device='cpu', return_weights=False):
    """ run ranking evaluation """

    model.eval()
    metrics = defaultdict(list)
    evaluator = Evaluator(name='ogbl-biokg')

    # keep track of all weights
    if return_weights:
        weights_list = list()

    for batch in tqdm(loader, desc='Evaluating', ncols=80):

        batch = batch.to(device)
        weights = model(batch.features)

        # add weights for current batch to list
        if return_weights:
            weights_list.append(weights.detach().cpu())

        # compute weighted average of model scores
        weight1, weight2 = torch.chunk(weights, 2, dim=1)
        scores_pos = weight1 * batch.pos1 + weight2 * batch.pos2
        scores_neg = weight1 * batch.neg1 + weight2 * batch.neg2

        # run evaluation, aggregate scores
        scores = {'y_pred_pos' : scores_pos.flatten().cpu(),
                  'y_pred_neg' : scores_neg.cpu()}
        metrics_batch = evaluator.eval(scores)

        for key, value in metrics_batch.items():
            metrics[key].append(value)

    # concatenate metrics into tensors
    for key, value in metrics.items():
        metrics[key] = torch.cat(value)

    # concatenate weights, reorder to have head-batch first
    if return_weights:
        weights = torch.cat(weights_list)
        head_idx = torch.arange(0, weights.size(0), 2)
        tail_idx = torch.arange(1, weights.size(0), 2)
        weights = torch.cat([weights[head_idx], weights[tail_idx]])

    if return_weights:
        return metrics, weights
    return metrics


def main(args):
    """ main script to train and evaluate adaptive weighting model """

    print(args)

    # make output directory, check for saved result
    if args.save:
        modelsdir1 = args.model_name1.replace('-', '').lower()
        modelsdir2 = args.model_name2.replace('-', '').lower()
        modelsdir = f'{modelsdir1}-{modelsdir2}'
        if args.subgraph is not None:
            subgraphdir, _ = os.path.splitext(os.path.basename(args.subgraph))
        else:
            subgraphdir = 'fullgraph'
        savedir = os.path.join(args.resdir, args.dataset, subgraphdir,
                               args.outdir, modelsdir)
        os.makedirs(savedir, exist_ok=True)
        savename = f'adawt-nl{args.num_layers}-nh{args.hidden_channels}-'
        savename += f'p{args.dropout:.0e}-neg{args.negative_samples}-'
        savename += f'e{args.epochs:02d}-b{args.batch_size}-lr{args.lr:.0e}'
        savename += f'-seed{args.seed}.json'
        if os.path.exists(os.path.join(savedir, savename)):
            print('found saved result, exiting...')
            sys.exit()

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # check if info_file is correct for dataset
    if args.dataset not in args.info_file:
        raise ValueError(f'info_file is incorrect: {args.info_file} does not '
                         f'contain {args.dataset}')

    # load edge split
    if args.subgraph is not None:
        split_edge = torch.load(args.subgraph)
    else:
        split_edge = dataset.get_edge_split()

    # load entity_dict
    entity_dict = split_edge['entity_dict']

    # set number of entities
    num_nodes = split_edge['num_nodes']

    # add offsets to nodes in validation and test triples
    valid_triples = add_node_offsets(split_edge['valid'], entity_dict)
    test_triples = add_node_offsets(split_edge['test'], entity_dict)

    # split validation set triples into training and evaluation sets
    nvalid = len(valid_triples['head'])
    num_train = int(args.fractrain * nvalid)
    permutation = np.random.permutation(nvalid)
    train_idx, eval_idx = permutation[:num_train], permutation[num_train:]
    train_triples, eval_triples = dict(), dict()
    for key, value in valid_triples.items():
        if isinstance(value, list):
            train_triples[key] = [value[i] for i in train_idx]
            if args.fractrain < 1:
                eval_triples[key] = [value[i] for i in eval_idx]
        elif isinstance(value, np.ndarray):
            train_triples[key] = valid_triples[key][train_idx]
            if args.fractrain < 1:
                eval_triples[key] = valid_triples[key][eval_idx]

    # load metadata file
    print('loading metadata...')
    info_file = pd.read_table(args.info_file, index_col=0, na_filter=False)

    # load precomputed scores for both models
    print('loading precomputed scores...')
    scores = {'model1' : dict(), 'model2' : dict()}

    scores1 = torch.load(args.scores_fname1)
    scores['model1']['valid'] = scores1['valid']['scores']
    scores['model1']['test'] = scores1['test']['scores']

    scores2 = torch.load(args.scores_fname2)
    scores['model2']['valid'] = scores2['valid']['scores']
    scores['model2']['test'] = scores2['test']['scores']

    # split precomputed scores according to training/evaluation splits
    for method in ('model1', 'model2'):
        scores[method]['train'] = dict()
        scores[method]['eval'] = dict()
        for key, value in scores[method]['valid'].items():
            scores[method]['train'][key] = value[train_idx]
            scores[method]['eval'][key] = value[eval_idx]

    # construct training dataset and dataloader for MLP
    print('setting up training dataset and data loader...')
    train_dataset = AveragingDataset(train_triples, scores['model1']['train'],
                                     scores['model2']['train'], info_file,
                                     num_nodes,
                                     negative_samples=args.negative_samples,
                                     tokens_fname=args.tokens_file)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              pin_memory=True, num_workers=args.num_workers,
                              collate_fn=AveragingDataset.collate_fn)

    # construct evaluation dataset and dataloader for MLP
    if args.fractrain < 1:
        print('setting up evaluation dataset and data loader...')
        eval_dataset = AveragingDataset(eval_triples, scores['model1']['eval'],
                                        scores['model2']['eval'], info_file,
                                        num_nodes,
                                        negative_samples=500,
                                        tokens_fname=args.tokens_file)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                                 pin_memory=True, num_workers=args.num_workers,
                                 collate_fn=AveragingDataset.collate_fn)

    # initialize model, optimizer, loss function
    print('setting up model and optimizer...')
    in_channels = train_dataset.features.size(1)
    model = AveragingModel(in_channels, args.hidden_channels, args.num_layers,
                           args.dropout).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)

    # set up best training loss and patience steps
    best_loss, patience_steps = np.inf, 0

    # training loop
    print('starting training...')
    model.train()
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, epoch, args.epochs,
                           device=device, margin=args.margin)

        # check for best loss, increment patience steps if loss not improved
        if loss - best_loss > args.tol:
            patience_steps += 1
        else:
            patience_steps = 0
        if loss < best_loss:
            best_loss = loss

        # print validation metrics, losses, and patience steps
        if args.fractrain < 1:
            metrics = evaluate(model, eval_loader, device=device)
            for metric, value in metrics.items():
                print(f'validation {metric}: {value.mean().item()}')
        print(f'epoch loss: {loss:.6f}')
        print(f'best loss: {best_loss:.6f}')
        print(f'patience steps: {patience_steps}')
        print('---')

        # stop early based on patience steps
        if patience_steps > args.patience:
            print('stopping early')
            break

    # final metrics on evaluation set
    if args.fractrain < 1:
        eval_metrics = metrics

    # construct test dataset and dataloader for MLP
    print('setting up test dataset and data loader...')
    dataset = AveragingDataset(test_triples, scores['model1']['test'],
                               scores['model2']['test'], info_file, num_nodes,
                               negative_samples=500,
                               tokens_fname=args.tokens_file)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        pin_memory=True, num_workers=args.num_workers,
                        collate_fn=AveragingDataset.collate_fn)

    # evaluate on test set
    print('starting evaluation...')
    test_metrics, weights = evaluate(model, loader, device=device,
                                     return_weights=True)

    # print final metrics
    print('---')
    for metric, value in test_metrics.items():
        print(f'test {metric}: {value.mean().item()}')

    # save result, if specified
    if args.save:
        result = {'args' : vars(args)}
        if args.fractrain < 1:
            result['eval_metrics'] = {key : value.mean().item()
                                      for key, value in eval_metrics.items()}
        result['test_metrics'] = {key : value.mean().item()
                                  for key, value in test_metrics.items()}
        with open(os.path.join(savedir, savename), 'w') as f:
            json.dump(result, f, indent=4)

if __name__ == '__main__':
    main(parser.parse_args())

