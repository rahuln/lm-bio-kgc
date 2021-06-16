""" script to train a router model to try to classify which of two different
    models will predict a higher score for each triple in the validation set
    of a specified KG, using a set of node-based and text-based features """

from argparse import ArgumentParser
from collections import defaultdict
from itertools import product
import json
import os
import pickle
import sys

import numpy as np
from ogb.linkproppred import Evaluator
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import torch
from xgboost import XGBClassifier

from features import compute_features
from preprocess import add_node_offsets


### command-line arguments

parser = ArgumentParser(description='train router model to predict which of '
                                    'two different KG completion models will '
                                    'score higher on a KG validation set')
parser.add_argument('--model-names', type=str, default=None, nargs='+',
                    help='ordered names of models to ensemble')
parser.add_argument('--scores-fnames', type=str, default=None, nargs='+',
                    help='filenames of precomputed scores, in same order as '
                         'model names')
parser.add_argument('--dataset', type=str, default='repodb',
                    help='name of KG completion dataset to use')
parser.add_argument('--info-file', type=str,
                    default='data/processed/repodb.tsv',
                    help='info file with KG metadata')
parser.add_argument('--subgraph', type=str, required=True,
                    help='precomputed subgraph of KG to use')
parser.add_argument('--num-neg-samples', type=int, default=None,
                    help='number of negative samples to use in ranking')
parser.add_argument('--classifier', type=str, default='xgboost',
                    choices=['xgboost', 'logreg', 'mlp', 'decisiontree',
                             'average'],
                    help='choice of classifier to use')
parser.add_argument('--ensemble-method', type=str, default='router',
                    choices=['router', 'average'],
                    help='use classifier as router or use predicted '
                         'probabilities for weighted average of scores')
parser.add_argument('--fractrain', type=float, default=1.,
                    help='fraction of validation set used to train router')
parser.add_argument('--tokens-file', type=str, default=None,
                    help='file containing pre-tokenized entity text')
parser.add_argument('--metric', type=str, default='mrr_list',
                    choices=['mrr_list', 'hits@1_list', 'hits@3_list',
                             'hits@10_list'],
                    help='which metric to use')
parser.add_argument('--verbose', action='store_true', help='verbose printing')
parser.add_argument('--save', action='store_true', help='save results to file')
parser.add_argument('--resdir', type=str, default='results',
                    help='global output directory for saved results')
parser.add_argument('--outdir', type=str, default='ensemble',
                    help='inner output directory for saved results')
parser.add_argument('--return-classifiers', action='store_true',
                    help='return trained classifiers with results')
parser.add_argument('--return-features', action='store_true',
                    help='return features with results')
parser.add_argument('--no-features', action='store_true',
                    help='do not use calculated features')
parser.add_argument('--no-scores', action='store_true',
                    help='do not use ranking scores as features')
parser.add_argument('--save-classifier', action='store_true',
                    help='save trained classifiers')
parser.add_argument('--seed', type=int, default=42, help='random seed')

# logistic regression hyperparameters
parser.add_argument('--penalty', type=str, default='l2', choices=['l1', 'l2'],
                    help='penalty for logistic regression')
parser.add_argument('--C', type=float, default=1.,
                    help='regularization parameter for penalty')

# xgboost hyperparameters
parser.add_argument('--n_estimators', type=int, default=100,
                    help='number of boosting rounds')
parser.add_argument('--max_depth', type=int, default=None,
                    help='maximum tree depth for base learners')
parser.add_argument('--learning_rate', type=float, default=None,
                    help='boosting learning rate')
parser.add_argument('--reg_alpha', type=float, default=None,
                    help='L1 regularization term on weights')
parser.add_argument('--reg_beta', type=float, default=None,
                    help='L2 regularization term on weights')
parser.add_argument('--importance_type', type=str, default='gain',
                    help='type for features importances')

# mlp hyperparameters
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of hidden layers')
parser.add_argument('--nhidden', type=int, default=100,
                    help='dimension of each hidden layer')
parser.add_argument('--alpha', type=float, default=0.0001,
                    help='L2 penalty regularization parameter')
parser.add_argument('--batch_size', type=int, default=200,
                    help='batch size for optimization')
parser.add_argument('--learning_rate_init', type=float, default=0.001,
                    help='initial learning rate for optimizer')
parser.add_argument('--max_iter', type=int, default=200,
                    help='number of epochs of training')


def simple_ensemble(scores, metric, fractrain, model_names):
    """ simple ensemble that computes a convex combination of multiple models'
        scores for ranking, with weights of the combination learned from the
        validation set """

    evaluator = Evaluator(name='ogbl-biokg')

    # choose subset of validation set indices
    num_scores = len(scores[model_names[0]]['valid']['scores']['y_pred_pos'])
    perm = np.random.permutation(num_scores)
    ntrain = int(fractrain * len(perm))
    inds = perm[:ntrain].tolist()

    # retrieve validation set scores for both models
    train = dict()
    for model_name in model_names:
        train[model_name] = dict()
        valid_scores = scores[model_name]['valid']['scores']
        train[model_name] = \
            {key : value[inds] for key, value in valid_scores.items()}

    # initialize grid of convex combinations of weights
    alphas = np.linspace(0.05, 0.95, 19)
    alphas_list = [alphas for _ in range(len(model_names))]
    grid = list(product(*alphas_list))
    grid = [elem for elem in grid if np.sum(elem) == 1.]
    results = list()

    for weights in grid:

        # compute weighted combination of scores
        pos = weights[0] * train[model_names[0]]['y_pred_pos']
        neg_head = weights[0] * train[model_names[0]]['y_pred_neg_head']
        neg_tail = weights[0] * train[model_names[0]]['y_pred_neg_tail']
        for i, model_name in enumerate(model_names):
            if i == 0: continue
            pos += weights[i] * train[model_name]['y_pred_pos']
            neg_head += weights[i] * train[model_name]['y_pred_neg_head']
            neg_tail += weights[i] * train[model_name]['y_pred_neg_tail']

        # compute metric for this set of weights
        pos, neg = torch.cat([pos, pos]), torch.cat([neg_head, neg_tail])
        values = {'y_pred_pos' : pos, 'y_pred_neg' : neg}
        results.append(evaluator.eval(values)[metric].mean().item())

    # find best weights
    weights = grid[np.argmax(results)]

    # compute weighted combination of scores for test examples
    test = {model_name : scores[model_name]['test']['scores']
            for model_name in model_names}
    pos = weights[0] * test[model_names[0]]['y_pred_pos']
    neg_head = weights[0] * test[model_names[0]]['y_pred_neg_head']
    neg_tail = weights[0] * test[model_names[0]]['y_pred_neg_tail']
    for i, model_name in enumerate(model_names):
        if i == 0: continue
        pos += weights[i] * test[model_name]['y_pred_pos']
        neg_head += weights[i] * test[model_name]['y_pred_neg_head']
        neg_tail += weights[i] * test[model_name]['y_pred_neg_tail']

    # compute metric for best weights
    pos, neg = torch.cat([pos, pos]), torch.cat([neg_head, neg_tail])
    values = {'y_pred_pos' : pos, 'y_pred_neg' : neg}
    metrics = evaluator.eval(values)
    final_metrics = {key : value.mean().item()
                     for key, value in metrics.items()}

    return final_metrics, weights


def main(args):
    """ fit routers to validation set to predict which of two models will have
        a higher metric for each example, then apply to the test set to boost
        inference speed/performance """

    # check to make sure some set of features are being used
    if args.no_features and args.no_scores:
        raise ValueError('must use either features or scores for ensembling')

    # set n_estimators for decision tree
    if args.classifier == 'decisiontree':
        args.n_estimators = 1

    if args.save:

        # construct output directory
        if args.subgraph is None:
            subgraphdir = 'fullgraph'
        else:
            subgraphdir, _ = os.path.splitext(os.path.basename(args.subgraph))

        # construct name for results file
        methodsdir = '-'.join(sorted(args.model_names)).lower()
        expdir = f'{args.classifier}'
        if args.classifier != 'average':
            expdir += f'-{args.ensemble_method}'

        # add hyperparameters to directory name
        if args.classifier == 'logreg':
            expdir += f'-pen{args.penalty}-C{args.C:.0e}'
        elif args.classifier in ('xgboost', 'decisiontree'):
            if args.classifier == 'xgboost':
                expdir += f'-ne{args.n_estimators}'
            if args.max_depth is not None:
                expdir += f'-md{args.max_depth}'
            if args.learning_rate is not None:
                expdir += f'-lr{args.learning_rate:.0e}'
            if args.reg_alpha is not None:
                expdir += f'-a{args.reg_alpha}'
            if args.reg_beta is not None:
                expdir += f'-b{args.reg_beta}'
        elif args.classifier == 'mlp':
            expdir += f'-nl{args.nlayers}-nh{args.nhidden}-a{args.alpha:.0e}'
            expdir += f'-bsz{args.batch_size}-lr{args.learning_rate_init:.0e}'
            expdir += f'-ne{args.max_iter}'

        expdir += '-tok' if args.tokens_file is not None else ''
        expdir += '-nofeat' if args.no_features else ''
        expdir += '-noscores' if args.no_scores else ''
        expdir += f'-s{args.seed}'

        savedir = os.path.join(args.resdir, args.dataset, subgraphdir,
                               args.outdir, methodsdir, expdir)

        # check for saved result
        if os.path.exists(os.path.join(savedir, 'results.json')):
            print('found saved result, exiting...')
            sys.exit()

        os.makedirs(savedir, exist_ok=True)

        # save args to config file
        with open(os.path.join(savedir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    # dictionary for saved results (if --save is set)
    results = dict()

    if args.verbose:
        print(args)

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

    # set number of entities and relations
    num_nodes = split_edge['num_nodes']
    nrelation = int(split_edge['train']['relation'].max()) + 1

    # set number of negative validation/test samples to use
    if args.num_neg_samples is None:
        args.num_neg_samples = split_edge['valid']['head_neg'].shape[1]

    # preprocess validation set
    valid = add_node_offsets(split_edge['valid'], entity_dict)
    valid['head_neg'] = valid['head_neg'][:, :args.num_neg_samples]
    valid['tail_neg'] = valid['tail_neg'][:, :args.num_neg_samples]

    # preprocess test set
    test = add_node_offsets(split_edge['test'], entity_dict)
    test['head_neg'] = test['head_neg'][:, :args.num_neg_samples]
    test['tail_neg'] = test['tail_neg'][:, :args.num_neg_samples]

    # convert ndarrays in triples to pytorch tensors
    for key in ('head', 'head_neg', 'relation', 'tail', 'tail_neg'):
        valid[key] = torch.from_numpy(valid[key]).to(device)
        test[key] = torch.from_numpy(test[key]).to(device)

    # load metadata file
    if args.verbose:
        print('loading metadata...')
    df = pd.read_table(args.info_file, index_col=0, na_filter=False)

    # load precomputed scores and metrics for both models
    if args.verbose:
        print('loading scores...')

    # retrieve validation and test scores and metrics for all models
    # metrics dictionary format is: model name -> subset -> metric
    scores = dict()
    metrics = dict()
    for model_name, scores_fname in zip(args.model_names, args.scores_fnames):
        scores[model_name] = torch.load(scores_fname)
        metrics[model_name] = {
            'valid' : scores[model_name]['valid']['metrics'],
            'test' : scores[model_name]['test']['metrics']
        }

    # add averaged validation and test metrics for all models to results
    # results dictionary format is: metric -> subset -> model name
    for metric in metrics[args.model_names[0]]['valid']:
        results[metric] = {
            'valid' : {name : metrics[name]['valid'][metric].mean().item()
                       for name in args.model_names},
            'test' : {name : metrics[name]['test'][metric].mean().item()
                      for name in args.model_names}
        }

    # get labels for validation and test set
    valid_metrics = \
        torch.cat([metrics[model_name]['valid'][args.metric].unsqueeze(1)
                   for model_name in args.model_names], dim=1)
    best_model_valid = torch.argmax(valid_metrics, dim=1)
    same_idx_valid = (torch.std(valid_metrics, dim=1) == 0)
    best_model_valid[same_idx_valid] = len(args.model_names)

    test_metrics = \
        torch.cat([metrics[model_name]['test'][args.metric].unsqueeze(1)
                   for model_name in args.model_names], dim=1)
    best_model_test = torch.argmax(test_metrics, dim=1)
    same_idx_test = (torch.std(test_metrics, dim=1) == 0)
    best_model_test[same_idx_test] = len(args.model_names)

    # seperate labels by head and tail batches, store in dictionary
    nhead_valid = len(split_edge['valid']['head'])
    nhead_test = len(split_edge['test']['head'])
    labels = {
        'valid' : {
            'head' : best_model_valid[:nhead_valid].numpy(),
            'tail' : best_model_valid[nhead_valid:].numpy()
        },
        'test' : {
            'head' : best_model_test[:nhead_test].numpy(),
            'tail' : best_model_test[nhead_test:].numpy()
        }
    }

    # print some information about the metrics
    if args.verbose or args.save:
        results['fraction_best'] = {'valid' : dict(), 'test' : dict()}

        # validation set
        if args.verbose:
            print('validation set statistics:')
        for i, model_name in enumerate(args.model_names):
            frac = best_model_valid.eq(i).float().mean().item()
            results['fraction_best']['valid'][model_name] = frac
            if args.verbose:
                print(f'fraction where {model_name} is best: {frac:.4f}')
        same_label = len(args.model_names)
        frac = best_model_valid.eq(same_label).float().mean().item()
        results['fraction_best']['valid']['same'] = frac
        if args.verbose:
            print(f'fraction of examples where all models same: {frac:.4f}')
            print('=' * 20)

        # test set
        if args.verbose:
            print('test set statistics:')
        for i, model_name in enumerate(args.model_names):
            frac = best_model_test.eq(i).float().mean().item()
            results['fraction_best']['test'][model_name] = frac
            if args.verbose:
                print(f'fraction where {model_name} is best: {frac:.4f}')
        same_label = len(args.model_names)
        frac = best_model_test.eq(same_label).float().mean().item()
        results['fraction_best']['test']['same'] = frac
        if args.verbose:
            print(f'fraction of examples where all models same: {frac:.4f}')
            print('=' * 20)

    # construct features
    if args.verbose:
        print('constructing features...')
    features = dict()
    for subset, triples in zip(('valid', 'test'), (valid, test)):
        if args.verbose:
            print(f'  {subset}...')

        # move tensors to CPU
        for key, value in triples.items():
            if torch.is_tensor(value):
                triples[key] = value.cpu()

        # compute and standardize features
        if not args.no_features:
            feat = compute_features(triples, df, num_nodes,
                                    tokens_fname=args.tokens_file)

        # add positive ranking score from each method to features
        if not args.no_scores:
            pos_scores = list()
            for model_name in args.model_names:
                y_pred_pos = scores[model_name][subset]['scores']['y_pred_pos']
                pos_scores.append(y_pred_pos.unsqueeze(1))
            pos_scores = torch.cat(pos_scores, dim=1).numpy()

        # use either features, scores, or both
        if args.no_features:
            feat = pos_scores
        elif args.no_scores:
            pass
        else:
            feat = np.concatenate((feat, pos_scores), axis=1)

        feat -= np.mean(feat, axis=0, keepdims=True)
        feat /= (np.std(feat, axis=0, keepdims=True) + 1e-8)

        features[subset] = feat

    if args.verbose:
        print(f'using {features["test"].shape[1]} features')

    ### Test of router to improve metric

    # fit router to validation set, then use it to route models on test set
    # and compare to each model separately

    # print test set metric for each model
    if args.verbose:
        for i, model_name in enumerate(args.model_names):
            value = metrics[model_name]['test'][args.metric].mean().item()
            print(f'{model_name} test {args.metric}: {value:.4f}')

    # define test labels, calculate fraction of majority class
    test_labels = np.concatenate((labels['test']['head'],
                                  labels['test']['tail']))
    majority = np.max(np.bincount(test_labels)) / len(test_labels)
    results['majority_frac'] = majority

    # print majority class fraction
    if args.verbose:
        print(f'majority class fraction: {majority:.4f}')
        print('=' * 20)

    if args.classifier == 'average':
        # compute simple ensemble with weighted average
        metrics_router, weights = \
            simple_ensemble(scores, args.metric, args.fractrain,
                            args.model_names)
        if args.verbose:
            print(f'router {args.metric}: {metrics_router[args.metric]:.4f}')
            weights_str = ', '.join([f'{w:.3f}' for w in weights])
            print(f'weights: {weights_str}')
        results['weights'] = list(weights)

    elif args.classifier in ('logreg', 'xgboost', 'decisiontree', 'mlp'):
        # use router with graph- and text-based features
        if args.verbose:
            clf_names = {'logreg' : 'LogisticRegression',
                         'xgboost' : 'XGBClassifier',
                         'decisiontree' : 'XGBClassifier',
                         'mlp' : 'MLPClassifier'}
            print(f'using {clf_names[args.classifier]} as router...')
            print()

            header_fmt = '%15s%15s%15s'
            print(header_fmt % ('train frac', 'router acc', args.metric))
            msg_fmt = '%15.4f%15.4f%15.4f'

        # initialize classifiers
        clf = dict()
        if args.classifier == 'logreg':
            kwargs = {'max_iter' : 1000, 'class_weight' : 'balanced',
                      'penalty' : args.penalty, 'C' : args.C,
                      'solver' : 'saga'}
            clf['head'] = LogisticRegression(**kwargs)
            clf['tail'] = LogisticRegression(**kwargs)
        elif args.classifier in ('xgboost', 'decisiontree'):
            kwargs = {'n_estimators' : args.n_estimators,
                      'max_depth' : args.max_depth,
                      'learning_rate' : args.learning_rate,
                      'reg_alpha' : args.reg_alpha,
                      'reg_beta' : args.reg_beta,
                      'importance_type' : args.importance_type,
                      'verbosity' : 0}
            clf['head'] = XGBClassifier(**kwargs)
            clf['tail'] = XGBClassifier(**kwargs)
        elif args.classifier == 'mlp':
            kwargs = {'hidden_layer_sizes' : (args.nhidden,) * args.nlayers,
                      'alpha' : args.alpha, 'batch_size' : args.batch_size,
                      'learning_rate_init' : args.learning_rate_init,
                      'max_iter' : args.max_iter}
            clf['head'] = MLPClassifier(**kwargs)
            clf['tail'] = MLPClassifier(**kwargs)

        # construct random training subset indices
        train_idx = dict()
        for key in ('head', 'tail'):
            perm = np.random.permutation(len(labels['valid'][key]))
            ntrain = int(args.fractrain * len(labels['valid'][key]))
            train_idx[key] = perm[:ntrain].tolist()

        # fit models to validation subset, prediction on full test set
        predicted = list()
        proba = list()
        cv_scores = list()
        for key in ('head', 'tail'):
            X = features['valid'][train_idx[key]]
            y = labels['valid'][key][train_idx[key]]
            cv_scores.extend(cross_val_score(clf[key], X, y))
            clf[key].fit(X, y)
            Xt = features['test']
            predicted.extend(clf[key].predict(Xt).tolist())
            proba.append(clf[key].predict_proba(Xt))
        predicted = np.array(predicted)
        proba = softmax(np.concatenate(proba, axis=0)[:, :-1], axis=1)
        results['cross_val_scores'] = cv_scores
        results['cross_val_mean'] = np.mean(cv_scores)

        # calculate metric when ensembling both models
        if args.ensemble_method == 'router':
            metrics_cat = defaultdict(list)
            for i, model_name in enumerate(args.model_names):
                idx = (predicted == i)
                for metric in metrics[model_name]['test']:
                    values = metrics[model_name]['test'][metric][idx]
                    metrics_cat[metric].append(values)
            # if predicted that all models have same metric, use first model
            idx = (predicted == len(args.model_names))
            for metric in metrics[args.model_names[0]]['test']:
                values = metrics[args.model_names[0]]['test'][metric][idx]
                metrics_cat[metric].append(values)
            metrics_router = {key : torch.cat(value).mean().item()
                              for key, value in metrics_cat.items()}
        elif args.ensemble_method == 'average':

            # initialize evaluator
            evaluator = Evaluator(name='ogbl-biokg')

            # initialize weights from probabilities for head, tail batches
            proba = torch.from_numpy(proba)
            num_head = proba.size(0) // 2
            w_head, w_tail = proba[:num_head], proba[num_head:]

            # calculate weighted average of positive scores
            name = args.model_names[0]
            scores_model = scores[name]['test']['scores']['y_pred_pos']
            pos_head = w_head[:, 0] * scores_model
            pos_tail = w_tail[:, 0] * scores_model
            for i, name in enumerate(args.model_names):
                if i == 0: continue
                scores_model = scores[name]['test']['scores']['y_pred_pos']
                pos_head += w_head[:, i] * scores_model
                pos_tail += w_tail[:, i] * scores_model

            # calculate weighted average of negative scores
            name = args.model_names[0]
            scores_head = scores[name]['test']['scores']['y_pred_neg_head']
            scores_tail = scores[name]['test']['scores']['y_pred_neg_tail']
            neg_head = w_head[:, 0].unsqueeze(1) * scores_head
            neg_tail = w_tail[:, 0].unsqueeze(1) * scores_tail
            for i, name in enumerate(args.model_names):
                if i == 0: continue
                scores_head = scores[name]['test']['scores']['y_pred_neg_head']
                scores_tail = scores[name]['test']['scores']['y_pred_neg_tail']
                neg_head += w_head[:, i].unsqueeze(1) * scores_head
                neg_tail += w_tail[:, i].unsqueeze(1) * scores_tail

            # run evaluation with weighted average scores to get new metric
            pos = torch.cat([pos_head, pos_tail], dim=0)
            neg = torch.cat([neg_head, neg_tail], dim=0)
            metrics_avg = evaluator.eval({'y_pred_pos' : pos,
                                          'y_pred_neg' : neg})
            metrics_router = {key : value.mean().item()
                              for key, value in metrics_avg.items()}

        # print router accuracy and ranking metric
        accuracy = accuracy_score(test_labels, predicted)
        results['router_accuracy'] = accuracy
        if args.verbose:
            final_metric = metrics_router[args.metric]
            print(msg_fmt % (args.fractrain, accuracy, final_metric))

        # get classification report, save/print if specified
        report = classification_report(test_labels, predicted)
        if args.verbose:
            print()
            print('\n=== classification report: ===\n')
            print(report)
        if args.save:
            with open(os.path.join(savedir, 'clf_report.txt'), 'w') as f:
                f.writelines(report)

    # store final ensemble value of metrics
    for metric, value in metrics_router.items():
        results[metric]['test']['ensemble'] = value

    # print upper bound (i.e., with perfect router)
    idx = test_metrics.argmax(dim=1)
    for metric in metrics[args.model_names[0]]['test']:
        values = \
            torch.cat([metrics[model_name]['test'][metric].unsqueeze(1)
                       for model_name in args.model_names], dim=1)
        metric_best = values[torch.arange(values.size(0)), idx].mean().item()
        results[metric]['test']['best'] = metric_best
    if args.verbose:
        best_value = results[args.metric]['test']['best']
        print(f'Best possible {args.metric}: {best_value:.4f}')

    # save results if specified
    if args.save:
        with open(os.path.join(savedir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

    # save classifiers if specified
    if args.save_classifier:
        with open(os.path.join(savedir, 'clf_head.pkl'), 'wb') as f:
            pickle.dump(clf['head'], f)
        with open(os.path.join(savedir, 'clf_tail.pkl'), 'wb') as f:
            pickle.dump(clf['tail'], f)

    # add trained classifiers to results dictionary
    if args.return_classifiers:
        results['classifiers'] = clf

    # add computed features to results dictionary
    if args.return_features:
        results['features'] = features

    # return metrics for each method, router, and best possible
    return results


if __name__ == '__main__':
    main(parser.parse_args())
