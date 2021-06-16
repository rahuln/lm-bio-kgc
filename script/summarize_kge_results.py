""" script to summarize KGE model experiment results in a given directory """


from argparse import ArgumentParser
from glob import glob
import json
import os
import sys

import pandas as pd


# command-line arguments
parser = ArgumentParser(description='summarize KGE model experiment results')
parser.add_argument('dirname', type=str,
                    help='location of all experiment results'
                         '(e.g., subgraph subdirectory)')
parser.add_argument('--save', action='store_true',
                    help='save results to .csv file')

if __name__ == '__main__':

    args = parser.parse_args()

    # check for saved summary, print results and exit if it exists
    savename = os.path.join(args.dirname, 'summary.csv')
    if os.path.exists(savename):
        print('Found saved summary file')
        df = pd.read_csv(savename, index_col=0)

        # print best result for each model
        df_grp = df.groupby('model')
        df_best = df.iloc[df_grp['valid_mrr'].idxmax()]
        print('Best results for each model (by validation set MRR):')
        print('=' * 50)
        subset = ['model', 'hidden_dim', 'gamma', 'learning_rate',
                  'negative_sample_size', 'loss_function', 'regularization',
                  'test_mrr', 'test_hits@1', 'test_hits@3', 'test_hits@10']
        print(df_best[subset])

        sys.exit()

    # load all config files
    files = glob(os.path.join(args.dirname, '**', 'config.json'),
                 recursive=True)

    # set up columns for final data frame
    columns = ['model', 'hidden_dim', 'gamma', 'learning_rate',
               'negative_sample_size', 'loss_function', 'regularization',
               'valid_mrr', 'valid_hits@1', 'valid_hits@3', 'valid_hits@10',
               'test_mrr', 'test_hits@1', 'test_hits@3', 'test_hits@10']

    # loop through files, retrieving necessary elements
    tuples = list()
    for fname in files:

        # load config file
        dirname = os.path.dirname(fname)
        with open(fname, 'r') as f:
            config = json.load(f)

        # keep track of model parameters
        model = config['model']
        hidden_dim = config['hidden_dim']
        gamma = config['gamma']
        learning_rate = config['learning_rate']
        negative_sample_size = config['negative_sample_size']
        loss_function = config['loss_function']
        regularization = config['regularization']
        tup = (model, hidden_dim, gamma, learning_rate, negative_sample_size,
               loss_function, regularization)

        # set up subgraph name
        subgraph = config['subgraph']
        if subgraph is not None:
            subgraphname, _ = os.path.splitext(os.path.basename(subgraph))
        else:
            subgraphname = None

        for subset in ('valid', 'test'):

            # set up metrics filename
            metrics_fname = f'{subset}_metrics'
            if subgraphname is not None:
                metrics_fname += f'_{subgraphname}'
            metrics_fname += '.json'
            metrics_fname = os.path.join(dirname, metrics_fname)

            # load metrics file
            try:
                with open(metrics_fname, 'r') as f:
                    metrics = json.load(f)
            except:
                continue

            # keep track of metrics
            mrr = metrics['mrr_list']
            hits1 = metrics['hits@1_list']
            hits3 = metrics['hits@3_list']
            hits10 = metrics['hits@10_list']
            tup = tup + (mrr, hits1, hits3, hits10)

        tuples.append(tup)

    # create data frame and save to specified directory
    df = pd.DataFrame(tuples, columns=columns)
    if args.save:
        df.to_csv(savename)

    # print best result for each model
    df_grp = df.groupby('model')
    df_best = df.iloc[df_grp['valid_mrr'].idxmax()]
    print('Best results for each model (by validation set MRR):')
    print('=' * 50)
    subset = ['model', 'hidden_dim', 'gamma', 'learning_rate',
              'negative_sample_size', 'loss_function', 'regularization',
              'test_mrr', 'test_hits@1', 'test_hits@3', 'test_hits@10']
    print(df_best[subset])

