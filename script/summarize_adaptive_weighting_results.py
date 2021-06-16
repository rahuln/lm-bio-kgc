""" script to summarize a set of results for the input-dependent adaptive
    weighted average ensemble trained using max-margin loss function """

from argparse import ArgumentParser
from glob import glob
import json
import os

import pandas as pd
from tqdm import tqdm


# command-line arguments
parser = ArgumentParser(description='summarize adaptive weighting results')
parser.add_argument('resdir', type=str, help='results directory')

def main(args):
    """ main script """

    # find all results files in directory
    files = glob(os.path.join(args.resdir, '**', '*.json'), recursive=True)
    print(f'found {len(files)} results files')

    # information to collect from each result
    arg_names = ['model_name1', 'model_name2', 'num_layers', 'hidden_channels',
                 'dropout', 'negative_samples', 'epochs', 'batch_size',
                 'lr', 'margin', 'tol', 'patience']
    metric_cols = ['eval_mrr', 'eval_hits1', 'eval_hits3', 'eval_hits10',
                   'test_mrr', 'test_hits1', 'test_hits3', 'test_hits10']
    metric_names = ['mrr_list', 'hits@1_list', 'hits@3_list', 'hits@10_list']
    tuples = list()

    # for each results files, collect hyperparameters and metrics
    for fname in tqdm(files, desc='processing results'):
        with open(fname, 'r') as f:
            res = json.load(f)
        entry = list()

        # collect hyperparameters
        args = res['args']
        for arg_name in arg_names:
            entry.append(args[arg_name])

        # collect evaluation and test set metrics
        for key in ('eval_metrics', 'test_metrics'):
            for metric in metric_names:
                entry.append(res[key][metric])

        tuples.append(tuple(entry))

    # construct data frame
    columns = arg_names + metric_cols
    df = pd.DataFrame(tuples, columns=columns)

    # find best result with and without KG-BERT
    print_col = ['model_name1', 'model_name2'] + metric_cols

    print('Best result without KG-BERT:')
    kge = df[df.model_name1 != 'KGBERT']
    print(kge.loc[kge.eval_mrr.idxmax()][print_col])

    print()

    print('Best result with KG-BERT:')
    kgbert = df[df.model_name1 == 'KGBERT']
    print(kgbert.loc[kgbert.eval_mrr.idxmax()][print_col])


if __name__ == '__main__':
    main(parser.parse_args())
