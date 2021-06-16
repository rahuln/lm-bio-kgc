""" script to summarize all ensembling results in a given directory """

from argparse import ArgumentParser
from glob import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')


# command-line arguments
parser = ArgumentParser(description='script to summarize all ensembling '
                                    'results in a given directory')
parser.add_argument('resdir', type=str, help='ensembling results directory')
parser.add_argument('--plot', action='store_true', help='plot results')
parser.add_argument('--dataset', type=str, default='',
                    help='name of dataset to add to plot')


def collect_results(resdir, multiple=False):
    """ collect all ensembling results in resdir into a single data frame """

    # info for data frame
    if multiple:
        models = ['model_names']
    else:
        models = ['model1', 'model2']
    hyperparams = ['classifier', 'ensemble_method', 'seed', 'penalty', 'C',
                   'n_estimators', 'max_depth', 'learning_rate', 'reg_alpha',
                   'reg_beta', 'nlayers', 'nhidden', 'alpha', 'batch_size',
                   'learning_rate_init', 'max_iter', ]
    metrics = ['mrr_model1', 'mrr_model2'] if not multiple else []
    metrics += ['mrr_ensemble', 'hits1_ensemble',
                'hits3_ensemble', 'hits10_ensemble', 'mrr_best', 'hits1_best',
                'hits3_best', 'hits10_best', 'majority_frac', 'cross_val_mean',
                'router_accuracy']
    columns = models + hyperparams + metrics

    # retrieve all results files from directory
    files = glob(f'{resdir}/**/config.json', recursive=True)
    tuples = list()

    # collection experiment information from file
    for fname in files:

        # load config, results files
        try:
            with open(fname, 'r') as f:
                config = json.load(f)
            with open(fname.replace('config', 'results'), 'r') as f:
                results = json.load(f)
        except:
            print(f'error loading file: {fname}')
            continue

        # collect model names
        if multiple:
            model_names = ', '.join(sorted(config['model_names']))
            model_values = [model_names]
        else:
            model1, model2 = config['model_names']
            model_values = [model1, model2]

        # collect hyperparameters
        hyperparam_values = [config[elem] for elem in hyperparams]

        # collect metrics for individual models
        metric_values = list()
        if not multiple:
            metric_values.append(results['mrr_list']['test'][model1])
            metric_values.append(results['mrr_list']['test'][model2])

        # collect metrics for ensemble
        metric_values.append(results['mrr_list']['test']['ensemble'])
        metric_values.append(results['hits@1_list']['test']['ensemble'])
        metric_values.append(results['hits@3_list']['test']['ensemble'])
        metric_values.append(results['hits@10_list']['test']['ensemble'])

        # collect metrics for oracle router
        metric_values.append(results['mrr_list']['test']['best'])
        metric_values.append(results['hits@1_list']['test']['best'])
        metric_values.append(results['hits@3_list']['test']['best'])
        metric_values.append(results['hits@10_list']['test']['best'])

        # collect class and classifier statistics
        metric_values.append(results.get('majority_frac', None))
        metric_values.append(results.get('cross_val_mean', None))
        metric_values.append(results.get('router_accuracy', None))

        # aggregate information for all columns
        values = model_values + hyperparam_values + metric_values
        tuples.append(tuple(values))

    # constuct and return data frame
    df = pd.DataFrame(tuples, columns=columns)
    return df


if __name__ == '__main__':

    args = parser.parse_args()

    # retrieve data frame with all results
    df = collect_results(args.resdir)

    # dictionary to save ensemble metric for each combination of classifier,
    # ensemble method, and pair of models
    results = dict()

    # columns to select when printing data frames
    print_col = ['model1', 'model2', 'classifier', 'mrr_model1',
                     'mrr_model2', 'mrr_ensemble', 'hits1_ensemble',
                     'hits3_ensemble', 'hits10_ensemble', 'mrr_best']

    # find best result across all pairs of ensembled models for global
    # weighted average
    df_avg = df[df.classifier == 'average']
    df_grp = df_avg.groupby(['model1', 'model2'])
    df_best = df_avg.loc[df_grp['mrr_ensemble'].idxmax()]
    print('average')
    df_print = df_best[print_col].sort_values('mrr_ensemble')
    print(df_print)
    print()

    # store metrics of global weighted average ensemble
    results['average'] = dict()
    for i, row in df_print.iterrows():
        model1, model2 = row['model1'], row['model2']
        key = tuple(sorted([model1, model2]))
        results['average'][key] = row['mrr_ensemble']

    # store metrics of oracle router (can use the data frame for the global
    # weighted average, since the oracle metrics will be the same for all
    # ensembling methods - they just depend on the model pair)
    results['oracle'] = dict()
    for i, row in df_print.iterrows():
        model1, model2 = row['model1'], row['model2']
        key = tuple(sorted([model1, model2]))
        results['oracle'][key] = row['mrr_best']

    # find best results across all pairs of ensembled models for each
    # combination of classifier and ensemble method
    models = ['ComplEx', 'DistMult', 'RotatE', 'TransE', 'KGBERT']
    classifiers = ['logreg', 'decisiontree', 'xgboost', 'mlp']
    ensemble_methods = ['router', 'average']
    extra = ['router_accuracy']   # extra column to print

    for classifier in classifiers:
        df_clf = df[df.classifier == classifier]
        results[classifier] = dict()
        for ensemble_method in ensemble_methods:
            df_clf_ens = df_clf[df_clf.ensemble_method == ensemble_method]
            df_grp = df_clf_ens.groupby(['model1', 'model2'])
            df_best = df_clf_ens.loc[df_grp['cross_val_mean'].idxmax()]
            print(classifier, ensemble_method)
            df_print = df_best[print_col + extra].sort_values('mrr_ensemble')
            print(df_print)
            print()

            # store metrics for this classifier and ensemble method
            results[classifier][ensemble_method] = dict()
            for i, row in df_print.iterrows():
                model1, model2 = row['model1'], row['model2']
                key = tuple(sorted([model1, model2]))
                value = row['mrr_ensemble']
                results[classifier][ensemble_method][key] = value
            results[classifier][ensemble_method]['router_accuracy'] = \
                df_print['router_accuracy'].mean().item()

    # find best results for each global weighted average, router, and
    # input-dependent weighted average model, with and without KG-BERT

    # global weighted average
    df_sub = df[df.classifier == 'average']
    df_kge = df_sub[df_sub.model1 != 'KGBERT']
    df_bert = df_sub[df_sub.model1 == 'KGBERT']
    print('best global weighted average\n')
    print('without KG-BERT\n')
    print(df_kge.loc[df_kge['mrr_ensemble'].idxmax()][print_col])
    print('with KG-BERT\n')
    print(df_bert.loc[df_bert['mrr_ensemble'].idxmax()][print_col])
    print()

    # router
    df_sub = df[(df.classifier != 'average') & (df.ensemble_method == 'router')]
    df_kge = df_sub[df_sub.model1 != 'KGBERT']
    df_bert = df_sub[df_sub.model1 == 'KGBERT']
    print('best router\n')
    print('without KG-BERT\n')
    print(df_kge.loc[df_kge['cross_val_mean'].idxmax()][print_col])
    print('with KG-BERT\n')
    print(df_bert.loc[df_bert['cross_val_mean'].idxmax()][print_col])
    print()

    # input-dependent weighted average
    df_sub = df[(df.classifier != 'average') & (df.ensemble_method == 'average')]
    df_kge = df_sub[df_sub.model1 != 'KGBERT']
    df_bert = df_sub[df_sub.model1 == 'KGBERT']
    print('best input-dependent weighted average\n')
    print('without KG-BERT\n')
    print(df_kge.loc[df_kge['cross_val_mean'].idxmax()][print_col])
    print('with KG-BERT\n')
    print(df_bert.loc[df_bert['cross_val_mean'].idxmax()][print_col])

    # plot results as grid of triangle plots with metric values for every
    # combination of classifier, ensemble method, and pair of models
    if args.plot:

        # mapping from classifier name to full name
        classifier_names = {'logreg' : 'logistic regression',
                            'decisiontree' : 'decision tree',
                            'xgboost' : 'XGBoost',
                            'mlp' : 'MLP'}

        # helper function to generate subplot with triangle matrix of metric
        # value for all pairs of methods
        def all_pairs_metric_subplot(metrics, methods, ax):

            # retrieve metric value for each pair of methods
            M = np.zeros((len(methods), len(methods)))
            for i in range(len(methods)):
                for j in range(i):
                    model1, model2 = methods[i], methods[j]
                    key = tuple(sorted([model1, model2]))
                    if key not in metrics:
                        M[i, j] = np.nan
                    else:
                        M[i, j] = metrics[key]

            # remove duplicated (ordered) pairs and plot matrix
            M[np.triu_indices_from(M)] = np.nan
            M = M[1:, :-1]
            img = ax.imshow(M, cmap='Reds')

            # add metric value label to each cell
            for i in range(M.shape[0]):
                for j in range(i + 1):
                    value = 100 * M[i, j]
                    cell_color = img.to_rgba(value / 100)
                    intensity = np.mean(cell_color[:3])
                    color = 'black' if intensity > 0.5 else 'white'
                    ax.text(j, i, f'{value:.1f}', color=color, fontsize=12,
                            fontweight='bold', ha='center', va='center')

            # remove spines
            for key in ('top', 'bottom', 'left', 'right'):
                ax.spines[key].set_visible(False)

            # add method names to xticks/yticks
            names = methods.copy()
            names[names.index('KGBERT')] = 'PubMedBERT'
            ax.set_xticks(np.arange(len(names) - 1))
            ax.set_yticks(np.arange(len(names) - 1))
            ax.set_xticklabels(names[:-1], rotation=-45)
            ax.set_yticklabels(names[1:])

        # plot global weighted average result in separate plot
        fig, ax = plt.subplots(figsize=(4, 4))
        all_pairs_metric_subplot(results['average'], models, ax)
        ax.set_title('global weighted average')
        plt.tight_layout()
        plt.savefig(os.path.join(args.resdir, 'fig-global-avg.png'), dpi=300)

        # plot oracle router result in separate plot
        fig, ax = plt.subplots(figsize=(4, 4))
        all_pairs_metric_subplot(results['oracle'], models, ax)
        ax.set_title('oracle router')
        plt.tight_layout()
        plt.savefig(os.path.join(args.resdir, 'fig-oracle.png'), dpi=300)

        # plot router with different classifiers
        fig, ax = plt.subplots(1, len(classifiers), figsize=(12.5, 3))
        for i, classifier in enumerate(classifiers):
            res = results[classifier]['router']
            acc = results[classifier]['router']['router_accuracy']
            all_pairs_metric_subplot(res, models, ax[i])
            name = classifier_names[classifier]
            ax[i].set_title(f'{name} ({100*acc:.1f}%)')
            if i > 0:
                ax[i].set_yticks([])
        ax[0].set_ylabel(f'{args.dataset:20s}', fontsize=14, fontweight='bold',
                         rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(args.resdir, 'fig-router.png'), dpi=300)

        # plot input-dependent weighted average with different classifiers
        fig, ax = plt.subplots(1, len(classifiers), figsize=(12.5, 3))
        for i, classifier in enumerate(classifiers):
            res = results[classifier]['average']
            acc = results[classifier]['average']['router_accuracy']
            all_pairs_metric_subplot(res, models, ax[i])
            name = classifier_names[classifier]
            ax[i].set_title(f'{name} ({100*acc:.1f}%)')
            if i > 0:
                ax[i].set_yticks([])
        ax[0].set_ylabel(f'{args.dataset:20s}', fontsize=14, fontweight='bold',
                         rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(args.resdir, 'fig-avg.png'), dpi=300)

