""" script to train router for ensembling combinations of two or more
    KG completion models """

from datetime import datetime
from itertools import product
import subprocess
import sys

import numpy as np

# dictionary of all models and their saved score files
scores_fnames = {
    'ComplEx' : 'results/repodb/'
                'repodb-edge-split-f0.2-neg500-s42/max-margin/ComplEx/'
                'd250-g0.1-lr1e-04-neg256-b512-r1e-05/'
                'scores-repodb-edge-split-f0.2-neg500-s42.pt',
    'RotatE' : 'results/repodb/'
               'repodb-edge-split-f0.2-neg500-s42/max-margin//RotatE/'
               'd1000-g1.0-lr1e-04-neg256-b512-r1e-06/'
               'scores-repodb-edge-split-f0.2-neg500-s42.pt',
    'PubMedBERT' : 'results/repodb/repodb-edge-split-f0.2-neg500-s42/'
                   'link-relation-ranking/kgbert-pubmedbert-finetune-neg32-'
                   'e20-b16-lr1e-05-gacc16-adam-desc/seed-42/'
                   'scores-repodb-edge-split-f0.2-neg500-s42-neg500.pt',
    'BioBERT' : 'results/repodb/repodb-edge-split-f0.2-neg500-s42/'
                'link-relation-ranking/kgbert-biobert-finetune-neg32-e20-b32-'
                'lr5e-05-gacc16-adam-desc/seed-42/'
                'scores-repodb-edge-split-f0.2-neg500-s42-neg500.pt'
}

# list combinations of models to ensemble
combinations = [['PubMedBERT', 'BioBERT'],
                ['ComplEx', 'RotatE', 'PubMedBERT'],
                ['RotatE', 'PubMedBERT', 'BioBERT']]

# grid of classifiers, ensembling methods, and model combinations
classifier_grid = ['average', 'logreg', 'decisiontree', 'xgboost', 'mlp']
ensemble_method_grid = ['router', 'average']
grid = list(product(classifier_grid, ensemble_method_grid, combinations))

# choose grid point based on command-line
classifier, ensemble_method, model_names = grid[int(sys.argv[1])]

# create grid of hyperparameters depending on which classifier is being used
if classifier == 'logreg':
    penalty_grid = [('penalty', value) for value in ('l1', 'l2')]
    C_grid = [('C', value) for value in list(np.logspace(-5, 3, 9))]
    param_grid = list(product(penalty_grid, C_grid))
elif classifier == 'decisiontree':
    max_depth_grid = [('max_depth', value) for value in (2, 4, 8)]
    lr_grid = [('learning_rate', value) for value in (1e-1, 1e-2, 1e-3)]
    param_grid = list(product(max_depth_grid, lr_grid))
elif classifier == 'xgboost':
    n_estimators_grid = [('n_estimators', value) for value in (100, 500, 1000)]
    max_depth_grid = [('max_depth', value) for value in (2, 4, 8)]
    lr_grid = [('learning_rate', value) for value in (1e-1, 1e-2, 1e-3)]
    param_grid = list(product(n_estimators_grid, max_depth_grid, lr_grid))
elif classifier == 'mlp':
    nlayers_grid = [('nlayers', value) for value in (1, 2)]
    nhidden_grid = [('nhidden', value) for value in (128, 256)]
    bsz_grid = [('batch_size', value) for value in (64, 128, 256)]
    lr_grid = [('learning_rate_init', value) for value in (1e-1, 1e-2, 1e-3)]
    param_grid = list(product(nlayers_grid, nhidden_grid, bsz_grid, lr_grid))
else:
    param_grid = [((None, None),)]

start = datetime.now()
print(f'\n=== {len(param_grid)} combinations of hyperparameters ===\n')

for i, params in enumerate(param_grid):

    # construct base command
    cmd = ['python', 'script/evaluate_router.py', '--model-names']
    cmd += model_names
    cmd += ['--scores-fnames']
    cmd += [scores_fnames[model_name] for model_name in model_names]
    cmd += ['--dataset=repodb',
            '--info-file=./data/processed/repodb.tsv',
            '--subgraph=./subgraph/repodb-edge-split-f0.2-neg500-s42.pt',
            f'--classifier={classifier}',
            f'--ensemble-method={ensemble_method}',
            '--tokens-file=./tokenized/tokens-repodb-pubmedbert.pt',
            '--verbose', '--save', '--outdir=ensemble-multi',
            '--save-classifier']

    # add classifier hyperparameters
    for name, param in params:
        if name is None:
            continue
        cmd.append(f'--{name}={param}')

    print(' '.join(cmd))
    subprocess.call(cmd)
    print(f'\n=== completed {i + 1} / {len(param_grid)} param settings ===\n')

print(f'total runtime: {datetime.now() - start}')
