""" script to train router for ensembling a pair of KG completion models """

from datetime import datetime
from itertools import product
import subprocess
import sys

import numpy as np

# list of all models and their saved score files
model_names = ['ComplEx', 'DistMult', 'RotatE', 'TransE', 'KGBERT']

scores_fnames = [
    'results/msi/msi-edge-split-f0.2-neg500-s42/'
        'max-margin/ComplEx/d1000-g0.1-lr1e-04-neg256-b512-r1e-05/'
        'scores-msi-edge-split-f0.2-neg500-s42.pt',
    'results/msi/msi-edge-split-f0.2-neg500-s42/'
        'max-margin/DistMult/d2000-g1.0-lr1e-03-neg256-b512-r1e-05/'
        'scores-msi-edge-split-f0.2-neg500-s42.pt',
    'results/msi/msi-edge-split-f0.2-neg500-s42/'
        'max-margin/RotatE/d1000-g1.0-lr1e-04-neg128-b512-r1e-06/'
        'scores-msi-edge-split-f0.2-neg500-s42.pt',
    'results/msi/msi-edge-split-f0.2-neg500-s42/'
        'max-margin/TransE/d500-g1.0-lr1e-04-neg128-b512-r1e-05/'
        'scores-msi-edge-split-f0.2-neg500-s42.pt',
    'results/msi/msi-edge-split-f0.2-neg500-s42/link-relation-ranking/'
        'kgbert-pubmedbert-finetune-neg8-e05-b32-lr1e-05-gacc2-adam-desc/'
        'seed-42/scores-msi-edge-split-f0.2-neg500-s42-neg500.pt'
]

# create all unordered pairs of models to ensemble
pairs = list()
for i in range(len(model_names)):
    for j in range(i):
        pairs.append((model_names[i], model_names[j],
                      scores_fnames[i], scores_fnames[j]))

# grid of classifiers and ensembling methods
classifier_grid = ['average', 'logreg', 'decisiontree', 'xgboost', 'mlp']
ensemble_method_grid = ['router', 'average']
grid = list(product(classifier_grid, ensemble_method_grid, pairs))

# choose grid point based on command-line
classifier, ensemble_method, pair = grid[int(sys.argv[1])]
model_name1, model_name2, scores_fname1, scores_fname2 = pair

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
    cmd = ['python', 'script/evaluate_router.py',
           '--model-names', model_name1, model_name2,
            '--scores-fnames', scores_fname1, scores_fname2,
            '--dataset=msi',
            '--info-file=./data/processed/msi.tsv',
            '--subgraph=./subgraph/msi-edge-split-f0.2-neg500-s42.pt',
            f'--classifier={classifier}',
            f'--ensemble-method={ensemble_method}',
            '--tokens-file=./tokenized/tokens-msi-pubmedbert.pt',
            '--verbose', '--save']

    # add classifier hyperparameters
    for name, param in params:
        if name is None:
            continue
        cmd.append(f'--{name}={param}')

    print(' '.join(cmd))
    subprocess.call(cmd)
    print(f'\n=== completed {i + 1} / {len(param_grid)} param settings ===\n')

print(f'\ntotal time: {datetime.now() - start}')

