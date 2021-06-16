""" script to train max-margin-based input-dependent weighted average for
    ensembling a pair of KG completion models """

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

# construct hyperparameter grid
nlayers_grid = [('num-layers', value) for value in (3, 4)]
nhidden_grid = [('hidden-channels', value) for value in (128, 256)]
bsz_grid = [('batch-size', value) for value in (64, 128, 256)]
lr_grid = [('lr', value) for value in (1e-1, 1e-2, 1e-3)]
negatives_grid = [('negative-samples', value) for value in (16, 32)]
param_grid = list(product(nlayers_grid, nhidden_grid, bsz_grid, lr_grid,
                          negatives_grid))

# grid of pairs of models and sets of parameters
grid = list(product(pairs, param_grid))

# choose grid point based on command-line
pair, params = grid[int(sys.argv[1])]
model_name1, model_name2, scores_fname1, scores_fname2 = pair

# construct base command
cmd = ['python', 'script/evaluate_adaptive_weighting.py',
       model_name1,
       scores_fname1,
       model_name2,
       scores_fname2,
       '--dataset=msi',
       '--info-file=./data/processed/msi.tsv',
       '--subgraph=./subgraph/msi-edge-split-f0.2-neg500-s42.pt',
       '--tokens-file=./tokenized/tokens-msi-pubmedbert.pt',
       '--save',
       '--dropout=0',
       '--epochs=200',
       '--tol=1e-4',
       '--patience=10']

# add classifier hyperparameters
for name, param in params:
    if name is None:
        continue
    cmd.append(f'--{name}={param}')

print(' '.join(cmd))

start = datetime.now()
print(f'started at: {start}')

subprocess.call(cmd)

end = datetime.now()
print(f'ended at {end}')
print(f'total time: {end - start}')
