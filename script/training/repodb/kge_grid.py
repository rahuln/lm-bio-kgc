""" script to run embedding-based methods for KG completion on repodb over a
    grid of hyperparameters """

import sys
import subprocess
from itertools import product


# set up grids of model, hidden dimension, gamma, and learning rate
model_grid = ['ComplEx', 'DistMult', 'RotatE', 'TransE']
hidden_dim_grid = [500, 1000, 2000]
gamma_grid = [0.1, 1.]
lr_grid = [1e-3, 1e-4]
neg_samp_grid = [128, 256]
regularization_grid = [1e-5, 1e-6]
options = list(product(model_grid, hidden_dim_grid, gamma_grid, lr_grid,
                       neg_samp_grid, regularization_grid))

# retrieve index from command line
model, d, g, lr, n, r = options[int(sys.argv[-1])]

# use half of hidden dim for models that double the dimension
if model in ('RotatE', 'ComplEx'):
    d = d // 2

# set up command
cmd = ['python', 'src/kge/run.py',
       '--cuda',
       '--do_train',
       '--do_valid',
       '--do_test',
       '--dataset=repodb',
       f'--model={model}',
       f'-n={n}',
       f'-d={d}',
       f'-g={g}',
       '-b=512',
       f'-r={r}',
       '--test_batch_size=32',
       f'-lr={lr}',
       '--max_steps=10000',
       '--warm_up_steps=20000',
       '--valid_steps=500',
       '--cpu_num=2',
       '--print_on_screen',
       '--subgraph=./subgraph/repodb-edge-split-f0.2-neg500-s42.pt',
       '--loss_function=max-margin']

# add extra options based on model 
if model in ('RotatE', 'ComplEx'):     # double entity embedding
    cmd.append('-de')
if model == 'ComplEx':                 # double relation embedding
    cmd.append('-dr')

# run command
subprocess.call(cmd)
