""" script to run embedding-based methods for KG completion on inductive
    split of msi dataset """

import sys
import subprocess
from itertools import product


# set up grid of models
model_grid = ['ComplEx', 'DistMult', 'RotatE', 'TransE']

# specify optimal embedding dimension, gamma, learning rate, and number of
# negative samples, and regularization for each model (from transductive split
# results)
params = {'ComplEx' : (1000, 0.1, 1e-4, 256, 1e-5),
          'DistMult' : (2000, 1., 1e-3, 256, 1e-5),
          'RotatE' : (1000, 1., 1e-4, 128, 1e-6),
          'TransE' : (500, 1., 1e-4, 128, 1e-5)}

# retrieve index from command line
model = model_grid[int(sys.argv[-1])]
d, g, lr, n, r = params[model]

# use half of hidden dim for models that double the dimension
if model in ('RotatE', 'ComplEx'):
    d = d // 2

# set up command
cmd = ['python', 'src/kge/run.py',
       '--cuda',
       '--do_train',
       '--do_valid',
       '--do_test',
       '--dataset=msi',
       f'--model={model}',
       f'-n={n}',
       f'-d={d}',
       f'-g={g}',
       '-b=512',
       f'-r={r}',
       '--test_batch_size=32',
       f'-lr={lr}',
       '--max_steps=50000',
       '--warm_up_steps=100000',
       '--valid_steps=5000',
       '--cpu_num=2',
       '--print_on_screen',
       '--subgraph=./subgraph/msi-edge-split-f0.2-neg500-ind-s42.pt',
       '--loss_function=max-margin']

# add extra options based on model 
if model in ('RotatE', 'ComplEx'):     # double entity embedding
    cmd.append('-de')
if model == 'ComplEx':                 # double relation embedding
    cmd.append('-dr')

# run command
subprocess.call(cmd)
