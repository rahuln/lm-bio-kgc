""" script to train multi-task KG-BERT for KG completion on msi over a grid
    of hyperparameters """

from itertools import product
import subprocess
import sys


# grids of models, learning rates, and batch sizes
model_grid = ['pubmedbert', 'pubmedbertfull', 'biobert', 'bioclinicalbert',
              'scibert', 'roberta']
lr_grid = [1e-5, 3e-5, 5e-5]
batch_size_grid = [16, 32]
grid = list(product(model_grid, lr_grid, batch_size_grid))

# choose hyperparameters using index from command-line argument
model, lr, batch_size = grid[int(sys.argv[-1])]

# set validation steps based on batch size to run validation 3 times per epoch
ntrain = 387724
valid_steps = ((2 * ntrain // batch_size) + 1) // 3

# set up and run command
cmd = ['python', '-u', 'src/lm/run.py',
       '--dataset=msi',
       f'--info_filename=./tokenized/tokens-msi-{model}-desc.pt',
       f'--model={model}',
       '--encoder-type=KGBERT',
       '--tokenized',
       f'--batch_size={batch_size}',
       '--test_batch_size=32',
       f'--lr={lr}',
       '--epochs=5',
       f'--valid_steps={valid_steps}',
       '--log_steps=100',
       '--gradient_steps=2',
       '--mode=finetune',
       '--subgraph=./subgraph/msi-edge-split-f0.2-neg500-s42.pt',
       '--eval-fraction=0.01',
       '--output-to-use=ranking_outputs',
       '--negative-sample-size=8',
       '--link-prediction',
       '--relation-prediction',
       '--relevance-ranking',
       '--use-descriptions',
       '--seed=42',
       '--save-metric=mrr_list']
subprocess.call(cmd)
