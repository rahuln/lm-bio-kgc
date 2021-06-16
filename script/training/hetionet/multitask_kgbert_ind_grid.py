""" script to train multi-task KG-BERT for KG completion on inductive split of
    hetionet using PubMedBERT with best hyperparameters from transductive
    experiments """

from itertools import product
import subprocess
import sys


# grids of models, learning rates, and batch sizes
model_grid = ['pubmedbert']
lr_grid = [3e-5]
batch_size_grid = [32]
grid = list(product(model_grid, lr_grid, batch_size_grid))

# choose hyperparameters using index from command-line argument
model, lr, batch_size = grid[int(sys.argv[-1])]

# set validation steps based on batch size to run validation 3 times per epoch
ntrain = 124416
valid_steps = ((2 * ntrain // batch_size) + 1) // 3

# set up and run command
cmd = ['python', '-u', 'src/lm/run.py',
       '--dataset=hetionet',
       f'--info_filename=./tokenized/tokens-hetionet-{model}-desc.pt',
       f'--model={model}',
       '--encoder-type=KGBERT',
       '--tokenized',
       f'--batch_size={batch_size}',
       '--test_batch_size=32',
       f'--lr={lr}',
       '--epochs=5',
       f'--valid_steps={valid_steps}',
       '--log_steps=100',
       '--gradient_steps=4',
       '--mode=finetune',
       '--subgraph=./subgraph/hetionet-edge-split-f0.2-neg80-ind-s42.pt',
       '--eval-fraction=0.01',
       '--output-to-use=ranking_outputs',
       '--negative-sample-size=16',
       '--num-neg-samples=80',
       '--link-prediction',
       '--relation-prediction',
       '--relevance-ranking',
       '--use-descriptions',
       '--seed=42',
       '--save-metric=mrr_list']
subprocess.call(cmd)
