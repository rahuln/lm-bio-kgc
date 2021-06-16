""" script to train DKRL on repoDB over a grid of hyperparameters """

from itertools import product
import subprocess
import sys


# grids of learning rates, batch sizes, embedding dimensions, and
# regularization
lr_grid = [1e-3, 1e-4, 1e-5]
embedding_dim_grid = [500, 1000, 2000]
regularization_grid = [0, 1e-2, 1e-3]
grid = list(product(lr_grid, embedding_dim_grid, regularization_grid))

# choose hyperparameters using index from command-line argument
lr, embedding_dim, regularization = grid[int(sys.argv[-1])]

# set validation steps based on batch size to run validation once per epoch
ntrain = 5325
batch_size = 64
valid_steps = ((2 * ntrain // batch_size) + 1)

# set up and run command
cmd = ['python', '-u', 'src/lm/run.py',
       '--dataset=repodb',
       '--info_filename=./tokenized/tokens-repodb-pubmedbert-desc.pt',
       '--model=pubmedbert',
       '--encoder-type=DKRLBiEncoder',
       '--tokenized',
       f'--batch_size={batch_size}',
       '--test_batch_size=64',
       f'--lr={lr}',
       '--epochs=40',
       f'--valid_steps={valid_steps}',
       '--log_steps=10',
       '--mode=finetune',
       '--subgraph=./subgraph/repodb-edge-split-f0.2-neg500-ind-s42.pt',
       '--output-to-use=ranking_outputs',
       '--negative-sample-size=32',
       '--link-prediction',
       '--relation-prediction',
       '--relevance-ranking',
       '--use-descriptions',
       '--seed=42',
       '--max-length=64',
       '--save-metric=mrr_list',
       '--score=TransE',
       f'--embedding-dim={embedding_dim}',
       f'--regularization={regularization}',
       '--update-embeddings']
subprocess.call(cmd)
