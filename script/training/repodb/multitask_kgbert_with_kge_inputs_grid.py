""" script to train multi-task KG-BERT with KGE (ComplEx) inputs for KG
    completion on repodb over a grid of hyperparameters """

from itertools import product
import subprocess
import sys


# grids of learning rates and batch sizes
lr_grid = [1e-5, 3e-5, 5e-5]
batch_size_grid = [16, 32]
grid = list(product(lr_grid, batch_size_grid))

# choose hyperparameters using index from command-line argument
lr, batch_size = grid[int(sys.argv[-1])]

# set validation steps based on batch size to run validation once per epoch
ntrain = 5342
valid_steps = ((2 * ntrain // batch_size) + 1)

# checkpoint file to load ComplEx embeddings from 
checkpoint_file = 'results/repodb/repodb-edge-split-f0.2-' \
                  + 'neg500-s42/negative-sampling-adv/ComplEx/d384-g20.0-' \
                  + 'lr1e-03-neg128/checkpoint'

# set up and run command
cmd = ['python', '-u', 'src/lm/run.py',
       '--dataset=repodb',
       '--info_filename=./tokenized/tokens-repodb-pubmedbert.pt',
       '--model=pubmedbert',
       '--encoder-type=KGBERTWithKGEInputs',
       '--tokenized',
       f'--batch_size={batch_size}',
       '--test_batch_size=32',
       f'--lr={lr}',
       '--epochs=20',
       f'--valid_steps={valid_steps}',
       '--log_steps=10',
       '--gradient_steps=8',
       '--mode=finetune',
       '--subgraph=./subgraph/repodb-edge-split-f0.2-neg500-s42.pt',
       '--output-to-use=ranking_outputs',
       '--negative-sample-size=16',
       '--link-prediction',
       '--relation-prediction',
       '--relevance-ranking',
       '--seed=42',
       '--save-metric=mrr_list',
       f'--checkpoint-file={checkpoint_file}']
subprocess.call(cmd)
