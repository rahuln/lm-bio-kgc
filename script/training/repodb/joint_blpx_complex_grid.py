""" script to train BLP cross-encoder for KG completion on repodb over a grid
    of hyperparameters """

from itertools import product
import subprocess
import sys


# grids of models, learning rates, and batch sizes
model_grid = ['pubmedbert']
lr_grid = [1e-5]
batch_size_grid = [16]
regularization_grid = [0, 1e-2, 1e-3]
grid = list(product(model_grid, lr_grid, batch_size_grid, regularization_grid))

# choose hyperparameters using index from command-line argument
model, lr, batch_size, regularization = grid[int(sys.argv[-1])]

# set validation steps based on batch size to run validation once per epoch
ntrain = 5342
valid_steps = ((2 * ntrain // batch_size) + 1)

# filenames of trained BLPCrossEncoder and ComplEx parameters to load
model_file = 'results/repodb/repodb-edge-split-f0.2-neg500-s42/' \
             + 'link-relation-ranking/blpcrossencoder-complex-d250-g0.0-cls' \
             + '-pubmedbert-finetune-neg32-e20-b16-lr1e-05-gacc8-adam-desc/' \
             + 'seed-42/best_model.pt'
checkpoint_file = 'results/repodb/repodb-edge-split-f0.2-' \
                  + 'neg500-s42/max-margin/ComplEx/d250-g0.1-lr1e-04-neg256' \
                  + '-b512-r1e-05/checkpoint'

# set up and run command
cmd = ['python', '-u', 'src/lm/run.py',
       '--dataset=repodb',
       f'--info_filename=./tokenized/tokens-repodb-{model}-desc.pt',
       f'--model={model}',
       '--encoder-type=JointBLPCrossEncoderAndKGEModel',
       '--tokenized',
       f'--batch_size={batch_size}',
       '--test_batch_size=32',
       f'--lr={lr}',
       '--epochs=5',
       f'--valid_steps={valid_steps}',
       '--log_steps=10',
       '--gradient_steps=8',
       '--mode=finetune',
       '--subgraph=./subgraph/repodb-edge-split-f0.2-neg500-s42.pt',
       '--output-to-use=ranking_outputs',
       '--negative-sample-size=32',
       '--link-prediction',
       '--relation-prediction',
       '--relevance-ranking',
       '--use-descriptions',
       '--seed=42',
       f'--model-file={model_file}',
       '--save-metric=mrr_list',
       '--score=ComplEx',
       '--embedding-dim=250',
       '--gamma=0',
       f'--regularization={regularization}',
       f'--checkpoint-file={checkpoint_file}',
       '--weighted-average']
subprocess.call(cmd)
