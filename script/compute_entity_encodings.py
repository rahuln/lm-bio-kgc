""" script to compute and save encodings of entity names from a KG using
    BERT-based model trained for KG completion """

from argparse import ArgumentParser
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def parse_args(args=None):
    parser = ArgumentParser(description='compute encodings of entity names '
                                        'from a KG dataset using KG-BERT-'
                                        'style model trained on that dataset')
    parser.add_argument('result_dir', type=str,
                        help='directory of saved results')
    parser.add_argument('--untrained', action='store_true',
                        help='use original BERT model rather than model '
                             'trained for KG completion')
    parser.add_argument('--info_filename', type=str,
                        default='data/processed/repodb.tsv',
                        help='info file for entities')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size for encoding')
    parser.add_argument('--max_length', default=64, type=int,
                        help='maximum length of entity name')
    parser.add_argument('--use_descriptions', action='store_true',
                        help='encode entity descriptions as well as names')

    return parser.parse_args(args)


def main(args):

    # set up device
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    # load config file from result directory
    with open(os.path.join(args.result_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    # check that dataset model was trained on matches info filename
    if config['dataset'] not in args.info_filename:
        raise ValueError('trained model\'s dataset and info file must match')

    # load info file and retrieve entity names and descriptions
    df = pd.read_table(args.info_filename, index_col=0, na_filter=False)
    names = df.name.tolist()
    descriptions = df.description.tolist()

    # load BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModel.from_pretrained(config['model_name']).to(device)

    # load state_dict for saved model, retrieve BERT encoder parameters
    if not args.untrained:
        saved = torch.load(os.path.join(args.result_dir, 'best_model.pt'))
        state_dict = {key[8:] : value for key, value in saved['model'].items()
                      if key.startswith('encoder')}
        model.load_state_dict(state_dict)

    # print info
    print(f'computing encodings for entities')
    print('model = %s' % config['model'])
    print('batch size = %d' % args.batch_size)

    # set model in eval mode
    model.eval()

    # initialize data loader, list of encodings for entity names
    loader = DataLoader(torch.arange(len(names)), batch_size=args.batch_size)
    encodings = list()

    # compute encodings for entity names
    for batch in tqdm(loader, desc='encoding entity names'):
        if args.use_descriptions:
            text = [names[i] + '; ' + descriptions[i] for i in batch.tolist()]
        else:
            text = [names[i] for i in batch.tolist()]
        tokenized = tokenizer(text, padding=True, truncation=True,
                              max_length=args.max_length,
                              return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**tokenized, return_dict=True)
        encodings.append(outputs.last_hidden_state[:, 0, :].cpu())
    encodings = torch.cat(encodings)

    print('saving computed encodings...')
    suffix = '-untrained' if args.untrained else ''
    desc = '-desc' if args.use_descriptions else ''
    savename = os.path.join(args.result_dir,
                            f'entity-encodings{desc}{suffix}.pt')
    torch.save(encodings, savename)

    print('done')


if __name__ == '__main__':
    main(parse_args())
