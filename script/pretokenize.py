""" pre-tokenize entity text and save to file """

from argparse import ArgumentParser
import os
import sys

import pandas as pd
from transformers import AutoTokenizer
import torch
from tqdm import tqdm


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='repodb',
                        help='name of dataset')
    parser.add_argument('--info_file', type=str,
                        default='data/processed/repodb.tsv',
                        help='file with entity information')
    parser.add_argument('--model', type=str, default='pubmedbert',
                        help='model whose tokenizer to use')
    parser.add_argument('--use-descriptions', action='store_true',
                        help='add entity descriptions')
    parser.add_argument('--separator', type=str, default='semicolon',
                        choices=['semicolon', 'sep'],
                        help='separator token between name/description text')
    parser.add_argument('--split', action='store_true',
                        help='save name and description tokens separately')
    args = parser.parse_args()

    # set up savename, check for saved precompute tokens
    use_desc = '-desc' if args.use_descriptions else ''
    rel = '-rel' if 'relations' in args.info_file else ''
    sep = '-sep' if args.separator == 'sep' else ''
    split = '-split' if (args.split and args.use_descriptions) else ''
    savename = os.path.join('tokenized',
        f'tokens-{args.dataset}-{args.model}{rel}{use_desc}{sep}{split}.pt')
    if os.path.exists(savename):
        print('tokenized results exist, exiting...')
        sys.exit()
    os.makedirs('tokenized', exist_ok=True)

    # check if info_file is correct for dataset
    if args.dataset not in args.info_file:
        raise ValueError(f'info_file is incorrect: {args.info_file} does not '
                         f'contain {args.dataset}')

    # map from model shortnames to huggingface names
    model_names = {
        'pubmedbert'  : 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'pubmedbertfull' : 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        'bert'        : 'bert-base-uncased',
        'bertlarge'   : 'bert-large-uncased',
        'roberta'      : 'roberta-base',
        'robertalarge' : 'roberta-large',
        'bioclinicalbert' : 'emilyalsentzer/Bio_ClinicalBERT',
        'biolm' : 'EMBO/bio-lm',
        'biobert' : 'dmis-lab/biobert-v1.1',
        'scibert' : 'allenai/scibert_scivocab_uncased'
    }

    # retrieve huggingface model name, load tokenizer
    args.model = model_names[args.model]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # load metadata file, check that node indices are in correct order
    df = pd.read_table(args.info_file, index_col=0, na_filter=False)
    assert sorted(df.index) == df.index.tolist(), 'index not sorted'

    # set separator token
    sep_token = '[SEP]' if args.separator == 'sep' else ';'

    # tokenize each entity name and store in list
    tokenized = list()
    for i, row in tqdm(df.iterrows(), total=len(df), desc='tokenizing'):
        text = row['name']
        if args.use_descriptions:
            if args.split:
                name_ids = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(text))
                desc_ids = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(row['description']))
                token_ids = [name_ids, desc_ids]
            else:
                text += f' {sep_token} ' + row['description']
                token_ids = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(text))
        else:
            token_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(text))
        tokenized.append(token_ids)

    # save to token IDs file
    torch.save(tokenized, savename)

