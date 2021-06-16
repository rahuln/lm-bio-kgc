""" script to preprocess multiscale interactome (msi) .tsv files and write
    to a single .tsv files with entity IDs and names """

from argparse import ArgumentParser
from collections import defaultdict
import os

import pandas as pd
from tqdm import tqdm, trange


# command-line arguments
parser = ArgumentParser(description='preprocess msi dataset to .tsv file')
parser.add_argument('dirname', type=str, help='location of msi files')
parser.add_argument('--outdir', type=str, default='data/processed',
                    help='output directory for combined .tsv file')


if __name__ == '__main__':

    args = parser.parse_args()

    # set up data structures to store entity IDs and info
    node_types = sorted(['drug', 'disease', 'function', 'protein'])
    ids = defaultdict(set)
    info = dict()

    # list of filenames that store msi dataset triples
    files = {('drug', 'protein') : '1_drug_to_protein.tsv',
             ('disease', 'protein') : '2_indication_to_protein.tsv',
             ('protein', 'protein') : '3_protein_to_protein.tsv',
             ('protein', 'function') : '4_protein_to_biological_function.tsv',
             ('function', 'function') : '5_biological_function_to_biological_function.tsv',
             ('drug', 'disease') : '6_drug_indication_df.tsv'}

    # process each file
    for (head, tail), fname in files.items():
        df = pd.read_table(os.path.join(args.dirname, fname), na_filter=False)
        # this file has different column names, change to match the others
        if fname == '6_drug_indication_df.tsv':
            df.columns = ['node_1', 'node_1_name', 'node_2', 'node_2_name']
        ids[head].update(df.node_1.tolist())
        ids[tail].update(df.node_2.tolist())
        for i, row in tqdm(df.iterrows(), total=len(df), desc=fname, ncols=100):
            info[row['node_1']] = (row['node_1_name'], head)
            info[row['node_2']] = (row['node_2_name'], tail)

    ids = {key : sorted(list(value)) for key, value in ids.items()}
    tuples = list()

    # collect entity ID, name, and type in list of tuples
    for node_type in node_types:
        for id in ids[node_type]:
            name, type = info[id]
            assert type == node_type, 'mismatched types'
            tuples.append((id, name, type))

    # convert to pandas data frame and save to .tsv file
    df = pd.DataFrame(tuples, columns=['id', 'name', 'type'])
    df.to_csv(os.path.join(args.outdir, 'msi.tsv'), sep='\t')

    # create data frame for relations, save to .tsv file
    relation_names = ['drug-protein interaction',
                      'disease-protein interaction',
                      'protein-protein interaction',
                      'protein-function relation',
                      'function-function relation',
                      'drug-disease interaction']
    relations = [(i, name) for i, name in enumerate(relation_names)]
    df = pd.DataFrame(relations, columns=['id', 'name'])
    df.to_csv(os.path.join(args.outdir, 'msi-relations.tsv'), sep='\t')

