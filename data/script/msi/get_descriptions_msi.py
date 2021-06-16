""" get descriptions for entities in multiscale interactome (msi) KG """

from argparse import ArgumentParser
import json
import os
import requests

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

import umls


parser = ArgumentParser('get descriptions for entities in msi KG')
parser.add_argument('msi_file', type=str, help='info file for msi entities')
parser.add_argument('go_file', type=str,
                    help='.obo file with all Gene Ontology definitions')
parser.add_argument('entrez_file', type=str,
                    help='pre-retrieved protein descriptions from Entrez')
parser.add_argument('--drugs-file', type=str,
                    default='data/processed/repodb.tsv',
                    help='info file for repodb entities')
parser.add_argument('--outdir', type=str, default='data/processed/msi',
                    help='output directory for intermediate results')
parser.add_argument('--skip-drugs', action='store_true',
                    help='skip processing drugs')
parser.add_argument('--skip-functions', action='store_true',
                    help='skip processing functions')
parser.add_argument('--skip-proteins', action='store_true',
                    help='skip processing proteins')
parser.add_argument('--skip-diseases', action='store_true',
                    help='skip processing diseases')


def save_json(data, fname):
    """ helper function to save structure to JSON file """
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4)


def get_drugbank_description(drug_id):
    """ helper function to scrape drug description from DrugBank for a given
        drug ID """
    url = f'https://go.drugbank.com/drugs/{drug_id}'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    dts = soup.findAll('dt')
    dds = soup.findAll('dd')
    if len(dts) == 0 or len(dds) == 0:
        raise RuntimeError('failed retrieving drug description')
    try:
        name = dds[0].text
        desc = dds[2].text
    except IndexError:
        raise RuntimeError('failed retrieving drug description')
    return name, desc


def get_drug_descriptions(df, info_file=None):
    """ get descriptions of all drugs in data frame """
    ids = df[df.type == 'drug'].id.tolist()
    if info_file is not None:
        if type(info_file) is str:
            info_file = pd.read_table(info_file, index_col=0, na_filter=False)
        tuples, failures = list(), list()
        drugs_desc = {row['id'] : row['description']
                      for i, row in info_file.iterrows()}
        for did in tqdm(ids, desc='drugs'):
            if did in drugs_desc:
                tuples.append((did, drugs_desc[did]))
            else:
                try:
                    _, desc = get_drugbank_description(did)
                    tuples.append((did, desc))
                except Exception:
                    failures.append(did)
        desc = pd.DataFrame(tuples, columns=['id', 'description'])
    else:
        tuples, failures = list(), list()
        for did in tqdm(ids, desc='drugs'):
            try:
                _, desc = get_drugbank_description(did)
                tuples.append((did, desc))
            except Exception:
                failures.append(did)
        desc = pd.DataFrame(tuples, columns=['id', 'description'])
    return desc, failures


def get_function_descriptions(df, go_file):
    """ get descriptions of all protein functions in data frame """
    ids = df[df.type == 'function'].id.tolist()
    definitions = dict()

    # construct definitions dictionary from GO file
    with open(go_file, 'r') as f:
        keys = list()
        for line in f.readlines():
            if line.startswith('id:') or line.startswith('alt_id:'):
                _, key = line.strip().split(' ')
                keys.append(key)
            elif line.startswith('def:'):
                value = line[5:].strip()
                start, end = value.index('"'), value.rindex('"')
                for key in keys:
                    definitions[key] = value[start + 1 : end - 1]
                keys = list()

    # get definitions for all IDs for functions in msi
    tuples, failures = list(), list()
    for fid in tqdm(ids, desc='functions'):
        if fid in definitions:
            tuples.append((fid, definitions[fid]))
        else:
            failures.append(fid)

    desc = pd.DataFrame(tuples, columns=['id', 'description'])
    return desc, failures


def get_protein_descriptions(df, entrez_file):
    """ get descriptions of all proteins in data frame """
    ids = df[df.type == 'protein'].id.tolist()

    # get preloaded definitions from Entrez
    with open(entrez_file, 'r') as f:
        definitions = json.load(f)
    tuples, failures = list(), list()
    for pid in tqdm(ids, desc='proteins'):
        if pid in definitions:
            tuples.append((pid, definitions[pid]))
        else:
            failures.append(pid)

    desc = pd.DataFrame(tuples, columns=['id', 'description'])
    return desc, failures


def get_one_cui(cui):
    """ get info for a single UMLS CUI """
    info = umls.get_cui(cui)
    retrieved_ui = info['data']['ui']
    assert cui == retrieved_ui
    name = info['data']['name']
    definitions = info['definitions']

    # sources that are not in English or include HTML tags
    sources_to_exclude = ['MSHCZE', 'MSHPOR', 'MSHNOR', 'MSHFRE', 'SCTSPA',
                          'AOT', 'MEDLINEPLUS']
    if len(definitions) > 0:
        definitions = [defn for defn in definitions
                       if defn['rootSource'] not in sources_to_exclude]
        definition = definitions[0]['value']    # take first in list
    else:
        definition = 'unknown disease'

    res = {'cui': cui,
           'name': name,
           'definition': definition}
    return res


def get_disease_descriptions(df):
    """ get descriptions of all diseases in data frame """
    ids = df[df.type == 'disease'].id.tolist()

    # get description using UMLS for each disease CUI
    tuples, failures = list(), list()
    for did in tqdm(ids, desc='diseases'):
        try:
            result = get_one_cui(did)
        except Exception:
            failures.append(did)
        defn = result['definition'].strip().replace('\n', ' ')
        tuples.append((did, defn))

    desc = pd.DataFrame(tuples, columns=['id', 'description'])
    return desc, failures


if __name__ == '__main__':
    args = parser.parse_args()
    df = pd.read_table(args.msi_file, index_col=0, na_filter=False)

    # create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # drugs
    if not args.skip_drugs:
        print('getting drug descriptions...')
        fname = os.path.join(args.outdir, 'msi-drugs.tsv')
        if os.path.exists(fname):
            drug_desc = pd.read_table(fname, index_col=0, na_filter=False)
        else:
            drug_desc, failures = get_drug_descriptions(df, args.drugs_file)
            drug_desc.to_csv(fname, sep='\t')
            save_json(failures, fname.replace('.tsv', '-failures.txt'))

    # functions
    if not args.skip_functions:
        print('getting function descriptions...')
        fname = os.path.join(args.outdir, 'msi-functions.tsv')
        if os.path.exists(fname):
            function_desc = pd.read_table(fname, index_col=0, na_filter=False)
        else:
            function_desc, failures = get_function_descriptions(df,
                                                                args.go_file)
            function_desc.to_csv(fname, sep='\t')
            save_json(failures, fname.replace('.tsv', '-failures.txt'))

    # proteins
    if not args.skip_proteins:
        print('getting protein descriptions...')
        fname = os.path.join(args.outdir, 'msi-proteins.tsv')
        if os.path.exists(fname):
            protein_desc = pd.read_table(fname, index_col=0, na_filter=False)
        else:
            protein_desc, failures = get_protein_descriptions(df,
                                                              args.entrez_file)
            protein_desc.to_csv(fname, sep='\t')
            save_json(failures, fname.replace('.tsv', '-failures.txt'))
        protein_desc = protein_desc.astype(str)

    # diseases
    if not args.skip_diseases:
        print('getting disease descriptions...')
        fname = os.path.join(args.outdir, 'msi-diseases.tsv')
        if os.path.exists(fname):
            disease_desc = pd.read_table(fname, index_col=0, na_filter=False)
        else:
            disease_desc, failures = get_disease_descriptions(df)
            disease_desc.to_csv(fname, sep='\t')
            save_json(failures, fname.replace('.tsv', '-failures.txt'))

    # construct final dataframe and save
    desc = pd.concat([drug_desc, function_desc, protein_desc, disease_desc])
    df = pd.merge(df, desc, how='left', on='id').fillna('')
    df.to_csv(args.msi_file, sep='\t')

