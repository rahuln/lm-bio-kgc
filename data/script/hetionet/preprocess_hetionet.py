""" script to preprocess hetionet dataset and construct entity info file """

from argparse import ArgumentParser
from collections import defaultdict
import json
import re
import requests
import os

from Bio import Entrez
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


# command-line arguments
parser = ArgumentParser(description='script to preprocess hetionet dataset '
                                    'and construct entity info file')
parser.add_argument('datafile', type=str,
                    help='path to hetionet knowledge graph JSON file')
parser.add_argument('--do-file', type=str, required=True,
                    help='path to disease ontology JSON file')
parser.add_argument('--mesh-file', type=str, required=True,
                    help='path to MeSH BIN file')
parser.add_argument('--email', type=str, required=True,
                    help='email for Entrez')
parser.add_argument('--drugbank-file', type=str, default=None,
                    help='path to TSV file with DrugBank names/descriptions')
parser.add_argument('--outdir', type=str, default='data/processed',
                    help='output directory for entity info file')


def get_compounds(ids, args):
    """ get compound names and descriptions from DrugBank """
    print('getting compounds...')
    mapping = dict()

    # get names and descriptions from DrugBank file if possible
    if args.drugbank_file is not None:
        df = pd.read_table(args.drugbank_file, index_col=0, na_filter=False)
        for drug_id in ids:
            elem = df[df.id == drug_id]
            if len(elem) == 1:
                mapping[drug_id] = {'name' : elem.name.item(),
                                    'description' : elem.description.item()}
    print(f'  found {len(mapping)} / {len(ids)} compounds in drugbank file')

    # scrape DrugBank directly for remaining names and descriptions
    remaining_ids = [drug_id for drug_id in ids if drug_id not in mapping]
    url_template = 'https://go.drugbank.com/drugs/{drug_id}'
    for drug_id in tqdm(remaining_ids, desc='  scraping drugbank'):
        page = requests.get(url_template.format(drug_id=drug_id))
        soup = BeautifulSoup(page.text, 'html.parser')

        # try to get name
        try:
            elem = soup.find('meta', {'name' : 'dc.title'})
            name = elem['content']
        except:
            name = 'unknown drug'

        # try to get description
        try:
            elem = soup.find('meta', {'name' : 'description'})
            desc = elem['content']
        except:
            desc = ''

        mapping[drug_id] = {'name' : name,
                            'description' : desc.strip().replace('\n', ' ')}

    # collect IDs, names, and descriptions into list of tuples
    tuples = list()
    for drug_id in ids:
        name = mapping[drug_id]['name']
        description = mapping[drug_id]['description']
        tuples.append((drug_id, name, 'compound', description))
    return tuples


def get_diseases(ids, args):
    """ get disease names and descriptions from Disease Ontology """
    print('getting diseases...')

    # process Disease Ontology file to get mapping from ID to name/description
    print('  processing Disease Ontology file...')
    with open(args.do_file, 'r') as f:
        do = json.load(f)
    entries = list(filter(lambda x: 'meta' in x, do['graphs'][0]['nodes']))
    mapping = dict()
    for entry in entries:
        if 'w3.org' in entry['id']:    # handle a metadata entry
            continue
        disease_id = entry['id'].split('/')[-1].replace('_', ':')
        mapping[disease_id] = {'name' : entry['lbl']}
        if 'definition' in entry['meta']:
            desc = entry['meta']['definition']['val'].replace('_', ' ')
            mapping[disease_id]['description'] = desc

    # create list of tuples with disease ID, name, and description
    tuples = list()
    for disease_id in ids:
        info = mapping[disease_id]
        name = info['name']
        description = info['description'] if 'description' in info else ''
        tuples.append((disease_id, name, 'disease', description))
    return tuples


def get_genes(ids, args):
    """ get gene names and descriptions from Entrez """
    Entrez.email = args.email
    id_list = ','.join([str(gene_id) for gene_id in ids])
    request = Entrez.epost('gene', id=id_list)
    result = Entrez.read(request)
    webEnv, queryKey = result['WebEnv'], result['QueryKey']
    data = Entrez.esummary(db='gene', webenv=webEnv, query_key=queryKey)
    annotations = Entrez.read(data)
    entries = annotations['DocumentSummarySet']['DocumentSummary']

    # create list of tuples with gene ID, name, and description
    citation_pattern = ' \([A-Za-z]+ et al..*\)'
    source_pattern = '\[.*\]$'
    tuples = list()
    for i, entry in enumerate(entries):
        gene_id = entry.attributes['uid']
        assert gene_id == str(ids[i]), 'gene IDs do not match'
        name, description = entry['Name'], entry['Summary']

        # remove citations and source information from description
        description = re.sub(citation_pattern, '', description)
        description = re.sub(source_pattern, '', description)

        tuples.append((gene_id, name, 'gene', description.strip()))
    return tuples


def get_side_effects(ids, args):
    """ get side effect names and descriptions from SIDER """
    print('getting side effects...')

    # scrape side effect name and definition from SIDER URL
    tuples = list()
    url_template = 'http://sideeffects.embl.de/se/{sideeffect_id}'
    for sideeffect_id in tqdm(ids, desc='  scraping sider'):
        page = requests.get(url_template.format(sideeffect_id=sideeffect_id))
        soup = BeautifulSoup(page.text, 'html.parser')
        name = soup.find('h1').text
        description = soup.find('div', {'class' : 'boxDiv'}).p.text
        if 'Definition:' not in description:
            description = ''
        else:
            description = description.replace('Definition: ', '')
        tuples.append((sideeffect_id, name, 'sideeffect', description))
    return tuples


def get_symptoms(ids, args):
    """ get symptom names and descriptions from MeSH """
    print('getting symptoms...')

    # process MeSH file to get mapping from ID to name and description
    with open(args.mesh_file, 'rb') as f:
        lines = f.readlines()
    lines = [line.decode('utf-8') for line in lines]
    mapping = dict()
    entry = dict()
    for line in lines:
        if 'NEWRECORD' in line:
            entry = dict()
        elif line.startswith('MH = '):
            entry['name'] = line[5:].strip()
        elif line.startswith('MS = '):
            entry['description'] = line[5:].strip()
        elif line.startswith('UI = '):
            if 'name' not in entry:
                raise ValueError('missing name for MeSH entry')
            if 'description' not in entry:
                entry['description'] = ''
            mapping[line[5:].strip()] = entry

    # construct list of tuples with symptom ID, name, and description
    tuples = list()
    for symptom_id in ids:
        name = mapping[symptom_id]['name']
        description = mapping[symptom_id]['description']
        tuples.append((symptom_id, name, 'symptom', description))
    return tuples


if __name__ == '__main__':
    args = parser.parse_args()

    # load hetionet JSON file, restrict to certain relation types, then get
    # entity IDs and types
    print('loading hetionet JSON file...')
    with open(args.datafile, 'r') as f:
        data = json.load(f)
    include = {'treats', 'presents', 'associates', 'causes'}
    edges = list(filter(lambda x: x['kind'] in include, data['edges']))
    types = {'Compound', 'Disease', 'Gene', 'Side Effect', 'Symptom'}
    entities = defaultdict(set)
    for edge in edges:
        source, target = edge['source_id'], edge['target_id']
        if source[0] in types:
            entities[source[0]].add(source[1])
        if target[0] in types:
            entities[target[0]].add(target[1])
    entities = {key : sorted(list(value)) for key, value in entities.items()}

    # construct directory for intermediate results
    tmp_dir = os.path.join(args.outdir, 'hetionet')
    os.makedirs(tmp_dir, exist_ok=True)

    # helper function to check for save tuples and load if they exist,
    # otherwise construct them with the specified function and save
    def load_saved_tuples(fname, func, ids, args):
        if os.path.exists(fname):
            print(f'found {fname}, loading...')
            with open(fname, 'r') as f:
                tuples = json.load(f)
        else:
            tuples = func(ids, args)
            with open(fname, 'w') as f:
                json.dump(tuples, f, indent=4)
        return tuples

    # get names and descriptions for entities
    compounds_fname = os.path.join(tmp_dir, 'compounds.json')
    compounds = load_saved_tuples(compounds_fname, get_compounds,
                                  entities['Compound'], args)

    diseases_fname = os.path.join(tmp_dir, 'diseases.json')
    diseases = load_saved_tuples(diseases_fname, get_diseases,
                                 entities['Disease'], args)

    genes_fname = os.path.join(tmp_dir, 'genes.json')
    genes = load_saved_tuples(genes_fname, get_genes,
                              entities['Gene'], args)

    sideeffects_fname = os.path.join(tmp_dir, 'sideeffects.json')
    sideeffects = load_saved_tuples(sideeffects_fname, get_side_effects,
                                    entities['Side Effect'], args)

    symptoms_fname = os.path.join(tmp_dir, 'symptoms.json')
    symptoms = load_saved_tuples(symptoms_fname, get_symptoms,
                                 entities['Symptom'], args)

    # construct data frame from all tuples
    print('saving entity names and descriptions to .tsv file...')
    tuples = compounds + diseases + genes + sideeffects + symptoms
    df = pd.DataFrame(tuples, columns=['id', 'name', 'type', 'description'])
    df.to_csv(os.path.join(args.outdir, 'hetionet.tsv'), sep='\t')

