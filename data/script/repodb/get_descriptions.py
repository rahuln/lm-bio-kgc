""" scrape descriptions for drugs and diseases in repoDB, save to file """

import os
import json
import requests
from argparse import ArgumentParser

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm, trange

import umls


# command-line arguments
parser = ArgumentParser(description='get descriptions for entities in repoDB')
parser.add_argument('--info-file', type=str, default='repodb.csv',
                    help='CSV file for repoDB')
parser.add_argument('--outfile', type=str,
                    default='data/processed/repodb.json',
                    help='filename for saved entity names and descriptions')
args = parser.parse_args()


# make sure outfile is a .json file
if not args.outfile.endswith('.json'):
    args.outfile = args.outfile + '.json'

# load saved descriptions, if they exists
if os.path.exists(args.outfile):
    with open(args.outfile, 'r') as f:
        results = json.load(f)
    descriptions = results['descriptions']
else:
    descriptions = dict()


### DRUGS

def match(name1, name2):
    # helper function to check if drug name in repoDB matches scraped name
    name1, name2 = name1.lower(), name2.lower()
    return (name1 == name2) or (name1 in name2) or (name2 in name1)

# load repodb file, get unique set of drug IDs and names
df = pd.read_csv(args.info_file)
drug_ids = sorted(df.drug_id.unique().tolist())
drug_names = [df[df.drug_id == did].iloc[0].drug_name for did in drug_ids]
names = {did : name for did, name in zip(drug_ids, drug_names)}

# set up DrugBank URL template, set of mismatched names
url_template = 'https://go.drugbank.com/drugs/{drug_id}'
mismatches = list()

# retrieve description for each drug in repoDB
progbar = tqdm(zip(drug_ids, drug_names), total=len(drug_ids), ncols=100,
               desc='getting drug descriptions', ascii=True)
for drug_id, drug_name in progbar:
    if drug_id in descriptions:
        continue
    url = url_template.format(drug_id=drug_id)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    name = soup.find(id='generic-name').nextSibling.text
    if not match(name, drug_name):
        mismatches.append((drug_id, drug_name, name))
    try:
        description = soup.find(id='description').nextSibling.text
    except AttributeError:
        description = 'unknown drug'
    descriptions[drug_id] = description.strip().replace('\n', ' ')

# save names and descriptions to file
results = {'names' : names, 'descriptions' : descriptions}
with open(args.outfile, 'w') as f:
    json.dump(results, f, indent=4)


### DISEASES

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

# get unique set of disease IDs and names
disease_ids = sorted(df.ind_id.unique().tolist())
disease_names = [df[df.ind_id == did].iloc[0].ind_name for did in disease_ids]
names.update({did : name for did, name in zip(disease_ids, disease_names)})

# keep track of failures
failures = list()

# retrieve description for each disease in repoDB
progbar = tqdm(zip(disease_ids, disease_names), total=len(disease_ids),
               ncols=100, desc='getting disease descriptions', ascii=True)
for disease_id, disease_name in progbar:
    if disease_id in descriptions:
        continue
    try:
        result = get_one_cui(disease_id)
    except Exception:
        failures.append((disease_id, disease_name))
    if not match(result['name'], disease_name):
        mismatches.append((disease_id, disease_name, result['name']))
    descriptions[disease_id] = result['definition'].strip().replace('\n', ' ')

# save names and descriptions to file
results = {'names' : names, 'descriptions' : descriptions}
with open(args.outfile, 'w') as f:
    json.dump(results, f, indent=4)

# convert dictionary of names/descriptions to pandas data frame, save to file
columns = ['id', 'name', 'ent_type', 'description']
tuples = list()
ent_ids = sorted(drug_ids) + sorted(disease_ids)
ent_types = (['drug'] * len(drug_ids)) + (['disease'] * len(disease_ids))
for ent_id, ent_type in zip(ent_ids, ent_types):
    name, desc = results['names'][ent_id], results['descriptions'][ent_id]
    tuples.append((ent_id, name, ent_type, desc))
df = pd.DataFrame(tuples, columns=columns)
df.to_csv(args.outfile.replace('.json', '.tsv'), sep='\t')

