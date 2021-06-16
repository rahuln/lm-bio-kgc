# Biomedical KG edge splits

This directory should be used to store the files containing the transductive and inductive edge splits for the RepoDB, Hetionet, and MSI datasets used in our paper. To download the files, see the links in the `README.md` in the main directory of this repository. Each file can be loaded using PyTorch, e.g.,

```
split_edge = torch.load('repodb-edge-split-f0.2-neg500-s42.pt')
```

The inductive edge splits are denoted with the phrase `ind` in the filename. You can recreate these edge splits or create your own using the scripts titled `script/create_*_dataset.py` in the main Github repository. For the training scripts, use the `--subgraph` command-line argument to specify one of these files in order to perform training with a particular edge split.

## Structure of data

The loaded file will be a nested dictionary with the following structure:

* `num_nodes` : `int`, number of nodes/entities in knowledge graph
* `entity_dict` : dictionary from `str` to `(int, int)` that maps each entity type to the range of indices (inclusive, exclusive) of entities that have that type
* `train` : training set, dictionary from `str` to `list` or `np.ndarray`
	* `head` : `np.ndarray` of shape `(ntrain,)`, indices of head entities
	* `tail` : `np.ndarray` of shape `(ntrain,)`, indices of tail entities
	* `relation` : `np.ndarray` of shape `(ntrain,)`, indices of relations
	* `head_type` : `list` of length `ntrain`, string names of types for head entities (match the keys of `entity_dict`)
	* `tail_type` : `list` of length `ntrain`, string names of types for tail entities (match the keys of `entity_dict`)
* `test` : test set, dictionary from `str` to `list` or `np.ndarray`
	* `head` : `np.ndarray` of shape `(ntest,)`, indices of head entity for each positive triple
	* `tail` : `np.ndarray` of shape `(ntest,)`, indices of tail entity for each positive triple
	* `relation` : `np.ndarray` of shape `(ntest,)`, indices of relation for each positive triple
	* `head_type` : `list` of length `ntrain`, string names of types for head entities (match the keys of `entity_dict`)
	* `tail_type` : `list` of length `ntrain`, string names of types for tail entities (match the keys of `entity_dict`)
	* `head_neg` : `np.ndarray` of shape `(ntest, num_negatives)`, indices of negative entities to replace head entity for each positive triple
	* `tail_neg` : `np.ndarray` of shape `(ntest, num_negatives)`, indices of negative entities to replace tail entity for each positive triple

There is also a validation set dictionary, with key `valid` and the same structure as the test set.

## Adding node offsets

By default, the indices for head and tail entities will range from `[0, N_type - 1]`, where `N_type` is the number of entities for a given type. In order to add offsets for each entity type so that the entity indices will be in the range `[0, num_nodes - 1]`, you will need to use the function `add_node_offsets` found in `src/lm/preprocess.py` or `src/kge/preprocess.py` in the main Github repository (if you are using the training scripts, they will do this for you):

```
train = add_node_offsets(split_edge['train'], split_edge['entity_dict'])
```
