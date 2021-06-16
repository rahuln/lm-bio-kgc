""" classes to wrap datasets """

from itertools import chain

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from util import tokenize, TextRetriever


class DataBatch:
    """ wrapper class for batch of training data, which performs collating,
        tokenization, and transferring to pinned memory """

    def __init__(self, inputs, indices, relations, labels, tokenizer,
        tokenized=False, biencoder=False, max_length=128):

        self.inputs = inputs
        self.indices = indices
        self.relations = relations
        self.labels = labels

        # tokenize inputs and get token ids, type ids, and attention mask
        self.tokenizing = (tokenizer is not None)
        tokens = tokenize(inputs, tokenizer, tokenized=tokenized,
                          biencoder=biencoder, max_length=max_length)
        if self.tokenizing:
            self.input_ids = tokens.input_ids
            self.token_type_ids = tokens.token_type_ids
            self.attention_mask = tokens.attention_mask

    @staticmethod
    def from_batch(data, tokenizer, tokenized=False, biencoder=False,
        max_length=128):

        # collate entity text pairs
        inputs = [_['pos_sample'] for _ in data]
        inputs += list(chain(*zip(*[_['neg_sample'] for _ in data])))

        # collate entity indices
        idx = [_['pos_idx'] for _ in data]
        idx += list(chain(*zip(*[_['neg_idx'] for _ in data])))
        indices = torch.tensor(idx)

        # collate relation indices
        rel = [_['pos_rel'] for _ in data]
        rel += list(chain(*zip(*[_['neg_rel'] for _ in data])))
        relations = torch.tensor(rel)

        # collate binary triple classification labels
        y = [_['pos_y'] for _ in data]
        y += list(chain(*zip(*[_['neg_y'] for _ in data])))
        labels = torch.tensor(y).unsqueeze(1)

        return DataBatch(inputs, indices, relations, labels, tokenizer,
                         tokenized=tokenized, biencoder=biencoder,
                         max_length=max_length)

    def pin_memory(self):
        # copy tensors to pinned memory
        if self.tokenizing:
            self.input_ids = self.input_ids.pin_memory()
            self.token_type_ids = self.token_type_ids.pin_memory()
            self.attention_mask = self.attention_mask.pin_memory()
        self.indices = self.indices.pin_memory()
        self.relations = self.relations.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

    def to(self, device):
        # copy all tensors to device
        if self.tokenizing:
            self.input_ids = self.input_ids.to(device)
            self.token_type_ids = self.token_type_ids.to(device)
            self.attention_mask = self.attention_mask.to(device)
        self.indices = self.indices.to(device)
        self.relations = self.relations.to(device)
        self.labels = self.labels.to(device)
        return self


class TrainDataset(Dataset):
    def __init__(self, info_filename, triples, count, entity_dict,
        negative_sample_size=1, tokenized=False, use_descriptions=False,
        negatives_file=None, relations_filename=None):

        # initialize helper class to retrieve text for triples
        self.retriever = TextRetriever(info_filename,
                                       relations_filename=relations_filename,
                                       use_descriptions=use_descriptions,
                                       tokenized=tokenized)

        self.len = 2 * len(triples['head'])
        self.triples = triples
        self.count = count
        self.entity_dict = entity_dict
        self.negative_sample_size = negative_sample_size
        self.epoch = 0
        self.negatives = None
        if negatives_file is not None:
            self.negatives = torch.load(negatives_file, map_location='cpu')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        mode = 'head-batch' if idx % 2 == 0 else 'tail-batch'
        idx = idx // 2

        head = self.triples['head'][idx]
        relation = self.triples['relation'][idx]
        tail = self.triples['tail'][idx]

        pos_idx = [head, tail]
        pos_sample = self.retriever.get_text(head, relation, tail)

        # negative samples of head or tail nodes of the same entity type
        if mode == 'head-batch':
            if self.negatives is not None:
                neg_head = self.negatives[mode][idx, self.epoch,
                                                :self.negative_sample_size]
                neg_head = neg_head.numpy()
            else:
                low, high = self.entity_dict[self.triples['head_type'][idx]]
                neg_head = np.random.randint(low, high,
                                             size=self.negative_sample_size)
            neg_sample = [self.retriever.get_text(idx, relation, pos_idx[1])
                          for idx in neg_head]
            neg_idx = [[idx, pos_idx[1]] for idx in neg_head]
        elif mode == 'tail-batch':
            if self.negatives is not None:
                neg_tail = self.negatives[mode][idx, self.epoch,
                                                :self.negative_sample_size]
                neg_tail = neg_tail.numpy()
            else:
                low, high = self.entity_dict[self.triples['tail_type'][idx]]
                neg_tail = np.random.randint(low, high,
                                             size=self.negative_sample_size)
            neg_sample = [self.retriever.get_text(pos_idx[0], relation, idx)
                          for idx in neg_tail]
            neg_idx = [[pos_idx[0], idx] for idx in neg_tail]

        # package data example in dictionary and return
        return_dict = {'pos_sample' : pos_sample, 'neg_sample' : neg_sample,
                       'pos_idx' : pos_idx, 'neg_idx' : neg_idx,
                       'pos_rel' : relation,
                       'neg_rel' : [relation] * self.negative_sample_size,
                       'pos_y' : 1.,
                       'neg_y' : [0.] * self.negative_sample_size}
        return return_dict

    def set_epoch(self, epoch):
        """ set training epoch (for precomputed negative samples) """
        self.epoch = epoch

    @staticmethod
    def collate_fn(data, tokenizer, tokenized, biencoder, max_length):
        return DataBatch.from_batch(data, tokenizer=tokenizer,
                                    tokenized=tokenized, biencoder=biencoder,
                                    max_length=max_length)


class TestDataset(Dataset):
    def __init__(self, info_filename, triples, entity_dict, seed,
        tokenized=False, use_descriptions=False):
        self.tokenized = tokenized
        if self.tokenized:
            self.info_file = torch.load(info_filename)
        else:
            self.info_file = pd.read_table(info_filename, index_col=0,
                                           na_filter=False)
        self.len = 2 * len(triples['head'])
        self.triples = triples
        self.entity_dict = entity_dict
        self.use_descriptions = use_descriptions

        # generate index of negative sample to use for each test example
        torch.manual_seed(seed)
        num_neg = len(self.triples['head_neg'][0])
        self.neg_indices = torch.randint(0, num_neg, (self.len,)).tolist()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        mode = 'head-batch' if idx % 2 == 0 else 'tail-batch'
        idx = idx // 2

        head = self.triples['head'][idx]
        relation = self.triples['relation'][idx]
        tail = self.triples['tail'][idx]

        pos_idx = [head, tail]
        pos_sample = [self.get_text(head), self.get_text(tail)]

        # negative samples of head or tail nodes of the same entity type
        if mode == 'head-batch':
            neg_head = self.triples['head_neg'][idx, self.neg_indices[idx]]
            neg_sample = [self.get_text(neg_head), pos_sample[1]]
            neg_idx = [neg_head, pos_idx[1]]
        elif mode == 'tail-batch':
            neg_tail = self.triples['tail_neg'][idx, self.neg_indices[idx]]
            neg_sample = [pos_sample[0], self.get_text(neg_tail)]
            neg_idx = [pos_idx[0], neg_tail]
        else:
            raise

        # package data example in dictionary and return
        return_dict = {'pos_sample' : pos_sample, 'neg_sample' : neg_sample,
                       'pos_idx' : pos_idx, 'neg_idx' : neg_idx,
                       'pos_rel' : relation + 1, 'neg_rel' : 0,
                       'pos_y' : 1., 'neg_y' : 0.}
        return return_dict

    def get_text(self, idx):
        """ retrieve text for entity at the specified index """
        if self.tokenized:
            return self.info_file[idx]
        else:
            if self.use_descriptions:
                return '; '.join([self.info_file.iloc[idx]['name'],
                                  self.info_file.iloc[idx]['description']])
            else:
                return self.info_file.iloc[idx]['name']

    @staticmethod
    def collate_fn(data):
        x = [_['pos_sample'] for _ in data] + [_['neg_sample'] for _ in data]
        idx = [_['pos_idx'] for _ in data] + [_['neg_idx'] for _ in data]
        r = [_['pos_rel'] for _ in data] + [_['neg_rel'] for _ in data]
        y = [_['pos_y'] for _ in data] + [_['neg_y'] for _ in data]
        return_dict = {'samples' : x, 'relations' : torch.tensor(r),
                       'indices' : torch.tensor(idx),
                       'labels' : torch.tensor(y).unsqueeze(1)}
        return return_dict


class RankingTestDataset(Dataset):
    """ dataset that returns entity and relation indices of positive and all
        associated negative triples in a set of validation/test triples, for
        both head-batch and tail-batch modes """

    def __init__(self, triples):
        self.len = 2 * len(triples['head'])
        self.triples = triples

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        mode = 'head-batch' if idx % 2 == 0 else 'tail-batch'
        orig_idx = idx
        idx = idx // 2

        head = self.triples['head'][idx]
        relation = self.triples['relation'][idx]
        tail = self.triples['tail'][idx]

        positive = torch.tensor([head, tail])
        relation = torch.tensor([relation])

        # negative samples of head or tail nodes of the same entity type
        if mode == 'head-batch':
            neg_head = self.triples['head_neg'][idx]
            negatives = torch.tensor([[neg, tail] for neg in neg_head])
        elif mode == 'tail-batch':
            neg_tail = self.triples['tail_neg'][idx]
            negatives = torch.tensor([[head, neg] for neg in neg_tail])
        else:
            raise

        return positive, negatives, relation, orig_idx


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        # This just repeats indefinitely, until the max number of training
        # iterations has been reached.
        while True:
            for data in dataloader:
                yield data
