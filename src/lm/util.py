""" utility functions and classes """

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers


class MLP(torch.nn.Module):
    """ class that defines a simple multilayer perception with NUM_LAYERS
        layers (minimum three), IN_CHANNELS input dimension, HIDDEN_CHANNELS
        hidden dimension, and OUT_CHANNELS output dimension, with ReLU
        nonlinearity and DROPOUT dropout rate. """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.dropouts.append(torch.nn.Dropout(p=dropout))
        for _ in range(num_layers - 3):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.dropouts.append(torch.nn.Dropout(p=dropout))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin, dropout in zip(self.lins[:-1], self.dropouts):
            x = lin(x)
            x = torch.relu(x)
            x = dropout(x)
        x = self.lins[-1](x)
        return x


class TextRetriever:
    """ helper class that retrieves text/token IDs for triples from
        provided files """

    def __init__(self, info_filename, relations_filename=None,
        use_descriptions=False, tokenized=False):

        # load file with entity text
        self.tokenized = tokenized
        if self.tokenized:
            self.info_file = torch.load(info_filename)
        else:
            self.info_file = pd.read_table(info_filename, index_col=0,
                                           na_filter=False)

        # load file with relation text
        self.use_relations = (relations_filename is not None)
        if self.use_relations:
            if self.tokenized:
                self.relations_file = torch.load(info_filename)
            else:
                self.relations_file = pd.read_table(relations_filename,
                                                    index_col=0,
                                                    na_filter=False)

        self.use_descriptions = use_descriptions

    def get_text(self, h, r, t):
        """ retrieve text for given head, relation, and tail indices """
        if self.tokenized:
            if self.use_relations:
                return [self.info_file[h],
                        self.relations_file[r],
                        self.info_file[t]]
            else:
                head_tokens = self.info_file[h]
                tail_tokens = self.info_file[t]
                if isinstance(head_tokens[0], list):
                    return [head_tokens[0], head_tokens[1],
                            tail_tokens[0], tail_tokens[1]]
                else:
                    return [head_tokens, tail_tokens]
        else:
            if self.use_descriptions:
                head_text = '; '.join([self.info_file.iloc[h]['name'],
                                       self.info_file.iloc[h]['description']])
                tail_text = '; '.join([self.info_file.iloc[t]['name'],
                                       self.info_file.iloc[t]['description']])
            else:
                head_text = self.info_file.iloc[h]['name']
                tail_text = self.info_file.iloc[t]['name']

            if self.use_relations:
                relation_text = self.relations_file.iloc[r]['name']
                return [head_text, relation_text, tail_text]
            else:
                return [head_text, tail_text]


def tokenize(x, tokenizer, tokenized=False, biencoder=False, max_length=128):
    """ return text in x processed into BERT input format, where x is a
        list of pairs of either entity text strings or pre-tokenized
        input IDs """
    if tokenizer is None:
        return x

    # tokenize all text first, then rest of function can be the same whether
    # tokenized is set to True or False
    if not tokenized:
        tokens = list()
        for example in x:
            tokens.append([tokenizer.convert_tokens_to_ids(
                               tokenizer.tokenize(elem)) for elem in example])
    else:
        tokens = x

    sep = tokenizer.sep_token_id

    # set token type IDs, which should both be zero for RoBERTa
    if isinstance(tokenizer, transformers.tokenization_roberta.RobertaTokenizer):
        type_id0, type_id1 = 0, 0
    else:
        type_id0, type_id1 = 0, 1

    if biencoder:
        head_input_ids, tail_input_ids = list(), list()
        head_token_type_ids, tail_token_type_ids = list(), list()
        # format is [CLS] head [SEP] relation [SEP],  [CLS] tail [SEP]
        #           0     0    0     1        1       0     0    0
        if len(tokens[0]) == 3:
            for head, relation, tail in tokens:

                # truncate head text
                num_rm = len(head) + len(relation) + 3 - max_length
                head, _, _ = \
                    tokenizer.truncate_sequences(head,
                        num_tokens_to_remove=num_rm,
                        truncation_strategy='longest_first')

                # contruct head, relation input IDs and token type IDs
                example = head + [sep] + relation
                head_input_ids.append(
                    tokenizer.build_inputs_with_special_tokens(example))
                head_type_ids = [type_id0] * (len(head) + 2)
                relation_type_ids = [type_id1] * (len(relation) + 1)
                head_token_type_ids.append(head_type_ids + relation_type_ids)

                # truncate tail text
                num_rm = len(tail) + 2 - max_length
                tail, _, _ = \
                    tokenizer.truncate_sequences(tail,
                        num_tokens_to_remove=num_rm,
                        truncation_strategy='longest_first')

                # construct tail input IDs and token type IDs
                tail_input_ids.append(
                    tokenizer.build_inputs_with_special_tokens(tail))
                tail_type_ids = [type_id0] * (len(tail) + 2)
                tail_token_type_ids.append(tail_type_ids)

        else:   # format is [CLS] head [SEP], [CLS] tail [SEP]
            for head, tail in tokens:

                # truncate head text
                num_rm = len(head) + 2 - max_length
                head, _, _ = \
                    tokenizer.truncate_sequences(head,
                        num_tokens_to_remove=num_rm,
                        truncation_strategy='longest_first')

                # contruct head, relation input IDs and token type IDs
                head_input_ids.append(
                    tokenizer.build_inputs_with_special_tokens(head))
                head_type_ids = [type_id0] * (len(head) + 2)
                head_token_type_ids.append(head_type_ids)

                # truncate tail text
                num_rm = len(tail) + 2 - max_length
                tail, _, _ = \
                    tokenizer.truncate_sequences(tail,
                        num_tokens_to_remove=num_rm,
                        truncation_strategy='longest_first')

                # construct tail input IDs and token type IDs
                tail_input_ids.append(
                    tokenizer.build_inputs_with_special_tokens(tail))
                tail_type_ids = [type_id0] * (len(tail) + 2)
                tail_token_type_ids.append(tail_type_ids)

        # concatenate input IDs and token type IDs for head and tail
        input_ids = head_input_ids + tail_input_ids
        token_type_ids = head_token_type_ids + tail_token_type_ids

    else:
        input_ids, token_type_ids = list(), list()
        # format is [CLS] head [SEP] relation [SEP] tail [SEP]
        #           0     0    0     1        1     0    0
        if len(tokens[0]) == 3:
            for head, relation, tail in tokens:
                num_rm = len(head) + len(tail) + len(relation) + 4 - max_length
                head, tail, _ = \
                    tokenizer.truncate_sequences(head, tail,
                        num_tokens_to_remove=num_rm,
                        truncation_strategy='longest_first')
                example = head + [sep] + relation + [sep] + tail
                input_ids.append(
                    tokenizer.build_inputs_with_special_tokens(example))
                head_type_ids = [type_id0] * (len(head) + 2)
                relation_type_ids = [type_id1] * (len(relation) + 1)
                tail_type_ids = [type_id0] * (len(tail) + 1)
                token_type_ids.append(
                    head_type_ids + relation_type_ids + tail_type_ids)

        elif len(tokens[0]) == 4:
            # format is [CLS] name1 [SEP] name2 [SEP] desc1 [SEP] desc2 [SEP]
            #           0     0     0     1     1     0     0     1     1
            for head, head_desc, tail, tail_desc in tokens:
                num_rm = len(head) + len(tail) + len(head_desc) + \
                    len(tail_desc) + 5 - max_length
                # only truncate descriptions, not names
                head_desc, tail_desc, _ = \
                    tokenizer.truncate_sequences(head_desc, tail_desc,
                        num_tokens_to_remove=num_rm,
                        truncation_strategy='longest_first')
                example = head + [sep] + tail + [sep] + \
                          head_desc + [sep] + tail_desc
                input_ids.append(
                    tokenizer.build_inputs_with_special_tokens(example))
                head_type_ids = [type_id0] * (len(head) + 2)
                head_desc_type_ids = [type_id0] * (len(head_desc) + 1)
                tail_type_ids = [type_id1] * (len(tail) + 1)
                tail_desc_type_ids = [type_id1] * (len(tail_desc) + 1)
                token_type_ids.append(head_type_ids + tail_type_ids + \
                    head_desc_type_ids + tail_desc_type_ids)

        else:   # format is [CLS] head [SEP] tail [SEP]
            for head, tail in tokens:
                num_rm = len(head) + len(tail) + 3 - max_length
                head, tail, _ = \
                    tokenizer.truncate_sequences(head, tail,
                        num_tokens_to_remove=num_rm,
                        truncation_strategy='longest_first')
                example = head + [sep] + tail
                input_ids.append(
                    tokenizer.build_inputs_with_special_tokens(example))
                head_type_ids = [type_id0] * (len(head) + 2)
                tail_type_ids = [type_id1] * (len(tail) + 1)
                token_type_ids.append(head_type_ids + tail_type_ids)

    # apply padding and return
    unpadded = {'input_ids': input_ids, 'token_type_ids': token_type_ids}
    tokens = tokenizer.pad(unpadded, return_tensors='pt')
    return tokens


def NegativeSamplingLoss(positive_score, negative_score):
    """ negative sampling loss for ranking """
    positive_loss = -F.logsigmoid(positive_score).mean()
    negative_loss = -F.logsigmoid(-negative_score).mean()
    return (positive_loss + negative_loss) / 2


def ComplEx(head, relation, tail, gamma):
    """ ComplEx score function """
    re_head, im_head = torch.chunk(head, 2, dim=-1)
    re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    score = re_score * re_tail + im_score * im_tail

    return torch.sum(score, dim=-1)


def DistMult(head, relation, tail, gamma):
    """ DistMult score function """
    return torch.sum(head * relation * tail, dim=-1)


def RotatE(head, relation, tail, gamma):
    """ RotatE score function """
    pi = 3.14159265358979323846
    embedding_dim = head.size(-1)
    embedding_range = (gamma + 2.0) / embedding_dim

    re_head, im_head = torch.chunk(head, 2, dim=-1)
    re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

    # make phases of relations uniformly distributed in [-pi, pi]
    phase_relation = relation / (embedding_range / pi)
    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    re_score = re_score - re_tail
    im_score = im_score - im_tail

    score = torch.stack([re_score, im_score], dim=0)
    score = torch.norm(score, dim=0)

    score = gamma - torch.sum(score, dim=-1)
    return score


def TransE(head, relation, tail, gamma):
    """ TransE score function """
    return gamma - torch.norm(head + relation - tail, dim=-1)


def sample_graph(triples, fraction):
    """ randomly sample a fraction of edges in a subset graph """
    ntotal = len(triples['head'])
    nsample = int(fraction * ntotal)
    indices = np.random.choice(ntotal, size=nsample, replace=False)
    sampled_triples = dict()
    for key, value in triples.items():
        if isinstance(value, list):
            sampled_triples[key] = [value[i] for i in indices]
        elif isinstance(value, np.ndarray):
            sampled_triples[key] = value[indices]
    return sampled_triples

