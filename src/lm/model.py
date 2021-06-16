""" pytorch modules """

import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, Embedding, Conv1d
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from util import MLP, ComplEx, DistMult, RotatE, TransE


class KGBERT(torch.nn.Module):
    """ BERT-based encoder with linear layer on top of CLS token for
        classification """

    def __init__(self, args):
        """ required arguments: model_name, link_prediction,
            relation_prediction, nrelation, relevance_ranking """
        super(KGBERT, self).__init__()

        self.model = args.model_name

        self.encoder = AutoModel.from_pretrained(self.model)
        hidden_size = self.encoder.config.hidden_size

        self.link_prediction = args.link_prediction
        if self.link_prediction:
            self.link_head = Linear(hidden_size, 1)

        self.relation_prediction = args.relation_prediction
        if self.relation_prediction:
            self.relation_head = Linear(hidden_size, args.nrelation)

        self.relevance_ranking = args.relevance_ranking
        if self.relevance_ranking:
            self.ranking_head = Linear(hidden_size, 1)

    def forward(self, data, return_encodings=False):
        output = self.encoder(input_ids=data.input_ids,
                              token_type_ids=data.token_type_ids,
                              attention_mask=data.attention_mask,
                              return_dict=True)
        cls_token = output.last_hidden_state[:, 0, :]
        outputs = dict()
        if self.link_prediction:
            outputs['link_outputs'] = self.link_head(cls_token)
        if self.relation_prediction:
            outputs['relation_outputs'] = self.relation_head(cls_token)
        if self.relevance_ranking:
            outputs['ranking_outputs'] = self.ranking_head(cls_token)

        if return_encodings:
            outputs['encodings'] = cls_token

        return outputs

    def reset_parameters(self):
        self.encoder = AutoModel.from_pretrained(self.model)
        self.linear.reset_parameters()


class KGBERTPlusEmbeddings(torch.nn.Module):
    """ BERT-based encoder combined with KG embedding-based entity and
        relation embeddings, with MLP on top for final output """

    def __init__(self, args):
        """ required arguments: model_name, checkpoint_file, num_hidden_layers,
            hidden_dim, dropout, average_emb """
        super(KGBERTPlusEmbeddings, self).__init__()

        self.model = args.model_name

        # load BERT model
        self.encoder = AutoModel.from_pretrained(self.model)

        # load entity and relation embeddings, set whether to average-pool
        checkpoint = torch.load(args.checkpoint_file, map_location='cpu')
        ent_emb = checkpoint['model_state_dict']['entity_embedding']
        rel_emb = checkpoint['model_state_dict']['relation_embedding']
        self.emb_dim = rel_emb.size(1)
        self.average_emb = args.average_emb

        # if entity embeddings dimension is twice relation embeddings
        # dimension, average across halves
        if ent_emb.size(1) == 2 * self.emb_dim:
            ent_emb = (ent_emb[:, :self.emb_dim] +
                       ent_emb[:, self.emb_dim:]) / 2.

        # convert entity and relation embeddings to parameters
        self.ent_emb = torch.nn.Parameter(ent_emb, requires_grad=False)
        self.rel_emb = torch.nn.Parameter(rel_emb, requires_grad=False)

        # MLP for combining entity/relation embeddings and LM encodings
        in_channels = self.emb_dim if self.average_emb else 3 * self.emb_dim
        in_channels += self.encoder.config.hidden_size
        self.mlp = MLP(in_channels, args.hidden_dim, 1,
                       args.num_hidden_layers + 2, args.dropout)

    def forward(self, data):

        # retrieve LM encodings
        output = self.encoder(input_ids=data.input_ids,
                              token_type_ids=data.token_type_ids,
                              attention_mask=data.attention_mask,
                              return_dict=True)
        cls_token = output.last_hidden_state[:, 0, :]

        # retrieve entity/relation embeddings
        head = self.ent_emb[data.indices[:, 0]]
        tail = self.ent_emb[data.indices[:, 1]]
        relation = self.rel_emb[data.relations]
        if self.average_emb:
            embeddings = (head + relation + tail) / 3.
        else:
            embeddings = torch.cat([head, relation, tail], dim=1)

        # concatenate embeddings with LM encodings, pass through MLP,
        # and return scores
        outputs = dict()
        mlp_inputs = torch.cat([embeddings, cls_token], dim=1)
        outputs['link_outputs'] = self.mlp(mlp_inputs)
        return outputs

    def reset_parameters(self):
        self.encoder = AutoModel.from_pretrained(self.model)
        self.linear.reset_parameters()


class KGBERTBiEncoder(torch.nn.Module):
    """ BERT-based encoder with linear layer on top of CLS token for
        classification, with head and tail entities passed through
        separate encoders and their CLS token representations
        concatenated """

    def __init__(self, args):
        """ required arguments: model_name, link_prediction,
            relation_prediction, nrelation, relevance_ranking """
        super(KGBERTBiEncoder, self).__init__()

        self.model = args.model_name

        self.encoder = AutoModel.from_pretrained(self.model)
        hidden_size = self.encoder.config.hidden_size

        self.link_prediction = args.link_prediction
        if self.link_prediction:
            self.link_head = Linear(2 * hidden_size, 1)

        self.relation_prediction = args.relation_prediction
        if self.relation_prediction:
            self.relation_head = Linear(2 * hidden_size, args.nrelation)

        self.relevance_ranking = args.relevance_ranking
        if self.relevance_ranking:
            self.ranking_head = Linear(2 * hidden_size, 1)

    def forward(self, data):
        output = self.encoder(input_ids=data.input_ids,
                              token_type_ids=data.token_type_ids,
                              attention_mask=data.attention_mask,
                              return_dict=True)
        cls_token = output.last_hidden_state[:, 0, :]
        e1, e2 = torch.chunk(cls_token, 2, dim=0)
        cls_token = torch.cat([e1, e2], dim=1)
        outputs = dict()
        if self.link_prediction:
            outputs['link_outputs'] = self.link_head(cls_token)
        if self.relation_prediction:
            outputs['relation_outputs'] = self.relation_head(cls_token)
        if self.relevance_ranking:
            outputs['ranking_outputs'] = self.ranking_head(cls_token)
        return outputs

    def reset_parameters(self):
        self.encoder = AutoModel.from_pretrained(self.model)
        self.linear.reset_parameters()


class KGBERTWithKGEInputs(torch.nn.Module):
    """ version of KG-BERT that prepends trained KG entity embeddings to text
        for each entity before doing forward pass """

    def __init__(self, args):
        """ required arguments: model_name, link_prediction,
            relation_prediction, nrelation, relevance_ranking,
            checkpoint_file """
        super(KGBERTWithKGEInputs, self).__init__()

        self.model = args.model_name

        self.encoder = AutoModel.from_pretrained(self.model)
        hidden_size = self.encoder.config.hidden_size

        # get CLS and SEP token IDs
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

        self.link_prediction = args.link_prediction
        if self.link_prediction:
            self.link_head = Linear(hidden_size, 1)

        self.relation_prediction = args.relation_prediction
        if self.relation_prediction:
            self.relation_head = Linear(hidden_size, args.nrelation)

        self.relevance_ranking = args.relevance_ranking
        if self.relevance_ranking:
            self.ranking_head = Linear(hidden_size, 1)

        # load KG embeddings from checkpoint file, modify BERT embeddings
        # to add KG embeddings after word embeddings
        checkpoint = torch.load(args.checkpoint_file, map_location='cpu')
        ent_emb = checkpoint['model_state_dict']['entity_embedding']

        # perform linear alignment to map entity embedding space to input
        # embedding space
        if args.align_embeddings:
            ent_emb = \
                self.align_embeddings(ent_emb, self.encoder, tokenizer, args)

        nentity = ent_emb.size(0)
        self.vocab_size = \
            self.encoder.embeddings.word_embeddings.weight.size(0)
        new_embeddings = torch.nn.Embedding(self.vocab_size + nentity,
                                            hidden_size)
        new_embeddings.weight.data[:self.vocab_size, :].copy_(
            self.encoder.embeddings.word_embeddings.weight.data)
        new_embeddings.weight.data[self.vocab_size:, :].copy_(ent_emb)
        self.encoder.embeddings.word_embeddings = new_embeddings

    def forward(self, data, return_encodings=False):

        # modify input IDs, token type IDs, and attention mask to account for
        # prepending entity embeddings
        input_ids = data.input_ids.tolist()
        token_type_ids = data.token_type_ids.tolist()
        attention_mask = data.attention_mask.tolist()
        ent_inds = (data.indices + self.vocab_size).tolist()
        device = data.input_ids.device

        for i in range(len(input_ids)):

            # handle head entity embedding
            idx = input_ids[i].index(self.cls_token_id)
            input_ids[i].insert(idx + 1, ent_inds[i][0])
            token_type_ids[i].insert(idx + 1, token_type_ids[i][idx + 1])
            attention_mask[i].insert(idx + 1, attention_mask[i][idx + 1])

            # handle tail entity embedding
            idx = input_ids[i].index(self.sep_token_id)
            input_ids[i].insert(idx + 1, ent_inds[i][1])
            token_type_ids[i].insert(idx + 1, token_type_ids[i][idx + 1])
            attention_mask[i].insert(idx + 1, attention_mask[i][idx + 1])

        # convert back to tensors and move to device
        input_ids = torch.tensor(input_ids).to(device)
        token_type_ids = torch.tensor(token_type_ids).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)

        # do forward pass
        output = self.encoder(input_ids=input_ids,
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask,
                              return_dict=True)
        cls_token = output.last_hidden_state[:, 0, :]
        outputs = dict()
        if self.link_prediction:
            outputs['link_outputs'] = self.link_head(cls_token)
        if self.relation_prediction:
            outputs['relation_outputs'] = self.relation_head(cls_token)
        if self.relevance_ranking:
            outputs['ranking_outputs'] = self.ranking_head(cls_token)

        if return_encodings:
            outputs['encodings'] = cls_token

        return outputs

    def align_embeddings(self, ent_emb, encoder, tokenizer, args):
        """ helper function to perform linear alignment to transform entity
            embedding space to LM input embedding space """
        if args.tokenized:
            tokens = torch.load(args.info_filename)
        else:
            df = pd.read_table(args.info_filename, index_col=0,
                               na_filter=False)
            names = df['name'].tolist()
            tokens = [tokenizer.convert_tokens_to_ids(
                          tokenizer.tokenize(name))
                      for name in names]

        # find indices of entity names that exist in encoder vocabulary
        lengths = np.array(list(map(len, tokens)))
        idx = np.where(lengths == 1)[0]

        # retrieve input and entity embeddings at the corresponding indices
        A = ent_emb[idx].numpy()
        input_ids = torch.tensor([tokens[i][0] for i in idx])
        B = encoder.embeddings.word_embeddings(input_ids).detach().numpy()

        # learn linear alignment, use to transform entity embeddings
        W, _, _, _ = np.linalg.lstsq(A, B)
        ent_emb = ent_emb @ torch.from_numpy(W)

        return ent_emb

    def reset_parameters(self):
        self.encoder = AutoModel.from_pretrained(self.model)
        self.linear.reset_parameters()


class BLP(torch.nn.Module):
    """ BERT for link prediction encoder, which encodes text for each entity as
        an entity embedding and applies a linear layer to the CLS token
        representation of each to learn an entity embedding, while also
        learning a matrix of relation embeddings. Superclass of BLPBiEncoder
        and BLPCrossEncoder, which each use a BERT bi-encoder and
        cross-encoder to encode entity text, respectively. """

    def __init__(self, args):
        """ required arguments: model_name, link_prediction,
            relation_prediction, nrelation, embedding_dim, score, gamma,
            regularization, entity_representation, checkpoint_file """
        super(BLP, self).__init__()

        self.model = args.model_name
        self.gamma = args.gamma
        self.regularization = args.regularization
        self.entity_representation = args.entity_representation

        # set up BERT encoder
        self.encoder = AutoModel.from_pretrained(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        hidden_size = self.encoder.config.hidden_size

        # check for valid score function
        if args.score not in ('ComplEx', 'DistMult', 'RotatE', 'TransE'):
            raise ValueError(f'score function {args.score} not available')
        score_functions = {'ComplEx' : ComplEx, 'DistMult' : DistMult,
                           'RotatE' : RotatE, 'TransE' : TransE}
        self.score_function = score_functions[args.score]

        # linear layer to convert CLS token to entity embedding
        if args.score in ('ComplEx', 'RotatE'):
            entity_dim = 2 * args.embedding_dim
        else:
            entity_dim = args.embedding_dim
        self.linear = Linear(hidden_size, entity_dim)

        # relation embeddings
        if args.score == 'ComplEx':
            relation_dim = 2 * args.embedding_dim
        else:
            relation_dim = args.embedding_dim
        self.relation_embedding = Embedding(args.nrelation, relation_dim)

        # load checkpoint file for KG embeddings (for regularization)
        self.loaded_kge = False
        if args.checkpoint_file is not None:
            self.loaded_kge = True
            checkpoint = torch.load(args.checkpoint_file, map_location='cpu')
            ent_emb = checkpoint['model_state_dict']['entity_embedding']
            rel_emb = checkpoint['model_state_dict']['relation_embedding']
            self.ent_emb = torch.nn.Parameter(ent_emb, requires_grad=False)

            # initialize relation embeddings from checkpoint
            self.relation_embedding.weight.data.copy_(rel_emb)

        # set up parameters for link prediction
        self.link_prediction = args.link_prediction
        if self.link_prediction:
            self.link_head = Linear(3 * hidden_size, 1)

        # set up parameters for relation prediction
        self.relation_prediction = args.relation_prediction
        if self.relation_prediction:
            self.relation_head = Linear(3 * hidden_size, args.nrelation)

    def forward(self, data):
        raise NotImplementedError

    def reset_parameters(self, data):
        raise NotImplementedError


class BLPBiEncoder(BLP):
    """ BERT for link prediction encoder, which encodes entities separately
        and applies a linear layer to the CLS token representation of each
        to learn an entity embedding, while also learning a matrix of
        relation embeddings. Based on 'Inductive Entity Representations from
        Text via Link Prediction', Daza et al. (2021) """

    def forward(self, data, return_entity_embeddings=False):
        output = self.encoder(input_ids=data.input_ids,
                              token_type_ids=data.token_type_ids,
                              attention_mask=data.attention_mask,
                              return_dict=True)

        if self.entity_representation == 'cls':
            cls_token = output.last_hidden_state[:, 0, :]
            head_emb, tail_emb = torch.chunk(cls_token, 2, dim=0)
        elif self.entity_representation == 'mean':
            # index of first and second [SEP] token for each item in batch
            inds = (data.input_ids == self.tokenizer.sep_token_id).nonzero()
            sep_inds = inds[:, 1]
            mean_pooled = list()
            outputs = output.last_hidden_state
            for i in range(sep_inds.size(0)):
                # tokens of entity text (between [CLS] and [SEP])
                vecs = outputs[i, 1 : sep_inds[i]]
                mean_pooled.append(vecs.mean(dim=0, keepdim=True))
            mean_pooled = torch.cat(mean_pooled)
            head_emb, tail_emb = torch.chunk(mean_pooled, 2, dim=0)

        head = self.linear(head_emb)
        tail = self.linear(tail_emb)

        relation = self.relation_embedding(data.relations)

        score = self.score_function(head, relation, tail, self.gamma)
        outputs = {'ranking_outputs' : score}

        # regularize BERT entity embeddings to match KG embeddings
        if self.loaded_kge and self.regularization > 0:
            kge_head = self.ent_emb[data.indices[:, 0]]
            kge_tail = self.ent_emb[data.indices[:, 1]]
            emb_norm = torch.norm(kge_head - head) ** 2 + \
                       torch.norm(kge_tail - tail) ** 2
            outputs['regularization'] = self.regularization * emb_norm

        # compute link prediction and relation prediction losses
        triple_embedding = torch.cat([head_emb, tail_emb,
                                      torch.abs(head_emb - tail_emb)], dim=1)
        if self.link_prediction:
            outputs['link_outputs'] = self.link_head(triple_embedding)
        if self.relation_prediction:
            outputs['relation_outputs'] = self.relation_head(triple_embedding)

        # add entity embeddings to output dictionary
        if return_entity_embeddings:
            outputs['head'] = head
            outputs['tail'] = tail

        return outputs

    def reset_parameters(self):
        self.encoder = AutoModel.from_pretrained(self.model)
        self.linear.reset_parameters()
        self.relation_embedding.reset_parameters()


class BLPCrossEncoder(BLP):
    """ BERT for link prediction encoder, which encodes entities jointly and
        applies a linear layer to the representation of each to learn an
        entity embedding, while also learning a matrix of relation embeddings.
        """

    def forward(self, data, return_entity_embeddings=False):
        output = self.encoder(input_ids=data.input_ids,
                              token_type_ids=data.token_type_ids,
                              attention_mask=data.attention_mask,
                              return_dict=True)

        # get representation of head and tail entity text from contextualized
        # token embeddings at the output layer
        if self.entity_representation == 'cls':
            head_emb = output.last_hidden_state[:, 0, :]

            # find [SEP] token just before second entity text
            mask = (data.input_ids == self.tokenizer.sep_token_id)
            sep_token = output.last_hidden_state[mask, :]
            # take every other [SEP] token since there are two per input
            tail_emb = sep_token[torch.arange(0, sep_token.size(0), 2)]
        elif self.entity_representation == 'mean':
            # index of first and second [SEP] token for each item in batch
            inds = (data.input_ids == self.tokenizer.sep_token_id).nonzero()
            sep_first = inds[torch.arange(0, inds.size(0), 2)][:, 1]
            sep_second = inds[torch.arange(1, inds.size(0), 2)][:, 1]
            head_emb = list()
            tail_emb = list()
            outputs = output.last_hidden_state
            for i in range(sep_first.size(0)):
                # tokens of head entity text (between [CLS] and first [SEP])
                head_vecs = outputs[i, 1 : sep_first[i]]
                head_emb.append(head_vecs.mean(dim=0, keepdim=True))
                # tokens of tail entity text (between first and second [SEP])
                tail_vecs = outputs[i, sep_first[i] + 1 : sep_second[i]]
                tail_emb.append(tail_vecs.mean(dim=0, keepdim=True))
            head_emb = torch.cat(head_emb)
            tail_emb = torch.cat(tail_emb)

        head = self.linear(head_emb)
        tail = self.linear(tail_emb)

        relation = self.relation_embedding(data.relations)

        score = self.score_function(head, relation, tail, self.gamma)
        outputs = {'ranking_outputs' : score}

        # regularize BERT entity embeddings to match KG embeddings
        if self.loaded_kge and self.regularization > 0:
            kge_head = self.ent_emb[data.indices[:, 0]]
            kge_tail = self.ent_emb[data.indices[:, 1]]
            emb_norm = torch.norm(kge_head - head) ** 2 + \
                       torch.norm(kge_tail - tail) ** 2
            outputs['regularization'] = self.regularization * emb_norm

        # compute link prediction and relation prediction losses
        triple_embedding = torch.cat([head_emb, tail_emb,
                                      torch.abs(head_emb - tail_emb)], dim=1)
        if self.link_prediction:
            outputs['link_outputs'] = self.link_head(triple_embedding)
        if self.relation_prediction:
            outputs['relation_outputs'] = self.relation_head(triple_embedding)

        # add entity embeddings to output dictionary
        if return_entity_embeddings:
            outputs['head'] = head
            outputs['tail'] = tail

        return outputs

    def reset_parameters(self):
        self.encoder = AutoModel.from_pretrained(self.model)
        self.linear.reset_parameters()
        self.relation_embedding.reset_parameters()


class KGEModel(torch.nn.Module):
    """ KG embedding model encoder, which learns an embedding for every
        entity and relation in the knowledge graph and computes the score
        for a pair of entities and a relation using their embeddings and
        a specified score function """

    def __init__(self, args):
        """ required arguments: embedding_dim, nentity, nrelation, score,
            gamma, checkpoint_file """
        super(KGEModel, self).__init__()

        self.gamma = args.gamma
        self.embedding_range = (self.gamma + 2.0) / args.embedding_dim

        # check for valid score function
        if args.score not in ('ComplEx', 'DistMult', 'RotatE', 'TransE'):
            raise ValueError(f'score function {args.score} not available')
        score_functions = {'ComplEx' : ComplEx, 'DistMult' : DistMult,
                           'RotatE' : RotatE, 'TransE' : TransE}
        self.score_function = score_functions[args.score]

        # load entity and relation embeddings from checkpoint file
        if args.checkpoint_file is not None:
            checkpoint = torch.load(args.checkpoint_file, map_location='cpu')

        # entity embeddings
        if args.score in ('ComplEx', 'RotatE'):
            entity_dim = 2 * args.embedding_dim
        else:
            entity_dim = args.embedding_dim
        self.entity_embedding = Embedding(args.nentity, entity_dim)
        if args.checkpoint_file is None:
            torch.nn.init.uniform_(
                tensor=self.entity_embedding.weight,
                a=-self.embedding_range,
                b=self.embedding_range)
        else:
            ent_emb = checkpoint['model_state_dict']['entity_embedding']
            self.entity_embedding.weight.data.copy_(ent_emb)

        # relation embeddings
        if args.score == 'ComplEx':
            relation_dim = 2 * args.embedding_dim
        else:
            relation_dim = args.embedding_dim
        self.relation_embedding = Embedding(args.nrelation, relation_dim)
        if args.checkpoint_file is None:
            torch.nn.init.uniform_(
                tensor=self.relation_embedding.weight,
                a=-self.embedding_range,
                b=self.embedding_range)
        else:
            rel_emb = checkpoint['model_state_dict']['relation_embedding']
            self.relation_embedding.weight.data.copy_(rel_emb)

    def forward(self, data, return_entity_embeddings=False):
        head = self.entity_embedding(data.indices[:, 0])
        tail = self.entity_embedding(data.indices[:, 1])
        relation = self.relation_embedding(data.relations)
        score = self.score_function(head, relation, tail, self.gamma)
        outputs = {'ranking_outputs' : score}

        # add entity embeddings to output dictionary
        if return_entity_embeddings:
            outputs['head'] = head
            outputs['tail'] = tail

        return outputs

    def reset_parameters(self):
        self.entity_embedding.reset_parameters()
        self.relation_embedding.reset_parameters()


class JointKGBERTAndKGEModel(KGBERT):
    """ model that computes ranking score as sum of KG-BERT and KGEModel
        ranking outputs, with scores combined using an optional weighted
        average with learned weights if specified """

    def __init__(self, args):
        """ required arguments (in addition to those of KGBERT and KGEModel):
            weighted_average """
        super(JointKGBERTAndKGEModel, self).__init__(args)
        self.kge_model = KGEModel(args)

        # create parameter for weighted average of scores
        self.weighted_average = args.weighted_average
        if self.weighted_average:
            self.alpha = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data):
        outputs = super().forward(data)
        kge_outputs = self.kge_model(data)
        kgbert_scores = outputs['ranking_outputs'].flatten()
        kge_scores = kge_outputs['ranking_outputs'].flatten()
        if self.weighted_average:
            weight = torch.sigmoid(self.alpha)
            scores = weight * kgbert_scores + (1 - weight) * kge_scores
        else:
            scores = kgbert_scores + kge_scores
        outputs['ranking_outputs'] = scores
        return outputs

    def reset_parameters(self):
        super().reset_parameters()
        self.kge_model.reset_parameters()


class JointBLPCrossEncoderAndKGEModel(BLPCrossEncoder):
    """ model that jointly trains BLPCrossEncoder and KGEModel with score
        function being a weighted average of the scores of each model, with
        additional (optional) regularization that encourages the models'
        entity embeddings to be similar """

    def __init__(self, args):
        # don't load KGE checkpoint in BLPCrossEncoder
        """ required arguments (in addition to those of BLPCrossEncoder and
            KGEModel): weighted_average """
        checkpoint_file = args.checkpoint_file
        args.checkpoint_file = None
        super(JointBLPCrossEncoderAndKGEModel, self).__init__(args)
        args.checkpoint_file = checkpoint_file

        self.kge_model = KGEModel(args)

        # create parameter for weighted average of scores
        self.weighted_average = args.weighted_average
        if self.weighted_average:
            self.alpha = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data):
        outputs = super().forward(data, return_entity_embeddings=True)
        kge_outputs = self.kge_model(data, return_entity_embeddings=True)
        blp_scores = outputs['ranking_outputs'].flatten()
        kge_scores = kge_outputs['ranking_outputs'].flatten()

        # compute weighted average of scores instead of adding, if specified
        if self.weighted_average:
            weight = torch.sigmoid(self.alpha)
            scores = weight * blp_scores + (1 - weight) * kge_scores
        else:
            scores = blp_scores + kge_scores
        outputs['ranking_outputs'] = scores

        # regularize BERT and KG entity embeddings to be similar to each other
        if self.regularization:
            head, tail = outputs['head'], outputs['tail']
            kge_head, kge_tail = kge_outputs['head'], kge_outputs['tail']
            emb_norm = torch.norm(kge_head - head) ** 2 + \
                       torch.norm(kge_tail - tail) ** 2
            outputs['regularization'] = self.regularization * emb_norm

        return outputs

    def reset_parameters(self):
        super().reset_parameters()
        self.kge_model.reset_parameters()


class DKRLBiEncoder(torch.nn.Module):
    """ Description-Embodied Knowledge Representation Learning (DKRL) with CNN
        encoder, from
        Xie et al., "Representation Learning of Knowledge Graphs with Entity
        Descriptions"
        Code adapted from https://github.com/dfdazac/blp """

    def __init__(self, args):
        """ required arguments: model_name, link_prediction,
            relation_prediction, nrelation, embedding_dim, score, gamma,
            regularization, update_embeddings """
        super(DKRLBiEncoder, self).__init__()

        self.model = args.model_name
        self.gamma = args.gamma
        self.regularization = args.regularization

        # set up BERT word embeddings
        encoder = AutoModel.from_pretrained(self.model)
        self.embeddings = encoder.embeddings.word_embeddings
        if not args.update_embeddings:
            self.embeddings.weight.requires_grad = False
        hidden_size = encoder.config.hidden_size

        # check for valid score function
        if args.score not in ('ComplEx', 'DistMult', 'RotatE', 'TransE'):
            raise ValueError(f'score function {args.score} not available')
        score_functions = {'ComplEx' : ComplEx, 'DistMult' : DistMult,
                           'RotatE' : RotatE, 'TransE' : TransE}
        self.score_function = score_functions[args.score]

        # double entity embedding size if using ComplEx or RotatE, set up
        # 1D conv layers
        if args.score in ('ComplEx', 'RotatE'):
            entity_dim = 2 * args.embedding_dim
        else:
            entity_dim = args.embedding_dim
        self.conv1 = Conv1d(hidden_size, entity_dim, kernel_size=2)
        self.conv2 = Conv1d(entity_dim, entity_dim, kernel_size=2)

        # double relation embedding size if using ComplEx, set up relation
        # embeddings matrix
        if args.score == 'ComplEx':
            relation_dim = 2 * args.embedding_dim
        else:
            relation_dim = args.embedding_dim
        self.relation_embedding = Embedding(args.nrelation, relation_dim)

        # set up parameters for link prediction
        self.link_prediction = args.link_prediction
        if self.link_prediction:
            self.link_head = Linear(3 * entity_dim, 1)

        # set up parameters for relation prediction
        self.relation_prediction = args.relation_prediction
        if self.relation_prediction:
            self.relation_head = Linear(3 * entity_dim, args.nrelation)

    def forward(self, data):
        # extract word embeddings and mask padding
        mask = data.attention_mask.float().unsqueeze(-1)
        embs = self.embeddings(data.input_ids) * mask

        # reshape to (batch size, embedding dim, seq length)
        embs = embs.transpose(1, 2)
        mask = mask.transpose(1, 2)

        # add padding for valid convolutions, pass through 1D conv layer, and
        # mask outputs due to padding
        embs = F.pad(embs, (0, 1))
        embs = self.conv1(embs)
        embs = embs * mask

        # compute kernel size depending on sequence length, apply max pooling
        if embs.size(2) >= 4:
            kernel_size = 4
        elif embs.size(2) == 1:
            kernel_size = 1
        else:
            kernel_size = 2
        embs = F.max_pool1d(embs, kernel_size=kernel_size)
        mask = F.max_pool1d(mask, kernel_size=kernel_size)

        # apply nonlinearity, add padding for valid convolutions, apply second
        # 1D conv layer, and mask outputs due to padding
        embs = torch.tanh(embs)
        embs = F.pad(embs, (0, 1))
        embs = self.conv2(embs)
        embs = embs * mask

        # final mean pooling and nonlinearity to get fixed-length vector
        lengths = torch.sum(mask, dim=-1)
        embs = torch.sum(embs, dim=-1) / lengths
        embs = torch.tanh(embs)

        # split embeddings into head and tail chunks, retrieve relation
        # embeddings and compute scores
        head, tail = torch.chunk(embs, 2, dim=0)
        relation = self.relation_embedding(data.relations)
        score = self.score_function(head, relation, tail, self.gamma)
        outputs = {'ranking_outputs' : score}

        # apply L2 regularization
        if self.regularization > 0:
            emb_norm = 0
            for elem in (head, tail, relation):
                emb_norm += torch.mean(elem ** 2)
            outputs['regularization'] = self.regularization * emb_norm / 3.

        # compute link prediction and relation prediction losses
        triple_embedding = torch.cat([head, tail,
                                      torch.abs(head - tail)], dim=1)
        if self.link_prediction:
            outputs['link_outputs'] = self.link_head(triple_embedding)
        if self.relation_prediction:
            outputs['relation_outputs'] = self.relation_head(triple_embedding)

        return outputs

