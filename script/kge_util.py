""" utility functions for ensembling experiments """

from collections import defaultdict

from ogb.linkproppred import Evaluator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_metric_kg(ent_emb, rel_emb, triples, method='RotatE', gamma=20.,
    eval_batch_size=64, return_scores=True):
    """ Compute score for each (head, relation, tail) triple using KG entity
        and relation embeddings. Note that the entity indices in `triples`
        must already have node type offsets added. """
    scores = {'metrics' : {'mrr_list' : list(),
                           'hits@1_list' : list(),
                           'hits@3_list' : list(),
                           'hits@10_list' : list()}}
    if return_scores:
        scores['scores'] = {'y_pred_pos' : list(),
                            'y_pred_neg_head' : list(),
                            'y_pred_neg_tail' : list()}

    evaluator = Evaluator(name='ogbl-biokg')

    head_all, tail_all = triples['head'], triples['tail']
    relation_all = triples['relation']
    head_neg_all, tail_neg_all = triples['head_neg'], triples['tail_neg']

    # define score function to use
    if method == 'RotatE':
        scorefn = lambda h, r, t, mode: RotatE(h, r, t, gamma=gamma, mode=mode)
    elif method == 'TransE':
        scorefn = lambda h, r, t, mode: TransE(h, r, t, gamma=gamma, mode=mode)
    elif method == 'ComplEx':
        scorefn = lambda h, r, t, mode: ComplEx(h, r, t, mode=mode)
    elif method == 'DistMult':
        scorefn = lambda h, r, t, mode: DistMult(h, r, t, mode=mode)
    else:
        raise ValueError(f'{args.method} is not a valid method')

    # handle head-batch
    loader = DataLoader(torch.arange(len(head_all)), batch_size=eval_batch_size)
    head_cat = torch.cat([head_all.unsqueeze(1), head_neg_all], dim=1)
    for batch in tqdm(loader, desc='head-batch'):
        head_part, tail_part = head_cat[batch], tail_all[batch]
        batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

        # [batch_size x negative_sample_size x embedding_dim]
        head = torch.index_select(
            ent_emb, dim=0, index=head_part.view(-1)
        ).view(batch_size, negative_sample_size, -1)

        # [batch_size x 1 x embedding_dim]
        relation = torch.index_select(
            rel_emb, dim=0, index=relation_all[batch],
        ).unsqueeze(1)

        # [batch_size x 1 x embedding_dim]
        tail = torch.index_select(
            ent_emb, dim=0, index=tail_part,
        ).unsqueeze(1)

        score = scorefn(head, relation, tail, 'head-batch')
        batch_results = evaluator.eval({'y_pred_pos': score[:, 0],
                                        'y_pred_neg': score[:, 1:]})
        for key in batch_results:
            scores['metrics'][key].append(batch_results[key])

        # keep track of scores
        if return_scores:
            scores['scores']['y_pred_pos'].append(score[:, 0])
            scores['scores']['y_pred_neg_head'].append(score[:, 1:])

    # handle tail-batch
    loader = DataLoader(torch.arange(len(tail_all)), batch_size=eval_batch_size)
    tail_cat = torch.cat([tail_all.unsqueeze(1), tail_neg_all], dim=1)
    for batch in tqdm(loader, desc='tail-batch'):
        head_part, tail_part = head_all[batch], tail_cat[batch]
        batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

        # [batch_size x 1 x embedding_dim]
        head = torch.index_select(
            ent_emb, dim=0, index=head_part,
        ).unsqueeze(1)

        # [batch_size x 1 x embedding_dim]
        relation = torch.index_select(
            rel_emb, dim=0, index=relation_all[batch],
        ).unsqueeze(1)

        # [batch_size x negative_sample_size x embedding_dim]
        tail = torch.index_select(
            ent_emb, dim=0, index=tail_part.view(-1),
        ).view(batch_size, negative_sample_size, -1)

        score = scorefn(head, relation, tail, 'tail-batch')
        batch_results = evaluator.eval({'y_pred_pos': score[:, 0],
                                        'y_pred_neg': score[:, 1:]})
        for key in batch_results:
            scores['metrics'][key].append(batch_results[key])

        # keep track of scores
        if return_scores:
            scores['scores']['y_pred_neg_tail'].append(score[:, 1:])

    for key in scores['metrics']:
        scores['metrics'][key] = torch.cat(scores['metrics'][key]).cpu()

    if return_scores:
        for key in scores['scores']:
            scores['scores'][key] = torch.cat(scores['scores'][key]).cpu()

    return scores


def RotatE(head, relation, tail, gamma, epsilon=2.0, mode='head-batch'):
    """ RotatE score function given entity and relation embeddings
        if mode == head-batch:
            head should be size (batch_size, num_neg_samples, 2 * hidden_dim)
            relation should be size (batch_size, 1, hidden_dim),
            tail should be size (batch_size, 1, 2 * hidden_dim)
         elif mode == tail-batch:
            head should be size (batch_size, 1, 2 * hidden_dim)
            relation should be size (batch_size, 1, hidden_dim)
            tail should be size (batch_size, num_neg_samples, 2 * hidden_dim)
    """
    pi = 3.14159265358979323846
    hidden_dim = relation.size(2)

    re_head, im_head = torch.chunk(head, 2, dim=2)
    re_tail, im_tail = torch.chunk(tail, 2, dim=2)

    embedding_range = torch.nn.Parameter(
        torch.Tensor([(gamma + epsilon) / hidden_dim]),
        requires_grad=False
    )

    phase_relation = relation / (embedding_range.item() / pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)

    if mode == 'head-batch':
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head
    else:
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

    score = torch.stack([re_score, im_score], dim = 0)
    score = score.norm(dim = 0)

    score = gamma - score.sum(dim = 2)
    return score


def ComplEx(head, relation, tail, mode):
    """ ComplEx score function given entity and relation embeddings
        if mode == head-batch:
            head should be size (batch_size, num_neg_samples, 2 * hidden_dim)
            relation should be size (batch_size, 1, hidden_dim),
            tail should be size (batch_size, 1, 2 * hidden_dim)
         elif mode == tail-batch:
            head should be size (batch_size, 1, 2 * hidden_dim)
            relation should be size (batch_size, 1, hidden_dim)
            tail should be size (batch_size, num_neg_samples, 2 * hidden_dim)
    """
    re_head, im_head = torch.chunk(head, 2, dim=2)
    re_relation, im_relation = torch.chunk(relation, 2, dim=2)
    re_tail, im_tail = torch.chunk(tail, 2, dim=2)

    if mode == 'head-batch':
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        score = re_head * re_score + im_head * im_score
    else:
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

    score = score.sum(dim = 2)
    return score


def TransE(head, relation, tail, gamma, mode='head-batch'):
    """ TransE score function """
    if mode == 'head-batch':
        score = head + (relation - tail)
    else:
        score = (head + relation) - tail

    score = gamma - torch.norm(score, p=1, dim=2)
    return score


def DistMult(head, relation, tail, mode='head-batch'):
    """ DistMult score function """
    if mode == 'head-batch':
        score = head * (relation * tail)
    else:
        score = (head * relation) * tail

    score = score.sum(dim = 2)
    return score

