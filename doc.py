import os
import json
import torch
import torch.utils.data.dataset
from typing import Optional, List
import random
from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger
from collections import defaultdict

entity_dict = get_entity_dict()
if args.use_link_graph:
    get_link_graph()

def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs

def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    return entity or ''

def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity

def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)

class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=self.relation)

        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, args, examples=None):
        self.path_list = path.split(',')
        self.task = task
        self.args = args

        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

        self.relation_tails = defaultdict(set)
        self._build_relation_tails()

    def _build_relation_tails(self):

        for ex in self.examples:
            r = ex.relation
            t_id = ex.tail_id
            self.relation_tails[r].add(t_id)



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        K = self.args.rs_negative_K
        r = example.relation
        t_id = example.tail_id
        candidates = list(self.relation_tails[r] - {t_id})
        sampled_t_ids = random.sample(candidates, min(K, len(candidates))) if candidates else []

        rs_negs = []
        for tid in sampled_t_ids:
            t_entity = entity_dict.get_entity_by_id(tid)
            t_text = _concat_name_desc(t_entity.entity, t_entity.entity_desc)
            encoded = _custom_tokenize(t_text)
            rs_negs.append({
                'input_ids': encoded['input_ids'],
                'mask': encoded['attention_mask'],
                'token_type_ids': encoded['token_type_ids']
            })
        return {
            **example.vectorize(),
            'rs_negs': rs_negs
        }

def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples

def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    rs_neg_inputs = [ex['rs_negs'] for ex in batch_data]
    all_rs_negs = []
    for sample_negs in rs_neg_inputs:
        all_rs_negs.extend(sample_negs)


    if len(all_rs_negs) > 0:
        rs_neg_ids, rs_neg_mask = to_indices_and_mask(
            [torch.LongTensor(n['input_ids']) for n in all_rs_negs],
            pad_token_id=get_tokenizer().pad_token_id
        )
        rs_neg_types = to_indices_and_mask(
            [torch.LongTensor(n['token_type_ids']) for n in all_rs_negs],
            need_mask=False
        )
    else:
        rs_neg_ids = rs_neg_mask = rs_neg_types = torch.tensor([])

    batch_dict.update({
        'rs_neg_token_ids': rs_neg_ids,
        'rs_neg_mask': rs_neg_mask,
        'rs_neg_token_type_ids': rs_neg_types,
    })

    return batch_dict

def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices