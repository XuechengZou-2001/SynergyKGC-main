import os
import json
import tqdm
import torch
import torch.utils.data
from typing import List
from collections import OrderedDict
from doc import collate, Example, Dataset
from config import args
from models import build_model
from utils import AttrDict, move_to_cuda
from dict_hub import build_tokenizer, get_dynamic_cache
from logger_config import logger

class BertPredictor:

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def load(self, ckt_path, use_data_parallel=True):
        assert os.path.exists(ckt_path)
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        build_tokenizer(self.train_args)
        self.model = build_model(self.train_args)

        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

        if torch.cuda.is_available():
            if use_data_parallel and self.device_count > 1:
                logger.info(f'Use {self.device_count} GPUs for parallel data processing')
                self.model = torch.nn.DataParallel(self.model)
                self.model.cuda()
                self.use_cuda = True
            else:
                logger.info('Using a single GPU')
                self.model.cuda()
                self.use_cuda = True
        logger.info('Load model from {} successfully'.format(ckt_path))

    def _setup_args(self):
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                logger.info('Set default attribute: {}={}'.format(k, v))
                self.train_args.__dict__[k] = v
        logger.info(
            'Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
        args.use_link_graph = self.train_args.use_link_graph
        args.is_test = True

    def _get_optimal_batch_size(self, base_batch_size):
        if self.device_count > 1:
            return base_batch_size * self.device_count
        return base_batch_size

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]):
        base_batch_size = 64
        inference_batch_size = self._get_optimal_batch_size(base_batch_size)

        logger.info(f"Triple prediction using batch_size={inference_batch_size} (enabling collaborative expert inference)")

        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task, args=args),
            num_workers=min(8, self.device_count * 2),
            batch_size=inference_batch_size,
            collate_fn=collate,
            shuffle=False,
            pin_memory=True)

        hr_tensor_list, tail_tensor_list = [], []
        for idx, batch_dict in enumerate(data_loader):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)

            if idx % 10 == 0:
                self._print_gpu_memory(f"predict_by_examples Batch {idx}")

            try:
                outputs = self.model(**batch_dict,
                                     use_gnn=True,
                                     use_head_gnn=True,
                                     use_tail_gnn=True)

                hr_tensor_list.append(outputs['hr_vector'].cpu())
                tail_tensor_list.append(outputs['tail_vector'].cpu())

                del outputs

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Insufficient inference memory in batch {idx}, current batch size: {len(batch_dict['batch_data'])}")
                self._print_gpu_memory("When OOM abnormality occurs")
                torch.cuda.empty_cache()
                raise e

            if idx % 10 == 0:
                torch.cuda.empty_cache()

        hr_result = torch.cat(hr_tensor_list, dim=0)
        tail_result = torch.cat(tail_tensor_list, dim=0)

        if self.use_cuda:
            hr_result = hr_result.cuda()
            tail_result = tail_result.cuda()

        return hr_result, tail_result

    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        examples = []
        for entity_ex in entity_exs:
            examples.append(Example(head_id='', relation='',
                                    tail_id=entity_ex.entity_id))

        base_batch_size = 128
        entity_batch_size = self._get_optimal_batch_size(base_batch_size)

        logger.info(f"Collaborative enhanced entity prediction using batch_size={entity_batch_size} (Synergy Expert)")

        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task, args=args),
            num_workers=min(8, self.device_count * 2),
            batch_size=entity_batch_size,
            collate_fn=collate,
            shuffle=False,
            pin_memory=True)

        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader, desc="Predict the entity library after collaborative enhancement")):
            batch_dict['only_ent_embedding'] = False

            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)

            try:
                outputs = self.model(**batch_dict,
                                     use_gnn=True,
                                     use_head_gnn=False,
                                     use_tail_gnn=True)

                ent_tensor_list.append(outputs['tail_vector'].cpu())

                del outputs

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Entity predicts insufficient memory in batch {idx}")
                self._print_gpu_memory("When Entity OOM occurs")
                torch.cuda.empty_cache()
                raise e

            if idx % 10 == 0:
                torch.cuda.empty_cache()

        result = torch.cat(ent_tensor_list, dim=0)
        if self.use_cuda:
            result = result.cuda()

        return result

    def init_Dynamic_cache(self, examples: List[Example]):
        base_batch_size = 32
        cache_batch_size = self._get_optimal_batch_size(base_batch_size)

        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task, args=args),
            num_workers=min(4, self.device_count),
            batch_size=cache_batch_size,
            collate_fn=collate,
            shuffle=False,
            pin_memory=True)

        logger.info(f"Initialize dynamic cache, with a total of {len(examples)} samples，batch_size={cache_batch_size}")

        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader, desc="Initialize dynamic cache")):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)

            if idx % 5 == 0:
                self._print_gpu_memory(f"Cache initialization Batch{idx}/{len(data_loader)}")

            try:
                outputs = self.model(**batch_dict, use_gnn=False)
                batch_data = batch_dict['batch_data']
                head_ids = [ex.head_id for ex in batch_data]
                tail_ids = [ex.tail_id for ex in batch_data]
                hr_vectors = outputs['hr_vector']
                tail_vectors = outputs['tail_vector']

                cache = get_dynamic_cache()
                cache.update_hr(head_ids, hr_vectors)
                cache.update_tail(tail_ids, tail_vectors)

                del outputs

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Insufficient cache initialization memory in batch {idx}")
                logger.error(f"Current batch size: {len(batch_dict['batch_data'])}")
                self._print_gpu_memory("When initializing OOM cache")
                torch.cuda.empty_cache()
                raise e

            if idx % 5 == 0:
                torch.cuda.empty_cache()

    def _print_gpu_memory(self, stage=""):
        if torch.cuda.is_available():
            for i in range(self.device_count):
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                logger.info(f"{stage} - GPU {i}: Assigned={allocated:.2f}GB, Cached={reserved:.2f}GB")