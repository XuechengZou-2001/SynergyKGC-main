import copy
import glob
import os
import json
import torch
import shutil
import time
import torch.nn as nn
import torch.utils.data
from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW
from doc import Dataset, collate
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer, get_dynamic_cache
from logger_config import logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class Trainer:
    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()

        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        train_dataset = Dataset(path=args.train_path, task=args.task, args=args)
        valid_dataset = Dataset(path=args.valid_path, task=args.task, args=args) if args.valid_path else None
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        metric_dict = self.eval_epoch(epoch)
        is_best = self.valid_loader and (self.best_metric is None or metric_dict['Acc@1'] > self.best_metric['Acc@1'])
        if is_best:
            self.best_metric = metric_dict

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs_dict = self.model(**batch_dict, use_gnn=(epoch >= self.args.gnn_start_epoch))

            logits_output = get_model_obj(self.model).compute_logits(output_dict=outputs_dict, batch_dict=batch_dict)
            model_output = ModelOutput(**logits_output)
            logits, labels = model_output.logits, model_output.labels

            main_loss = self.criterion(logits, labels)

            align_loss = 0
            if epoch >= self.args.gnn_start_epoch:
                align_loss_hr = torch.nn.functional.mse_loss(outputs_dict['hr_vector'], outputs_dict['hr_semantic_raw'])
                align_loss_tail = torch.nn.functional.mse_loss(outputs_dict['tail_vector'],
                                                               outputs_dict['tail_semantic_raw'])
                align_loss = (align_loss_hr + align_loss_tail) * 0.1

            total_loss = main_loss + align_loss
            losses.update(total_loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}

        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))

        model_obj = get_model_obj(self.model)
        if hasattr(model_obj, 'reset_cache_stats'):
            model_obj.reset_cache_stats()

        epoch_start_time = time.time()

        for i, batch_dict in enumerate(self.train_loader):
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs_dict = self.model(**batch_dict, use_gnn=(epoch >= self.args.gnn_start_epoch))
            else:
                outputs_dict = self.model(**batch_dict, use_gnn=(epoch >= self.args.gnn_start_epoch))

            batch_data = batch_dict['batch_data']
            head_ids = [ex.head_id for ex in batch_data]
            tail_ids = [ex.tail_id for ex in batch_data]
            hr_vectors = outputs_dict['hr_vector']
            tail_vectors = outputs_dict['tail_vector']

            cache_store_start = time.time()
            cache = get_dynamic_cache()
            cache.update_hr(head_ids, hr_vectors)
            cache.update_tail(tail_ids, tail_vectors)
            cache_store_time = time.time() - cache_store_start

            memory_usage = cache.get_memory_usage()
            if hasattr(model_obj, 'cache_profiler'):
                model_obj.cache_profiler.record_store_time(cache_store_time)
                model_obj.cache_profiler.record_cache_memory(memory_usage)

            logits_output = get_model_obj(self.model).compute_logits(output_dict=outputs_dict, batch_dict=batch_dict)
            model_output = ModelOutput(**logits_output)
            logits, labels = model_output.logits, model_output.labels
            assert logits.size(0) == batch_size

            main_loss = self.criterion(logits, labels)
            main_loss += self.criterion(logits, labels)

            align_loss = 0
            if epoch >= self.args.gnn_start_epoch:
                align_loss_hr = torch.nn.functional.mse_loss(outputs_dict['hr_vector'], outputs_dict['hr_semantic_raw'])
                align_loss_tail = torch.nn.functional.mse_loss(outputs_dict['tail_vector'],
                                                               outputs_dict['tail_semantic_raw'])
                align_loss = (align_loss_hr + align_loss_tail) * 0.1

            total_loss = main_loss + align_loss

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            inv_t.update(model_output.inv_t, 1)
            losses.update(total_loss.item(), batch_size)

            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            if i % self.args.print_freq == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)

        epoch_total_time = time.time() - epoch_start_time
        if hasattr(model_obj, 'get_cache_stats'):
            train_stats = model_obj.get_cache_stats()
            logger.info('=' * 60)
            logger.info(f'EPOCH {epoch} TRAINING PERFORMANCE ANALYSIS')
            logger.info('=' * 60)
            logger.info('Epoch Total Time: {:.4f}s'.format(epoch_total_time))
            mem_usage = train_stats['cache_memory_usage']
            logger.info('Cache Memory Usage:')
            logger.info('  - HR Cache: {:.2f} MB'.format(mem_usage['current']['hr_cache_memory'] / (1024 * 1024)))
            logger.info('  - Tail Cache: {:.2f} MB'.format(mem_usage['current']['tail_cache_memory'] / (1024 * 1024)))
            logger.info('  - Total: {:.2f} MB'.format(mem_usage['current']['total_memory'] / (1024 * 1024)))
            logger.info('  - Average per Batch: {:.2f} MB'.format(mem_usage['average_total'] / (1024 * 1024)))

            logger.info('Cache Performance:')
            logger.info('  - Cache Hits: {}'.format(train_stats['cache_hits']))
            logger.info('  - Cache Misses: {}'.format(train_stats['cache_misses']))
            logger.info('  - Hit Rate: {:.2%}'.format(train_stats['hit_rate']))
            logger.info('Time Breakdown:')
            logger.info('  - BERT Single Calls: {} calls, Avg: {:.4f}s, Total: {:.4f}s'.format(
                train_stats['bert_call_count'], train_stats['bert_avg_single_time'],
                sum(train_stats['bert_single_call_times'])))
            logger.info('  - BERT Batch Total Time: {:.4f}s'.format(train_stats['bert_batch_total_time']))
            logger.info('  - Cache Lookup Time: {:.4f}s'.format(train_stats['cache_lookup_time']))
            logger.info('  - Cache Store Time: {:.4f}s'.format(train_stats['cache_store_time']))
            logger.info('  - GNN Computation Time: {:.4f}s'.format(train_stats['gnn_computation_time']))
            logger.info('  - Neighbor Retrieval Time: {:.4f}s'.format(train_stats['neighbor_retrieval_time']))
            logger.info('  - Total Cache Time: {:.4f}s'.format(train_stats['total_cache_time']))

            total_compute_time = (sum(train_stats['bert_single_call_times']) + train_stats['gnn_computation_time'] +
                                  train_stats['total_cache_time'] + train_stats['neighbor_retrieval_time'])
            if total_compute_time > 0:
                logger.info('Time Percentage Breakdown:')
                logger.info('  - BERT: {:.1%}'.format(sum(train_stats['bert_single_call_times']) / total_compute_time))
                logger.info(
                    '  - Synergy Expert (GNN): {:.1%}'.format(train_stats['gnn_computation_time'] / total_compute_time))
                logger.info('  - Cache: {:.1%}'.format(train_stats['total_cache_time'] / total_compute_time))
                logger.info('  - Neighbor Retrieval: {:.1%}'.format(
                    train_stats['neighbor_retrieval_time'] / total_compute_time))
            logger.info('=' * 60)

        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)