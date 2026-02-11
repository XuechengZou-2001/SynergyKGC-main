from abc import ABC
from copy import deepcopy
import torch
import torch.nn as nn
import time
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from dict_hub import get_cotail_graph, get_dynamic_cache, get_cotail_graph_valid
from triplet_mask import construct_mask
import torch.nn.functional as F

def build_model(args) -> nn.Module:
    return CustomBertModel(args)

@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor

class GNNLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):

        for layer in [self.attention, self.ffn]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, query: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:

        if neighbors.size(0) == 0:
            return query

        query_3d = query.view(1, 1, -1)
        neighbors_3d = neighbors.unsqueeze(1)  # [K, 1, D]

        attn_output, _ = self.attention(
            query=query_3d,  # [1, 1, D]
            key=neighbors_3d,  # [K, 1, D]
            value=neighbors_3d,  # [K, 1, D]
            need_weights=False
        )

        query_3d = self.norm1(query_3d + self.dropout(attn_output))  # [1, 1, D]

        ffn_output = self.ffn(query_3d)
        output = self.norm2(query_3d + self.dropout(ffn_output))  # [1, 1, D]

        return output.squeeze(0).squeeze(0)  # [D]

class CacheProfiler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_lookup_time = 0.0
        self.cache_store_time = 0.0
        self.gnn_computation_time = 0.0
        self.neighbor_retrieval_time = 0.0
        self.bert_single_call_times = []
        self.bert_batch_total_time = 0.0
        self.cache_memory_usage = []

    def record_cache_memory(self, memory_usage):
        self.cache_memory_usage.append(memory_usage)

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def record_lookup_time(self, time_taken):
        self.cache_lookup_time += time_taken

    def record_store_time(self, time_taken):
        self.cache_store_time += time_taken

    def record_gnn_time(self, time_taken):
        self.gnn_computation_time += time_taken

    def record_neighbor_time(self, time_taken):
        self.neighbor_retrieval_time += time_taken

    def record_bert_single_call(self, time_taken):
        self.bert_single_call_times.append(time_taken)

    def record_bert_batch_time(self, time_taken):
        self.bert_batch_total_time += time_taken

    def get_stats(self):
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        total_memory = sum(usage['total_memory'] for usage in self.cache_memory_usage)
        avg_memory = total_memory / len(self.cache_memory_usage) if self.cache_memory_usage else 0

        current_memory = self.cache_memory_usage[-1] if self.cache_memory_usage else {
            'hr_cache_memory': 0,
            'tail_cache_memory': 0,
            'total_memory': 0
        }

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_lookup_time': self.cache_lookup_time,
            'cache_store_time': self.cache_store_time,
            'gnn_computation_time': self.gnn_computation_time,
            'neighbor_retrieval_time': self.neighbor_retrieval_time,
            'total_cache_time': self.cache_lookup_time + self.cache_store_time,
            'bert_single_call_times': self.bert_single_call_times,
            'bert_batch_total_time': self.bert_batch_total_time,
            'bert_avg_single_time': sum(self.bert_single_call_times) / len(self.bert_single_call_times) if self.bert_single_call_times else 0,
            'bert_call_count': len(self.bert_single_call_times),
            'cache_memory_usage': {
                'current': current_memory,
                'average_total': avg_memory,
                'all_entries': self.cache_memory_usage
            }
        }

cache_profiler = CacheProfiler()

class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)

        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin

        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

        self.hr_synergy_expert = CrossModalSynergyLayer(
            self.config.hidden_size,
            use_anchor=getattr(args, 'use_hr_anchor', args.use_identity_anchor),
            density_threshold=getattr(args, 'density_threshold', -1)
        )
        self.tail_synergy_expert = CrossModalSynergyLayer(
            self.config.hidden_size,
            use_anchor=getattr(args, 'use_tail_anchor', args.use_identity_anchor),
            density_threshold=getattr(args, 'density_threshold', -1)
        )

        self.cache_profiler = cache_profiler

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        start_time = time.time()

        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)

        encoding_time = time.time() - start_time
        self.cache_profiler.record_bert_single_call(encoding_time)

        return cls_output

    def _get_cached_vectors_with_profiling(self, cache, entity_ids, vector_type):
        start_time = time.time()

        if vector_type == 'hr':
            vectors = [vec for vec in cache.get_hr_vectors(entity_ids) if vec is not None]
        else:
            vectors = [vec for vec in cache.get_tail_vectors(entity_ids) if vec is not None]

        lookup_time = time.time() - start_time
        self.cache_profiler.record_lookup_time(lookup_time)

        hit_count = len(vectors)
        miss_count = len(entity_ids) - hit_count

        for _ in range(hit_count):
            self.cache_profiler.record_cache_hit()
        for _ in range(miss_count):
            self.cache_profiler.record_cache_miss()

        return vectors

    def _get_neighbors_with_profiling(self, graph, entity_id):
        start_time = time.time()
        neighbors = graph.get_cotail_neighbors(entity_id)
        neighbor_time = time.time() - start_time
        self.cache_profiler.record_neighbor_time(neighbor_time)
        return neighbors

    def _apply_gnn_with_profiling(self, gnn_layer, target_vector, neighbor_tensor):
        start_time = time.time()

        result = gnn_layer(target_vector, neighbor_tensor)

        gnn_time = time.time() - start_time

        self.cache_profiler.record_gnn_time(gnn_time)

        return result

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                use_gnn=True,
                use_head_gnn=True,
                use_tail_gnn=True,
                only_ent_embedding=False, **kwargs) -> dict:

        def get_current_batch_data(full_batch_data, current_batch_size):
            if not full_batch_data:
                return []

            if torch.is_tensor(hr_token_ids):
                device_count = torch.cuda.device_count()
                if device_count > 1 and len(full_batch_data) > current_batch_size:
                    device_idx = torch.cuda.current_device()
                    start_idx = device_idx * current_batch_size
                    end_idx = start_idx + current_batch_size
                    return full_batch_data[start_idx:end_idx]
            return full_batch_data[:current_batch_size]

        batch_bert_start = time.time()

        if only_ent_embedding:
            result = self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                                tail_mask=tail_mask,
                                                tail_token_type_ids=tail_token_type_ids)
            batch_bert_time = time.time() - batch_bert_start
            self.cache_profiler.record_bert_batch_time(batch_bert_time)
            return result

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        batch_bert_time = time.time() - batch_bert_start
        self.cache_profiler.record_bert_batch_time(batch_bert_time)

        hr_semantic_raw = hr_vector.clone().detach()
        tail_semantic_raw = tail_vector.clone().detach()

        if not use_gnn:
            return {'hr_vector': hr_vector,
                    'tail_vector': tail_vector,
                    'head_vector': head_vector,
                    'hr_semantic_raw': hr_semantic_raw,
                    'tail_semantic_raw': tail_semantic_raw}

        current_batch_size = hr_vector.size(0)
        cache = get_dynamic_cache()
        full_batch_data = kwargs.get('batch_data', [])
        batch_data = get_current_batch_data(full_batch_data, current_batch_size)

        if use_gnn and not self.training:
            if use_head_gnn:
                updated_hr = []
                for i, ex in enumerate(batch_data):
                    cotail_heads = self._get_neighbors_with_profiling(get_cotail_graph_valid(), ex.head_id)
                    neighbor_vectors = self._get_cached_vectors_with_profiling(cache, cotail_heads, 'hr')

                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(hr_vector.device)
                        updated, _ = self._apply_gnn_with_profiling(
                            self.hr_synergy_expert, hr_vector[i], neighbor_tensor)
                        updated_hr.append(updated)
                    else:
                        updated_hr.append(hr_vector[i])
                hr_vector = torch.stack(updated_hr)

            if use_tail_gnn:
                updated_tail = []
                for i, ex in enumerate(batch_data):
                    cotail_tails = self._get_neighbors_with_profiling(get_cotail_graph_valid(), ex.tail_id)
                    neighbor_vectors = self._get_cached_vectors_with_profiling(cache, cotail_tails, 'tail')

                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(tail_vector.device)
                        updated, _ = self._apply_gnn_with_profiling(
                            self.tail_synergy_expert, tail_vector[i], neighbor_tensor)
                        updated_tail.append(updated)
                    else:
                        updated_tail.append(tail_vector[i])
                tail_vector = torch.stack(updated_tail)

        if use_gnn and self.training:
            if use_head_gnn:
                updated_hr = []
                for i, ex in enumerate(batch_data):
                    cotail_heads = self._get_neighbors_with_profiling(get_cotail_graph(), ex.head_id)
                    neighbor_vectors = self._get_cached_vectors_with_profiling(cache, cotail_heads, 'hr')

                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(hr_vector.device)
                        updated, _ = self._apply_gnn_with_profiling(
                            self.hr_synergy_expert, hr_vector[i], neighbor_tensor)
                        updated_hr.append(updated)
                    else:
                        updated_hr.append(hr_vector[i])
                hr_vector = torch.stack(updated_hr)

            if use_tail_gnn:
                updated_tail = []
                for i, ex in enumerate(batch_data):
                    cotail_tails = self._get_neighbors_with_profiling(get_cotail_graph(), ex.tail_id)
                    neighbor_vectors = self._get_cached_vectors_with_profiling(cache, cotail_tails, 'tail')

                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(tail_vector.device)
                        updated, _ = self._apply_gnn_with_profiling(
                            self.tail_synergy_expert, tail_vector[i], neighbor_tensor)
                        updated_tail.append(updated)
                    else:
                        updated_tail.append(tail_vector[i])
                tail_vector = torch.stack(updated_tail)

        hr_vector = F.normalize(hr_vector, p=2, dim=1)
        tail_vector = F.normalize(tail_vector, p=2, dim=1)

        return {
            'hr_vector': hr_vector,
            'tail_vector': tail_vector,
            'head_vector': head_vector,
            'hr_semantic_raw': hr_semantic_raw,
            'tail_semantic_raw': tail_semantic_raw
        }

    def get_cache_stats(self):
        return self.cache_profiler.get_stats()

    def reset_cache_stats(self):
        self.cache_profiler.reset()

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
        if self.args.use_rs_negative and self.training:
            rs_neg_ids = batch_dict['rs_neg_token_ids']
            if rs_neg_ids.size(0) > 0:
                rs_neg_vec = self._encode(
                    self.tail_bert,
                    rs_neg_ids,
                    batch_dict['rs_neg_mask'],
                    batch_dict['rs_neg_token_type_ids']
                )
                K = rs_neg_vec.size(0) // batch_size
                rs_neg_vec = rs_neg_vec.view(batch_size, K, -1)
                rs_logits = torch.bmm(
                    hr_vector.unsqueeze(1),  # (B,1,D)
                    rs_neg_vec.transpose(1, 2)  # (B,D,K)
                ).squeeze(1) * self.log_inv_t.exp()
                logits = torch.cat([logits, rs_logits], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector


class CrossModalSynergyLayer(nn.Module):
    def __init__(self, hidden_size, use_anchor=True, density_threshold=-1, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_anchor = use_anchor
        self.density_threshold = density_threshold

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_proj = nn.Linear(hidden_size, hidden_size)

        self.synergy_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.constant_(self.gate[-1].bias, 2.0)

    def forward(self, semantic_query: torch.Tensor, structural_neighbors: torch.Tensor):
        density = structural_neighbors.size(0)

        if self.density_threshold >= 0:
            current_use_anchor = (density < self.density_threshold)
        else:
            current_use_anchor = self.use_anchor

        if current_use_anchor:
            target_self = semantic_query.unsqueeze(0)
            kv_pool = torch.cat([target_self, structural_neighbors], dim=0)
        else:
            kv_pool = structural_neighbors if density > 0 else semantic_query.unsqueeze(0)

        q = self.q_proj(semantic_query).view(1, 1, -1)  # [1, 1, D]
        kv = self.kv_proj(kv_pool).unsqueeze(0)  # [1, K_pool, D]

        synergy_context, attn_weights = self.synergy_attn(
            query=q, key=kv, value=kv, need_weights=True
        )  # [1, 1, D]

        gate_input = torch.cat([q, synergy_context], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input))

        fused_x = alpha * q + (1 - alpha) * synergy_context

        out = self.norm1(semantic_query + self.dropout(fused_x.squeeze(0).squeeze(0)))

        return out, attn_weights