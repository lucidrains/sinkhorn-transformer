import math
import torch
from torch import nn
from operator import mul
from math import gcd
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce

from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from product_key_memory import PKM
from sinkhorn_transformer.reversible import ReversibleSequence, SequentialSequence

# helper functions

def identity(x, *args, **kwargs): return x

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)

def divisible_by(num, divisor):
    return num % divisor == 0

def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def all_none(*arr):
    return all(el is None for el in arr)

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def rotate_left(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(n, None))
    r = (*pre_slices, slice(0, n))
    return torch.cat((t[l], t[r]), dim=dim)

def rotate_right(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(-n, None))
    r = (*pre_slices, slice(None, -n))
    return torch.cat((t[l], t[r]), dim=dim)

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def merge_heads(h, v):
    b, t, d = v.shape
    return v.view(b, t, h, -1).transpose(1, 2).reshape(b, h, t, -1)

def split_heads(h, v):
    *_, t, d = v.shape
    return v.view(-1, h, t, d).transpose(1, 2).reshape(-1, t, d * h)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+1] = [buckets, -1]
    return t.reshape(*shape)

def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+2] = [-1]
    return t.reshape(*shape)

def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)

def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)

def reorder_buckets(t, r):
    return torch.einsum('buv,bvtd->butd', r, t)

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def cumavg(t, dim):
    r = torch.arange(1, t.shape[dim] + 1, device=t.device, dtype=t.dtype)
    expand_slice = [None] * len(t.shape)
    expand_slice[dim] = slice(None, None)
    return t.cumsum(dim=dim) / r[tuple(expand_slice)]

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def expand_batch_and_merge_head(b, t):
    shape = list(t.squeeze(0).shape)
    t = expand_dim(t, 0, b)
    shape[0] = shape[0] * b
    return t.reshape(*shape)

def differentiable_topk(x, k, temperature=1.):
    *_, n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(*_, k * n, dim)

# helper classes

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x):
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c) for c in chunks], dim = self.dim)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class ProjectInOut(nn.Module):
    def __init__(self, fn, dim_in, dim_out, project_out = True):
        super().__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else identity

    def forward(self, x, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, **kwargs)
        x = self.project_out(x)
        return x

# non-causal sortnet and sinkhorn attention

class SimpleSortNet(nn.Module):
    def __init__(self, heads, bucket_size, max_buckets, dim, non_permutative, temperature, sinkhorn_iter):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.max_buckets = max_buckets
        self.bucket_size = bucket_size
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.linear = nn.Parameter(torch.randn(1, heads, dim, max_buckets))
        self.act = nn.ReLU()

    def forward(self, q, k, topk=1):
        bh, t, _ = q.shape
        b = bh // self.heads
        buckets = t // self.bucket_size

        b_q, b_k = bucket(buckets, q), bucket(buckets, k)
        x = torch.cat((b_q.sum(dim=2), b_k.sum(dim=2)), dim=-1)

        W = expand_batch_and_merge_head(b, self.linear)
        R = self.act(x @ W)

        return differentiable_topk(R, k=topk, temperature=self.temperature) if self.non_permutative else gumbel_sinkhorn(R, self.sinkhorn_iter, self.temperature)

class AttentionSortNet(nn.Module):
    def __init__(self, heads, bucket_size, kv_bucket_size, dim, non_permutative, temperature, sinkhorn_iter, n_sortcut = 0):
        super().__init__()
        self.heads = heads
        self.bucket_size = bucket_size
        self.kv_bucket_size = kv_bucket_size
        self.dim = dim
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

    def forward(self, q, k, topk=1):
        bh, *_, bucket_size, kv_bucket_size, device, dtype, dim = *q.shape, self.bucket_size, self.kv_bucket_size, q.device, q.dtype, self.dim
        b = bh // self.heads

        buckets = q.shape[1] // bucket_size
        kv_buckets = k.shape[1] // kv_bucket_size

        b_q = bucket(buckets, q) if self.n_sortcut == 0 else bucket(1, q)
        b_k = bucket(kv_buckets, k)

        sq = b_q.mean(dim=2)
        sk = b_k.mean(dim=2)

        R = torch.einsum('bie,bje->bij', sq, sk).to(q) * (dim ** -0.5)

        if self.non_permutative:
            k = topk if self.n_sortcut == 0 else self.n_sortcut
            return differentiable_topk(R, k=k)

        return gumbel_sinkhorn(F.relu(R), self.sinkhorn_iter, self.temperature)

class SinkhornAttention(nn.Module):
    def __init__(self, bucket_size, dim, dim_heads, heads, max_seq_len, temperature = 0.75, non_permutative = True, sinkhorn_iter = 7, n_sortcut = 0, dropout = 0., kv_bucket_size = None, use_simple_sort_net = False, n_top_buckets = 1):
        super().__init__()
        self.bucket_size = bucket_size
        self.kv_bucket_size = default(kv_bucket_size, bucket_size)

        self.dim = dim
        self.heads = heads
        self.temperature = temperature
        self.non_permutative = non_permutative
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

        if use_simple_sort_net:
            self.sort_net = SimpleSortNet(heads, self.kv_bucket_size, max_seq_len // self.kv_bucket_size, dim_heads * 2, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter)
        else:
            self.sort_net = AttentionSortNet(heads, self.bucket_size, self.kv_bucket_size, dim_heads, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut)

        self.n_top_buckets = n_top_buckets
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        b, h, t, d_h, n_top, d, heads, temperature, bucket_size, kv_bucket_size, device = *q.shape, self.n_top_buckets, self.dim, self.heads, self.temperature, self.bucket_size, self.kv_bucket_size, q.device

        bh = b * h
        buckets = q.shape[2] // bucket_size
        kv_buckets = k.shape[2] // kv_bucket_size
        n_top = min(n_top, kv_buckets)

        merge_batch_head = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_head, (q, k, v))

        # bucket query, key, values

        b_q = bucket(buckets, q)
        b_k, b_v = map(partial(bucket, kv_buckets), (k, v))

        bsz = b_k.shape[2]

        # calculate reordering matrix R with simple sort net

        R = self.sort_net(q, k, topk=n_top)
        R = R.type_as(q).to(q)

        # concatenate reordered buckets

        b_k_r = reorder_buckets(b_k, R)
        b_v_r = reorder_buckets(b_v, R)

        # choose the top n ranked buckets for all query buckets

        if self.n_sortcut > 0:
            b_k_r = b_k_r[:, 0:self.n_sortcut].reshape(bh, 1, -1, d_h)
            b_v_r = b_v_r[:, 0:self.n_sortcut].reshape(bh, 1, -1, d_h)
            b_k_r = expand_dim(b_k_r, 1, buckets)
            b_v_r = expand_dim(b_v_r, 1, buckets)
        else:
            b_k_r = b_k_r.reshape(bh, buckets, -1, d_h)
            b_v_r = b_k_r.reshape(bh, buckets, -1, d_h)

        b_k = torch.cat((b_k_r, b_k), dim=2) if buckets == kv_buckets else b_k_r
        b_v = torch.cat((b_v_r, b_v), dim=2) if buckets == kv_buckets else b_v_r

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d_h ** -0.5)

        # mask 
        mask_value = max_neg_value(dots)

        if not all_none(q_mask, kv_mask):
            q_mask = default(q_mask, lambda: torch.ones((b, t), device=device).bool())
            kv_mask = default(kv_mask, q_mask)
            mq, mk = bucket(buckets, q_mask), bucket(kv_buckets, kv_mask)
            expand_head_and_merge_into_batch = lambda x: merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, h))
            mq, mk = map(expand_head_and_merge_into_batch, (mq, mk))

            mk_r = batched_index_select(mk, R.abs().argmax(dim=-1))

            if self.n_sortcut > 0:
                mk_r = mk_r[:, 0:self.n_sortcut].reshape(-1, 1, bsz * self.n_sortcut)
                mk_r = expand_dim(mk_r, 1, buckets)
            else:
                mk_r = mk_r.reshape(bh, buckets, -1)

            mk = torch.cat((mk_r, mk), dim=2) if buckets == kv_buckets else mk_r
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask            

        # attention
        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        out = torch.einsum('buij,buje->buie', dots, b_v)
        out = unbucket(out)

        out = out.reshape(b, h, t, d_h)
        return out

# causal sort net and reordered bucketing attention

def mask_reordering_matrix(R, topk, temperature):
    buckets = R.shape[1]

    mask_value = max_neg_value(R)
    mask = torch.zeros(R.shape, device=R.device).bool()
    i, j = torch.triu_indices(buckets, buckets)
    mask[:, i, j + topk] = True

    R.masked_fill_(mask, mask_value)
    return differentiable_topk(R, topk, temperature)

class CausalSimpleSortNet(nn.Module):
    def __init__(self, heads, bucket_size, max_buckets, n_top_buckets, dim, temperature):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.bucket_size = bucket_size
        self.max_buckets = max_buckets
        self.n_top_buckets = n_top_buckets
        self.temperature = temperature
        self.linear = nn.Parameter(torch.randn(1, heads, dim, max_buckets + n_top_buckets))
        self.act = nn.LeakyReLU()

    def forward(self, q, k, topk=1):
        bh, *_, h, max_buckets = *q.shape, self.heads, self.max_buckets
        b = bh // h
        buckets = k.shape[1] // self.bucket_size

        k_r = torch.cat((cumavg(k, dim=1), k), dim=-1)
        k_r = bucket(buckets, k_r)

        # for causal sort net, take the first token of each bucket to prevent leaking of future to past
        x = k_r[:, :, 0]

        W = expand_batch_and_merge_head(b, self.linear)
        R = self.act(x @ W)
        R = R[:, 0:buckets, 0:(buckets + self.n_top_buckets)]

        return mask_reordering_matrix(R, topk, self.temperature)

class CausalAttentionSortNet(nn.Module):
    def __init__(self, heads, bucket_size, dim, temperature):
        super().__init__()
        self.heads = heads
        self.bucket_size = bucket_size
        self.dim = dim
        self.temperature = temperature

    def forward(self, q, k, topk=1):
        bh, *_, h, dim = *q.shape, self.heads, self.dim

        b = bh // h
        buckets = q.shape[1] // self.bucket_size
        kv_buckets = k.shape[1] // self.bucket_size

        q_r = bucket(buckets, cumavg(q, dim=1))
        k_r = bucket(kv_buckets, cumavg(k, dim=1))

        sq = q_r[:, :, 0]
        sk = k_r.sum(dim=2)
        sk = F.pad(sk, (0, 0, topk, 0))

        R = torch.einsum('bie,bje->bij', sq, sk) * (dim ** -0.5)
        return mask_reordering_matrix(R, topk, self.temperature)

def apply_fn_after_split_ind(dim, ind, fn, t):
    l, r = split_at_index(dim, ind, t)
    return torch.cat((l, fn(r)), dim=dim)

class SinkhornCausalAttention(nn.Module):
    def __init__(self, bucket_size, dim, dim_heads, heads, max_seq_len, dropout = 0., kv_bucket_size = None, use_simple_sort_net = False, n_top_buckets = 2, temperature = 1.):
        super().__init__()
        assert kv_bucket_size is None or bucket_size == kv_bucket_size, 'different bucketing for key/values for causal reordering not supported yet'

        self.dim = dim
        self.heads = heads
        self.bucket_size = bucket_size

        # a learned null key / value for the first bucket (which has nothing in the past to sort to)
        self.null_keys = nn.Parameter(torch.randn(heads, 1, dim_heads))
        self.null_values = nn.Parameter(torch.randn(heads, 1, dim_heads))

        if use_simple_sort_net:
            self.sort_net = CausalSimpleSortNet(heads, bucket_size, max_seq_len // bucket_size, n_top_buckets, dim_heads * 2, temperature)
        else:
            self.sort_net = CausalAttentionSortNet(heads, bucket_size, dim_heads, temperature)

        self.n_top_buckets = n_top_buckets
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        b, h, t, d_h, n_top, d, bsz, device = *q.shape, self.n_top_buckets, self.dim, self.bucket_size, q.device

        bh = b * h
        hh = h // 2
        buckets = t // bsz
        n_top = min(n_top, buckets)

        hh_slice = (slice(None), slice(hh, None))

        rotate_fn = partial(apply_fn_after_split_ind, 1, hh, lambda t: rotate_left(t, bsz-1, dim=2))
        q, k, v = map(rotate_fn, (q, k, v))

        # merge batch and head
        merge_batch_head = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_head, (q, k, v))

        # bucket qkv
        b_q, b_k, b_v = map(partial(bucket, buckets), (q, k, v))

        # calculate R
        R = self.sort_net(q, k, topk=n_top)
        R = R.type_as(q).to(q)

        # add null key / values
        b_null_k = self.null_keys[None, :, None, :, :].expand(b, h, n_top, bsz, -1).reshape(bh, n_top, bsz, -1).to(k)
        b_null_v = self.null_values[None, :, None, :, :].expand(b, h, n_top, bsz, -1).reshape(bh, n_top, bsz, -1).to(v)

        b_k_r = torch.cat((b_null_k, b_k), dim=1)
        b_v_r = torch.cat((b_null_v, b_v), dim=1)

        # reorder buckets to buckets of the past
        b_k_r = reorder_buckets(b_k_r, R)
        b_v_r = reorder_buckets(b_v_r, R)

        b_k_r = b_k_r.reshape(bh, buckets, bsz * n_top, -1)
        b_v_r = b_v_r.reshape(bh, buckets, bsz * n_top, -1)

        # and concatenate to original buckets themselves for local attention
        b_k = torch.cat((b_k_r, b_k), dim=2)
        b_v = torch.cat((b_v_r, b_v), dim=2)

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d_h ** -0.5)

        # mask
        mask_value = max_neg_value(q)

        if not all_none(q_mask, kv_mask):
            q_mask = default(q_mask, lambda: torch.ones((b, t), device=device).bool())
            kv_mask = default(kv_mask, q_mask)

            expand_head = lambda x: x.unsqueeze(1).repeat(1, h, 1)
            q_mask, kv_mask = map(expand_head, (q_mask, kv_mask))

            q_mask[hh_slice] = rotate_left(q_mask[hh_slice], bsz-1, dim=2)
            kv_mask[hh_slice] = rotate_left(kv_mask[hh_slice], bsz-1, dim=2)

            q_mask, kv_mask = map(lambda x: merge_dims(0, 1, x), (q_mask, kv_mask))
            mq, mk = bucket(buckets, q_mask), bucket(buckets, kv_mask)

            mk_with_null = F.pad(mk, (0, 0, 2, 0), value=True)
            mk_r = batched_index_select(mk_with_null, R.abs().argmax(dim=-1))

            mk_r = mk_r.reshape(bh, buckets, -1)
            mk = torch.cat((mk_r, mk), dim=2)
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # masking for half head rotations
        shift = n_top * bsz
        total_shift = shift + bsz

        mask = torch.ones((b, h, buckets, bsz, total_shift), device=device).bool()
        i, j = torch.triu_indices(bsz, bsz, 1)
        mask[:, :, :, i, j + shift] = False
        mask[:, hh:, -1, 0:shift, 0:shift+1] = False
        mask[:, hh:, -1, 0, 0:shift+1] = True
        mask = mask.reshape(b * h, buckets, bsz, total_shift)

        dots.masked_fill_(~mask, mask_value)
        del mask

        # attention
        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        out = torch.einsum('buij,buje->buie', dots, b_v)
        out = unbucket(out)

        out = out.reshape(b, h, t, d_h)
        out = apply_fn_after_split_ind(1, hh, lambda t: rotate_right(t, bsz-1, dim=2), out)
        return out

class SinkhornSelfAttention(nn.Module):
    def __init__(self, dim, bucket_size, max_seq_len, heads = 8, dim_head = None, kv_bucket_size = None, causal = False, non_permutative = True, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, attn_dropout = 0., dropout = 0., context_only = False, use_simple_sort_net = False, n_local_attn_heads = 0, n_top_buckets = 1):
        super().__init__()
        assert dim_head or divisible_by(dim, heads), f'If dim_head is None, dimension {dim} must be divisible by the number of heads {heads}'
        assert not (causal and n_sortcut > 0), 'sortcut can only be used for non causal attention'
        assert not (causal and context_only), 'context only self attention layer cannot be causal'
        assert n_local_attn_heads <= heads, 'number of local attention heads cannot exceed total heads'

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads
        self.dim_head = dim_head

        self.heads = heads
        self.bucket_size = bucket_size
        self.kv_bucket_size = default(kv_bucket_size, bucket_size)

        self.context_only = context_only
        self.to_q = nn.Linear(dim, dim_heads, bias=False)
        self.to_kv = nn.Linear(dim, dim_heads * 2, bias=False) if not context_only else None

        self.to_out = nn.Linear(dim_heads, dim)

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attention = LocalAttention(bucket_size, causal, dropout = attn_dropout, look_forward=(1 if not causal else 0))

        sink_heads = heads - n_local_attn_heads

        if causal:
            attn = SinkhornCausalAttention(bucket_size, dim, dim_head, sink_heads, max_seq_len, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets, temperature = temperature)
        else:
            attn = SinkhornAttention(bucket_size, dim, dim_head, sink_heads, max_seq_len, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets)

        self.sinkhorn_attention = attn

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_mask = None, context = None, context_mask = None):
        b, t, d, h, dh, l_h = *x.shape, self.heads, self.dim_head, self.n_local_attn_heads
        assert divisible_by(t, self.bucket_size), f'sequence {t} needs to be divisible by bucket size {self.bucket_size}'
        assert not (self.context_only and context is None), 'context key / values must be supplied if context self attention layer'
        assert not (context is not None and (context.shape[0], context.shape[2]) !=  (b, d)), 'contextual key / values must have the same batch and dimensions as the decoder'

        q = self.to_q(x)

        kv = self.to_kv(x).chunk(2, dim=-1) if not self.context_only else (context, context)
        kv_mask = input_mask if not self.context_only else context_mask

        assert divisible_by(kv[0].shape[1], self.kv_bucket_size), 'key/value sequences need to be divisible by key/value bucket size'

        qkv = (q, *kv)
        merge_heads_fn = partial(merge_heads, h)
        q, k, v = map(merge_heads_fn, qkv)

        split_index_fn = partial(split_at_index, 1, l_h)
        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))
        has_local, has_sinkhorn = map(lambda x: x.shape[1] > 0, (lq, q))

        out = []

        if has_local > 0:
            out.append(self.local_attention(lq, lk, lv, input_mask = input_mask))

        if has_sinkhorn > 0:
            out.append(self.sinkhorn_attention(q, k, v, q_mask = input_mask, kv_mask = kv_mask))

        out = torch.cat(out, dim=1)
        out = split_heads(h, out)
        out = self.to_out(out)
        out = self.dropout(out)
        return out

class SinkhornTransformer(nn.Module):
    def __init__(self, dim, depth, max_seq_len = None, causal = False, heads = 8, dim_head = None, bucket_size = 64, kv_bucket_size = None, context_bucket_size = None, non_permutative = False, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, reversible = False, ff_chunks = 1, ff_dropout = 0., attn_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., weight_tie = False, ff_glu = False, use_simple_sort_net = None, receives_context = False, context_n_sortcut = 2, n_local_attn_heads = 0, use_rezero = False, n_top_buckets = 1,  pkm_layers = tuple(), pkm_num_keys = 128):
        super().__init__()
        layers = nn.ModuleList([])

        kv_bucket_size = default(kv_bucket_size, bucket_size)
        context_bucket_size = default(context_bucket_size, bucket_size)

        get_attn = lambda: SinkhornSelfAttention(dim, bucket_size, max_seq_len, causal = causal, heads = heads, dim_head = dim_head, kv_bucket_size = kv_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, use_simple_sort_net = use_simple_sort_net, n_local_attn_heads = n_local_attn_heads, n_top_buckets = n_top_buckets)
        get_ff = lambda: Chunk(ff_chunks, FeedForward(dim, dropout = ff_dropout, glu = ff_glu), along_dim=1)
        get_pkm = lambda: PKM(dim, num_keys = pkm_num_keys)

        get_attn_context = lambda: SinkhornSelfAttention(dim, bucket_size, max_seq_len, context_only = True, heads = heads, dim_head = dim_head, kv_bucket_size = context_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = context_n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, n_top_buckets = n_top_buckets)
        get_ff_context = lambda: FeedForward(dim, dropout = ff_dropout, glu = ff_glu)

        if weight_tie:
            get_attn, get_attn_context, get_ff, get_ff_context = map(cache_fn, (get_attn, get_attn_context, get_ff, get_ff_context))

        fn_wrapper = partial(PreNorm, nn.LayerNorm, dim) if not use_rezero else ReZero

        for ind in range(depth):
            layer_num = ind + 1
            use_pkm = layer_num in pkm_layers

            get_parallel_fn = get_ff if not use_pkm else get_pkm

            layers.append(nn.ModuleList([
                fn_wrapper(get_attn()),
                fn_wrapper(get_parallel_fn())
            ]))

            if not receives_context:
                continue

            layers.append(nn.ModuleList([
                fn_wrapper(get_attn_context()),
                fn_wrapper(get_ff_context())
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        attn_context_layer = ((True, False),) if receives_context else tuple()
        route_attn = ((True, False), *attn_context_layer) * depth
        route_context = ((False, False), *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
        attn_route_map = {'input_mask': route_attn}

        self.layers = execute_type(layers, args_route = {**context_route_map, **attn_route_map}, layer_dropout = layer_dropout)
        self.receives_context = receives_context

        self.max_seq_len = max_seq_len
        self.pad_to_bucket_size = lcm(bucket_size, kv_bucket_size)
        self.context_bucket_size = context_bucket_size

        self.is_fixed_length = use_simple_sort_net and not causal

        # if not using attention sort and also not causal, force fixed sequence length
        assert not (self.is_fixed_length and self.max_seq_len is None), 'maximum sequence length must be specified if length is fixed'

    def forward(self, x, **kwargs):
        assert not (self.is_fixed_length and x.shape[1] != self.max_seq_len), f'you must supply a sequence of length {self.max_seq_len}'
        assert ('context' not in kwargs or self.receives_context), 'needs to be initted with receives_context True if passing contextual key / values'
        return self.layers(x, **kwargs)

class SinkhornTransformerLM(nn.Module):
    def __init__(self, num_tokens, dim, max_seq_len, depth, heads = 8, dim_head = None, bucket_size = 64, kv_bucket_size = None, context_bucket_size = None, causal = False, non_permutative = True, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, reversible = False, ff_chunks = 1, ff_glu = False, return_embeddings = False, ff_dropout = 0., attn_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., emb_dropout = 0., weight_tie = False, emb_dim = None, use_simple_sort_net = None, receives_context = False, context_n_sortcut = 0, n_local_attn_heads = 0, use_rezero = False, n_top_buckets = 2, pkm_layers = tuple(), pkm_num_keys = 128):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len

        self.to_token_emb = nn.Embedding(num_tokens, emb_dim)
        self.axial_pos_emb = AxialPositionalEmbedding(emb_dim, axial_shape = (max_seq_len // bucket_size, bucket_size))
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.sinkhorn_transformer = SinkhornTransformer(dim, depth, max_seq_len = max_seq_len, causal = causal, heads = heads, dim_head = dim_head, bucket_size = bucket_size, kv_bucket_size = kv_bucket_size, context_bucket_size = context_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, reversible = reversible, ff_chunks = ff_chunks, ff_dropout = ff_dropout, attn_dropout = attn_dropout, attn_layer_dropout = attn_layer_dropout, layer_dropout = layer_dropout, weight_tie = weight_tie, ff_glu = ff_glu, use_simple_sort_net = use_simple_sort_net, receives_context = receives_context, context_n_sortcut = context_n_sortcut, n_local_attn_heads = n_local_attn_heads, use_rezero = use_rezero, n_top_buckets = n_top_buckets,  pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys)

        if emb_dim != dim:
            self.sinkhorn_transformer = ProjectInOut(self.sinkhorn_transformer, emb_dim, dim, project_out =(not return_embeddings))

        self.norm = nn.LayerNorm(emb_dim)
        self.to_logits = identity if return_embeddings else nn.Linear(emb_dim, num_tokens)

    def forward(self, x, **kwargs):
        _, t, device = *x.shape, x.device
        assert t <= self.max_seq_len, f'sequence length {t} is greater than maximum sequence length {self.max_seq_len}'

        x = self.to_token_emb(x)
        x = self.axial_pos_emb(x) + x
        x = self.emb_dropout(x)
        x = self.sinkhorn_transformer(x, **kwargs)
        x = self.norm(x)
        return self.to_logits(x)
