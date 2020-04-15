import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce
from sinkhorn_transformer.reversible import ReversibleSequence, SequentialSequence

# helper functions

def identity(x, *args, **kwargs): return x

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def divisible_by(num, divisor):
    return num % divisor == 0

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
    shape[arr_slice] = [reduce((lambda x, y: x * y), shape[arr_slice])]
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
    r = torch.arange(1, t.shape[dim] + 1, device=t.device)
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

def zero_all_but_top(x, dim, k=1):
    values, indices = torch.topk(x, k, dim=dim)
    return torch.zeros_like(x).scatter_(dim, indices, values)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

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

# local attention

class LocalAttention(nn.Module):
    def __init__(self, buckets, causal = False, look_backward = 1, look_forward = 0, dropout = 0.):
        super().__init__()
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'
        self.buckets = buckets
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        buckets, causal, look_backward, look_forward = self.buckets, self.causal, self.look_backward, self.look_forward
        b, h, t, e, device, dtype = *q.shape, q.device, q.dtype

        ticker = torch.arange(t, device=device, dtype=dtype)[None, :]
        b_t = bucket(buckets, ticker)

        merge_batch_and_heads = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_and_heads, (q, k, v))

        bucket_fn = partial(bucket, buckets)
        bq, bk, bv = map(bucket_fn, (q, k, v))

        look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        dots = torch.einsum('bhie,bhje->bhij', bq, bk)
        mask_value = max_neg_value(dots)

        if causal:
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]
            dots.masked_fill_(mask, mask_value)
            del mask

        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask

        if not all_none(q_mask, kv_mask):
            q_mask = default(q_mask, lambda: torch.ones((b, t), device=device).bool())
            kv_mask = default(kv_mask, q_mask)
            mq, mk = map(bucket_fn, (q_mask, kv_mask))

            mk = look_around(mk, pad_value = False, **look_around_kwargs)
            mask = (mq[:, None, :, :, None] * mk[:, None, :, None, :])
            mask = merge_batch_and_heads(mask.expand(-1, h, -1, -1, -1))
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        out = out.reshape(b, h, t, e)
        return out

# non-causal sortnet and sinkhorn attention

class SimpleSortNet(nn.Module):
    def __init__(self, heads, buckets, dim, non_permutative, temperature, sinkhorn_iter):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.buckets = buckets
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.linear = nn.Parameter(torch.randn(1, heads, dim, buckets))
        self.act = nn.ReLU()

    def forward(self, q, k):
        bh, *_, buckets = *q.shape, self.buckets
        b = bh // self.heads

        b_q, b_k = bucket(buckets, q), bucket(buckets, k)
        x = torch.cat((b_q.sum(dim=2), b_k.sum(dim=2)), dim=-1)

        W = expand_dim(self.linear, 0, b).reshape(b * self.heads, self.dim, buckets)
        R = self.act(x @ W)

        return R.softmax(dim=-1) if self.non_permutative else gumbel_sinkhorn(R, self.sinkhorn_iter, self.temperature)

class AttentionSortNet(nn.Module):
    def __init__(self, heads, buckets, dim, non_permutative, temperature, sinkhorn_iter, n_sortcut = 0):
        super().__init__()
        self.heads = heads
        self.buckets = buckets
        self.dim = dim
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

        self.q_pos_emb = nn.Parameter(torch.randn(1, heads, buckets if n_sortcut == 0 else 1, dim))
        self.k_pos_emb = nn.Parameter(torch.randn(1, heads, buckets, dim))

    def forward(self, q, k):
        bh, *_, buckets, device, dtype, dim = *q.shape, self.buckets, q.device, q.dtype, self.dim
        b = bh // self.heads

        b_q = bucket(buckets, q) if self.n_sortcut == 0 else bucket(1, q)
        b_k = bucket(buckets, k)

        pos_q, pos_k = map(partial(expand_batch_and_merge_head, b), (self.q_pos_emb, self.k_pos_emb))

        sq = b_q.mean(dim=2) + pos_q
        sk = b_k.mean(dim=2) + pos_k

        R = torch.einsum('bie,bje->bij', sq, sk).to(q)

        if self.n_sortcut > 0:
            values, indices = torch.topk(R, self.n_sortcut)
            values = values.reshape(bh, self.n_sortcut, -1)
            indices = indices.reshape(bh, self.n_sortcut, -1)
            R = torch.zeros(bh, self.n_sortcut, buckets, device=device, dtype=dtype).scatter(2, indices, values)

        return R.softmax(dim=-1) if self.non_permutative else gumbel_sinkhorn(F.relu(R), self.sinkhorn_iter, self.temperature)

class SinkhornAttention(nn.Module):
    def __init__(self, buckets, dim, dim_heads, heads, temperature = 0.75, non_permutative = False, sinkhorn_iter = 7, n_sortcut = 0, dropout = 0., kv_buckets = None, attn_sort_net = False):
        super().__init__()
        self.buckets = buckets
        self.kv_buckets = default(kv_buckets, buckets)
        assert not (self.buckets != self.kv_buckets and n_sortcut == 0), 'sortcut must be used if the query buckets do not equal the key/value buckets'

        self.dim = dim
        self.heads = heads
        self.temperature = temperature
        self.non_permutative = non_permutative
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

        if attn_sort_net:
            self.sort_net = AttentionSortNet(heads, self.kv_buckets, dim_heads, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut)
        else:
            self.sort_net = SimpleSortNet(heads, self.kv_buckets, dim_heads * 2, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        b, h, t, d_h, d, heads, temperature, buckets, kv_buckets, device = *q.shape, self.dim, self.heads, self.temperature, self.buckets, self.kv_buckets, q.device

        merge_batch_head = lambda x: x.reshape(b * h, t, d_h)
        q, k, v = map(merge_batch_head, (q, k, v))

        # bucket query, key, values

        b_q = bucket(buckets, q)
        b_k, b_v = map(partial(bucket, kv_buckets), (k, v))

        bsz = b_k.shape[2]

        # calculate reordering matrix R with simple sort net

        R = self.sort_net(q, k)
        R = R.type_as(q).to(q)

        # only allow one bucket to be reordered to, needed for input masking to work

        R = zero_all_but_top(R, dim=2, k=1)

        # concatenate reordered buckets

        b_k_r = reorder_buckets(b_k, R)
        b_v_r = reorder_buckets(b_v, R)

        # choose the top n ranked buckets for all query buckets

        if self.n_sortcut > 0:
            b_k_r = b_k_r[:, 0:self.n_sortcut].reshape(-1, 1, bsz * self.n_sortcut, d_h)
            b_v_r = b_v_r[:, 0:self.n_sortcut].reshape(-1, 1, bsz * self.n_sortcut, d_h)
            b_k_r = expand_dim(b_k_r, 1, buckets)
            b_v_r = expand_dim(b_v_r, 1, buckets)
        
        b_k = torch.cat((b_k_r, b_k), dim=2) if buckets == kv_buckets else b_k_r
        b_v = torch.cat((b_v_r, b_v), dim=2) if buckets == kv_buckets else b_v_r

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d ** -0.5)

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

def mask_reordering_matrix(R):
    buckets = R.shape[1]

    mask_value = max_neg_value(R)
    mask = torch.zeros(R.shape, device=R.device).bool()
    i, j = torch.triu_indices(buckets, buckets)
    mask[:, i, j + 1] = True

    R.masked_fill_(mask, mask_value)
    del mask

    R = R.softmax(dim=-1)
    R = R.tril(diagonal=-1) # extra insurance
    return R

class CausalSimpleSortNet(nn.Module):
    def __init__(self, heads, buckets, dim):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.buckets = buckets
        self.linear = nn.Parameter(torch.randn(1, heads, dim, buckets + 1))
        self.act = nn.LeakyReLU()

    def forward(self, q, k):
        bh, *_, h, buckets = *q.shape, self.heads, self.buckets
        b = bh // h

        k_r = torch.cat((cumavg(k, dim=1), k), dim=-1)
        k_r = bucket(buckets, k_r)

        # for causal sort net, take the first token of each bucket to prevent leaking of future to past
        x = k_r[:, :, 0]

        W = expand_dim(self.linear, 0, b).reshape(bh, self.dim, buckets + 1)
        R = self.act(x @ W)

        return mask_reordering_matrix(R)

class CausalAttentionSortNet(nn.Module):
    def __init__(self, heads, buckets, dim):
        super().__init__()
        self.heads = heads
        self.buckets = buckets
        self.dim = dim

        self.q_pos_emb = nn.Parameter(torch.randn(1, heads, buckets, dim))
        self.k_pos_emb = nn.Parameter(torch.randn(1, heads, buckets, dim))

    def forward(self, q, k):
        bh, *_, h, buckets, dim = *q.shape, self.heads, self.buckets, self.dim
        b = bh // h

        pos_q, pos_k = map(partial(expand_batch_and_merge_head, b), (self.q_pos_emb, self.k_pos_emb))

        q_r = bucket(buckets, cumavg(q, dim=1))
        k_r = bucket(buckets, cumavg(k, dim=1))

        b_q_r = q_r[:, :, 0]
        b_k_r = k_r.sum(dim=2)

        sq = b_q_r + pos_q
        sk = b_k_r + pos_k

        sk = F.pad(sk, (0, 0, 1, 0))

        R = torch.einsum('bie,bje->bij', sq, sk)
        return mask_reordering_matrix(R)

class SinkhornCausalAttention(nn.Module):
    def __init__(self, buckets, dim, dim_heads, heads, dropout = 0., kv_buckets = None, attn_sort_net = False):
        super().__init__()
        assert kv_buckets is None, 'different bucketing for key/values for causal reordering not supported yet'

        self.dim = dim
        self.buckets = buckets
        self.heads = heads

        # a learned null key / value for the first bucket (which has nothing in the past to sort to)
        self.null_keys = nn.Parameter(torch.randn(heads, 1, dim_heads))
        self.null_values = nn.Parameter(torch.randn(heads, 1, dim_heads))

        if attn_sort_net:
            self.sort_net = CausalAttentionSortNet(heads, buckets, dim_heads)
        else:
            self.sort_net = CausalSimpleSortNet(heads, buckets, dim_heads * 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        b, h, t, d_h, d, buckets, device = *q.shape, self.dim, self.buckets, q.device
        bh = b * h

        bsz = t // buckets
        hh = h // 2

        hh_slice = (slice(None), slice(hh, None))

        q[hh_slice] = rotate_left(q[hh_slice], bsz-1, dim=2)
        k[hh_slice] = rotate_left(k[hh_slice], bsz-1, dim=2)
        v[hh_slice] = rotate_left(v[hh_slice], bsz-1, dim=2)

        # merge batch and head

        merge_batch_head = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_head, (q, k, v))

        # bucket qkv

        b_q, b_k, b_v = map(partial(bucket, buckets), (q, k, v))

        # calculate R
        R = self.sort_net(q, k)
        R = R.type_as(q).to(q)

        # only allow one bucket to be reordered to, needed for input masking to work
        R = zero_all_but_top(R, dim=2, k=1)

        # add null key / values
        b_null_k = self.null_keys[None, :, None, :, :].expand(b, h, 1, bsz, -1).reshape(bh, 1, bsz, -1)
        b_null_v = self.null_values[None, :, None, :, :].expand(b, h, 1, bsz, -1).reshape(bh, 1, bsz, -1)

        b_k_r = torch.cat((b_null_k, b_k), dim=1)
        b_v_r = torch.cat((b_null_v, b_v), dim=1)

        # reorder buckets to buckets of the past
        b_k_r = reorder_buckets(b_k_r, R)
        b_v_r = reorder_buckets(b_v_r, R)

        # and concatenate to original buckets themselves for local attention
        b_k = torch.cat((b_k_r, b_k), dim=2)
        b_v = torch.cat((b_v_r, b_v), dim=2)

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d ** -0.5)

        # mask
        mask_value = max_neg_value(q)

        if not all_none(q_mask, kv_mask):
            q_mask = default(q_mask, lambda: torch.ones((b, t), device=device).bool())
            kv_mask = default(kv_mask, q_mask)
            mq, mk = bucket(buckets, q_mask), bucket(buckets, kv_mask)
            expand_head_and_merge_into_batch = lambda x: merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, h))
            mq, mk = map(expand_head_and_merge_into_batch, (mq, mk))

            mk_with_null = F.pad(mk, (0, 0, 1, 0), value=True)
            mk_r = batched_index_select(mk_with_null, R.abs().argmax(dim=-1))

            mk = torch.cat((mk_r, mk), dim=2)
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask

        mask = torch.ones((b, h, buckets, bsz, bsz * 2), device=device).bool()
        i, j = torch.triu_indices(bsz, bsz, 1)
        mask[:, :, :, i, j + bsz] = False
        mask[:, hh:, -1, 0:bsz, 0:bsz+1] = False
        mask[:, hh:, -1, 0, 0:bsz+1] = True
        mask = mask.reshape(b * h, buckets, bsz, bsz * 2)

        dots.masked_fill_(~mask, mask_value)
        del mask

        # attention
        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        out = torch.einsum('buij,buje->buie', dots, b_v)
        out = unbucket(out)

        out = out.reshape(b, h, t, d_h)
        out[hh_slice] = rotate_right(out[hh_slice], bsz-1, dim=2)
        return out

class SinkhornSelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, buckets = 64, kv_buckets = None, causal = False, non_permutative = False, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, attn_dropout = 0., dropout = 0., context_only = False, attn_sort_net = False, n_local_attn_heads = 0):
        super().__init__()
        assert divisible_by(dim, heads), f'dimension {dim} must be divisible by the number of heads {heads}'
        assert not (causal and n_sortcut > 0), 'sortcut can only be used for non causal attention'
        assert not (causal and context_only), 'context only self attention layer cannot be causal'
        assert n_local_attn_heads <= heads, 'number of local attention heads cannot exceed total heads'

        self.heads = heads
        self.buckets = buckets
        self.kv_buckets = default(kv_buckets, buckets)

        self.context_only = context_only
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False) if not context_only else None

        self.to_out = nn.Linear(dim, dim)

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attention = LocalAttention(buckets, causal, dropout = attn_dropout, look_forward=(1 if not causal else 0))

        sink_heads = heads - n_local_attn_heads
        dim_heads = dim // heads
        if causal:
            attn = SinkhornCausalAttention(buckets, dim, dim_heads, sink_heads, dropout = attn_dropout, kv_buckets = kv_buckets, attn_sort_net = attn_sort_net)
        else:
            attn = SinkhornAttention(buckets, dim, dim_heads, sink_heads, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, dropout = attn_dropout, kv_buckets = kv_buckets, attn_sort_net = attn_sort_net)

        self.sinkhorn_attention = attn

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_mask = None, context = None, context_mask = None):
        b, t, d, h, l_h = *x.shape, self.heads, self.n_local_attn_heads
        assert divisible_by(t, self.buckets), f'sequence {t} needs to be divisible by bucket size {self.buckets}'
        assert not (self.context_only and context is None), 'context key / values must be supplied if context self attention layer'

        q = self.to_q(x)

        kv = self.to_kv(x).chunk(2, dim=-1) if not self.context_only else (context, context)
        kv_mask = input_mask if not self.context_only else context_mask

        assert divisible_by(kv[0].shape[1], self.kv_buckets), 'key/value sequences need to be divisible by key/value bucket size'

        qkv = (q, *kv)
        merge_heads_fn = partial(merge_heads, h)
        q, k, v = map(merge_heads_fn, qkv)

        split_index_fn = partial(split_at_index, 1, l_h)
        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))
        has_local, has_sinkhorn = map(lambda x: x.shape[1] > 0, (lq, q))

        out = []

        if has_local > 0:
            out.append(self.local_attention(lq, lk, lv, q_mask = input_mask, kv_mask = kv_mask))

        if has_sinkhorn > 0:
            out.append(self.sinkhorn_attention(q, k, v, q_mask = input_mask, kv_mask = kv_mask))

        out = torch.cat(out, dim=1)
        out = split_heads(h, out)
        out = self.to_out(out)
        out = self.dropout(out)
        return out

class SinkhornTransformer(nn.Module):
    def __init__(self, dim, depth, causal = False, heads = 8, buckets = 64, kv_buckets = None, non_permutative = False, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, reversible = False, ff_chunks = 1, ff_dropout = 0., attn_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., weight_tie = False, ff_glu = False, attn_sort_net = False, receives_context = False, context_buckets = None, context_n_sortcut = 2, n_local_attn_heads = 0):
        super().__init__()
        layers = nn.ModuleList([])        
        context_buckets = default(context_buckets, buckets)

        get_attn = lambda: SinkhornSelfAttention(dim, causal = causal, heads = heads, buckets = buckets, kv_buckets = kv_buckets, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, attn_sort_net = attn_sort_net, n_local_attn_heads = n_local_attn_heads)
        get_ff = lambda: FeedForward(dim, dropout = ff_dropout, glu = ff_glu)

        get_attn_context = lambda: SinkhornSelfAttention(dim, context_only = True, heads = heads, buckets = buckets, kv_buckets = context_buckets, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = context_n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, attn_sort_net = attn_sort_net)
        get_ff_context = lambda: FeedForward(dim, dropout = ff_dropout, glu = ff_glu)

        if weight_tie:
            get_attn, get_attn_context, get_ff, get_ff_context = map(cache_fn, (get_attn, get_attn_context, get_ff, get_ff_context))

        for _ in range(depth):
            layers.append(nn.ModuleList([
                PreNorm(nn.LayerNorm, dim, get_attn()),
                PreNorm(nn.LayerNorm, dim, Chunk(ff_chunks, get_ff(), along_dim=1))
            ]))

            if receives_context:
                layers.append(nn.ModuleList([
                    PreNorm(nn.LayerNorm, dim, get_attn_context()),
                    PreNorm(nn.LayerNorm, dim, Chunk(ff_chunks, get_ff_context(), along_dim=1))
                ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        attn_context_layer = ((True, False),) if receives_context else tuple()
        route_attn = ((True, False), *attn_context_layer) * depth
        route_context = ((False, False), *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
        attn_route_map = {'input_mask': route_attn}

        self.layers = execute_type(layers, args_route = {**context_route_map, **attn_route_map}, layer_dropout = layer_dropout)
        self.receives_context = receives_context

    def forward(self, x, **kwargs):
        assert ('context' not in kwargs or self.receives_context), 'needs to be initted with receives_context True if passing contextual key / values'
        return self.layers(x, **kwargs)

class SinkhornTransformerLM(nn.Module):
    def __init__(self, num_tokens, dim, max_seq_len, depth, buckets = 64, kv_buckets = None, heads = 8, causal = False, non_permutative = False, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, reversible = False, ff_chunks = 1, ff_glu = False, return_embeddings = False, ff_dropout = 0., attn_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., weight_tie = False, emb_dim = None, attn_sort_net = False, receives_context = False, context_buckets = None, context_n_sortcut = 2, n_local_attn_heads = 0):
        super().__init__()
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.to_token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = nn.Embedding(max_seq_len, emb_dim)
        self.sinkhorn_transformer = SinkhornTransformer(dim, depth, causal = causal, heads = heads, buckets = buckets, kv_buckets = kv_buckets, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, reversible = reversible, ff_chunks = ff_chunks, ff_dropout = ff_dropout, attn_dropout = attn_dropout, attn_layer_dropout = attn_layer_dropout, layer_dropout = layer_dropout, weight_tie = weight_tie, ff_glu = ff_glu, attn_sort_net = attn_sort_net, receives_context = receives_context, context_buckets = context_buckets, context_n_sortcut = context_n_sortcut, n_local_attn_heads = n_local_attn_heads)

        if emb_dim != dim:
            self.sinkhorn_transformer = ProjectInOut(self.sinkhorn_transformer, emb_dim, dim, project_out =(not return_embeddings))

        self.to_logits = identity if return_embeddings else nn.Linear(emb_dim, num_tokens)

    def forward(self, x, **kwargs):
        _, t, device = *x.shape, x.device
        assert t <= self.max_seq_len, f'sequence length {t} is greater than maximum sequence length {self.max_seq_len}'

        x = self.to_token_emb(x)
        x = self.pos_emb(torch.arange(t, device=device)) + x
        x = self.sinkhorn_transformer(x, **kwargs)
        return self.to_logits(x)
