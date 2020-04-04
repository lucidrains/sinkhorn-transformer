import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from sinkhorn_transformer.reversible import ReversibleSequence

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

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

def merge_heads(h, v):
    b, t, d = v.shape
    return v.view(b, t, h, -1).transpose(1, 2).reshape(b, h, t, -1)

def split_heads(h, v):
    *_, t, d = v.shape
    return v.view(-1, h, t, d).transpose(1, 2).reshape(-1, t, d * h)

def bucket(buckets, v):
    b, t, d = v.shape
    return v.reshape(b, buckets, -1, d)

def unbucket(v):
    b, *_, e = v.shape
    return v.reshape(b, -1, e)

def logsumexp(tensor, dim, keepdim=True):
    return torch.log(torch.exp(tensor).sum(dim, keepdim=keepdim))

def sample_gumbel(shape, device, eps=1e-4):
    u = torch.empty(shape, device = device).uniform_(0, 1)
    return -torch.log(-torch.log(u + eps) + eps)

def identity(x, *args, **kwargs): return x

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def sinkhorn_sorting_operator(r, n_iters = 8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - logsumexp(r, dim=2)
        r = r - logsumexp(r, dim=1)
    return torch.exp(r)

def reorder_buckets(t, r):
    return torch.einsum('buv,bvtd->butd', r, t)

def cumavg(t, dim):
    r = torch.arange(1, t.shape[dim] + 1, device=t.device)
    expand_slice = [None] * len(t.shape)
    expand_slice[dim] = slice(None, None)
    return t.cumsum(dim=dim) / r[tuple(expand_slice)]

def each(fn, arr):
    for el in arr:
        fn(el)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x):
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c) for c in chunks], dim = self.dim)

class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.LeakyReLU(),
            nn.Linear(ff_mult * dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class WithNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class SinkhornAttention(nn.Module):
    def __init__(self, buckets, dim, heads, temperature = 0.75, sinkhorn_iter = 7, n_sortcut = 0):
        super().__init__()
        self.dim = dim
        self.buckets = buckets
        self.heads = heads
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

        d_head = dim // heads
        self.sort_w = nn.Parameter(torch.randn(1, heads, d_head, buckets))

    def forward(self, q, k, v, context=None):
        b, h, t, d_h, d, heads, temperature, buckets, device = *q.shape, self.dim, self.heads, self.temperature, self.buckets, q.device
        bsz = t // buckets

        q = q.reshape(b * h, t, d_h)
        k = k.reshape(b * h, t, d_h)
        v = v.reshape(b * h, t, d_h)

        bucket_fn = partial(bucket, buckets)
        b_q, b_k, b_v = map(bucket_fn, (q, k, v))

        # calculate R

        k_r = bucket_fn(k)
        b_k_r = k_r.sum(dim=2)

        e_W_r = self.sort_w.expand(b, -1, -1, -1).reshape(b * heads, d_h, buckets)
        R = log(F.relu(b_k_r @ e_W_r))

        # gumbel sinkhorn

        gumbel_noise = sample_gumbel(R.shape, device)
        R  = (R + gumbel_noise) / temperature

        R = sinkhorn_sorting_operator(R, self.sinkhorn_iter)
        R = torch.tril(R, diagonal=-1)
        R = R.type(q.type())

        k_bucketed = bucket_fn(k)
        v_bucketed = bucket_fn(v)

        k_reordered_buckets = reorder_buckets(k_bucketed, R)
        v_reordered_buckets = reorder_buckets(v_bucketed, R)

        if self.n_sortcut > 0:
            k_reordered_buckets = k_reordered_buckets[:, 0:self.n_sortcut].reshape(-1, 1, bsz * self.n_sortcut, d_h).expand(-1, buckets, -1, -1)
            v_reordered_buckets = v_reordered_buckets[:, 0:self.n_sortcut].reshape(-1, 1, bsz * self.n_sortcut, d_h).expand(-1, buckets, -1, -1)

        b_k = torch.cat((k_reordered_buckets, b_k), dim=2)
        b_v = torch.cat((v_reordered_buckets, b_v), dim=2)

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d ** -0.5)
        dots = dots.softmax(dim=-1)

        out = torch.einsum('buij,buje->buie', dots, b_v)
        out = unbucket(out)

        out = out.reshape(b, h, t, d_h)
        return out

class SinkhornCausalAttention(nn.Module):
    def __init__(self, buckets, dim, heads, temperature = 0.75, sinkhorn_iter = 7, n_sortcut = 0):
        super().__init__()
        self.dim = dim
        self.buckets = buckets
        self.heads = heads
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        d_head = dim // heads
        self.sort_w = nn.Parameter(torch.randn(1, heads, d_head * 2, buckets))

    def forward(self, q, k, v):
        b, h, t, d_h, d, heads, temperature, buckets, device = *q.shape, self.dim, self.heads, self.temperature, self.buckets, q.device
        bsz = t // buckets
        hh = h // 2

        hh_slice = (slice(None), slice(hh, None))

        q[hh_slice] = rotate_left(q[hh_slice], bsz-1, dim=2)
        k[hh_slice] = rotate_left(k[hh_slice], bsz-1, dim=2)
        v[hh_slice] = rotate_left(v[hh_slice], bsz-1, dim=2)

        q = q.reshape(b * h, t, d_h)
        k = k.reshape(b * h, t, d_h)
        v = v.reshape(b * h, t, d_h)

        bucket_fn = partial(bucket, buckets)
        b_q, b_k, b_v = map(bucket_fn, (q, k, v))

        # calculate R

        k_r = torch.cat((cumavg(k, dim=1), k), dim=-1)
        k_r = bucket_fn(k_r)

        b_k_r = k_r[:, :, 0]

        e_W_r = self.sort_w.expand(b, -1, -1, -1).reshape(b * heads, d_h * 2, buckets)
        R = log(F.relu(b_k_r @ e_W_r))

        # gumbel sinkhorn

        gumbel_noise = sample_gumbel(R.shape, device)
        R  = (R + gumbel_noise) / temperature

        R = R.softmax(dim=-1)
        R = torch.tril(R, diagonal=-1)
        R = R.type(q.type())

        k_bucketed = bucket_fn(k)
        v_bucketed = bucket_fn(v)

        k_reordered_buckets = reorder_buckets(k_bucketed, R)
        v_reordered_buckets = reorder_buckets(v_bucketed, R)

        b_k = torch.cat((k_reordered_buckets, b_k), dim=2)
        b_v = torch.cat((v_reordered_buckets, b_v), dim=2)

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d ** -0.5)

        mask = torch.ones((b, h, buckets, bsz, bsz * 2), device=device).bool()
        i, j = torch.triu_indices(bsz, bsz, 1)
        mask[:, :, :, i, j + bsz] = False
        mask[:, hh:, -1, 0:bsz, 0:bsz+1] = False
        mask[:, hh:, -1, 0, 0:bsz+1] = True
        mask = mask.reshape(b * h, buckets, bsz, bsz * 2)

        mask_value = max_neg_value(dots)
        dots.masked_fill_(~mask, mask_value)
        del mask

        dots = dots.softmax(dim=-1)

        out = torch.einsum('buij,buje->buie', dots, b_v)
        out = unbucket(out)

        out = out.reshape(b, h, t, d_h)
        out[hh_slice] = rotate_right(out[hh_slice], bsz-1, dim=2)
        return out

class SinkhornSelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, buckets = 64, causal = False, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        if causal:
            attn = SinkhornCausalAttention(buckets, dim, heads, temperature = temperature)
        else:
            attn = SinkhornAttention(buckets, dim, heads, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature)

        self.sinkhorn_attention = attn

    def forward(self, x):
        b, t, d, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=2)

        merge_heads_fn = partial(merge_heads, h)
        q, k, v = map(merge_heads_fn, qkv)
        out = self.sinkhorn_attention(q, k, v)
        out = split_heads(h, out)
        out = self.to_out(out)
        return out

class SinkhornTransformer(nn.Module):
    def __init__(self, dim, depth, causal = False, heads = 8, buckets = 64, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, ff_chunks = 1):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            layers.append(nn.ModuleList([
                SinkhornSelfAttention(dim, causal = causal, heads = heads, buckets = buckets, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature),
                Chunk(ff_chunks, FeedForward(dim), along_dim=1)
            ]))
        self.layers = ReversibleSequence(layers)
    def forward(self, x, input_mask = None):
        x = torch.cat([x, x], dim = -1)
        x = self.layers(x)
        return torch.stack(x.chunk(2, dim=-1)).sum(dim=0)

class SinkhornTransformerLM(nn.Module):
    def __init__(self, num_tokens, dim, max_seq_len, depth, buckets = 64, heads = 8, causal = False, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, ff_chunks = 1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.to_token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.sinkhorn_transformer = SinkhornTransformer(dim, depth, causal = causal, heads = heads, buckets = buckets, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, ff_chunks = ff_chunks)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, input_mask = None):
        x = self.to_token_emb(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = self.sinkhorn_transformer(x)
        return self.to_logits(x)
