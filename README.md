## Sinkhorn Transformer

[![PyPI version](https://badge.fury.io/py/sinkhorn-transformer.svg)](https://badge.fury.io/py/sinkhorn-transformer)

<img src="./sinkhorn.png" width="500">

<img src="./sortcut.png" width="500">

---

This is a reproduction of the work outlined in <a href="https://arxiv.org/abs/2002.11296">Sparse Sinkhorn Attention</a>, with additional enhancements.

It includes a parameterized sorting network, using sinkhorn normalization to sample a permutation matrix that matches the most relevant buckets of keys to the buckets of queries.

This work also brings in reversible networks and feed forward chunking (concepts introduced from <a href="https://openreview.net/forum?id=rkgNKkHtvB">Reformer</a>) to bring about further memory savings.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Eej8U4pP5ldZOz3tHwpoBFgmQqLhQLUq) 204k tokens (demonstration purposes)

## Install

```bash
$ pip install sinkhorn_transformer
```

## Use

A Sinkhorn Transformer based language model

```python
import torch
from sinkhorn_transformer import SinkhornTransformerLM

s = SinkhornTransformerLM(
    num_tokens = 20000,
    dim = 1024,
    heads = 8,
    depth = 12,
    buckets = 64,
    max_seq_len = 8192,
    causal = False,           # auto-regressive or not
    n_sortcut = 2,            # use sortcut to reduce memory complexity to linear
    ff_chunks = 10,           # feedforward chunking, from Reformer paper
    reversible = True,        # make network reversible, from Reformer paper
    ff_dropout = 0.1,         # feedforward dropout
    attn_dropout = 0.1,       # post attention dropout
    attn_layer_dropout = 0.1, # post attention layer dropout
    layer_dropout = 0.1,      # add layer dropout, from 'Reducing Transformer Depth on Demand' paper
    weight_tie = True,        # tie layer parameters, from Albert paper
    emb_dim = 128,            # embedding factorization, from Albert paper
    ff_glu = True,            # use GLU in feedforward, from paper 'GLU Variants Improve Transformer'
    n_local_attn_heads = 2,   # replace N heads with local attention, suggested to work well from Routing Transformer paper
)

x = torch.randint(0, 20000, (1, 2048))
s(x) # (1, 2048, 20000)
```

A plain Sinkhorn Transformer, layers of sinkhorn attention

```python
import torch
from sinkhorn_transformer import SinkhornTransformer

s = SinkhornTransformer(
    dim = 1024,
    heads = 8,
    depth = 12,
    buckets = 64
)

x = torch.randn(1, 2048, 1024)
s(x) # (1, 2048, 1024)
```

Sinkhorn Encoder / Decoder Transformer

```python
import torch
from sinkhorn_transformer import SinkhornTransformerLM

DE_SEQ_LEN = 4096
EN_SEQ_LEN = 4096

enc = SinkhornTransformerLM(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    heads = 8,
    buckets = 32,
    max_seq_len = DE_SEQ_LEN,
    reversible = True,
    return_embeddings = True
).cuda()

dec = SinkhornTransformerLM(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    causal = True,
    buckets = 32,
    max_seq_len = EN_SEQ_LEN,
    receives_context = True,
    reversible = True
).cuda()

x = torch.randint(0, 20000, (1, DE_SEQ_LEN)).cuda()
y = torch.randint(0, 20000, (1, EN_SEQ_LEN)).cuda()

x_mask = torch.ones_like(x).bool().cuda()
y_mask = torch.ones_like(y).bool().cuda()

context = enc(x, input_mask=x_mask)
dec(y, context=context, input_mask=y_mask, context_mask=x_mask) # (1, 4096, 20000)
```

## Sinkhorn

This repository has diverged from the paper and is now using attention in place of the original sorting net + gumbel sinkhorn sampling. I have not found a noticeable difference in performance yet, and the new scheme allows me to generalize the network to flexible sequence lengths. If you would like to try Sinkhorn, please use the following settings, which only works for non-causal networks.

```python
import torch
from sinkhorn_transformer import SinkhornTransformerLM

s = SinkhornTransformerLM(
    num_tokens = 20000,
    dim = 1024,
    heads = 8,
    depth = 12,
    buckets = 64,
    max_seq_len = 8192,
    use_simple_sort_net = True, # turn off attention sort net
    sinkhorn_iter = 7,          # number of sinkhorn iterations - default is set at reported best in paper
    n_sortcut = 2,              # use sortcut to reduce complexity to linear time
    temperature = 0.75,         # gumbel temperature - default is set at reported best in paper
    non_permutative = False,    # allow buckets of keys to be sorted to queries more than once
)

x = torch.randint(0, 20000, (1, 8192))
s(x) # (1, 8192, 20000)
```

## Citations

```bibtex
@misc{tay2020sparse,
    title   = {Sparse Sinkhorn Attention},
    author  = {Yi Tay and Dara Bahri and Liu Yang and Donald Metzler and Da-Cheng Juan},
    year    = {2020},
    url.    = {https://arxiv.org/abs/2002.11296}
}
```

```bibtex
@inproceedings{kitaev2020reformer,
    title       = {Reformer: The Efficient Transformer},
    author      = {Nikita Kitaev and Lukasz Kaiser and Anselm Levskaya},
    booktitle   = {International Conference on Learning Representations},
    year        = {2020},
    url         = {https://openreview.net/forum?id=rkgNKkHtvB}
}
```

```bibtex
@misc{lan2019albert,
    title       = {ALBERT: A Lite BERT for Self-supervised Learning of Language Representations},
    author      = {Zhenzhong Lan and Mingda Chen and Sebastian Goodman and Kevin Gimpel and Piyush Sharma and Radu Soricut},
    year        = {2019},
    url         = {https://arxiv.org/abs/1909.11942}
}
```

```bibtex
@misc{shazeer2020glu,
    title   = {GLU Variants Improve Transformer},
    author  = {Noam Shazeer},
    year    = {2020},
    url     = {https://arxiv.org/abs/2002.05202}
}
```

```bibtex
@misc{roy*2020efficient,
    title   = {Efficient Content-Based Sparse Attention with Routing Transformers},
    author  = {Aurko Roy* and Mohammad Taghi Saffar* and David Grangier and Ashish Vaswani},
    year    = {2020},
    url     = {https://openreview.net/forum?id=B1gjs6EtDr}
}
```

```bibtex
@inproceedings{fan2020reducing,
    title     ={Reducing Transformer Depth on Demand with Structured Dropout},
    author    ={Angela Fan and Edouard Grave and Armand Joulin},
    booktitle ={International Conference on Learning Representations},
    year      ={2020},
    url       ={https://openreview.net/forum?id=SylO2yStDr}
}
```

<img src="./divine.png"></img>