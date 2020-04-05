## Sinkhorn Transformer

[![PyPI version](https://badge.fury.io/py/sinkhorn-transformer.svg)](https://badge.fury.io/py/sinkhorn-transformer)

<img src="./sinkhorn.png" width="500">

---

This will eventually be a reproduction of the work outlined in <a href="https://arxiv.org/abs/2002.11296">Sparse Sinkhorn Attention</a>.

It includes a parameterized sorting network, using sinkhorn normalization to sample a permutation matrix that matches the most relevant buckets of keys to the buckets of queries.

This work also brings in reversible networks and feed forward chunking (concepts introduced from <a href="https://openreview.net/forum?id=rkgNKkHtvB">Reformer</a>) to bring about further memory savings.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Eej8U4pP5ldZOz3tHwpoBFgmQqLhQLUq) 204k tokens (demonstration purposes)

## Install

```bash
$ pip install sinkhorn_transformer
```

## Use

A SinkhornTransformer based language model

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
    causal = False,          # auto-regressive or not
    sinkhorn_iter = 7,       # number of sinkhorn iterations - default is set at reported best in paper
    n_sortcut = 2,           # use sortcut to reduce complexity to linear time
    temperature = 0.75,      # gumbel temperature - default is set at reported best in paper
    non_permutative = False, # allow buckets of keys to be sorted to queries more than once
    ff_chunks = 10,          # feedforward chunking, from Reformer paper
    reversible = True,       # make network reversible, from Reformer paper
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

## Todo

1. Contextual key / values
2. Full encoder / decoder, with argument routing for reversible network
3. Find solution for input masking, potentially with topk of sorting matrix rows
4. Add ability to add local attention heads

## Citations

```bibtex
@misc{tay2020sparse,
    title   ={Sparse Sinkhorn Attention},
    author  ={Yi Tay and Dara Bahri and Liu Yang and Donald Metzler and Da-Cheng Juan},
    year    ={2020},
    url.    ={https://arxiv.org/abs/2002.11296}
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
