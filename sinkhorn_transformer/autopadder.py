import math
import torch
from torch import nn
import torch.nn.functional as F
from sinkhorn_transformer.sinkhorn_transformer import SinkhornTransformer, SinkhornTransformerLM

def pad_to_multiple(tensor, seqlen, multiple, dim=-1, pad_left = False):
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    padding = math.ceil(m) * multiple - seqlen
    pre_pad_offset = (0,) * (-1 - dim) * 2
    offset = (padding, 0) if pad_left else (0, padding)
    padded_tensor = F.pad(tensor, (*pre_pad_offset, *offset), value=0)
    return padded_tensor

class Autopadder(nn.Module):
    def __init__(self, net, pad_left=False):
        super().__init__()
        assert isinstance(net, (SinkhornTransformer, SinkhornTransformerLM)), 'only modules SinkhornTransformer and SinkhornTransformerLM accepted'
        self.net = net

        self.bucket_size = net.bucket_size
        self.pad_dim = -1 if isinstance(net, SinkhornTransformerLM) else -2
        self.pad_left = pad_left

    def forward(self, x, **kwargs):
        b, t, device = *x.shape[:2], x.device

        input_mask = kwargs.get('input_mask')

        if input_mask is None:
            input_mask = torch.full_like(x, True, device=x.device, dtype=torch.bool)

        x = pad_to_multiple(x, t, self.bucket_size, dim=self.pad_dim, pad_left=self.pad_left)
        padding = x.shape[1] - t

        if input_mask is not None:
            offset = (0, padding) if not self.pad_left else (padding, 0)
            new_mask = F.pad(input_mask, offset, value=False)
            kwargs.update(input_mask=new_mask)

        out = self.net(x, **kwargs)

        output_slice = slice(0, t) if not self.pad_left else slice(padding, None)
        return out[:, output_slice]
