import torch
from sinkhorn_transformer.sinkhorn_transformer import SinkhornTransformerLM
from sinkhorn_transformer.autoregressive_wrapper import AutoregressiveWrapper

N_BATCH = 16
SRC_SEQ_LEN = 512
TGT_SEQ_LEN = 512

enc = SinkhornTransformerLM(
    num_tokens = 64,
    dim = 512,
    depth = 1,
    heads = 8,
    max_seq_len = SRC_SEQ_LEN,
    bucket_size = 64,
    return_embeddings = True
).cuda()

dec = SinkhornTransformerLM(
    num_tokens = 64,
    dim = 512,
    depth = 2,
    heads = 8,
    max_seq_len = TGT_SEQ_LEN,
    bucket_size = 64,
    causal = True,
    receives_context = True
).cuda()

dec = AutoregressiveWrapper(dec, ignore_index = 0, pad_value = 0)
opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=2e-4)

bos = 1 * torch.ones(N_BATCH, 1).long()
eos = 2 * torch.ones(N_BATCH, 1).long()
pos = 3 * torch.ones(N_BATCH, 1).long()

for i in range(10000):
    train_seq_in = torch.randint(4, 63, (N_BATCH, SRC_SEQ_LEN-2)).long()
    train_seq_out = train_seq_in

    x = torch.cat([bos, train_seq_in, eos], dim=1).cuda()
    y = torch.cat([bos, train_seq_out, eos, eos], dim=1).cuda()

    context = enc(x)
    loss = dec(y, context = context, return_loss = True)
    loss.backward()

    opt.step()
    opt.zero_grad()
    print(i, loss.item())