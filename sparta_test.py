import sparta
import torch

M, N, K = 1024, 1024, 1024
SPARSITY = 0.8
BLOCK = (8, 8)
has_bias = True

A = torch.rand((M, K), dtype=torch.float32).cuda()
B = torch.rand((N, K), dtype=torch.float32).cuda()
bias = torch.rand((N,), dtype=torch.float32).cuda()
# generate and apply mask
B_mask = sparta.testing.block_mask(B.shape, block=BLOCK, sparsity=SPARSITY).cuda()
B = torch.mul(B, B_mask)
# dense operator
linear = torch.nn.Linear(K, N, bias=has_bias).cuda()
linear.load_state_dict(dict(weight=B, bias=bias) if has_bias else dict(weight=B))
# sparse operator
splinear = sparta.nn.SparseLinear(linear, weight_mask=B_mask)
best_cfg = sparta.nn.tune(splinear, sample_inputs=[A], max_trials=10, algo="rand")
torch.testing.assert_close(splinear(A), linear(A))
