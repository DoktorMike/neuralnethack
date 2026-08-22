# Width sweep, torch eager, same setup as sweep_nnh.cc.
# Prints: torch,IN,H,train_s_per_epoch,infer_us
import sys
import time

import torch
import torch.nn as nn

IN = int(sys.argv[1]) if len(sys.argv) > 1 else 64
H = int(sys.argv[2]) if len(sys.argv) > 2 else 128
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 64

torch.manual_seed(42)
X = 2 * torch.rand(N, IN, dtype=torch.float64) - 1
y = (X.sum(dim=1) > 0).double().unsqueeze(1)

model = nn.Sequential(nn.Linear(IN, H), nn.Tanh(), nn.Linear(H, 1), nn.Sigmoid()).double()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# one warm epoch outside timer
perm = torch.randperm(N)
for b in range(0, N - BATCH + 1, BATCH):
    idx = perm[b : b + BATCH]
    opt.zero_grad(set_to_none=True)
    loss_fn(model(X[idx]), y[idx]).backward()
    opt.step()

t0 = time.perf_counter()
for _ in range(EPOCHS):
    perm = torch.randperm(N)
    for b in range(0, N - BATCH + 1, BATCH):
        idx = perm[b : b + BATCH]
        opt.zero_grad(set_to_none=True)
        loss_fn(model(X[idx]), y[idx]).backward()
        opt.step()
per_epoch = (time.perf_counter() - t0) / EPOCHS

model.eval()
x0 = X[:1]
with torch.no_grad():
    model(x0)
    reps = 2000
    i0 = time.perf_counter()
    for _ in range(reps):
        model(x0)
    infer_us = 1e6 * (time.perf_counter() - i0) / reps

print(f"torch,{IN},{H},{per_epoch:.5f},{infer_us:.2f}", flush=True)
