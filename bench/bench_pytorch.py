# PyTorch benchmark, same protocol and CSV format as bench_nnh:
#   pima:    8-32-1 (tanh + sigmoid), MSE, Adam lr=0.01
#   covtype: 54-128-7 (tanh + softmax), cross-entropy, Adam lr=0.01
# Reports eager and torch.compile as separate lib rows. Compile/warmup
# time is excluded from train_s (one-time cost; noted in README).
#
# Run via uv with the CPU wheel index (see run.sh) or any env with torch.

import sys
import time

import torch
import torch.nn as nn


def load_pima(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = [float(v) for v in line.split()]
            X.append(row[:8])
            y.append(int(row[8]))
    return torch.tensor(X, dtype=torch.float64), torch.tensor(y)


def load_covtype(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = [float(v) for v in line.split(",")]
            X.append(row[:54])
            y.append(int(row[54]) - 1)  # 1..7 -> 0..6
    return torch.tensor(X, dtype=torch.float64), torch.tensor(y)


def z_normalise(trn_X, tst_X):
    mean = trn_X.mean(dim=0)
    sd = trn_X.std(dim=0, unbiased=False)
    keep = sd > 1e-12
    for X in (trn_X, tst_X):
        X[:, keep] = (X[:, keep] - mean[keep]) / sd[keep]


def make_model(dataset):
    if dataset == "pima":
        return nn.Sequential(
            nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid()
        ).double()
    return nn.Sequential(nn.Linear(54, 128), nn.Tanh(), nn.Linear(128, 7)).double()


def emit(lib, dataset, arch, epochs, batch, threads, blas, trial, train_s, infer_us, acc):
    print(
        f"{lib},{dataset},{arch},{epochs},{batch},{threads},{blas},{trial},"
        f"{train_s:.4f},{infer_us:.3f},{acc:.4f}",
        flush=True,
    )


def run(lib, dataset, root, epochs, batch, trials, compiled):
    if dataset == "pima":
        trn_X, trn_y = load_pima(f"{root}/pima.trn.tab")
        tst_X, tst_y = load_pima(f"{root}/pima.tst.tab")
        arch = "8-32-1"
        loss_fn = nn.MSELoss()
    else:
        trn_X, trn_y = load_covtype(f"{root}/covtype.trn.csv")
        tst_X, tst_y = load_covtype(f"{root}/covtype.tst.csv")
        arch = "54-128-7"
        loss_fn = nn.CrossEntropyLoss()
    z_normalise(trn_X, tst_X)
    trn_t = trn_y.double().unsqueeze(1) if dataset == "pima" else trn_y

    n = trn_X.shape[0]
    threads = torch.get_num_threads()

    for t in range(trials):
        torch.manual_seed(42 + t)
        model = make_model(dataset)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        step = model
        if compiled:
            step = torch.compile(model)
            # warm up compile outside the timer
            loss_fn(step(trn_X[:batch]), trn_t[:batch]).backward()
            opt.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for b in range(0, n - batch + 1, batch):
                idx = perm[b : b + batch]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(step(trn_X[idx]), trn_t[idx])
                loss.backward()
                opt.step()
        train_s = time.perf_counter() - t0

        model.eval()
        infer = step if compiled else model
        reps = 20
        correct = 0
        rows = [tst_X[i : i + 1] for i in range(tst_X.shape[0])]
        with torch.no_grad():
            infer(rows[0])  # warm up batch-1 recompile outside the timer
            i0 = time.perf_counter()
            for r in range(reps):
                for i, x in enumerate(rows):
                    y = infer(x)
                    if r == 0:
                        if dataset == "pima":
                            pred = 1 if y.item() >= 0.5 else 0
                        else:
                            pred = int(y.argmax())
                        if pred == int(tst_y[i]):
                            correct += 1
            i1 = time.perf_counter()
        infer_us = 1e6 * (i1 - i0) / (reps * tst_X.shape[0])
        acc = correct / tst_X.shape[0]
        emit(lib, dataset, arch, epochs, batch, threads, "torch", t + 1, train_s, infer_us, acc)


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "pima"
    root = sys.argv[2] if len(sys.argv) > 2 else f"datasets/{dataset}"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    trials = int(sys.argv[5]) if len(sys.argv) > 5 else 10

    run("pytorch", dataset, root, epochs, batch, trials, compiled=False)
    run("pytorch-compiled", dataset, root, epochs, batch, trials, compiled=True)


if __name__ == "__main__":
    main()
