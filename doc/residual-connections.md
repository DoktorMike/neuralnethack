# Residual (skip) connections

Each layer can optionally take a residual input from an earlier layer. The
skip source's output is added element-wise into the target layer's
pre-activation, before the activation function:

```
z = W · y_prev + b + y_skip       // skip added before activation
y = act(z)
```

Pre-activation rather than post-activation, because the existing
activation-derivative formulas all express f'(z) in terms of f(z). Putting the
skip in pre-activation means that bookkeeping keeps working without any extra
plumbing.

Two hard constraints:

- **Source must come earlier in the chain.** A layer can only skip from a layer with a smaller index. `skipFrom()` aborts otherwise.
- **Source and target must have the same width.** The merge is element-wise, so the shapes have to line up.

## Layer indexing

This is the part that trips people up. Indices count up from the first hidden
layer. The input vector is *not* a layer. So for an architecture
`[n_in, n_h1, n_h2, n_h3, n_out]`:

```
arch:    [n_in,    n_h1,    n_h2,    n_h3,    n_out]
                    ^        ^        ^        ^
                  layer 0  layer 1  layer 2  layer 3 (output)
```

Which means in `arch = [2, 4, 4, 1]` (input plus two width-4 hidden plus
width-1 output), layers 0 and 1 are both width 4 and can be wired together
with a skip.

## From C++

```cpp
std::vector<uint> arch = {2, 4, 4, 1};
std::vector<std::string> types = {"tansig", "tansig", "logsig"};
Mlp mlp(arch, types, false);

// Layer 1's pre-activation gets layer 0's output added in.
mlp.skipFrom(/*target=*/1, /*source=*/0);
```

Pass `-1` as the source to clear an existing skip on a given target.

## From a TOML config

Under `[network]`, add `skip_connections` as an array of `[target, source]`
pairs:

```toml
[network]
size = [2, 4, 4, 1]
activations = ["tansig", "tansig", "logsig"]
error_fcn = "kullback"
skip_connections = [[1, 0]]   # layer 1 receives skip from layer 0
```

One skip source per target layer (later entries for the same target overwrite
earlier ones). Multiple targets are free to share the same source.

For a full worked example with an ensemble of residual MLPs, see
`examples/xor_residual_ensemble.cc` ([examples.md](examples.md)).
