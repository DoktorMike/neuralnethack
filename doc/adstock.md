# Lag structures / adstock (marketing-mix models)

Time-series regression where today's input keeps affecting the response for
many periods (ad spend, promotions, pricing) usually gets modeled by feeding L
lags per channel into the network — L free weights per channel per neuron that
then need pruning. The `Adstock` stage collapses each channel's lag window
through a **normalized parametric carryover kernel** instead, with the kernel
parameters trained jointly with the weights by any of the optimizers:

```cpp
#include "mlp/Adstock.hh"

// Input per pattern: [c0 lag0..lag27, c1 lag0..lag27, c2 lag0..lag27, seasonality]
// arch[0] = channels + passthrough covariates = 4
Mlp mlp({4, 8, 1}, {"tansig", "purelin"}, false);
mlp.adstock(Adstock(/*channels=*/3, /*lags=*/28, /*passthrough=*/1,
                    Adstock::Kernel::Geometric)); // or Kernel::Weibull

// train exactly as usual -- Adam / SGD / L-BFGS all update the kernel params
SummedSquare loss(mlp, data);
Adam opt(mlp, data, loss, 0.0, 32, 0.01);
opt.train(std::cout);

auto w = mlp.adstock()->kernelWeights(0); // fitted carryover curve, channel 0
```

Kernels: **geometric** (1 param/channel, monotone decay) and **Weibull**
(2 params/channel, allows a delayed peak). 84 lagged inputs above cost 3
trained lag parameters instead of hundreds of free weights — nothing to prune,
and the fitted kernels are directly interpretable as carryover curves. The MLP
behind the stage learns saturation and channel interactions. Gradients flow
through one extra GEMM; the stage serializes with the model (`NNH2` format,
old `NNH1` files still load). Worked example with recovered kernels and
holdout R²: `examples/mmm_adstock.cc`.

## Boxed mode

**Boxed mode** scales this to many channels (say 100 channels, 156 weekly
observations — where 100 free lambdas would be hopeless): K shared kernel
"boxes" (short / medium / long carryover) plus a learned per-channel routing
`pi_c = softmax(logits_c / tau)` that mixes the boxes. One routing gates both
the carryover kernel and an optional per-box Hill saturation
`a^n / (a^n + s^n)` (trainable half-saturation and exponent; n > 1 learns an
S-shaped response), so 100 media insertion types collapse to K effective ones.
Each box's parameters pool ~C/K channels' worth of data — the pooling is the
prior:

```cpp
Mlp mlp({100 + P, 1}, {"purelin"}, false);
mlp.adstock(Adstock(/*channels=*/100, /*lags=*/28, /*passthrough=*/P,
                    Adstock::Kernel::Weibull, /*nBoxes=*/3,
                    Adstock::Saturation::Hill));

// train with the entropy penalty OFF until routing stabilizes...
opt.train(std::cout);
// ...then enable it to harden the assignments toward one-hot
mlp.adstock()->entropyPenalty(0.01);
opt.train(std::cout);

auto box = mlp.adstock()->boxAssignments(); // channel -> box
auto pi  = mlp.adstock()->routingProbs(7);  // soft routing, channel 7
```

Do not enable the entropy penalty from the first epoch — it hardens the
routing before the boxes separate and locks in chance-level assignments; warm
up with it off (measured: 12/12 channels routed correctly with warmup, chance
without). `summarizeBoxedAdstock` extends the ensemble summary to boxed stages
with label-switching-safe box bands plus **assignment stability** — "channel 7
routed long in 9/10 members". Full worked example: `examples/mmm_boxed.cc`
(50 insertion types = 10 media × 5 messages, 156 weekly obs, seasonality +
unemployment + trend; recovers the delayed peak and the S-shaped Hill
exponent, and shows where 156 rows stop identifying middle carryover regimes —
the stability readout flags exactly those channels). The example's data
process also ships as ready-made files with ground-truth tables in
[`datasets/mmm/`](../datasets/mmm/README.md). Design rationale and V2
(feature-based routing) in [spec-boxed-adstock.md](spec-boxed-adstock.md).

## Kernel-parameter uncertainty

Kernel-parameter uncertainty comes from the ensemble machinery: train members
on bootstrap resamples, then summarize the spread of fitted kernels across
members:

```cpp
#include "evaltools/Uncertainty.hh"

auto s = EvalTools::Uncertainty::summarizeAdstock(ensemble, /*alpha=*/0.1);
// s.paramMean / s.paramLower / s.paramUpper: 90% band per channel on the
// natural scale (geometric lambda, or Weibull shape and scale)
// s.weightMean / s.weightLower / s.weightUpper: per-lag bands on the
// carryover curve itself
```

## Choosing the window length

The kernel is truncated and renormalized over L lags, so carryover beyond the
window is invisible to the model. For geometric decay the truncated tail mass
is ≈ λ^L: λ = 0.8 loses 4.4% at L = 14 but 0.2% at L = 28; λ = 0.9 needs
L ≈ 44 for a 1% tail. Size L for the slowest decay you consider plausible
(L ≥ ln ε / ln λ_max) — the cost is only `channels × L` inputs per pattern and
O(L) per GEMM row, and an over-long window costs nothing statistically since
the kernel stays a 1-2 parameter family regardless of L. The fixed window is a
deliberate trade against recursive adstock (`a_t = x_t + λa_{t-1}`): infinite
memory, but stateful patterns would break bootstrap/cross-split resampling and
shuffled batches.

## Input scaling

Real spends live on 10^4-plus scales, which saturates Hill immediately and
wrecks gradient magnitudes. Use `normalization = "maxabs"`: inputs are
divided by max|x| with no centering, so zero spend stays exactly zero and
the a >= 0 Hill domain survives — Z-normalisation would break both. All lag
columns of one channel share a single scale (per-column maxima differ near
the series edges and would warp the kernel); with an `[adstock]` config
section the grouping is automatic via `Factory::adstockColumnGroups`. The
target is never scaled — rescaling the loss destabilises training
(measured: holdout R^2 0.90 vs junk on the mmm dataset). Interpretation:
a shared box half-saturation s then reads "saturates at fraction s of the
channel's max spend"; real-unit half-sat per channel = s x max spend_c.
