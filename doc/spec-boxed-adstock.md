# Spec: Boxed adstock — routed carryover and saturation

Status: ACCEPTED (2026-08-22) with decisions:
1. Hill exponent is a trainable per-box parameter, init 1 — n > 1 gives
   the S-shape; without it a plain tanh response would do as well.
2. No per-channel Hill mode.
3. Entropy penalty (not post-hoc snap-and-refit).
4. K-selection recipe documented in V1; a helper ships with V2.
Depends on: the shipped `mlp/Adstock` stage (geometric/Weibull kernels,
joint training, NNH2 serialization, ensemble summaries).

## Problem

A weekly MMM with C ≈ 100 media channels and T ≈ 156 observations cannot
support a carryover kernel (1-2 params) and a saturation curve per
channel: 100 lambdas / (k, s) pairs plus 100 half-saturations is
unidentifiable at this sample size. But media behave in a small number of
carryover regimes (short / medium / long). The model should learn K
regime "boxes" and learn which box each channel belongs to, jointly with
everything else — collapsing 100 insertion types into K effective ones.

## V1: free-logit routing (mixture of boxes)

### Model

Per channel c with routing weights `pi_c = softmax(logits_c / tau)` over
K boxes (K is an architectural choice, like layer width):

    kernel_c   = sum_k pi_ck * w(theta_k)          carryover, mixed kernel
    a_c        = sum_l kernel_c[l] * x_{c,t-l}     adstocked spend
    out_c      = sum_k pi_ck * hill(a_c; s_k)      gated saturation

- `theta_k`: box kernel params — geometric lambda_k, or Weibull (k_k,
  s_k). Family is shared across boxes, chosen at construction (as today).
- `hill(a; s, n) = a^n / (a^n + s^n)` with half-saturation
  `s_k = exp(sigma_k)` and exponent `n_k = exp(nu_k)` per box, `nu`
  initialized to 0 (n = 1, plain diminishing returns); n > 1 yields the
  S-shaped response. Saturation gating is optional (`Saturation::None`
  keeps today's behavior of letting the MLP learn it).
- One routing `pi_c` gates BOTH carryover and saturation: a channel is
  one insertion type, so it gets one regime end to end.
- The channel's effect scale (beta) stays in the dense layer behind the
  stage, as today.

### Parameter count and why it identifies

- Box params: K kernel params (geometric) or 2K (Weibull), + K
  saturations. K=3 Weibull + Hill: 9.
- Routing logits: C x K free params (300 for C=100, K=3). Each channel
  only chooses among K options (~log2 K bits), not a continuous lambda;
  with the entropy penalty below, the effective content is a categorical
  assignment. Each box kernel pools ~C/K channels' worth of data. The
  pooling IS the prior.

### Regularization

- Temperature `tau` (default 1.0, settable): anneal down during training
  to harden assignments. V1 exposes the knob; scheduling is the caller's
  loop.
- Entropy penalty `beta_H * sum_c H(pi_c)` added to the loss, pushing
  routing toward one-hot. Off by default. Implemented alongside the
  existing weight-elimination hook in `Error`.
  IMPLEMENTATION FINDING: the penalty must NOT be on from step one — it
  creates positive feedback toward the nearest vertex and hardens the
  routing before the boxes separate (measured: 12/12 channels routed
  correctly with a warmup, chance-level without one). Correct schedule:
  train with beta = 0 until routing stabilizes, then enable beta to
  harden (measured 0.82 -> 1.00 max pi without losing a single
  assignment). Documented in the class docs and exercised by the test.
- The dense head behind the stage is the real overfitting risk at
  T=156, C=100 — recommend linear head or tiny H plus weight
  elimination. The spec does not change the head; document the
  recommendation.

### Gradients (all closed form, no BPTT)

Same chaining as today (`Error::chainAdstock`, one extra GEMM gives
`dE/d out_c` per pattern). Then:

- d out_c / d sigma_k    = pi_ck * dhill/ds (a_c; s_k, n_k) * s_k
- d out_c / d nu_k       = pi_ck * dhill/dn (a_c; s_k, n_k) * n_k
- d out_c / d theta_k    = [sum_j pi_cj * dhill/da(a_c; s_j)] *
                           pi_ck * sum_l dw/dtheta_k[l] * x_{c,t-l}
- d out_c / d logit_cj   = softmax Jacobian applied to the vector of
                           per-box contributions (both the kernel mix and
                           the saturation mix terms).

The `computeKernels` cache grows box kernels `w(theta_k)` and their
derivatives once per call — cheaper than today's per-channel kernels
when C >> K.

### API sketch

    Adstock(uint channels, uint lags, uint passthrough,
            Kernel family,            // Geometric | Weibull, as today
            uint nBoxes,              // K >= 1; K == 0/absent = current per-channel mode
            Saturation sat = Saturation::None);  // None | Hill

    // routing + reporting
    std::vector<double> routingProbs(uint c) const;   // pi_c, length K
    std::vector<uint>   boxAssignments() const;       // argmax per channel
    std::vector<double> boxKernel(uint k) const;      // w(theta_k), length L
    double              boxSaturation(uint k) const;  // s_k (natural scale)
    void temperature(double tau);

Params layout (flat, channel-major after box block):
`[theta_1..theta_K | sigma_1..sigma_K (if Hill) | logits_11..logits_CK]`.
Everything rides `Mlp::weights()/gradients()` as today, so Adam / SGD /
L-BFGS keep working unchanged.

### Serialization

NNH3 block: family, C, L, P, K, saturation flag, then params in the flat
weight vector as today. NNH1/NNH2 continue to load.

### Ensemble uncertainty and label switching

`summarizeAdstock` gains the boxed case. Boxes are canonicalized per
member by sorting on mean carryover lag (sum_l l * w_k[l]) before
banding, otherwise box permutation across members corrupts the intervals.
New reporting: assignment stability — the fraction of members routing
channel c to its modal box. "Channel 7 routed long in 9/10 members" is
the client-facing statement.

### Tests

1. Finite-difference gradient check through the full stage (both
   families, Hill on and off, K=3) — tolerance 1e-6 as today.
2. Recovery: C=12 channels generated from 3 known boxes (lambda 0.2 /
   0.5 / 0.8), assert >= 10/12 channels routed correctly and box lambdas
   within 0.05.
3. Saturation recovery: known per-box half-saturations, assert ordering
   and rough magnitude.
4. Entropy penalty drives mean max(pi_c) above 0.9 on the recovery
   problem.
5. Ensemble: label-switching canonicalization (permute one member's
   boxes manually, assert identical summary), assignment stability
   output.
6. NNH3 round-trip.

### K selection (recipe, V1)

Fit K = 2..5. Compare (a) holdout error, (b) assignment stability across
ensemble members, (c) whether any box receives < ~C/(3K) channels
(collapsed box = K too large). Prefer the smallest K whose stability
holds; a helper automating this ships with V2.

### Out of scope for V1

- Routing from channel features (V2).
- Per-box kernel families (all boxes share one family).
- Learned temperature schedule.

## V2: amortized routing (feature-based, attention-ready)

Replace free logits with a routing function over per-channel features:

    logits_c = r(f_c)

- `f_c`: channel covariates — insertion type one-hot, spend statistics
  (mean, CV, burstiness, autocorrelation), price tier. Cheap to compute
  from the spend series plus metadata.
- `r`: small shared net. Two stages:
  - V2a: linear / one-hidden-layer MLP on `f_c`. Likely sufficient.
  - V2b: attention over the channel set — queries from `f_c`, keys/values
    from box embeddings, so channels inform each other's routing
    (e.g. "route like the other TV-shaped channels"). Only worth it if
    V2a routing is demonstrably noisy and channels share structure the
    features miss.

What it buys: routing generalizes to channels unseen in training
(new campaign types get a box from their features, not from 156 obs of
their own), and C x K free logits shrink to the routing net's params,
shared across channels.

Cost: feature pipeline in datatools (per-channel covariate matrix
alongside the lag window), routing-net gradients (plain backprop through
`r`), and the identifiability questions move into `r`. V2 reuses V1's
box/gating math unchanged — only the source of the logits changes, so
V1 is not throwaway.

## Resolved questions

1. Hill exponent: trainable per box, init 1. (S-shape needs n > 1;
   fixed n = 1 would make plain tanh just as good a response.)
2. Per-channel Hill without boxing: no.
3. Entropy penalty inside Error (weight-elim precedent); snap-and-refit
   kept as fallback idea only.
4. K selection: recipe documented in V1 (above); helper in V2.
