# Configuration and CLI

Run the library without writing any C++: the `neuralnethack` binary takes a
single config file and does the whole thing — parses the data, normalises it,
trains an ensemble (with model selection if you ask for one), evaluates on the
test set, and writes everything to disk.

```sh
./build/neuralnethack config.toml
```

Working examples ship with the datasets:

```sh
cd datasets/pima   && ../../build/neuralnethack config-pima.toml
cd datasets/iris   && ../../build/neuralnethack config-iris.toml
cd datasets/wine   && ../../build/neuralnethack config-wine.toml
```

The other CLI tools (`ann`, `modelselector`, `featureselector`, `saliency`,
`auc`) all read the same config format. Pick the one that matches what you're
after.

## Output files

Every output file is suffixed with whatever you put in the `suffix` field, so
you can run a few experiments side by side without clobbering each other:

- `result.<suffix>.txt`: train/test AUC (binary) or accuracy (multi-class).
- `networks.<suffix>.xml`: the trained ensemble, ready to reload.
- `outputlist.<suffix>.txt`: per-pattern model outputs (toggle with `save_output_list`).
- `saliencies.<suffix>.txt`: input saliencies, handy for feature selection.
- `myconfig.debug`: the parsed config, so you can sanity-check what was actually used.
- `<curve>_NNN.dat` (when `output.learning_curve_file` is set): per-member learning curves, one row per epoch with `epoch  trainErr  valErr`. The validation error comes from each member's out-of-bag split.

## Config file format

Configs are TOML. Sections group related settings, named keys replace the old
positional tuples (no more counting arguments), and comments use `#`. A
minimal binary-classification config looks like this:

```toml
suffix = "myrun"
seed = 42
normalization = "Z"          # "Z" or "no"
problem_type = "class"       # "class" or "regr"

[data.train]
file = "data/train.tab"
id_col = 0                   # 0 = no id column
in_cols = "1-8"              # range string, 1-indexed
out_cols = "9"
row_range = "0"              # "0" = all rows

[data.test]
file = "data/test.tab"
id_col = 0
in_cols = "1-8"
out_cols = "9"
row_range = "0"

[network]
size = [8, 4, 1]
activations = ["relu", "logsig"]   # one per non-input layer
error_fcn = "kullback"             # "sumsqr" or "kullback"
softmax = false                    # true for multi-class with linear output
weight_init = "glorot"             # "glorot" (default) or "legacy_uniform".
                                   # glorot picks Xavier uniform for saturating
                                   # activations and He uniform for ReLU-family,
                                   # both scaled to fan-in / fan-out. Biases
                                   # initialise to zero. legacy_uniform is the
                                   # pre-4.1.0 U(-0.5, 0.5) draw, kept for
                                   # back-compat with serialised models.
# Optional residual connections: each entry is [target_layer, source_layer]
# (0-indexed, source < target, both layers must have matching width).
# skip_connections = [[2, 0]]

# Optional adstock (parametric lag kernel) input stage — see adstock.md.
# Raw data rows carry channels*lags + passthrough input columns; the stage
# feeds channels + passthrough values into the first layer, so
# network.size[0] must equal channels + passthrough (Factory validates).
# [adstock]
# channels = 50
# lags = 13
# passthrough = 3            # trailing covariate columns, passed through
# kernel = "weibull"         # "geometric" (1 param) or "weibull" (2 params)
# boxes = 5                  # 0 / omitted = per-channel mode
# saturation = "hill"        # "none" (default) or "hill"; boxed mode only
# temperature = 1.0          # routing softmax temperature
# entropy_penalty = 0.0      # KEEP 0 in configs: it applies from epoch one,
#                            # which hardens routing before the boxes separate.
#                            # Harden in a second phase via the API instead.
# nonnegative_betas = false  # constrain first-layer media columns >= 0

[training]
method = "adam"              # "gd", "adam", "qn"
max_epochs = 2000

[training.adam]
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.01

[training.early_stopping]
patience = 0                 # 0 disables (default). When > 0 the trainer stops
min_delta = 0.0              # if val loss has not improved by min_delta for
                             # `patience` recorded epochs, and the model weights
                             # are restored to the best-val snapshot.

[regularization.weight_elim]
enabled = false
alpha = 0.01
w0 = 1.0

[ensemble]
method = "bagg"              # "bagg", "cs"
runs = 5
parts = 2
split = "rnd"                # "rnd" or "ser"
vary_weights = false

[model_selection]
method = "cv"                # "cv", "boot", "hold", "none"
runs = 3
parts = 5
split = "rnd"
fraction = 0.2

[output]
save_session = true
save_output_list = true
# learning_curve_file = "curve.dat"   # optional, per-member files <stem>_NNN.<ext>
```

See `datasets/pima/config-pima.toml` for a fully commented version with every
field. Type strings (`relu`, `adam`, `kullback`, ...) are listed in
[architecture.md](architecture.md#type-strings).

## Migrating from the legacy format

Configs from version 2.x and earlier used a space-separated
`{Identifier} {Value} {Value} ...` format with `%` comments. There's a script
for that:

```sh
scripts/migrate-config.py old-config.txt -o new-config.toml
```

It handles the field rename, splits the positional tuples (`GDParam`,
`AdamParam`, `EnsParam`, `MSParam`, `WeightElim`, `Vary`) into named keys, and
drops the result into the right section. Eyeball the output before running it
for real, since the legacy format had a few oddities.
