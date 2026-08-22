# Multi-class classification (softmax)

For K-way classification, use a linear output layer of width K and turn
softmax on. Pair it with the cross-entropy loss and the (target - output)
shortcut at the output layer gives you exactly the right gradient (no
derivative on softmax to apply explicitly, the math cancels).

## From C++

```cpp
std::vector<uint> arch = {4, 8, 3};                  // 4-feature input, 3 classes
std::vector<std::string> types = {"tansig", "purelin"};
Mlp mlp(arch, types, /*softmax=*/true);
```

## From a TOML config

```toml
[network]
size = [4, 8, 3]
activations = ["tansig", "purelin"]
softmax = true
error_fcn = "kullback"
```

Targets should be one-hot encoded (one column per class in the data file,
`out_cols = "6-8"` for example).

Worked examples in `examples/multiclass_iris.cc`, `examples/multiclass_wine.cc`,
and `examples/multiclass_synthetic.cc` ([examples.md](examples.md)).
