# NeuralNetHack Architecture

This document describes the current architecture of NeuralNetHack: a C++23
library for training and evaluating weighted ensembles of multi-layer
perceptrons. Diagrams are [Mermaid](https://mermaid.js.org/) and render
directly on GitHub.

It replaces the old Umbrello/Visual-Paradigm models (`*.xmi`, `nethack.vpp`),
which predated the `std::variant` activation refactor, the Sampler hierarchy,
conformal prediction, and the Factory-based orchestration.

## Contents

- [System overview](#system-overview)
- [mlp — the MLP engine](#mlp--the-mlp-engine)
- [datatools — data handling](#datatools--data-handling)
- [evaltools — evaluation](#evaltools--evaluation)
- [Orchestration — ensembles, estimation, selection](#orchestration--ensembles-estimation-selection)
- [Training data flow](#training-data-flow)
- [CLI binaries](#cli-binaries)

## Legend

Across the class diagrams:

| Arrow | Meaning |
|---|---|
| `A <\|-- B` | B inherits from A |
| `A *-- B` | A owns B by value / `unique_ptr` (composition) |
| `A o-- B` | A shares B via `shared_ptr` (aggregation) |
| `A ..> B` | A references B without owning (raw pointer / ref / uses) |

---

## System overview

Five subsystems sit under a thin orchestration layer. Arrows point in the
direction of the dependency (caller to callee).

```mermaid
flowchart TD
    subgraph orchestration["orchestration (top level)"]
        Factory
        Config
        EnsembleBuilder
        ModelEstimator
        ModelSelector
        FeatureSelector
        Ensemble
        Saliency
        OddsRatio
    end

    subgraph mlp["mlp"]
        Mlp
        Layer
        Activation
        Trainer
        Error
        Serialization
    end

    subgraph datatools["datatools"]
        DataSet
        CoreDataSet
        Pattern
        Sampler
        Normaliser
    end

    subgraph evaltools["evaltools"]
        Roc
        Gof
        Conformal
        ConfusionMatrix
        EvalTools
    end

    subgraph parser["parser"]
        Parser
        TomlParser
        NetworkParser
    end

    matrixtools["matrixtools (BLAS-backed vec/mat ops)"]

    Config --> Factory
    Factory --> mlp
    Factory --> datatools
    EnsembleBuilder --> mlp
    EnsembleBuilder --> datatools
    EnsembleBuilder --> Ensemble
    ModelEstimator --> EnsembleBuilder
    ModelSelector --> ModelEstimator
    FeatureSelector --> Ensemble
    Ensemble --> Mlp
    evaltools --> Ensemble
    evaltools --> datatools
    Saliency --> Ensemble
    OddsRatio --> Ensemble
    parser --> Config
    parser --> Ensemble
    mlp --> matrixtools
    evaltools --> matrixtools
```

---

## mlp — the MLP engine

The core engine. An `Mlp` is a sequence of `Layer`s held **by value**; each
`Layer` carries its activation as a `std::variant` tag (no class hierarchy,
no virtual dispatch on the hot path). `Trainer` and `Error` are the only
polymorphic hierarchies here.

```mermaid
classDiagram
    class Mlp {
        -vector~uint~ theArch
        -vector~string~ theTypes
        -bool theSoftmax
        -vector~Layer~ theLayers
        -vector~int~ theSkipFrom
        +propagate(vector~double~) vector~double~
        +propagateBatch(double*, uint B) double*
        +weights() vector~double~
        +training(bool)
        +skipFrom(uint target, int source)
    }
    class Layer {
        -uint ncurr
        -uint nprev
        -Activation theAct
        -vector~double~ theWeights
        -vector~double~ theOutputs
        -vector~double~ theGradients
        -NormType theNormType
        -vector~double~ theGamma
        -vector~double~ theBeta
        +propagate(vector~double~, double* preactSkip) vector~double~
        +propagateBatch(double*, uint B, ...) double*
        +applyDerivative(vector~double~)
        +activation() Activation
    }
    class Activation {
        <<variant>>
        Sigmoid | TanH | Linear
        ReLU | LeakyReLU | ELU
    }
    note for Activation "Free functions dispatch via std::visit:\nfire / firePrime / firePrimePrime,\napplyActivation / applyDerivScale,\nactivationFromTag / activationToTag"

    class Trainer {
        <<abstract>>
        #Mlp* theMlp
        #DataSet* theData
        #Error* theError
        #unique_ptr~Error~ theOwnedError
        #uint theNumEpochs
        #uint theBatchSize
        #early-stopping state
        +train(ostream) *
        +clone() unique_ptr~Trainer~ *
        +trainNew(DataSet, ostream) unique_ptr~Mlp~
        +earlyStopping(uint patience, double minDelta)
    }
    class GradientDescent {
        -double theLearningRate
        -double theDecLearningRate
        -double theMomentum
    }
    class Adam {
        -double theLearningRate
        -double theBeta1
        -double theBeta2
        -vector~double~ theM
        -vector~double~ theV
    }
    class QuasiNewton {
        -vector~vector~double~~ sHistory
        -vector~vector~double~~ yHistory
        -LBFGS_M = 20
    }

    class Error {
        <<abstract>>
        #Mlp* theMlp
        #unique_ptr~Mlp~ theOwnedMlp
        #DataSet* theDset
        #bool theWeightElimOn
        +gradient(Mlp, DataSet) double *
        +outputError(Mlp, DataSet) double *
    }
    class CrossEntropy
    class SummedSquare

    class Weights {
        -vector~double~ theWeights
        -vector~uint~ arch
    }
    class Serialization {
        <<free functions>>
        +saveMlpBinary(Mlp, path)
        +loadMlpBinary(path) unique_ptr~Mlp~
        +saveEnsembleBinary(Ensemble, path)
        +loadEnsembleBinary(path) unique_ptr~Ensemble~
    }

    Mlp *-- Layer
    Layer *-- Activation
    Trainer <|-- GradientDescent
    Trainer <|-- Adam
    Trainer <|-- QuasiNewton
    Error <|-- CrossEntropy
    Error <|-- SummedSquare
    Trainer ..> Mlp
    Trainer ..> Error
    Error ..> Mlp
    Serialization ..> Mlp
```

Notes:

- **Activation devirtualized.** `Layer` holds one `Activation` variant by
  value; both scalar and batch paths dispatch through `std::visit`, letting
  the compiler inline the per-element kernel. Parameterized activations carry
  their own params (`LeakyReLU::alpha = 0.01`, `ELU::alpha = 1.0`).
- **Layers do more than activation.** Each `Layer` also holds optional batch
  / layer normalization state (`NormType`, gamma/beta + running stats),
  inverted-dropout buffers, and weight-init scheme. Skip connections are
  recorded per target layer in `Mlp::theSkipFrom` (-1 = none).
- **Ownership is flexible at the Trainer/Error seam.** Both default to raw
  non-owning pointers to their collaborators, but each also has an *owning*
  constructor (`Trainer` can own its `Error`; `Error` can own its `Mlp`)
  used when a trainer must outlive the caller's stack frame.
- **`trainNew()` returns `unique_ptr<Mlp>`** — the trained net is freshly
  owned by the caller (typically `EnsembleBuilder`).

---

## datatools — data handling

Pattern storage is shared and copy-free: many `DataSet` index-views point
into one `CoreDataSet` via `shared_ptr`. `Sampler` is the resampling
hierarchy that drives ensemble training and model estimation.

```mermaid
classDiagram
    class Sampler {
        <<abstract>>
        #unique_ptr~DataManager~ theDataManager
        #DataSet* theData
        #unique_ptr~vector~DataSet~~ theSplits
        +next() pair~DataSet,DataSet~ *
        +hasNext() bool *
        +howMany() uint *
        +reset() *
        +data(DataSet*)
    }
    class BootstrapSampler {
        -uint n
        -uint index
    }
    class CrossSplitSampler {
        -uint n
        -uint k
        -uint runCntr
    }
    class HoldOutSampler {
        -double ratio
        -uint n
    }
    class DummySampler {
        -uint n
    }

    class DataSet {
        -vector~uint~ theIndices
        -shared_ptr~CoreDataSet~ theCoreDataSet
        +pattern(uint) Pattern
        +nInput() uint
        +nOutput() uint
        +size() uint
    }
    class CoreDataSet {
        -vector~Pattern~ patterns
        +addPattern(Pattern)
        +pattern(uint) Pattern
        +size() uint
    }
    class Pattern {
        -string id
        -vector~double~ in
        -vector~double~ out
        +input() vector~double~
        +output() vector~double~
    }
    class DataManager {
        -vector~uint~ indices
        -bool isRandom
        +split(DataSet, double) pair~DataSet,DataSet~
        +split(DataSet, uint) vector~DataSet~
        +split(DataSet) pair~DataSet,DataSet~
        +join(vector~DataSet~) DataSet
    }
    class Normaliser {
        -vector~double~ theMean
        -vector~double~ theStdDev
        -vector~bool~ theSkip
        +calcAndNormalise(DataSet, bool) DataSet
        +normalise(Pattern) Pattern
        +unnormalise(DataSet) DataSet
    }

    Sampler <|-- BootstrapSampler
    Sampler <|-- CrossSplitSampler
    Sampler <|-- HoldOutSampler
    Sampler <|-- DummySampler
    Sampler *-- DataManager
    Sampler ..> DataSet
    DataSet o-- CoreDataSet
    CoreDataSet *-- Pattern
    DataManager ..> DataSet
    Normaliser ..> DataSet
    Normaliser ..> Pattern
```

Notes:

- **`DataSet` is a lightweight view.** It owns only an index vector plus a
  `shared_ptr<CoreDataSet>`. Splits produced by `DataManager` are new views
  over the same underlying patterns — no pattern copying.
- **`Sampler::next()`** yields a `(train, test/validation)` pair per call;
  `hasNext()`/`howMany()`/`reset()` drive the resampling loop. The concrete
  type fixes the resampling scheme (bootstrap, k-fold cross-split, hold-out,
  or dummy pass-through).
- **`Normaliser`** caches per-feature mean/stddev and a skip mask (binary
  features are left untouched), and normalizes in place bidirectionally.

---

## evaltools — evaluation

Mostly stateless calculators plus a metrics namespace. Conformal prediction
and the confusion-matrix metrics are the recent additions absent from the old
design model.

```mermaid
classDiagram
    class Roc {
        -vector~pair~double,double~~ theRoc
        -double theAuc
        -unique_ptr~Evaluator~ theEval
        +calcRoc(vector~double~, vector~uint~)
        +calcAucWmwFast(...) double
        +auc() double
    }
    class Evaluator {
        -double theCut
        -uint nTp
        -uint nTn
        +evaluate(vector~double~, vector~uint~)
        +tpf() double
        +fpf() double
    }
    class Gof {
        -uint numBins
        +goodnessOfFit(vector~double~, vector~uint~) double
    }
    class Conformal {
        -Mode theMode
        -double theAlpha
        -vector~double~ theQ
        +calibrate(Ensemble, DataSet)
        +interval(Ensemble, vector~double~) vector~Interval~
        +set(Ensemble, vector~double~) vector~uint~
        +coverage(Ensemble, DataSet) double
    }
    class ConfusionMatrix {
        -uint n
        -vector~vector~uint~~ m
        +add(uint actual, uint predicted)
        +fromEnsemble(Ensemble, DataSet, double cut)$ ConfusionMatrix
        +fromBinary(...)$ ConfusionMatrix
    }
    class EvalTools {
        <<namespace ErrorMeasures>>
        +crossEntropy(Ensemble, DataSet) double
        +auc(Ensemble, DataSet) double
        +accuracy / precision / recall / f1 / mcc
        +mae / mape / smape / rmse / r2
    }

    Roc *-- Evaluator
    Conformal ..> Ensemble
    Conformal ..> DataSet
    ConfusionMatrix ..> Ensemble
    EvalTools ..> Ensemble
    EvalTools ..> DataSet
    EvalTools ..> ConfusionMatrix
```

Notes:

- **`Roc`** computes the ROC curve and AUC three ways
  (Wilcoxon-Mann-Whitney, a fast WMW variant, and trapezoidal); it owns an
  `Evaluator` for the threshold sweep.
- **`Conformal`** does split-conformal prediction: per-dimension residual
  quantiles for regression intervals, LAC scores for classification sets,
  plus an empirical `coverage()` check. It consumes an `Ensemble` + a
  calibration `DataSet`, owning neither.
- **`EvalTools::ErrorMeasures`** is the one-stop metrics namespace, with
  overloads taking either raw vectors or an `(Ensemble, DataSet)` pair, plus
  the confusion-matrix-derived classification metrics.

---

## Orchestration — ensembles, estimation, selection

`Config` is reflected by `Factory` into the polymorphic object graph. An
`EnsembleBuilder` drives a `Sampler` + `Trainer` to populate an `Ensemble`;
`ModelEstimator` wraps that with cross-validation/bootstrap error estimation;
`ModelSelector` grid-searches over `Config`. Results travel as `Session`s.

```mermaid
classDiagram
    class Config {
        <<all params>>
        data paths / column selection
        arch + activations + softmax
        weight-elim + skip connections
        optimizer params (gd / adam)
        early stopping
        ensemble / selection params
        theVary : map~string,vector~double~~
    }
    class Factory {
        <<free functions>>
        +createMlp(Config) unique_ptr~Mlp~
        +createError(Config, DataSet) unique_ptr~Error~
        +createTrainer(Config, DataSet) unique_ptr~Trainer~
        +createSampler(Config, DataSet) unique_ptr~Sampler~
        +createEnsembleBuilder(Config, DataSet) unique_ptr~EnsembleBuilder~
        +createModelEstimator(Config, DataSet) unique_ptr~ModelEstimator~
    }
    class Ensemble {
        -vector~unique_ptr~Mlp~~ theEnsemble
        -vector~double~ theScales
        +addMlp(unique_ptr~Mlp~, double)
        +propagate(vector~double~) vector~double~
        +size() uint
    }
    class Session {
        +unique_ptr~Ensemble~ ensemble
        +unique_ptr~DataSet~ trnData
        +unique_ptr~DataSet~ valData
    }
    class EnsembleBuilder {
        <<abstract>>
        -unique_ptr~Trainer~ theTrainer
        -unique_ptr~Sampler~ theSampler
        -TrainerFactory theTrainerFactory
        -uint64 theBaseSeed
        -vector~Session~ theSessions
        +buildEnsemble() Ensemble*
        +sessions() vector~Session~
    }
    class ModelEstimator {
        <<abstract>>
        -unique_ptr~EnsembleBuilder~ theEnsembleBuilder
        -unique_ptr~Sampler~ theSampler
        -vector~Session~ theSessions
        +runAndEstimateModel(errorFunc) pair~double,double~
    }
    class ModelSelector {
        +findBestModel(DataSet, Config) pair~Config,double~
        +Auc632PlusRule(double, double, double) double
    }
    class FeatureSelector {
        -uint minFeatures
        -uint maxFeatures
        -Config best
        +run(Config, errorFunc) Config
    }

    Config ..> Factory
    Factory ..> Mlp
    Factory ..> Trainer
    Factory ..> Error
    Factory ..> Sampler
    Factory ..> EnsembleBuilder
    Factory ..> ModelEstimator
    Ensemble *-- Mlp
    Session *-- Ensemble
    Session *-- DataSet
    EnsembleBuilder *-- Trainer
    EnsembleBuilder *-- Sampler
    EnsembleBuilder *-- Session
    EnsembleBuilder ..> Ensemble
    ModelEstimator *-- EnsembleBuilder
    ModelEstimator *-- Sampler
    ModelSelector ..> ModelEstimator
    ModelSelector ..> Config
    FeatureSelector ..> Ensemble
    FeatureSelector ..> Config
```

Notes:

- **`Ensemble`** owns its members as `vector<unique_ptr<Mlp>>` with a parallel
  `theScales` vector; `propagate()` returns the scale-weighted average of
  member outputs.
- **`Session`** bundles a built `Ensemble` with the train/validation `DataSet`
  views it was produced from; it is the unit of result passed up the stack and
  out to `PrintUtils`.
- **`EnsembleBuilder`** can train members in parallel: `theTrainerFactory`
  hands each worker its own `Trainer`, and `nnh::rand` (thread-local
  `mt19937_64`, seeded from `theBaseSeed`) gives each worker an independent
  RNG stream.
- **`Saliency`** and **`OddsRatio`** (free-function namespaces, not shown as
  classes) compute input-sensitivity and binary-classification feature impact
  over an `Ensemble`.

---

## Training data flow

End-to-end path from a config file to a saved, evaluated ensemble.

```mermaid
sequenceDiagram
    participant CLI as CLI (src/*.cc)
    participant P as TomlParser / Parser
    participant Cfg as Config
    participant F as Factory
    participant EB as EnsembleBuilder
    participant S as Sampler
    participant T as Trainer
    participant E as Ensemble
    participant Ev as evaltools

    CLI->>P: read config + data file
    P->>Cfg: populate Config
    P->>CLI: CoreDataSet (wrapped in DataSet)
    CLI->>F: createEnsembleBuilder(Config, DataSet)
    F->>EB: build with Sampler + Trainer
    loop per resample
        EB->>S: next()
        S-->>EB: (train, val) DataSet views
        EB->>T: trainNew(train)
        T-->>EB: unique_ptr<Mlp>
        EB->>E: addMlp(mlp, scale)
    end
    EB-->>CLI: Ensemble (inside Session)
    CLI->>Ev: AUC / accuracy / ROC / conformal
    Ev-->>CLI: metrics
    CLI->>CLI: Serialization.saveEnsembleBinary()
```

For model selection the same flow nests one level deeper: `ModelSelector`
sweeps `Config.theVary`, and for each variant a `ModelEstimator` runs the
resampling loop above to produce a cross-validated error estimate, scored with
the `.632+` rule.

---

## CLI binaries

`src/` builds the user-facing executables on top of the library:

| Binary | Source | Role |
|---|---|---|
| `neuralnethack` | `neuralnethack.cc` | Main driver: train + estimate from a config file |
| `ann` | `ann.cc` | Train/apply a single network |
| `modelselector` | `modelselector.cc` | Grid search over hyperparameters |
| `featureselector` | `featureselector.cc` | Backward-elimination feature selection |
| `featureselector2` | `featureselector2.cc` | Variant feature-selection driver |
| `auc` | `auc.cc` | Compute ROC/AUC for predictions |
| `saliency` | `saliency.cc` | Input-sensitivity analysis |

---

## Maintaining this document

These diagrams are hand-maintained Mermaid, kept in sync with the headers.
When you change ownership, inheritance, or add a class, update the relevant
diagram here. There is no generator and no binary model file to regenerate.
