# Graph Report - nnzig  (2026-06-19)

## Corpus Check
- 29 files · ~24,773 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 238 nodes · 285 edges · 28 communities (21 shown, 7 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2f412b6b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Eigen Wrappers|Eigen Wrappers]]
- [[_COMMUNITY_NN Lifecycle|NN Lifecycle]]
- [[_COMMUNITY_Eigen Linear Algebra|Eigen Linear Algebra]]
- [[_COMMUNITY_Normalization Core|Normalization Core]]
- [[_COMMUNITY_Parameter Config|Parameter Config]]
- [[_COMMUNITY_Binary IO|Binary I/O]]
- [[_COMMUNITY_PyTorch Benchmark|PyTorch Benchmark]]
- [[_COMMUNITY_MLP Layer|MLP Layer]]
- [[_COMMUNITY_Docs & CI Workflows|Docs & CI Workflows]]
- [[_COMMUNITY_Build System|Build System]]
- [[_COMMUNITY_Eigen Activations|Eigen Activations]]
- [[_COMMUNITY_Benchmark Plotting|Benchmark Plotting]]
- [[_COMMUNITY_nnzig Benchmark Runner|nnzig Benchmark Runner]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]

## God Nodes (most connected - your core abstractions)
1. `NN` - 14 edges
2. `NNzig` - 11 edges
3. `main()` - 9 edges
4. `f_type` - 9 edges
5. `MLP` - 8 edges
6. `NNzig` - 8 edges
7. `run()` - 7 edges
8. `load_config()` - 6 edges
9. `write_params_zon()` - 6 edges
10. `Norm` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Test Workflow` --conceptually_related_to--> `NNzig`  [INFERRED]
  .github/workflows/test.yml → README.md
- `Docs Workflow` --conceptually_related_to--> `NNzig`  [INFERRED]
  .github/workflows/docs.yml → README.md
- `set_batch_size_compute()` --calls--> `write_params_zon()`  [INFERRED]
  benchmarks/bench_resources.py → benchmarks/config.py
- `set_training_seed()` --calls--> `load_config()`  [INFERRED]
  benchmarks/multi_seed.py → benchmarks/config.py
- `set_training_seed()` --calls--> `write_params_zon()`  [INFERRED]
  benchmarks/multi_seed.py → benchmarks/config.py

## Import Cycles
- None detected.

## Communities (28 total, 7 thin omitted)

### Community 0 - "Eigen Wrappers"
Cohesion: 0.08
Nodes (38): activateNone(), activateRelu(), activateSigmoid(), activateTanh(), adamUpdate(), computeMeanStd(), computeMSE(), denormalize() (+30 more)

### Community 2 - "Eigen Linear Algebra"
Cohesion: 0.29
Nodes (10): eigen_adamUpdate(), eigen_divScalar(), eigen_matrixVectorMulAdd(), eigen_setZero(), eigen_updateGradBiases(), eigen_updateGradWeights(), eigen_vectorInit(), eigen_vectorMatrixMul() (+2 more)

### Community 4 - "Parameter Config"
Cohesion: 0.25
Nodes (6): activation, baseParams, convertStringToEnum(), convertTupleToEnumArray(), loss, norm

### Community 6 - "PyTorch Benchmark"
Cohesion: 0.27
Nodes (10): _check(), _format_zon_value(), _get(), Shared benchmark configuration helpers.  All benchmark parameters live in ``conf, Return the contents of params.zon (no comments) for the given config., Write benchmarks/params.zon from the config., Validate the config against the same constraints enforced at compile     time in, render_params_zon() (+2 more)

### Community 9 - "Docs & CI Workflows"
Cohesion: 0.15
Nodes (13): Docs Workflow, GitHub Pages, Benchmarks, Configuration, Documentation, Features, License, NNzig (+5 more)

### Community 10 - "Build System"
Cohesion: 0.60
Nodes (4): build(), createTree(), getOpenMPPaths(), Tree

### Community 12 - "Eigen Activations"
Cohesion: 0.53
Nodes (5): eigen_none(), eigen_relu(), eigen_sigmoid(), eigen_tanh(), f_type

### Community 14 - "nnzig Benchmark Runner"
Cohesion: 0.18
Nodes (19): ensure_app_realized(), main(), parse_args(), parse_time_output(), parse_wall_clock(), Return the store path of the app's wrapper script (its 'program').      Uses sep, GNU time can only wrap a path that already exists, so if the resolved     progra, Parse GNU time's 'm:ss' or 'h:mm:ss' into seconds (float). (+11 more)

### Community 18 - "Community 18"
Cohesion: 0.22
Nodes (8): Architecture, Benchmark setup, Commands, Configuration model — read this before editing params, graphify, NNzig, Toolchain, Zig 0.16 conventions used here

### Community 20 - "Community 20"
Cohesion: 0.60
Nodes (4): eigen_computeMeanStd(), eigen_denormalize(), eigen_normalize(), f_type

### Community 21 - "Community 21"
Cohesion: 0.14
Nodes (11): compute_loss(), init_normal(), make_step(), MLP, Replace every Linear weight and bias with an independent sample from     N(0, 1), # NOTE: optax.adam uses a global step counter (incremented every, Read the benchmark dataset written by generate_data.py.      The header's first, Write the training/validation loss curves.      The on-disk layout matches nnzig (+3 more)

### Community 22 - "Community 22"
Cohesion: 0.22
Nodes (6): MLP, # NOTE: torch.optim.Adam uses a global step counter (incremented every, Read the benchmark dataset written by generate_data.py.      The header's first, Write the training/validation loss curves.      The on-disk layout matches nnzig, read_numpy_binary(), write_losses_binary()

### Community 23 - "Community 23"
Cohesion: 0.28
Nodes (8): _draw_main(), _draw_ratio(), _mean_std(), Return (mean, sample-std ddof=1) of a list; std is 0.0 for n=1., Per-N (means, stds) arrays from a list of sample lists., Log-scale main panel: mean line + std band for every library., Linear ratio panel: (lib - ref) / ref * 100 for every lib in     _RATIO_LIBS (Py, _stats_arrays()

### Community 26 - "Community 26"
Cohesion: 0.22
Nodes (7): MLP, # NOTE: tf.keras.optimizers.Adam uses a global step counter (incremented, Read the benchmark dataset written by generate_data.py.      The header's first, Write the training/validation loss curves.      The on-disk layout matches nnzig, Multi-layer perceptron matching nnzig's mlp.zig layer stack.      A Dense layer, read_numpy_binary(), write_losses_binary()

### Community 27 - "Community 27"
Cohesion: 0.43
Nodes (6): main(), Read a loss-curve file ([precision_bytes, n_epochs, 1, 1] + train + val)., Point training.seed at `seed` and regenerate params.zon (nnzig reads it     at c, read_losses(), run(), set_training_seed()

## Knowledge Gaps
- **24 isolated node(s):** `Tree`, `activation`, `loss`, `norm`, `baseParams` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `nnzig Benchmark Runner` to `Community 27`, `PyTorch Benchmark`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Why does `set_training_seed()` connect `Community 27` to `nnzig Benchmark Runner`, `PyTorch Benchmark`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `NNzig` (e.g. with `Docs Workflow` and `Test Workflow`) actually correct?**
  _`NNzig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Run cmd, streaming the last few lines of combined output. Returns     the Comple`, `Return the store path of the app's wrapper script (its 'program').      Uses sep`, `GNU time can only wrap a path that already exists, so if the resolved     progra` to the rest of the system?**
  _55 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Eigen Wrappers` be split into smaller, more focused modules?**
  _Cohesion score 0.07692307692307693 - nodes in this community are weakly interconnected._
- **Should `Community 21` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._