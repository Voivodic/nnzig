# Graph Report - nnzig  (2026-06-15)

## Corpus Check
- 27 files · ~21,659 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 216 nodes · 261 edges · 25 communities (18 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5f3a3ddf`
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

## God Nodes (most connected - your core abstractions)
1. `NN` - 17 edges
2. `f_type` - 11 edges
3. `NNzig` - 11 edges
4. `main()` - 8 edges
5. `NNzig` - 8 edges
6. `run()` - 7 edges
7. `Norm` - 6 edges
8. `run_timed()` - 5 edges
9. `set_n_neurons()` - 5 edges
10. `load_config()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Test Workflow` --conceptually_related_to--> `NNzig`  [INFERRED]
  .github/workflows/test.yml → README.md
- `Docs Workflow` --conceptually_related_to--> `NNzig`  [INFERRED]
  .github/workflows/docs.yml → README.md
- `set_n_neurons()` --calls--> `load_config()`  [INFERRED]
  benchmarks/bench_resources.py → benchmarks/config.py
- `main()` --calls--> `load_config()`  [INFERRED]
  benchmarks/bench_resources.py → benchmarks/config.py
- `set_training_seed()` --calls--> `load_config()`  [INFERRED]
  benchmarks/multi_seed.py → benchmarks/config.py

## Import Cycles
- None detected.

## Communities (25 total, 7 thin omitted)

### Community 0 - "Eigen Wrappers"
Cohesion: 0.08
Nodes (38): activateNone(), activateRelu(), activateSigmoid(), activateTanh(), computeMeanStd(), computeMSE(), denormalize(), eigen_computeMeanStd() (+30 more)

### Community 2 - "Eigen Linear Algebra"
Cohesion: 0.28
Nodes (12): eigen_matrixVectorMulAdd(), eigen_matrixVectorMulAddBatch(), eigen_setZero(), eigen_updateGradBiases(), eigen_updateGradBiasesBatch(), eigen_updateGradWeights(), eigen_updateGradWeightsBatch(), eigen_vectorInit() (+4 more)

### Community 4 - "Parameter Config"
Cohesion: 0.25
Nodes (6): activation, baseParams, convertStringToEnum(), convertTupleToEnumArray(), loss, norm

### Community 6 - "PyTorch Benchmark"
Cohesion: 0.16
Nodes (17): _check(), _format_zon_value(), _get(), load_config(), Shared benchmark configuration helpers.  All benchmark parameters live in ``conf, Return the contents of params.zon (no comments) for the given config., Write benchmarks/params.zon from the config., Validate the config against the same constraints enforced at compile     time in (+9 more)

### Community 9 - "Docs & CI Workflows"
Cohesion: 0.15
Nodes (13): Docs Workflow, GitHub Pages, Benchmarks, Configuration, Documentation, Features, License, NNzig (+5 more)

### Community 10 - "Build System"
Cohesion: 0.67
Nodes (3): build(), createTree(), Tree

### Community 12 - "Eigen Activations"
Cohesion: 0.53
Nodes (5): eigen_none(), eigen_relu(), eigen_sigmoid(), eigen_tanh(), f_type

### Community 14 - "nnzig Benchmark Runner"
Cohesion: 0.21
Nodes (16): ensure_app_realized(), main(), parse_args(), parse_time_output(), parse_wall_clock(), Parse GNU time's 'm:ss' or 'h:mm:ss' into seconds (float)., Extract (wall_seconds, max_rss_kbytes) from GNU `time -v` output., Run cmd under `time -v`, return (wall_seconds, max_rss_kbytes).      GNU time wr (+8 more)

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

## Knowledge Gaps
- **23 isolated node(s):** `Tree`, `activation`, `loss`, `norm`, `baseParams` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `PyTorch Benchmark` to `nnzig Benchmark Runner`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `main()` connect `nnzig Benchmark Runner` to `PyTorch Benchmark`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `NNzig` (e.g. with `Docs Workflow` and `Test Workflow`) actually correct?**
  _`NNzig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Run cmd, streaming the last few lines of combined output. Returns     the Comple`, `Return the store path of the app's wrapper script (its 'program').      Uses sep`, `GNU time can only wrap a path that already exists, so if the resolved     progra` to the rest of the system?**
  _46 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Eigen Wrappers` be split into smaller, more focused modules?**
  _Cohesion score 0.07692307692307693 - nodes in this community are weakly interconnected._
- **Should `Community 21` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._