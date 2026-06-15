# Graph Report - nnzig  (2026-06-14)

## Corpus Check
- 21 files · ~16,259 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 156 nodes · 180 edges · 21 communities (15 shown, 6 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6c86cd9f`
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
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]

## God Nodes (most connected - your core abstractions)
1. `NN` - 16 edges
2. `f_type` - 11 edges
3. `NNzig` - 11 edges
4. `NNzig` - 8 edges
5. `Norm` - 6 edges
6. `MLP` - 5 edges
7. `f_type` - 4 edges
8. `MLP` - 3 edges
9. `f_type` - 3 edges
10. `Docs Workflow` - 3 edges

## Surprising Connections (you probably didn't know these)
- `Test Workflow` --conceptually_related_to--> `NNzig`  [INFERRED]
  .github/workflows/test.yml → README.md
- `Docs Workflow` --conceptually_related_to--> `NNzig`  [INFERRED]
  .github/workflows/docs.yml → README.md

## Import Cycles
- None detected.

## Communities (21 total, 6 thin omitted)

### Community 0 - "Eigen Wrappers"
Cohesion: 0.08
Nodes (38): activateNone(), activateRelu(), activateSigmoid(), activateTanh(), computeMeanStd(), computeMSE(), denormalize(), eigen_computeMeanStd() (+30 more)

### Community 2 - "Eigen Linear Algebra"
Cohesion: 0.28
Nodes (12): eigen_matrixVectorMulAdd(), eigen_matrixVectorMulAddBatch(), eigen_setZero(), eigen_updateGradBiases(), eigen_updateGradBiasesBatch(), eigen_updateGradWeights(), eigen_updateGradWeightsBatch(), eigen_vectorInit() (+4 more)

### Community 4 - "Parameter Config"
Cohesion: 0.25
Nodes (6): activation, baseParams, convertStringToEnum(), convertTupleToEnumArray(), loss, norm

### Community 9 - "Docs & CI Workflows"
Cohesion: 0.15
Nodes (13): Docs Workflow, GitHub Pages, Benchmarks, Configuration, Documentation, Features, License, NNzig (+5 more)

### Community 10 - "Build System"
Cohesion: 0.67
Nodes (3): build(), createTree(), Tree

### Community 12 - "Eigen Activations"
Cohesion: 0.53
Nodes (5): eigen_none(), eigen_relu(), eigen_sigmoid(), eigen_tanh(), f_type

### Community 18 - "Community 18"
Cohesion: 0.22
Nodes (8): Architecture, Benchmark setup, Commands, Configuration model — read this before editing params, graphify, NNzig, Toolchain, Zig 0.16 conventions used here

### Community 20 - "Community 20"
Cohesion: 0.60
Nodes (4): eigen_computeMeanStd(), eigen_denormalize(), eigen_normalize(), f_type

## Knowledge Gaps
- **23 isolated node(s):** `Tree`, `activation`, `loss`, `norm`, `baseParams` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Are the 2 inferred relationships involving `NNzig` (e.g. with `Docs Workflow` and `Test Workflow`) actually correct?**
  _`NNzig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Tree`, `activation`, `loss` to the rest of the system?**
  _23 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Eigen Wrappers` be split into smaller, more focused modules?**
  _Cohesion score 0.07692307692307693 - nodes in this community are weakly interconnected._