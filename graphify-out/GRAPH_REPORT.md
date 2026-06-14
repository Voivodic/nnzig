# Graph Report - .  (2026-06-13)

## Corpus Check
- Corpus is ~13,772 words - fits in a single context window. You may not need a graph.

## Summary
- 119 nodes · 134 edges · 18 communities (12 shown, 6 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.8)
- Token cost: 1,281 input · 575 output

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

## God Nodes (most connected - your core abstractions)
1. `NN` - 16 edges
2. `f_type` - 11 edges
3. `Norm` - 6 edges
4. `MLP` - 5 edges
5. `MLP` - 3 edges
6. `Docs Workflow` - 3 edges
7. `createTree()` - 2 edges
8. `build()` - 2 edges
9. `computeMeanStd()` - 2 edges
10. `convertStringToEnum()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `Docs Workflow` --conceptually_related_to--> `NNzig`  [INFERRED]
  .github/workflows/docs.yml → README.md
- `Test Workflow` --conceptually_related_to--> `NNzig`  [INFERRED]
  .github/workflows/test.yml → README.md

## Import Cycles
- None detected.

## Communities (18 total, 6 thin omitted)

### Community 0 - "Eigen Wrappers"
Cohesion: 0.13
Nodes (22): eigen_matrixVectorMulAdd(), eigen_matrixVectorMulAddBatch(), eigen_setZero(), eigen_updateGradBiases(), eigen_updateGradBiasesBatch(), eigen_updateGradWeights(), eigen_updateGradWeightsBatch(), eigen_vectorInit() (+14 more)

### Community 2 - "Eigen Linear Algebra"
Cohesion: 0.28
Nodes (12): eigen_matrixVectorMulAdd(), eigen_matrixVectorMulAddBatch(), eigen_setZero(), eigen_updateGradBiases(), eigen_updateGradBiasesBatch(), eigen_updateGradWeights(), eigen_updateGradWeightsBatch(), eigen_vectorInit() (+4 more)

### Community 4 - "Parameter Config"
Cohesion: 0.25
Nodes (6): activation, baseParams, convertStringToEnum(), convertTupleToEnumArray(), loss, norm

### Community 9 - "Docs & CI Workflows"
Cohesion: 0.50
Nodes (5): Docs Workflow, GitHub Pages, NNzig, Test Workflow, Zig Compiler

### Community 10 - "Build System"
Cohesion: 0.67
Nodes (3): build(), createTree(), Tree

## Knowledge Gaps
- **8 isolated node(s):** `Tree`, `activation`, `loss`, `norm`, `baseParams` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `Tree`, `activation`, `loss` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Eigen Wrappers` be split into smaller, more focused modules?**
  _Cohesion score 0.13043478260869565 - nodes in this community are weakly interconnected._