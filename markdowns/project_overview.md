# Project Overview: অলীকবচন

## Competition Goal
Build a Bengali hallucination detector that predicts whether a response is:

- `1` = faithful
- `0` = hallucinated

The model must work with and without supporting `context`.

## Why It Matters
Bengali LLMs can sound fluent while still giving wrong answers. This competition focuses on catching those errors, especially in culturally specific Bengali knowledge where mistakes are harder to notice.

## Data Summary
The benchmark includes:

- Bengali prompts and candidate responses
- Some rows with supporting context, some without context
- Multiple domains and task types
- Cultural-distance bands:
  - `C0`: globally stable facts
  - `C1`: Bangladesh-specific or culturally situated facts
  - `C2`: recent, contested, or time-sensitive facts

The released sample set is for local validation. The test set is hidden.

## Evaluation
Submissions are scored with F1 on the hallucinated class. In practice, this is the main metric to optimize.

## Competition Format
The competition has two phases:

1. Phase 1: Kaggle leaderboard predictions
2. Phase 2: runnable offline solution package, paper, and README for top teams

## Practical Constraints
Phase 2 runs inside Kaggle with these limits:

- offline inference only
- open-weight models only
- under 9 hours total runtime
- under 50 GB total model weights

## Suggested System Direction
The architecture note in this workspace proposes a multi-track ensemble:

- context-based NLI for rows with passages
- retrieval-based verification for rows without passages
- cross-lingual consistency checks
- optional self-consistency sampling
- a lightweight meta-classifier to combine scores

## Files In This Workspace

- [olikbochon_architecture.md](olikbochon_architecture.md)
- [app/dataset/dataset samples.json](app/dataset/dataset%20samples.json)
- [app/dataset/test set.csv](app/dataset/test%20set.csv)

## Short Takeaway
This is a Bengali hallucination detection research competition, not a standard supervised classification task. The strongest solutions will likely combine factual verification, multilingual consistency signals, and careful handling of culturally specific examples.