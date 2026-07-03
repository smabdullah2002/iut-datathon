# অলীকবচন — Hallucination Detection System
### Full Architecture & Bird's-Eye View

## Recommended Winning Direction

The first draft of this architecture was intentionally broad. After checking the competition rules and data shape more carefully, the better strategy is to keep the system simpler and center it on a calibrated binary classifier.

The key revision is this:

- **Primary scorer:** a multilingual encoder-based classifier
- **Support signals:** retrieval features, band/task metadata, and context availability
- **Auxiliary signal:** cross-lingual LLM consistency, used as a feature rather than a gate
- **Final layer:** a lightweight meta-model with band-aware calibration

This better matches the metric, which rewards hallucination detection on the negative class and gives extra importance to C1.

---

## 1. High-Level Pipeline

```
                              ┌─────────────────────────┐
                              │  Input: (prompt, response,│
                              │  context?, task_type,     │
                              │  domain, C-band)          │
                              └────────────┬─────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │                                             │
              context present?                            context absent
                     │                                             │
                     ▼                                             ▼
        ┌────────────────────────┐                  ┌──────────────────────────┐
        │   TRACK A: NLI/         │                  │  TRACK B: Closed-book    │
        │   Entailment Scoring    │                  │  Fact Verification       │
        │   (BanglaBERT)          │                  │  (Retrieval + Encoder)   │
        └────────────┬────────────┘                  └────────────┬─────────────┘
                     │                                             │
                     └─────────────────┬───────────────────────────┘
                                       │
                                      ▼
                                     ┌──────────────────────────────────┐
                                     │  AUXILIARY FEATURE LAYER          │
                                     │  Cross-lingual consistency         │
                                     │  + optional self-consistency       │
                                     │  + claim-extraction confidence      │
                                     └────────────────┬────────────────────┘
                                           │
                                           ▼
                                     ┌──────────────────────────────────┐
                                     │   META-CLASSIFIER                  │
                                     │   LightGBM / Logistic Regression   │
                                     │   over encoder, retrieval, band,   │
                                     │   context flag, and LLM features   │
                                     └────────────────┬────────────────────┘
                                           │
                                           ▼
                                      label ∈ {0 = hallucinated,
                                        1 = faithful}
```

---

## 2. Track-by-Track Breakdown

### Track A — Encoder-Centered Faithfulness Scoring
**Purpose:** Standard faithfulness check — does the response follow from the passage?

- **Model:** BanglaBERT or XLM-R fine-tuned as the main binary classifier
- **Input:** `(context, response)` when context exists, otherwise `(prompt, response)` plus a `context_available` flag
- **Output:** direct faithfulness probability, not only a 3-way entailment label
- **Training data:** sample split plus public Bengali/multilingual NLI data for warm-starting the decision boundary
- **Why this model:** it matches the actual scoring objective better than a generation-first stack and is easier to calibrate offline

### Track B — Retrieval-Augmented Verification Features
**Purpose:** Catch fabricated facts when there's no passage to check against — this is the hardest and highest-value part of the pipeline.

- **Retrieval corpus:** Bengali Wikipedia dump, Banglapedia if licensing permits, and other legal public Bengali fact sources packaged as Kaggle data
- **Retrieval tooling:** dense or hybrid retrieval with FAISS plus a sentence embedding model
- **Features:** top-k similarity scores, best-match entailment score, contradiction score, retrieval coverage, and evidence confidence
- **Role in the system:** these features support the encoder rather than becoming a standalone branch
- **Why this is better:** it reduces pipeline duplication and lets the classifier learn when retrieval is informative versus noisy

### Track C — Cross-Lingual Consistency as an Auxiliary Feature
**Purpose:** Catch the "fluent but wrong in Bengali" failure mode the competition is explicitly built around (e.g. correct in English, wrong in Bengali).

- **Model:** TituLLM (1B or 3B — test both if compute allows)
- **Method:**
  1. Extract the core factual claim from the response
  2. Ask TituLLM the same underlying question in English and in Bengali
  3. Compare the two answers (string/semantic similarity, or a small entailment check between them)
  4. Large divergence → strong hallucination signal
- **Role in the system:** always compute this where a claim can be extracted, but feed it into the meta-model as a feature
- **Calibration note:** allow the final model to learn how much to trust this feature by band, task type, and retrieval confidence
- **Why this stays valuable:** it captures cases where the answer is fluent and plausible but factually wrong in a way retrieval may miss

### Track D — Self-Consistency Sampling (optional, if budget allows)
**Purpose:** Auxiliary signal — hallucinated/fabricated facts tend to be less stable across resampling than true facts.

- **Method:** Generate the same answer 3–5 times at moderate temperature (via TituLLM or another compact open-weight model), measure agreement
- Keep it as a low-priority feature, only if it fits the runtime budget after the core encoder and retrieval passes

### Meta-Classifier — Final Decision Layer
- **Model:** LightGBM or logistic regression
- **Features:** encoder score, top-k retrieval scores, evidence entailment, cross-lingual consistency, self-consistency, context flag, cultural band, task type, and domain
- **Training strategy:** stratified validation by band and task type, with threshold calibration focused on hallucinated-class F1
- **Why this works:** it is easier to reproduce offline and better aligned with the competition metric than a large ensemble of loosely coupled models

---

## 3. Suggested Tools & Models Summary

| Layer | Tool / Model | Notes |
|---|---|---|
| NLI / entailment | **BanglaBERT** or **XLM-R** (fine-tuned) | Primary faithfulness scorer |
| Secondary encoder | **XLM-R** or **BanglaBERT** | Comparison/ensemble candidate |
| Generative Bangla-native LLM | **TituLLM (1B / 3B)** | Cross-lingual consistency check; continual-pretrained on Llama-3.2, ~37B token corpus |
| Alternative generative model | **TigerLLM** | Also explicitly permitted; worth benchmarking against TituLLM |
| General multilingual backup | **Qwen / Gemma / Llama (small variants)** | If Bengali-native models underperform on a task type |
| Retrieval index | **FAISS** | Local, no external calls needed at inference — code-competition safe |
| Sentence embeddings | Bengali/multilingual sentence-transformer (verify license/size) | For building the retrieval corpus |
| Fact corpus | Bengali Wikipedia dump (+ Banglapedia if permitted) | Attach as a Kaggle dataset |
| Ensemble/meta-model | **LightGBM** or Logistic Regression | Final classifier, calibrated on band-aware features |
| Quantization (if size-constrained) | GGUF / Q4_K_M via Ollama-style tooling | Keeps you under the 50GB combined weight cap |

---

## 4. Compute Budget Sanity Check (Phase 2 constraints)

- Hardware: single P100 or 2×T4
- Runtime: < 9 hours total
- On-disk weights: < 50GB combined
- No internet at inference — everything must be pre-attached as a Kaggle dataset

**Rough allocation:**
- BanglaBERT (Track A+B): tiny footprint, near-instant inference
- TituLLM 1B or 3B (Track C): the main GPU/time cost
- FAISS index + embeddings: CPU-side, minimal GPU cost
- LightGBM ensemble: negligible

This split leaves comfortable headroom under both the size and runtime caps, especially if the LLM is used as a feature rather than a full decision branch.

---

## 5. Validation Strategy

- Build a **stratified local split** by cultural-distance band (C0/C1/C2) × task type from the sample data
- Track **overall F1** and **C1 F1 separately** — C1 is the tie-breaker and where the competition's core phenomenon lives
- Calibrate thresholds per band or include band as an explicit categorical feature in the final model
- Read every C1 sample row manually before finalizing the fact-verification corpus — qualitative pattern recognition matters more here than in a typical Kaggle setup

---

## 6. Open Item to Verify Before Committing

- Confirm TituLLM and TigerLLM weight licenses and Hugging Face availability, so they can legally be pulled once and attached as a Kaggle dataset for offline Phase 2 use.

## 7. Revised Recommendation Summary

If the goal is to win under the competition rules, the best version of this architecture is:

1. A strong encoder-based classifier as the main prediction engine
2. Retrieval as supporting evidence generation, not a separate branch
3. Cross-lingual consistency as a real feature, especially for C1
4. A lightweight meta-model with band-aware calibration
5. Optional self-consistency only if runtime allows

That version is simpler to execute, easier to validate, and better matched to the scoring rule than the original multi-track generative design.
