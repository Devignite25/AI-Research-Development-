# QLSM-NS: Quantized Latent State Model — Neuro-Symbolic Variant

**A Hybrid Architecture for Intent-Controlled Python Code Generation**

**Jorge A. Maure**  
DevIgnite LLC — AI Research & Development  
The Wider Lens Research Initiative  
Jacksonville, Florida, USA  
January 2026

---

## Abstract

We present **QLSM-NS (Quantized Latent State Model — Neuro-Symbolic Variant)**, a hybrid architecture for efficient, intent-controlled Python code generation that combines neural generation with symbolic reasoning. QLSM-NS addresses the central failure mode of pure neural code generation — syntactically invalid, semantically inconsistent output — by integrating a Bridge Layer that connects a neural VQ-VAE + Mamba predictor pipeline with a pure-Python symbolic reasoning system (Hindley-Milner type inference, AST pattern matching, 15 algebraic simplification rules).

Training proceeded through eight phases on an NVIDIA RTX 3060 12GB, resolving codebook collapse, predictor overfitting, and encoder-decoder mismatch. Key results: VQ-VAE codebook utilization from 16.3% to **100%**; validation loss from 0.019 to **0.0001**; syntax validity from 0% to **93.3%** via symbolic refinement; and a REINFORCE policy gradient breakthrough from 6.3% to **78.5%** syntax validity without symbolic assistance. The neuro-symbolic thesis is confirmed: symbolic refinement produces >20% execution improvement over neural-only output.

**Keywords:** code generation, neuro-symbolic AI, VQ-VAE, Mamba, Hindley-Milner types, REINFORCE, policy gradient, intent conditioning

---

## 1. Introduction

### 1.1 The Problem With Pure Neural Code Generation

Neural code generation systems learn to produce syntactically plausible character sequences by exposure to code corpora. They have no internal representation of what makes code *valid* — type consistency, scope correctness, balanced syntax. When they fail, they fail silently: the output looks like code but cannot execute.

The standard response is scale: more parameters, more data, more compute. QLSM-NS explores a different response: encode what we know symbolically, and let the neural system learn what we cannot encode.

### 1.2 The Neuro-Symbolic Thesis

**Hypothesis:** A neural generator constrained and corrected by symbolic reasoning will produce more executable code than either system alone.

This is precisely the claim HierHypGen proved for image generation: structural inductive bias (Sheaf Laplacian regularization) outperforms statistical learning at high compression. QLSM-NS applies the same principle to code generation: structural symbolic constraints outperform learned statistical regularities for correctness guarantees.

### 1.3 QLSM-NS in the DevIgnite Research Program

QLSM-NS is the seventh DevIgnite paper and the only one that is primarily empirical engineering rather than architectural theory. It provides the working proof of the neuro-symbolic principle that Genesis's Symbolic Layer depends on. Where Genesis specifies what a grounded self-model should do, QLSM-NS demonstrates that grounding symbolic constraints to runtime outcomes produces measurable gains.

### 1.4 Contributions

1. **Four-layer hybrid architecture** — Executive, Bridge, Symbolic, Neural — with clean separation of concerns and five configurable generation strategies.
2. **Codebook collapse resolution** — Three-phase VQ-VAE development resolving the utilization collapse from 16.3% to 100% via Gumbel-softmax quantization.
3. **Symbolic refinement pipeline** — Hindley-Milner type checker + AST pattern matching + 15 algebraic rules lifting syntax validity from 0% to 93.3%.
4. **REINFORCE breakthrough** — Policy gradient training with frozen encoder, trainable decoder lifting syntax validity from 6.3% to 78.5% without symbolic assistance.
5. **Neuro-symbolic thesis confirmed** — Symbolic refinement produces >20% execution improvement over neural-only output, validated empirically.

---

## 2. Architecture

### 2.1 System Overview

QLSM-NS consists of four layers with strict separation of concerns:

```
Natural Language Intent
        │
        ▼
EXECUTIVE LAYER
├── GoalManager (track intent goals, measure completion)
└── StrategySelector (choose generation strategy by complexity)
        │
        ▼
NEURAL LAYER
├── VQ-VAE Compressor (code → discrete latent codes, 4x compression)
├── Mamba/LSTM Predictor (next-code prediction)
├── Intent Leash (frozen MiniLM embeddings for intent conditioning)
└── Grounding Decoder (latent codes → Python characters)
        │
        ▼
BRIDGE LAYER
├── SymbolicInterface (unified validate / parse / simplify / infer API)
├── ConstrainedGenerator (per-step symbolic constraint application)
└── ConflictResolver (ACCEPT | RETRY | SYMBOLIC_FIX | FALLBACK)
        │
        ▼
SYMBOLIC LAYER
├── AST Nodes (custom representation, pattern matching)
├── Type System (Hindley-Milner with unification)
├── Inference Engine (15 algebraic simplification rules)
└── Knowledge Base (builtin types, common patterns)
        │
        ▼
Validated Python Code
```

### 2.2 Neural Layer

**VQ-VAE.** Compresses Python source code to discrete latent codes at 4x compression. Codebook size 512 (reduced from 2048 after collapse analysis). Gumbel-softmax quantization with temperature annealing. Entropy regularization weight 0.1 (reduced from 0.5 after reconstruction-utilization tradeoff analysis). Dead code restart threshold 0.1%.

**Mamba/LSTM Predictor.** Autoregressive next-code prediction over the VQ codebook. 9.0M parameters (expanded from 4.8M in v1). Dropout 0.3, label smoothing 0.1, 80/20 train/val split, early stopping patience 10.

**Intent Leash.** Frozen MiniLM sentence embeddings project natural language intent into the predictor's conditioning space. Intent similarity 0.95 on benchmark tasks.

**Grounding Decoder.** Expands latent codes back to Python characters. The primary bottleneck identified in Phase 6 — VQ-VAE reconstruction quality caps end-to-end execution success.

### 2.3 Bridge Layer

**SymbolicInterface.** Unified API over all symbolic operations:

```python
result = bridge.validate("def add(a, b): return a + b")
# ValidationResult(valid=True, syntax_ok=True, types_ok=True, confidence=1.0)
```

Validation fields: `valid`, `syntax_ok`, `types_ok`, `semantics_ok`, `errors`, `suggestions`, `confidence`.

**ConflictResolver.** Four resolution actions applied when neural output fails validation:

| Action | When Applied |
|---|---|
| ACCEPT | Validation passed |
| RETRY | Recoverable error — regenerate with guidance |
| SYMBOLIC_FIX | Apply targeted symbolic correction |
| FALLBACK | Use template from knowledge base |

**Strategy Selection.** The Executive Layer selects one of five strategies based on complexity score:

| Strategy | Bridge Behavior | Use Case |
|---|---|---|
| NEURAL_ONLY | No validation | Fast, low-stakes generation |
| NEURAL_THEN_VERIFY | Validate after generation | Default |
| ITERATIVE_REFINEMENT | Validate → fix → repeat (max 5) | Complex tasks |
| CONSTRAINED_NEURAL | Per-step constraint checking | High correctness requirement |
| SYMBOLIC_THEN_NEURAL | Template + neural infill | Structured patterns |

### 2.4 Symbolic Layer

The Symbolic Layer operates entirely in pure Python — no PyTorch required. This is architecturally significant: symbolic reasoning is available even when the neural layer is absent or unavailable (MockQLSM mode for testing).

**AST Nodes.** Custom AST representation supporting pattern matching:

```python
pattern = BinaryOp("+", Capture("left"), Capture("right"))
target = BinaryOp("+", Var("x"), Literal(1))
matches = match_pattern(target, pattern)
# {"left": Var(x), "right": Literal(1)}
```

Node types: Literal, Var, BinaryOp, FunctionDef, Call, Return, If, For, Module.

**Type System.** Hindley-Milner with unification:

```
unify(List[T], List[int]) = {T: int}
unify(int, str) → UnificationError
```

**Inference Engine.** 15 algebraic simplification rules applied iteratively:

| Rule | Pattern | Result |
|---|---|---|
| add_zero_right | x + 0 | x |
| mul_one_right | x * 1 | x |
| mul_zero | x * 0 | 0 |
| sub_self | x - x | 0 |
| div_one | x / 1 | x |
| add_self | x + x | 2 * x |
| *...9 more* | | |

---

## 3. Training — Eight Phases

### 3.1 Phase 1: Baseline VQ-VAE (Failure Analysis)

**Configuration:** Codebook size 2048, hidden dim 256, hard VQ + EMA updates, 100 epochs.

**Failure:** Codebook collapse — only 16.3% of codes used (336/2048). EMA update mechanism causes codes to cluster around frequently-seen patterns. Reconstruction accuracy 50.8% on test.

**Root cause:** EMA updates in hard VQ create a positive feedback loop — popular codes receive more updates, improving further, drawing more assignments. Unpopular codes are never updated and become dead.

### 3.2 Phase 2: VQ-VAE v2 (Fix Collapse)

**Changes:** Codebook reduced to 512, Gumbel-softmax quantization, entropy loss weight 0.5, dead code restart threshold 0.5%.

**Results:** Codebook utilization 100% ✅. Reconstruction accuracy dropped to 25.6% — entropy loss too aggressive.

**Finding:** The reconstruction-utilization tradeoff. High entropy weight forces all codes to be used equally, but prevents the network from learning which codes to use for what — every code becomes generic.

### 3.3 Phase 3: VQ-VAE v3 (Balanced — Best)

**Changes from v2:** Entropy weight reduced 0.5 → 0.1, hidden dim increased 256 → 384, training data expanded to 12K mixed synthetic + real, slower Gumbel annealing.

| Epoch | Recon Loss | Val Loss | Codebook Util |
|---|---|---|---|
| 1 | 1.331 | 0.556 | 99.8% |
| 10 | 0.015 | 0.008 | 100% |
| 20 | 0.003 | 0.0004 | 100% |
| 30 | **0.001** | **0.0001** | **100%** |
| 45 (early stop) | 0.005 | 0.002 | 100% |

**Final:** 100% codebook utilization, 0.0001 validation loss, 39% reconstruction accuracy. This is the VQ-VAE checkpoint used in all downstream phases.

### 3.4 Phase 4: Predictor v2 (Fix Overfit)

**v1 failure:** Loss collapsed to 0.0000 — the model memorized the training set.

**Fixes:** Dropout 0.3, label smoothing 0.1, 80/20 train/val split, early stopping patience 10.

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 6.24 | 6.23 |
| 10 | 5.69 | 5.60 |
| 27 (early stop) | 5.32 | **5.58** |

Train and val loss track together — healthy generalization. No collapse to memorization.

### 3.5 Phase 5: Symbolic Integration (Neuro-Symbolic Thesis Confirmation)

**Setup:** Symbolic refinement pipeline applied to neural outputs. 15 patterns tested: simple arithmetic, control flow (if-else), recursion (factorial, fibonacci), list comprehensions, slice notation, method calls.

| Metric | Pre-Refinement | Post-Refinement | Improvement |
|---|---|---|---|
| Syntax Validity | 0% | **93.3%** | +93.3pp |
| Execution Success | 0% | **40.0%** | +40.0pp |

**Conclusion: The neuro-symbolic thesis holds.** Symbolic refinement converts syntactically invalid neural output into executable code across all tested pattern classes. The >20% execution improvement validates the architecture's central claim.

### 3.6 Phase 6: End-to-End Finetuning (Bottleneck Identification)

**Approach:** Joint VQ-VAE + Predictor training on syntax validity signal.

**Result:** Best val syntax 6.3%. VQ-VAE reconstruction quality is the bottleneck — the decoder cannot reliably recover Python characters from latent codes regardless of predictor quality.

**Finding:** Supervised finetuning cannot bypass the VQ-VAE reconstruction ceiling. A different training signal is needed.

### 3.7 Phase 7: REINFORCE Policy Gradient (Breakthrough)

**Approach:** Policy gradient training with frozen encoder, trainable decoder only. Syntax validity as reward. Baseline variance reduction for stable learning.

**Configuration:** 50 epochs, lr=5e-6, simple pattern curriculum, Gumbel-softmax for differentiable sampling.

| Metric | Phase 6 (E2E Finetune) | Phase 7 (REINFORCE) | Improvement |
|---|---|---|---|
| Val Syntax | 6.3% | **78.5%** | +72.2pp |

**Benchmark (4 samples):** 100% syntax validity, 75% execution correctness.

**Why REINFORCE works where supervised finetuning failed:** Supervised training propagates gradients through the VQ-VAE bottleneck, which is effectively noise at this quality level. REINFORCE bypasses the bottleneck entirely — it treats the encoder as fixed and trains the decoder directly on the reward signal it actually cares about (syntax validity). The gradient flows from outcome to decoder, not through latent space.

### 3.8 Phase 8: Execution Reward Training

**Approach:** Combined reward: 0.3 × syntax + 0.7 × execution. Training with test cases for verification. Lower lr (2e-6) for fine-tuning stability.

| Epoch | Syntax | Execution |
|---|---|---|
| 1 | 87.5% | 43.0% |
| 50 | 97.0% | 59.5% |
| 95 | **97.0%** | **68.0%** |
| 100 | 96.5% | 61.5% |

**Key insight:** The execution-trained model produces 40% syntactically valid code *directly* — without symbolic refinement — versus 0% from the base VQ-VAE. It has internalized some syntactic structure via the combined reward signal.

**Tradeoff:** Symbolic refinement patterns were tuned for VQ-VAE output characteristics. Execution-trained output is cleaner and different — refinement needs retuning. **Refine v2** (tuned for execution output): 100% syntax, 45% execution.

---

## 4. Final Results

| Component | Status | Key Metric |
|---|---|---|
| VQ-VAE v3 | ✅ | 100% codebook utilization, 0.0001 val loss |
| Predictor v2 | ✅ | 5.58 val loss, no overfit |
| Symbolic Refinement | ✅ | 93.3% syntax, 40% execution |
| Intent Leash | ✅ | 0.95 intent similarity |
| REINFORCE E2E | ✅ | 78.5% syntax without symbolic assistance |
| Execution Reward | ✅ | 40% raw syntax, 68% training execution |
| Refine v2 | ✅ | **100% syntax, 45% execution** |
| Test Suite | ✅ | **23/23 tests passing** |

---

## 5. Bug Fixes Documented

| Bug | File | Fix |
|---|---|---|
| Codebook collapse | vqvae.py | Gumbel-softmax replaces EMA |
| Reconstruction-utilization tradeoff | vqvae.py | Entropy weight 0.5 → 0.1 |
| Predictor memorization | predictor.py | Dropout + label smoothing + early stopping |
| Leash device mismatch | leash.py | Dynamic device placement for projection layer |
| Tensor view error | losses.py | reshape() instead of view() |

---

## 6. Discussion

### 6.1 Why the Symbolic Layer Cannot Be Replaced by Scale

The symbolic refinement pipeline operates on structural properties of Python code — scope, type consistency, bracket balance. These properties are discrete and compositional: a missing comma is either present or absent, and adding it either fixes or doesn't fix the parse error. Neural generators learn probability distributions over tokens; they have no representation of these structural properties.

This is exactly the lesson HierHypGen proved for images: the Sheaf Laplacian enforces regional consistency that the network cannot reliably learn statistically. QLSM-NS proves the same principle for code: the type checker enforces correctness that the predictor cannot reliably learn from corpus statistics alone.

### 6.2 Why REINFORCE Worked Where Supervised Finetuning Failed

Supervised finetuning propagates gradients through every layer, including the VQ-VAE bottleneck. At VQ-VAE reconstruction accuracy of 39%, the gradient signal passing through the bottleneck is substantially noise. The decoder receives contradictory updates — improve this latent code in one direction while the reconstruction loss pushes it in another.

REINFORCE bypasses the bottleneck by treating the encoder as fixed. The decoder trains directly on syntax validity. This is the same principle as the FAA productive conflict resolution: when two gradient signals are in conflict, structural separation of the optimization targets is more effective than joint training.

### 6.3 The Remaining Bottleneck

VQ-VAE reconstruction at 39% character accuracy is the ceiling. At higher reconstruction accuracy, the predictor's learned code distributions would translate more reliably into valid Python. Phase 8 showed that execution reward training can partially compensate — internalized syntax structure reduces dependence on reconstruction quality. Full resolution requires either better VQ-VAE reconstruction or moving to a continuous latent space.

### 6.4 Connection to Genesis

Genesis's Symbolic Layer (structural safety priors, domain-aware thresholds, productive conflict) directly inherits the QLSM-NS principle: **encode what you know symbolically, learn what you cannot encode.** The Critic's grounded self-model predicts observable outcomes rather than internal states — this is the same architectural decision as QLSM-NS's symbolic layer predicting structural validity rather than learning it from gradient pressure.

---

## 7. Conclusion

QLSM-NS proves the neuro-symbolic thesis empirically for Python code generation. Symbolic refinement produces >20% execution improvement over neural-only output. REINFORCE policy gradient produces a 72.2 percentage point improvement in syntax validity over supervised finetuning by bypassing the VQ-VAE bottleneck. The full system achieves 100% syntax validity and 45% execution correctness with Refine v2 and 23/23 tests passing.

The architecture demonstrates a general principle with implications beyond code generation: wherever output validity can be specified structurally, symbolic constraints should be applied structurally rather than learned statistically. Neural systems learn distributions. Symbolic systems enforce rules. The combination outperforms either alone.

---

**Code:** https://github.com/Devignite25/AI-Research-Development  
**Hardware:** NVIDIA RTX 3060 12GB | PyTorch 2.5.1+cu121

---

## References

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
2. van den Oord, A., et al. (2017). Neural Discrete Representation Learning (VQ-VAE). NeurIPS.
3. Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning. (REINFORCE)
4. Damas, L., & Milner, R. (1982). Principal Type-Schemes for Functional Programs. POPL. (Hindley-Milner)
5. Maure, J. A. (2026). HierHypGen: Sheaf-Theoretic Diffusion. DevIgnite LLC. [This repository]
6. Maure, J. A. (2026). Genesis: A Unified Foundation for Developmental AGI. DevIgnite LLC. [This repository]
7. Wang, Y., et al. (2021). CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation. EMNLP.

---

*© 2026 DevIgnite LLC — AI Research & Development / The Wider Lens Research Initiative. Licensed under CC BY 4.0.*
