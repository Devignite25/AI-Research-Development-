# QLSM-NS: Quantized Latent State Model — Neuro-Symbolic Variant

**Author:** Jorge A. Maure — DevIgnite LLC / The Wider Lens Research Initiative  
**Date:** January 2026  
**Hardware:** NVIDIA RTX 3060 12GB | PyTorch 2.5.1+cu121  
**Status:** ✅ Trained and validated — 23/23 tests passing

---

## Overview

QLSM-NS is the **empirical proof paper** in the DevIgnite research program — the only paper that is primarily engineering rather than theory. It proves the neuro-symbolic thesis for Python code generation: a neural VQ-VAE + Mamba pipeline constrained and corrected by a pure-Python symbolic reasoning system (Hindley-Milner types, AST pattern matching, 15 algebraic rules) produces measurably more executable code than either system alone.

> **>20% execution improvement from symbolic refinement. Confirmed.**

---

## Final Results

| Component | Key Metric |
|---|---|
| VQ-VAE v3 | 100% codebook utilization, 0.0001 val loss |
| Predictor v2 | 5.58 val loss — no overfit |
| Symbolic Refinement | **93.3% syntax, 40% execution** |
| REINFORCE E2E | **78.5% syntax** — 72pp jump from supervised |
| Execution Reward | 68% training execution |
| Refine v2 | **100% syntax, 45% execution** |
| Test Suite | **23/23 passing** |

---

## Architecture

```
Natural Language Intent
        │
EXECUTIVE LAYER (GoalManager + StrategySelector)
        │
NEURAL LAYER (VQ-VAE → Mamba → Intent Leash → Decoder)
        │
BRIDGE LAYER (validate → fix → resolve conflicts)
        │
SYMBOLIC LAYER (AST + Hindley-Milner types + 15 rules)
        │
Validated Python Code
```

Five generation strategies selectable by complexity: NEURAL_ONLY, NEURAL_THEN_VERIFY, ITERATIVE_REFINEMENT, CONSTRAINED_NEURAL, SYMBOLIC_THEN_NEURAL.

---

## Training Arc (8 Phases)

| Phase | Focus | Key Result |
|---|---|---|
| 1 | Baseline VQ-VAE | 16.3% codebook use — failed |
| 2 | Gumbel-softmax | 100% utilization — fixed |
| 3 | Balanced entropy | 0.0001 val loss — best VQ-VAE |
| 4 | Predictor overfit | Healthy 5.58 val loss |
| 5 | Symbolic integration | **0% → 93.3% syntax** ← thesis confirmed |
| 6 | E2E finetuning | 6.3% — VQ-VAE is bottleneck |
| 7 | REINFORCE | **6.3% → 78.5% syntax** ← breakthrough |
| 8 | Execution reward | **68% execution** |

---

## Connection to the Wider Research Program

QLSM-NS proves the same principle HierHypGen proved for images: **structural inductive bias outperforms statistical learning for correctness guarantees.** HierHypGen used Sheaf Laplacian regularization for image coherence. QLSM-NS uses Hindley-Milner type checking for code validity. Same principle, different domain, both confirmed empirically.

This principle is directly inherited by Genesis's Symbolic Layer and Grounded Self-Model design.

---

## Quick Start

```bash
# Without PyTorch (Mock)
from qlsm.models.mock_qlsm import MockQLSM
mock = MockQLSM()
code = mock.generate("Write a function to add two numbers")[0]

# With PyTorch (Full)
from qlsm import QLSM_NS
model = QLSM_NS()
result = model.generate("Write a function to add two numbers")
print(result.code, result.validation.valid)

# Tests
python tests/test_symbolic_standalone.py   # 5/5
python tests/test_integration_mock.py      # 8/8
python tests/test_qlsm_ns.py               # 10/10
```

---

## Read the Paper

[QLSM_NS_Paper.pdf](./QLSM_NS_Paper.pdf)

---

## Citation

```
Maure, J. A. (2026). QLSM-NS: Quantized Latent State Model — Neuro-Symbolic
Variant. DevIgnite LLC / The Wider Lens Research Initiative.
https://github.com/Devignite25/AI-Research-Development
```

---

*© 2026 DevIgnite LLC. Licensed under CC BY 4.0.*
