# The Competence-Conservatism Paradox

**Author:** Jorge A. Maure — DevIgnite LLC  
**Date:** February 2026  
**Keywords:** AI safety, modular architectures, multi-agent RL, alignment tax, variance in policy gradients

---

## Abstract

We identify and characterize the **Competence-Conservatism Paradox** in modular multi-agent reinforcement learning architectures for safety-aligned decision-making. In a triad consisting of a high-variance risk-taking module (H-Module), a low-variance conservative baseline (L-Module), and a learned safety critic (F-Module), increasing the competence of the critic systematically suppresses the risk-taking module.

Empirical results in simplified poker and blackjack environments show that an untrained critic permits H-Module engagement in ~55% of decisions, while a trained critic reduces H-selection to 11.6%. Ablation studies demonstrate that lightweight architectural interventions restore H-engagement to ~40% with no accuracy penalty. Attempts to replace hard-coded arbitration with meta-learned conflict resolution via REINFORCE collapse to 0% H-selection, revealing the **Variance Asymmetry Problem**.

> *"Intelligent safety mechanisms that detect disagreement will systematically suppress risk-taking agents unless explicitly trained to tolerate productive conflict."*

---

## Read the Paper

[competence_conservatism_paradox.md](./competence_conservatism_paradox.md)

---

## Key Finding

| Configuration | F-Module State | H-Module Selection | Result |
|---|---|---|---|
| Untrained | Passive | **55%** | H engaged |
| Trained (20 epochs) | Active — 100% CONFLICT | **11.6%** | H suppressed |
| Asymmetric Fix | Absolute threshold | **39.6%** | Restored |

---

## Citation
```
Maure, J. A. (2026). The Competence-Conservatism Paradox: Emergent Risk-Aversion
in Multi-Agent Safety Architectures. DevIgnite LLC AI Research & Development.
https://github.com/Devignite25/AI-Research-Development
```
