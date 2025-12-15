# A Simple Parameter Inference with Transformers

Semestral project for the course *Computational Intelligence Methods* (MVI), FIT CTU.
This repository contains code and experiments accompanying the paper:

**“A Simple Parameter Inference with Transformers”**

## Overview

Starting from a single-frequency sinusoid, the project progressively increases signal complexity by:
- adding Gaussian noise,
- adding a second frequency,
- introducing nonlinear interaction terms.

Both **deterministic regression** and **probabilistic inference using normalizing flows** are investigated.

---

## Original formulated assignment
Consider the simple periodic function:
$$y_i(A_i,\omega_i;t)=A_i \sin\left(\omega_i t \right)$$

where _A_ represents the amplitude, _ω_ the frequency, and _t_ is the time variable. Restrict the time variable to the interval t ∈ [0,2\π].

To create the dataset, produce _N_ discrete representations of $y_i$, i.e., N-number of 1D vectors, called $V_i(A_i,\omega_i)$, by creating a grid of reasonable combinations of _A_ and _ω_. Assume that it is sampled with 100 points, meaning each 1D vector consists of 100 points. Initially, N can be on the order of 10³ and then progressively increase to test the model’s accuracy against this choice.

Inference:
Using the dataset, start with an encoder-only transformer model (e.g., a Bert-like model) and train it to predict $\left(A,\omega\right)$ from a given $V_i$. In this case, the vector $V_i$ is the input, and the output (or “labels”) will be the pairs$\left(A_i,\omega \right)$, to $Vi\left(A_i,\omega_i \right)$.

Later was amplitude fixed to $A = 1$ and increasing dataset samples $N=10^3$ was done within optimization.

### Signal types

1. **Single-frequency signal**
$$y(t) = \sin(\omega \cdot t)$$

2. **Two-frequency linear mixture**
$$y(t) = \sin(\omega_1 \cdot t) + \sin(\omega_2 \cdot t)$$

3. **Two-frequency nonlinear mixture**
$$y(t) = \sin(\omega_1 \cdot t) + \sin(\omega_2 \cdot t) + \sin(\omega_1 \cdot t) \cdot \sin(\omega_2 \cdot t)$$
 
The goal is to infer:
- $\omega$ in the single-frequency case,
- unordered pairs $\{\omega_1, \omega_2\}$ in the two-frequency cases.

## Implemented models

### Transformer encoder
- Encoder-only Transformer
- Linear input embedding
- Positional encoding

### Output heads

1. **Deterministic regression**
   - Fully connected head
   - Trained with MSE loss

2. **Probabilistic head (Normalizing Flows)**
   - Masked Autoregressive Flow (MAF)
   - Transformer output used as conditioning context
   - Trained by **minimizing negative log-likelihood**
   - Supports uncertainty estimation via sampling

Permutation ambiguity for two-frequency inference is handled during training and evaluation.


## Note on code style

The scripts in this repo were written for experimentation and reporting.
They are not professionally organized as reusable Python modules and may require manual edits
(paths, configs, plotting calls) depending on your setup.
Everything was runned with cude device

## Repository structure

```text
.
├── dataset_creation.py      # Synthetic dataset generation
├── train_and_test.py        # Training and evaluation loops
├── visualizations.py        # Plotting utilities
├── configuration.py         # Experiment configuration and settings
├── analysis.py              # Post-processing and metrics (messy file)
├── main.py                  # Prepared final models
├── notebooks/               # Exploratory notebooks
├── plots/                   # Figures and tables
└── README.md