# vps-virtual-parameter-synthesis
Virtual Parameter Synthesis (VPS) experiments for improving the internal mathematical and reasoning capabilities of LLMs

Here is link of colab's whole testing and experiments: https://colab.research.google.com/drive/1WpErVF1rGpzTJi_vm1doaG3N1_aBT1eS
This VPS configuration is **also used in our Idef-mathematics A6 model**, which is built on top of **Qwen**.  
In internal evaluations, **A6** achieved a **+4–5% improvement in reasoning accuracy** on **IMO-inspired geometric problems** compared to the previous **A5** model.

🔗 Learn more about our **Geometry Intelligence** system here:  
https://idef-mathematics.com/



## Abstract

Virtual Parameter Sharpening (VPS) is an inference-time technique that augments frozen transformer linear layers with dynamic low-rank perturbations. Unlike fine-tuning or standard LoRA, VPS constructs its low-rank factors *on-the-fly* from activation statistics and optional gradient signals, enabling test-time adaptation without persistent parameter updates. This document provides a rigorous analysis of the mathematical foundations, architectural design, and mechanisms by which VPS may improve reasoning performance.

---

## 1. Core Mathematical Framework

### 1.1 Modified Forward Pass

For a standard linear layer with weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$, the VPS-augmented forward pass is:

$$y = Wx + \gamma \cdot (xA)B^\top$$

where:
- $A \in \mathbb{R}^{d_{in} \times r}$ and $B \in \mathbb{R}^{d_{out} \times r}$ are dynamically constructed low-rank factors
- $\gamma \in [0, 1]$ is a scaling coefficient (adaptive or fixed)
- $r \ll \min(d_{in}, d_{out})$ is the perturbation rank

The factors are derived from the frozen weights via:

$$A = W^\top V, \quad B = WU$$

where $U \in \mathbb{R}^{d_{in} \times r}$ and $V \in \mathbb{R}^{d_{out} \times r}$ are **selector matrices** constructed by the builder subsystem.

### 1.2 Effective Perturbation Structure

Substituting the definitions, the full output becomes:

$$y = Wx + \gamma \cdot x(W^\top V)(WU)^\top = Wx + \gamma \cdot xW^\top V U^\top W^\top$$

This is equivalent to:

$$y = \left(W + \gamma \cdot (W^\top V U^\top W^\top)^\top \right) x = \left(W + \gamma \cdot W U V^\top W\right)x$$

**Key observation**: The perturbation $\Delta W = \gamma \cdot W U V^\top W$ is *weight-dependent*, meaning VPS does not add arbitrary noise but rather amplifies directions that are already encoded in $W$. This is fundamentally different from additive LoRA ($W + BA$) where $B, A$ are independent parameters.

### 1.3 Interpretation via Projection Operators

When $U$ and $V$ are sparse one-hot selectors (as in the SK builder), they act as projection operators onto subspaces of the input and output feature dimensions. Let $\Pi_U = UU^\top$ and $\Pi_V = VV^\top$ denote the projection matrices. Then:

$$\Delta W \propto W \Pi_U \Pi_V W$$

This selectively amplifies the interaction between specific input features (selected by $U$) and output features (selected by $V$) through the lens of the existing weight structure.

---

## 2. Builder Subsystem: Constructing U and V

The builder subsystem determines which input/output dimensions receive perturbation. Three strategies are implemented:

### 2.1 SK Builder (Sparse Selector)

**Algorithm**:
1. Compute input activation scores: $s_{in}^{(j)} = \frac{1}{N}\sum_{i=1}^N |x_{ij}|$
2. Compute output activation scores: $s_{out}^{(k)} = \frac{1}{N}\sum_{i=1}^N |[xW^\top]_{ik}|$
3. Select top-$k$ indices for each: $\mathcal{I}_{in} = \text{top-}k(s_{in})$, $\mathcal{I}_{out} = \text{top-}k(s_{out})$
4. Construct one-hot matrices: $U_{j,c} = \mathbf{1}[j = \mathcal{I}_{in}^{(c)}]$, $V_{k,c} = \mathbf{1}[k = \mathcal{I}_{out}^{(c)}]$

**Mathematical rationale**: By selecting high-activation dimensions, VPS focuses its perturbation budget on the feature subspace most relevant to the current input. This is a form of *activation-guided sparsity* that avoids wasting capacity on dormant dimensions.

### 2.2 SC Builder (Sylvester Coupling)

The SC builder refines the SK selection by solving a ridge regression that couples input activations to output activations:

**Algorithm** (after obtaining SK selections):
1. Extract compact views: $X_A \in \mathbb{R}^{N \times r}$ (activations at selected input indices), $Y \in \mathbb{R}^{N \times r}$ (outputs at selected output indices)
2. Solve the ridge system: $(X_A^\top X_A + \alpha I)T = X_A^\top Y$
3. Mix columns of $V$: $\tilde{V} = V T^\top$, then column-normalize

**Mathematical rationale**: This solves a local least-squares problem that finds a linear coupling $T$ between the selected input and output subspaces. The resulting $\tilde{V}$ aligns the output selector with directions that best predict the output from the input, effectively learning a task-relevant rotation within the selected subspace.

The ridge regularization $\alpha I$ ensures numerical stability and prevents overfitting to the current batch.

### 2.3 Hybrid Builder

Uses SC when gradient information is available (indicating an optimization context), otherwise falls back to SK. This provides adaptive complexity based on available signals.

---

## 3. Spectral Norm Control

### 3.1 Per-Component Clipping

To prevent the low-rank perturbation from destabilizing the forward pass, each rank-1 component is clipped:

$$\text{For each } i: \quad \text{if } \|A_{:,i}\| \cdot \|B_{:,i}\| > \tau, \quad \text{scale both by } \sqrt{\frac{\|A_{:,i}\| \cdot \|B_{:,i}\|}{\tau}}$$

This ensures $\|A_{:,i}\| \cdot \|B_{:,i}\| \leq \tau$ for all $i$.

**Mathematical rationale**: The spectral norm of the perturbation $\Delta = AB^\top$ satisfies:

$$\|\Delta\|_2 \leq \|A\|_F \|B\|_F \leq \sqrt{r} \cdot \max_i (\|A_{:,i}\| \cdot \|B_{:,i}\|)$$

By bounding each rank-1 term, we obtain $\|\Delta\|_2 \leq \sqrt{r} \cdot \tau$. For small $r$ and $\tau < 1$, this keeps the perturbation in a controlled regime where it modifies but does not overwhelm the base computation.

---

## 4. Adaptive Policy System

### 4.1 Energy-Based Scaling

The policy module computes a batch-level "energy" statistic:

$$E = \frac{1}{N \cdot d}\sum_{i,j} h_{ij}^2$$

where $h = Wx$ is the hidden activation. A scale factor is derived via:

$$\sigma = 1 - e^{-E}$$

This maps energy $E \in [0, \infty)$ to scale $\sigma \in [0, 1)$, with the following behavior:
- Low energy (near-zero activations) → $\sigma \approx 0$ → minimal perturbation
- High energy (strongly activated) → $\sigma \approx 1$ → full perturbation

### 4.2 Entropy-Aware Adjustment

Token-level entropy $H$ from the output logits is incorporated:

$$\sigma \leftarrow \max(\sigma, \min(1, H/3))$$

**Rationale**: High entropy indicates model uncertainty. In uncertain states, increasing the perturbation may help the model escape local optima in its implicit reasoning trajectory. This is analogous to simulated annealing where temperature is raised in regions of high uncertainty.

### 4.3 Adaptive Hyperparameter Interpolation

Given bounds $[r_{lo}, r_{hi}]$, $[\gamma_{lo}, \gamma_{hi}]$, etc., the policy interpolates:

$$r = r_{lo} + (r_{hi} - r_{lo}) \cdot \sigma$$
$$\gamma = \gamma_{lo} + (\gamma_{hi} - \gamma_{lo}) \cdot \sigma$$

This provides smooth, input-dependent hyperparameter scheduling without discrete mode switching.

---

## 5. Ephemeral L-BFGS Preconditioning

### 5.1 Mechanism

The system maintains a small memory ($m=5$) of curvature pairs $(s_i, y_i)$ where:
- $s_i$ = step direction from previous iterations
- $y_i$ = gradient difference

The standard L-BFGS two-loop recursion is applied to precondition the delta:

$$\tilde{\delta} = H_k \delta$$

where $H_k$ is the implicit inverse Hessian approximation.

### 5.2 Rationale

This is an unusual application of L-BFGS—typically used in optimization, here repurposed for inference-time conditioning. The intuition is that the curvature information accumulated across inference steps captures second-order structure of the loss landscape. Preconditioning the perturbation by this approximation may align it with directions of high curvature (rapid change), potentially amplifying informative gradient directions.

**Caveat**: In the provided code, this is applied with a small scale factor ($10^{-3}$) and without backpropagation, making it a heuristic enhancement rather than a rigorous optimization step.

---

## 6. Iterative Verification Loop

### 6.1 Multi-Objective Verifier

The `CompositeVerifier` computes a weighted loss across multiple objectives:

| Component | Description | Weight (default) |
|-----------|-------------|------------------|
| `numeric` | $(pred - gold)^2$ on extracted numeric answers | 1.0 |
| `units` | Dimensional consistency check via `pint` | 0.5 |
| `self_consistency` | Variance across multiple samples | 0.3 |
| `algebraic` | Structural match via `sympy` simplification | 0.2 |

Total loss: $\mathcal{L} = \sum_i w_i \cdot \ell_i$

### 6.2 Feedback Loop

The verification loss provides a signal for the policy:

```
improved = (L_current < L_previous)
policy.update_outcome(improved, L_previous - L_current)
```

Recent improvement history influences future scaling decisions, implementing a form of meta-learning across inference iterations.

### 6.3 Gradient-Informed Refinement

A short cross-entropy surrogate is computed:

$$\mathcal{L}_{CE} = \text{CrossEntropy}(\text{logits}[:, -T:], \text{gold\_tokens}[-T:])$$

Backpropagating this populates `grad_h` buffers, which the SC builder can then use to construct more informed selectors. This creates a feedback path from the verification objective to the perturbation construction.

---

## 7. Q/K Coupling in Attention

### 7.1 Mechanism

Query and Key projection layers in attention blocks are paired:
```python
q._peer = k; k._peer = q
q.is_Q = True; k.is_K = True
```

### 7.2 Rationale

In self-attention, $Q = xW_Q$ and $K = xW_K$ interact via $\text{softmax}(QK^\top / \sqrt{d})$. Perturbing $W_Q$ and $W_K$ independently may create inconsistent attention patterns. By coupling them, perturbations can be coordinated to maintain coherent query-key interactions.

This is motivated by the observation that effective attention requires *alignment* between query and key representations—random independent perturbations would likely degrade this alignment.

---

## 8. Why Might VPS Improve Reasoning?

### 8.1 Activation-Conditioned Computation

Unlike static weights, VPS modulates computation based on the *current input*. For reasoning tasks, different problems activate different feature subspaces. VPS's activation-guided selection focuses perturbation on the currently-relevant subspace, potentially:
- Amplifying task-specific signal pathways
- Suppressing irrelevant dimensions that might introduce noise

### 8.2 Implicit Ensemble Effect

The dynamic nature of the perturbation means that each input effectively sees a slightly different "effective model." This provides an implicit ensemble effect without multiple forward passes, which may improve robustness on out-of-distribution reasoning patterns.

### 8.3 Iterative Refinement as Search

The verification loop implements a form of search in output space:
1. Generate candidate
2. Evaluate via multi-objective verifier
3. Update perturbation policy based on feedback
4. Generate refined candidate

This resembles beam search or MCTS in that it explores multiple trajectories, but operates through perturbation of the forward pass rather than explicit tree expansion.

### 8.4 Second-Order Information via L-BFGS

The ephemeral L-BFGS component captures local curvature information. In optimization, preconditioning by the inverse Hessian converts gradient descent into Newton's method, which has superior convergence in convex regions. The heuristic application here may help the perturbation align with "informative" directions in the representation space.

---

## 9. Architectural Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        VPS Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input x ─────────────────────┬─────────────────────► Wx        │
│                               │                         │       │
│                               ▼                         │       │
│                    ┌──────────────────┐                 │       │
│                    │   SK/SC Builder  │                 │       │
│                    │  (constructs U,V)│                 │       │
│                    └────────┬─────────┘                 │       │
│                             │                           │       │
│                             ▼                           │       │
│                    ┌──────────────────┐                 │       │
│                    │  A = W^T V       │                 │       │
│                    │  B = W U         │                 │       │
│                    └────────┬─────────┘                 │       │
│                             │                           │       │
│                             ▼                           │       │
│                    ┌──────────────────┐                 │       │
│                    │ Spectral Clip    │                 │       │
│                    │ (||A_i||·||B_i|| │                 │       │
│                    │    ≤ τ)          │                 │       │
│                    └────────┬─────────┘                 │       │
│                             │                           │       │
│                             ▼                           │       │
│                    ┌──────────────────┐                 │       │
│                    │ δ = (xA)B^T      │                 │       │
│                    └────────┬─────────┘                 │       │
│                             │                           │       │
│                             ▼                           ▼       │
│                    ┌──────────────────────────────────────┐    │
│                    │     y = Wx + γ·δ                     │    │
│                    └──────────────────────────────────────┘    │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────────┐                         │
│                    │  Adaptive Policy │◄──── Token Entropy      │
│                    │  (γ, r, order)   │◄──── Improvement Hist   │
│                    └──────────────────┘◄──── Activation Energy  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rank` | int | 2 | Base rank of low-rank perturbation |
| `topk` | int | 32 | Number of features selected by SK builder |
| `gamma` | float | 0.5 | Perturbation scaling coefficient |
| `tau` | float | 0.8 | Spectral norm clip threshold |
| `builder` | str | "hybrid" | Builder type: "sk", "sc", or "hybrid" |
| `order` | int | 1 | Order of delta expansion (1 or 2) |
| `qk_coupling` | bool | True | Enable Q/K pairing in attention |
| `lbfgs_enabled` | bool | True | Enable ephemeral L-BFGS |
| `adaptive_rank` | bool | True | Enable energy-based rank adaptation |
| `adaptive_gamma` | bool | True | Enable energy-based gamma adaptation |
| `alpha` | float | 1e-3 | Ridge regularization for SC builder |

---



---

## 12. References

- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer. (L-BFGS)
- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.

---

## Appendix A: Module Dependencies

```
vps/
├── vpscore/
│   ├── vps_linear.py      # Core VPSLinear wrapper
│   ├── builders.py        # SK, SC, Hybrid builders
│   ├── policy.py          # Adaptive policy system
│   ├── math_utils.py      # Entropy, spectral utilities
│   ├── ephemeral_lbfgs.py # Curvature memory
│   ├── hooks.py           # Activation/gradient capture
│   ├── patch_hf.py        # HuggingFace model patching
│   ├── config.py          # Configuration dataclass
│   └── verifiers/
│       └── composite_verifier.py  # Multi-objective verification
└── scripts/
    ├── infer_vps.py       # Inference with iterative refinement
    └── run_ablations.py   # Ablation study harness
```

---

*Document prepared for technical review. All mathematical formulations derived directly from source code analysis.*
