# Chapter 07 — Normalization and Residuals

> Two structural choices — residual connections and layer normalization — are what make training 96-layer networks possible. Without them, gradients vanish or explode before reaching the first layer.

---

## 7.1 The deep network training problem

As neural networks get deeper, two pathological things happen during backpropagation:

**Vanishing gradients:** gradients shrink as they flow backward through many layers. With 96 layers, a gradient from the loss that starts at magnitude 1 may arrive at layer 1 with magnitude $10^{-20}$. Parameters in early layers receive no update signal.

**Exploding gradients:** the opposite. Gradients multiply as they flow back; with unfortunate weight initialization or scaling, they become astronomically large and destabilize training.

The solutions: **residual connections** address vanishing gradients; **layer normalization** addresses both, keeping activations in a stable range throughout training.

---

## 7.2 Residual connections (skip connections)

Introduced by He et al. (ResNets, 2015) for image classification, immediately adopted for Transformers.

**Without residuals:**
$$\mathbf{x}^{(l+1)} = F^{(l)}(\mathbf{x}^{(l)})$$

**With residuals:**
$$\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + F^{(l)}(\mathbf{x}^{(l)})$$

The gradient through a residual connection:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l+1)}} \cdot \frac{\partial \mathbf{x}^{(l+1)}}{\partial \mathbf{x}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l+1)}} \cdot \left(1 + \frac{\partial F^{(l)}}{\partial \mathbf{x}^{(l)}}\right)$$

The $+1$ term ensures that even if $\frac{\partial F}{\partial \mathbf{x}} \approx 0$ (the sublayer contributes nothing), the gradient still flows back with magnitude equal to $\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l+1)}}$.

**Unrolling across all $L$ layers:**

The gradient from the loss back to layer 1 is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(0)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(L)}} \cdot \prod_{l=0}^{L-1} \left(1 + \frac{\partial F^{(l)}}{\partial \mathbf{x}^{(l)}}\right)$$

This product can contain many terms $> 1$ as well as terms $< 1$. The $+1$ makes it much less likely to vanish than a pure product $\prod \frac{\partial F}{\partial \mathbf{x}}$.

**Intuition:** the residual connection creates a "highway" for gradients — they can travel directly from the loss to any layer without passing through all the sublayer Jacobians.

---

## 7.3 Layer Normalization

### Why not batch normalization?

**BatchNorm** (Ioffe & Szegedy, 2015): normalize across the *batch* dimension for each feature. For a batch of sequences of length $n$ with $d$-dimensional features:

$$\text{BN}(x_{b,t,d}) = \gamma_d \cdot \frac{x_{b,t,d} - \mu_d}{\sigma_d + \varepsilon} + \beta_d$$

where $\mu_d, \sigma_d$ are computed across the batch dimension $b$.

**Problems for sequences:**
- Batch statistics depend on batch size — unstable with small batches
- Sequences have variable length — you'd need padding and masking
- At inference, you use running statistics from training, which may not match the test distribution
- Doesn't work at all for autoregressive inference (you're processing one token at a time)

#### BatchNorm vs. LayerNorm: Architectural Comparison

| Dimension | Batch Normalization (BatchNorm) | Layer Normalization (LayerNorm) |
|---|---|---|
| **Normalization Axis** | Across batch samples ($B$) for each feature | Across feature dimension ($D$) for each token |
| **Batch Size Sensitivity** | High (degrades with small batch sizes) | Zero (identical computation whether batch size is 1 or 1,000) |
| **Variable Sequence Lengths** | Prone to corruption from zero-padding tokens | Clean (computes statistics strictly within each token's vector) |
| **Inference Behavior** | Requires tracking running mean/variance during training | Identical exact calculation during training and inference |
| **Autoregressive Inference** | Inapplicable (single-token generation has $B=1, L=1$) | Native fit for sequential token generation |

Because language models must handle dynamic context lengths and perform single-token autoregressive generation at inference, Layer Normalization is mathematically and operationally superior to BatchNorm for sequence modeling.

### Layer Normalization

**LayerNorm** (Ba et al., 2016): normalize across the *feature* dimension for each token independently.

$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \qquad \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$$

$$\text{LN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \boldsymbol{\beta}$$

where:
- $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ are learned scale and shift (initialized to $\mathbf{1}$ and $\mathbf{0}$ respectively)
- $\varepsilon \approx 10^{-5}$ for numerical stability
- Normalization is per-token, across the $d_{\text{model}}$ dimension

**Properties:**
- Independent of batch size — works with batch size 1 or at inference
- Independent of sequence length — each token is normalized by its own statistics
- Differentiable everywhere

After normalization, the output has zero mean and unit variance *before* the learned scale $\gamma$ and shift $\beta$ are applied. $\gamma$ and $\beta$ allow the model to learn the optimal scale and offset for each dimension.

---

## 7.4 Pre-LN vs Post-LN

There are two places to put LayerNorm relative to the sublayer and residual.

### Post-LN (original Transformer)

$$\mathbf{x}^{(l+1)} = \text{LN}\!\left(\mathbf{x}^{(l)} + \text{Attn}(\mathbf{x}^{(l)})\right)$$
$$\mathbf{x}^{(l+2)} = \text{LN}\!\left(\mathbf{x}^{(l+1)} + \text{FFN}(\mathbf{x}^{(l+1)})\right)$$

The normalization is applied *after* adding the residual. This was the original formulation.

**Problem:** early in training, the residual $\mathbf{x}^{(l)} + \text{Attn}(\mathbf{x}^{(l)})$ can have very large magnitude, causing the normalization to see extreme values and produce unstable gradients. Post-LN models require careful warmup (slowly increasing learning rate from near-zero) to stabilize early training.

### Pre-LN (GPT-2 onwards)

$$\mathbf{x}^{(l)} \leftarrow \mathbf{x}^{(l)} + \text{Attn}\!\left(\text{LN}(\mathbf{x}^{(l)})\right)$$
$$\mathbf{x}^{(l)} \leftarrow \mathbf{x}^{(l)} + \text{FFN}\!\left(\text{LN}(\mathbf{x}^{(l)})\right)$$

The normalization is applied *inside* the residual, before the sublayer. The residual path ($\mathbf{x}^{(l)} +$) is clean — no normalization on it.

**Advantage:** the gradient through the residual path is unimpeded by normalization operations. This makes training much more stable without requiring careful warmup. Most large models (GPT-2, GPT-3, Llama, PaLM) use Pre-LN.

**Disadvantage:** Pre-LN models have slightly worse final perplexity than Post-LN models when both are trained to convergence with perfect tuning. The stability gain of Pre-LN is usually worth this small cost.

### Final LayerNorm

Pre-LN models add one more LayerNorm at the very end (after all transformer blocks, before the output head). Without it, the final hidden state magnitude could vary widely across tokens and positions.

---

## 7.5 RMSNorm (Root Mean Square Layer Normalization)

While LayerNorm solved the sequence dependency problem, modern LLMs (LLaMA 1–3, Mistral, Gemma, Qwen, DeepSeek, PaLM) have almost universally replaced LayerNorm with **RMSNorm** (Zhang & Sennrich, 2019).

### The Mathematical Formulation

Standard LayerNorm computes both the mean ($\mu$) and variance ($\sigma^2$), performing two transformations:
1. **Centering:** $\mathbf{x} - \mu$ (shifts distribution to zero mean)
2. **Scaling:** $\frac{1}{\sqrt{\sigma^2 + \varepsilon}}$ (scales variance to 1)

Zhang & Sennrich demonstrated that the primary performance stabilization of LayerNorm originates entirely from the **scaling** operation (which regulates gradient scale across layers), while the **centering** operation adds computational overhead with negligible regularizing benefit.

RMSNorm discards the mean calculation entirely and normalizes solely by the root-mean-square of the activations:

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \varepsilon}$$

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}$$

Notice that RMSNorm also typically eliminates the learned bias vector $\boldsymbol{\beta}$ (only learning the scaling parameter $\boldsymbol{\gamma}$), reducing parameter counts and memory transfers.

### Why Modern LLMs Prefer RMSNorm

1. **Elimination of Synchronization & Memory Passes:** In modern GPU kernels (CUDA / Triton), computing LayerNorm requires calculating the mean, writing/caching it, calculating variance, subtracting the mean, and scaling. RMSNorm requires only a single sum-of-squares reduction pass, reducing GPU memory access overhead.
2. **Speed:** Yields a **10% to 50% speedup** in normalization kernel runtime without any degradation in language modeling perplexity or downstream benchmark accuracy.
3. **Scale Invariance:** If all activations in $\mathbf{x}$ are multiplied by a scalar $\alpha$, $\text{RMS}(\alpha \mathbf{x}) = \alpha \text{RMS}(\mathbf{x})$, so $\text{RMSNorm}(\alpha \mathbf{x}) = \text{RMSNorm}(\mathbf{x})$—preserving scale invariance.

### Step-by-Step Numeric Comparison: LayerNorm vs. RMSNorm

Suppose a 4-dimensional hidden vector for a token is $\mathbf{x} = [2.0, 4.0, 6.0, 8.0]$, with $\boldsymbol{\gamma} = [1, 1, 1, 1]$ and $\varepsilon = 0$.

* **LayerNorm Calculation:**
  * Mean: $\mu = (2 + 4 + 6 + 8) / 4 = 5.0$
  * Centered values: $\mathbf{x} - \mu = [-3.0, -1.0, 1.0, 3.0]$
  * Variance: $\sigma^2 = [(-3)^2 + (-1)^2 + 1^2 + 3^2] / 4 = [9 + 1 + 1 + 9] / 4 = 20 / 4 = 5.0$
  * Standard deviation: $\sigma = \sqrt{5.0} \approx 2.236$
  * Normalized vector: $\frac{\mathbf{x} - \mu}{\sigma} \approx [-1.342, -0.447, 0.447, 1.342]$

* **RMSNorm Calculation:**
  * Mean of squared values: $\frac{1}{4} (2^2 + 4^2 + 6^2 + 8^2) = \frac{1}{4} (4 + 16 + 36 + 64) = \frac{120}{4} = 30.0$
  * Root Mean Square: $\text{RMS}(\mathbf{x}) = \sqrt{30.0} \approx 5.477$
  * Normalized vector: $\frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} = [2/5.477, 4/5.477, 6/5.477, 8/5.477] \approx [0.365, 0.730, 1.095, 1.461]$

Both maintain activation stability, but RMSNorm achieves it without the subtractive mean-centering step, saving valuable GPU bandwidth.

---

## 7.6 The full block

Putting it all together, one Transformer block in modern models (Pre-LN + RMSNorm):

```
Input: x

Attention block:
  x_norm = RMSNorm(x)
  attn_out = MultiHeadAttention(x_norm)
  x = x + attn_out          ← residual connection

FFN block:
  x_norm = RMSNorm(x)
  ffn_out = FFN(x_norm)
  x = x + ffn_out           ← residual connection

Output: x
```

The residual connections ensure gradient highways. The RMSNorm ensures stable activation magnitudes going into each sublayer.

---

## 7.7 Initialization matters

Residuals and normalization help, but initialization still matters significantly.

**GPT-2 initialization** (Radford et al., 2019): the output projection $W_O$ in attention and $W_2$ in FFN are scaled by $1/\sqrt{N_L}$ where $N_L$ is the number of layers. This ensures that at initialization, the sum of many residual contributions doesn't grow with depth:

$$W_O \sim \mathcal{N}\!\left(0, \frac{\sigma^2}{N_L}\right)$$

Without this scaling, deeper models would have larger activation norms at initialization, requiring smaller learning rates and longer warmups.

---

## Summary

| Concept | Key point |
|---------|-----------|
| Residual connection | $\mathbf{x} \leftarrow \mathbf{x} + F(\mathbf{x})$ — gradient highway via $+1$ in chain rule |
| LayerNorm | Normalize per-token across $d_{\text{model}}$, then learn $\gamma, \beta$ |
| Post-LN | Original; slightly better quality; training instability |
| Pre-LN | Modern standard; stable training without warmup |
| RMSNorm | Simplified LN without mean subtraction; 30% faster; used in Llama |
| Initialization | Scale residual outputs by $1/\sqrt{N_L}$ to control activation magnitude |

**Next:** [Chapter 08 — Training](08-training.md)
