# Chapter 06 — Feed-Forward Networks

> After attention gathers information across tokens, each token "thinks" about what it learned. This is the FFN's job — and where most of the model's factual knowledge lives.

---

## 6.1 Architecture

The feed-forward network (FFN) applied to each token independently:

$$\text{FFN}(\mathbf{x}) = \sigma(\mathbf{x} W_1 + \mathbf{b}_1)\, W_2 + \mathbf{b}_2$$

where:
- $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ — one token's hidden state
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ — expand
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ — compress
- $\sigma$ — activation function (ReLU, GELU, or SwiGLU)
- $d_{\text{ff}} = 4 \times d_{\text{model}}$ — the standard ratio

**Why independent per token?** Attention already handled cross-token communication. The FFN applies the same transformation to each token's representation, refining it using the gathered context but without mixing tokens further.

**Why $d_{\text{ff}} = 4 \times d_{\text{model}}$?** This is empirical. The original Transformer paper used this ratio, subsequent scaling studies confirmed it's close to optimal given the attention/FFN parameter allocation. The expansion creates a bottleneck structure: project up (giving more dimensions to work in), apply nonlinearity (apply complex function), project back down.

---

## 6.2 Activation functions

### ReLU (original)

$$\text{ReLU}(x) = \max(0, x)$$

Simple and fast. Hard gate: any negative value is zero. Problem: the "dying ReLU" — if a unit's pre-activation is always negative, it never activates and its incoming weights receive no gradient signal. They die.

### GELU (GPT-2, BERT, most modern models)

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

where $\Phi$ is the CDF of the standard normal distribution.

GELU is a **soft gate**: instead of a hard threshold at 0, GELU smoothly weights the input based on how likely it is to be positive under the standard normal. For $x \gg 0$, $\text{GELU}(x) \approx x$. For $x \ll 0$, $\text{GELU}(x) \approx 0$. The transition is smooth.

Practical approximation (used in code):
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{2/\pi}\left(x + 0.044715x^3\right)\right]\right)$$

GELU outperforms ReLU empirically on language tasks. The smooth transition preserves gradient flow through near-zero activations.

### SwiGLU (Llama, PaLM)

$$\text{SwiGLU}(\mathbf{x}) = \text{Swish}(\mathbf{x} W_1) \odot (\mathbf{x} W_3)$$
$$\text{Swish}(x) = x \cdot \sigma(x)$$

where $\sigma$ is the sigmoid function and $\odot$ is element-wise multiplication.

SwiGLU uses a **gating mechanism**: one linear projection computes "values" and another computes "gates". The gate (Swish activation) modulates the values element-wise. This is a form of the **Gated Linear Unit (GLU)** family.

SwiGLU requires a third weight matrix $W_3 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ but uses a smaller $d_{\text{ff}}$ to compensate:
- Standard: $d_{\text{ff}} = 4 d_{\text{model}}$ with two matrices ($W_1, W_2$)
- SwiGLU: $d_{\text{ff}} \approx 2.67 d_{\text{model}}$ with three matrices ($W_1, W_2, W_3$) — same total params

Noam Shazeer (2020) showed SwiGLU consistently outperforms GELU by ~1-2% perplexity. It's now the default in most state-of-the-art models.

---

## 6.3 Why expand then compress?

The two-layer structure with expansion might seem inefficient. Why not a single square matrix $W \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$?

**Universal approximation requires nonlinearity.** A single linear layer, regardless of size, is still linear — it can only compute linear functions of its input. Two linear layers composed = still one linear layer. The nonlinearity $\sigma$ in the middle is what allows the FFN to compute non-linear functions.

**The expansion creates representational capacity.** Consider: the model wants to detect whether the input contains the pattern "animal at position subject + verb + pronoun at position subject_of_later_clause". This is a complex boolean-like combination. In the $d_{\text{ff}}$-dimensional space after the first projection, there are enough dimensions to represent many such patterns simultaneously.

**Bottleneck structure forces compression.** Projecting back to $d_{\text{model}}$ forces the network to discard irrelevant information and keep only what's useful for the next layer. This is an implicit form of regularization.

---

## 6.4 FFN as key-value memory

Geva et al. (2021) — *"Transformer Feed-Forward Layers Are Key-Value Memories"* — showed that FFN layers can be interpreted as storing factual knowledge as a key-value store.

**The construction:**

Write $W_1$ column-wise as keys: $W_1 = [\mathbf{k}_1 | \mathbf{k}_2 | \cdots | \mathbf{k}_{d_{\text{ff}}}]$ where $\mathbf{k}_i \in \mathbb{R}^{d_{\text{model}}}$.

The first layer computes:
$$\mathbf{m} = \sigma(\mathbf{x} W_1) = \sigma\!\left([\mathbf{x} \cdot \mathbf{k}_1, \ldots, \mathbf{x} \cdot \mathbf{k}_{d_{\text{ff}}}]\right)$$

Entry $m_i$ is high when input $\mathbf{x}$ has high dot product with key $\mathbf{k}_i$ — i.e., when the input matches the $i$-th "pattern."

Write $W_2$ row-wise as values: $W_2 = [\mathbf{v}_1; \mathbf{v}_2; \cdots; \mathbf{v}_{d_{\text{ff}}}]$.

The second layer computes:
$$\text{FFN}(\mathbf{x}) = \sum_{i=1}^{d_{\text{ff}}} m_i \cdot \mathbf{v}_i$$

This is a **soft retrieval**: each "memory slot" $i$ has a key $\mathbf{k}_i$ (pattern) and a value $\mathbf{v}_i$ (what to output when pattern matches). The output is a weighted sum of values, weighted by how well the input matches each key.

**Empirical findings:** Geva et al. found that specific FFN neurons (memory slots) respond to identifiable linguistic patterns. One neuron might fire for "female first names following 'Mrs.'", storing as its value the representation for "female human". Another fires for "country capitals" and stores geographic information. The model uses these memories to complete factual statements.

This is why factual editing (changing "The capital of France is **Paris**" to "**Berlin**") requires modifying FFN weights — not attention weights. ROME (Meng et al., 2022) is an algorithm that directly edits specific FFN memories to change model facts.

---

## 6.5 Parameter dominance

For one Transformer block, the parameter counts:

| Component | Parameters |
|-----------|-----------|
| Attention ($W_Q, W_K, W_V, W_O$) | $4 d_{\text{model}}^2$ |
| FFN ($W_1, W_2$) | $2 \cdot d_{\text{model}} \cdot d_{\text{ff}} = 8 d_{\text{model}}^2$ |
| **Ratio** | **FFN has 2× more params than attention** |

Across a full model, FFN parameters account for about 2/3 of total parameters:

- GPT-3 (175B): ~116B in FFN layers, ~58B in attention
- Llama 3 70B: ~47B in FFN (SwiGLU), ~23B in attention

This has implications for model compression: pruning or quantizing FFN layers more aggressively than attention may be acceptable.

---

## 6.6 Mixture of Experts (MoE)

Standard FFN: every token uses the same weight matrices. **Mixture of Experts** (MoE) replaces the single FFN with $n$ expert FFNs, activating only $k$ of them per token.

$$\text{MoE}(\mathbf{x}) = \sum_{i \in \text{top-k}(\text{router}(\mathbf{x}))} g_i(\mathbf{x}) \cdot \text{FFN}_i(\mathbf{x})$$

where $\text{router}(\mathbf{x}) = \text{softmax}(\mathbf{x} W_r) \in \mathbb{R}^n$ assigns scores to experts, and the top-$k$ experts (by score) are activated.

**Mixtral 8x7B** (Mistral AI, 2023): 8 experts, 2 active per token. Total parameters ≈ 46.7B; active parameters per token ≈ 12.9B. Performance similar to Llama 2 70B (a dense model) at 1/3 the inference compute.

**GPT-4** reportedly uses MoE — likely 16 experts with 2 active, enabling the ~1.8T parameter count without proportionally higher inference cost.

**Challenges with MoE:**
- **Load balancing:** without extra loss terms, all tokens route to the same few experts. Auxiliary losses (e.g., encouraging uniform routing) are needed.
- **Communication overhead:** in distributed training, different experts may be on different GPUs — tokens must be routed across devices.
- **Increased total memory:** even though only $k/n$ experts run per token, all $n$ experts must fit in memory.

---

## Summary

| Concept | Key point |
|---------|-----------|
| Architecture | Expand ($\times 4$) → nonlinearity → compress → same dimension |
| ReLU | Hard gate at 0; dying neuron problem |
| GELU | Smooth gate; default for BERT/GPT-2 |
| SwiGLU | Gated variant; default for Llama/PaLM; ~1-2% better |
| Key-value memory | FFN = soft retrieval over learned patterns (Geva 2021) |
| Parameter share | FFN ≈ 2/3 of total model parameters |
| MoE | $n$ expert FFNs, $k$ active per token; same quality, less compute |

**Next:** [Chapter 07 — Normalization and Residuals](07-normalization-residuals.md)
