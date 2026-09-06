# Chapter 05 — Positional Encoding

> Self-attention is order-blind. Adding position information is not optional — it's what makes the model see sentences instead of bags of words.

---

## 5.1 The permutation equivariance problem

Self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$. All three are linear functions of $X$.

**Claim:** if you permute the rows of $X$ (i.e., reorder the tokens), the output rows permute in exactly the same way.

**Proof sketch:** let $P$ be a permutation matrix. Then $PX$ is the reordered input.

$$Q' = PXW_Q = PQ, \quad K' = PK, \quad V' = PV$$
$$S' = Q'K'^\top = PQK^\top P^\top = PSP^\top$$
$$A' = \text{softmax}(S') = P \cdot \text{softmax}(S) \cdot P^\top = PAP^\top$$
$$\text{Output}' = A'V' = PAP^\top PV = PAV = P \cdot \text{Output}$$

The output is permuted in the same way as the input. This means the model cannot distinguish "Dog bites man" from "Man bites dog" — they produce the same set of output vectors, just reordered.

Position information must be injected explicitly.

---

## 5.2 Sinusoidal positional encoding

Vaswani et al. (2017) proposed adding a fixed, non-learned position signal to each token embedding:

$$\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

where:
- $\text{pos} \in \{0, 1, 2, \ldots, n-1\}$ is the token index
- $i \in \{0, 1, \ldots, d_{\text{model}}/2 - 1\}$ is the dimension index

The input to layer 1 is:

$$\mathbf{x}_{\text{pos}} = E[\text{token\_id}] + \text{PE}(\text{pos})$$

### Why sinusoids?

**Different frequencies per dimension:** dimension $2i$ oscillates with period $2\pi \cdot 10000^{2i/d_{\text{model}}}$:
- $i=0$: period = $2\pi \approx 6.3$ tokens (high frequency, captures adjacent tokens)
- $i=d/4$: period ≈ $2\pi \cdot 100 \approx 628$ tokens (medium)
- $i=d/2-1$: period ≈ $2\pi \cdot 10000 \approx 62{,}832$ tokens (very low frequency, captures long-range)

This creates a unique binary-like fingerprint for each position across many scales simultaneously — similar to how binary numbers encode position using bits of different "frequencies" (2⁰, 2¹, 2², ...).

**Worked example** ($d_{\text{model}} = 4$, positions 0–4):

| pos | dim 0 (sin, period≈6) | dim 1 (cos) | dim 2 (sin, period≈628) | dim 3 (cos) |
|-----|----------------------|-------------|------------------------|-------------|
| 0 | 0.000 | 1.000 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 | 0.010 | 1.000 |
| 2 | 0.909 | -0.416 | 0.020 | 1.000 |
| 3 | 0.141 | -0.990 | 0.030 | 1.000 |
| 4 | -0.757 | -0.654 | 0.040 | 1.000 |

Each row is a distinct vector, and nearby positions have similar (but not identical) encodings.

---

## 5.3 The linear relationship between positions

A key theoretical property: for any fixed offset $k$, there exists a linear transformation $T_k$ such that:

$$\text{PE}(\text{pos} + k) = T_k \cdot \text{PE}(\text{pos})$$

**Proof:** For a single frequency $\omega$:

$$\begin{pmatrix} \sin((\text{pos}+k)\omega) \\ \cos((\text{pos}+k)\omega) \end{pmatrix} = \begin{pmatrix} \cos(k\omega) & \sin(k\omega) \\ -\sin(k\omega) & \cos(k\omega) \end{pmatrix} \begin{pmatrix} \sin(\text{pos}\cdot\omega) \\ \cos(\text{pos}\cdot\omega) \end{pmatrix}$$

This is a 2D rotation matrix with angle $k\omega$. Since each pair of dimensions follows this pattern, the full PE transformation is block-diagonal rotation matrices — a linear operation.

**Why this matters for attention:** the dot product $\text{PE}(\text{pos}_i) \cdot \text{PE}(\text{pos}_j)$ depends only on $|\text{pos}_i - \text{pos}_j|$, not on the absolute positions. Attention weights can therefore encode relative distance without needing to learn absolute position representations.

---

## 5.4 Learned positional embeddings

BERT and some early GPT models used **learned positional embeddings**: a trainable matrix $P \in \mathbb{R}^{L_{\max} \times d_{\text{model}}}$ where each row is learned during training.

$$\mathbf{x}_{\text{pos}} = E[\text{token\_id}] + P[\text{pos}]$$

Advantages:
- Can learn position patterns specific to the training data
- Slightly better performance than sinusoidal on fixed-length tasks

Disadvantages:
- **Fixed maximum length:** $L_{\max}$ is a hyperparameter. The model cannot generalize to sequences longer than seen during training.
- Sinusoidal PE can theoretically handle arbitrary length (though performance degrades in practice)

---

## 5.5 Rotary Position Embedding (RoPE)

RoPE (Su et al., 2021) is the most widely used positional encoding in modern LLMs (Llama, Mistral, Qwen, Falcon, etc.). It encodes position by **rotating** the query and key vectors, not by adding to the input.

### Core idea

For a pair of dimensions $(q_{2i}, q_{2i+1})$ in the query vector at position $m$, apply a rotation:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

where $\theta_i = 10000^{-2i/d_k}$ (same base as sinusoidal PE).

Apply the same rotation to keys at their respective positions.

### The key property

The dot product between a rotated query at position $m$ and a rotated key at position $n$ depends only on their **relative distance** $m - n$:

$$Q_m^\top K_n = f(\mathbf{q}, \mathbf{k}, m-n)$$

This is exactly what we want: attention weights are a function of relative position, not absolute position. The model learns to attend based on "how far away" a token is, not "what absolute index" it has.

### Why RoPE beats sinusoidal

1. **Relative by construction:** dot products encode relative position inherently. With sinusoidal PE, the model has to *learn* to use relative position from absolute encodings.
2. **Length extrapolation:** with techniques like NTK-aware scaling or YaRN, RoPE models can be extended to contexts much longer than trained on. Llama 3.1 was trained at 128k context with extended RoPE.
3. **No overhead:** rotation is applied to Q and K inside attention, adding no extra parameters or computation beyond the rotation itself.

---

## 5.6 ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2021) takes a different approach: instead of modifying embeddings, it adds a **penalty to attention scores** based on distance.

$$S_{ij} \leftarrow S_{ij} - m_h \cdot |i - j|$$

where $m_h$ is a head-specific slope (different for each of the $h$ heads). Closer tokens get less penalty, so the model naturally attends more to nearby tokens.

Advantages:
- Zero additional parameters
- Excellent length extrapolation (MPT-7B was trained at 2k context but works at 65k)
- Simple to implement

Disadvantages:
- Hardcoded locality bias — may underperform RoPE for tasks requiring long-range dependencies
- Less flexible than RoPE for different types of positional signals

---

## 5.7 NTK-aware scaling for context extension

A common problem: you want to use a model at 4× its training context length. Just using the model breaks down because positions > $L_{\text{train}}$ are out of distribution.

**NTK-aware scaling** (bloc97, 2023): scale the base of the RoPE frequencies. Instead of $\theta_i = 10000^{-2i/d}$, use:

$$\theta_i' = (10000 \cdot \alpha)^{-2i/d}$$

where $\alpha = L_{\text{extended}} / L_{\text{train}}$. This preserves the high-frequency dimensions (which encode local structure) while stretching the low-frequency dimensions (which encode long-range structure) to cover the larger context.

This allows inference at 4–8× the training context with minimal performance degradation, without any fine-tuning.

---

## Summary

| Method | Encoding | Learned? | Max length | Relative? | Used by |
|--------|----------|----------|------------|-----------|---------|
| Sinusoidal | Add to input | No | Arbitrary | Implicit | Original Transformer |
| Learned absolute | Add to input | Yes | Fixed | No | BERT, early GPT |
| RoPE | Rotate Q, K | No (base fixed) | Extendable | Yes | Llama, Mistral, Qwen |
| ALiBi | Bias scores | No | Extendable | Yes | MPT, Falcon |

**Next:** [Chapter 06 — Feed-Forward Networks](06-feed-forward.md)
