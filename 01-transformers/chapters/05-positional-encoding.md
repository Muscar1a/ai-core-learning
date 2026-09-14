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

## 5.5 Rotary Position Embedding (RoPE) — Complete Mathematical Breakdown

RoPE (Su et al., 2021) has become the undisputed universal standard for positional representation in modern Large Language Models (including Meta's LLaMA 1–3, Mistral, Gemma, Qwen, and DeepSeek).

### Why Traditional Encodings Fail

Traditional absolute encodings (both sinusoidal and learned) add a position vector directly to the token embedding:
$$\mathbf{x}_{\text{input}} = \mathbf{x}_{\text{token}} + \mathbf{p}_{\text{pos}}$$

This design suffers from three fundamental weaknesses:
1. **Vector Corruption:** Position and semantic meaning are forcibly summed into the exact same vector space, forcing semantic features and positional signals to compete for representation capacity.
2. **Translation Blindness:** Absolute encodings force the network to independently learn from scratch how position pairs relate. There is no built-in inductive bias indicating that tokens at positions $(5, 10)$ share the identical syntactic distance as tokens at $(105, 110)$.
3. **Sequence Length Rigidity:** Learned embeddings cannot process sequences beyond the predetermined training horizon ($L > L_{\text{max}}$).

### The Core Geometric Intuition

RoPE does not add anything. Instead, it **rotates** the Query and Key vectors in 2D coordinate planes:
* Every token at position $m$ has its $Q$ and $K$ vectors rotated by an angle directly proportional to $m$.
* When computing attention $Q \cdot K$, the dot product between the two rotated vectors preserves the angle difference $(n - m)$, inherently capturing **relative distance** without requiring learned parameters.
* **$V$ remains unrotated:** Value vectors hold content to be aggregated and do not participate in pairwise distance matching.

### 2D Rotation Matrix Formulation

For any 2D vector $\mathbf{v} = [x, y]^\top$, rotating counter-clockwise by angle $\theta$ preserves its Euclidean length while adjusting direction:
$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

In high-dimensional spaces ($d_k = 64$ or $128$), RoPE splits the vector into $d_k / 2$ orthogonal 2D subspaces:
```
Q = [ q_1, q_2 | q_3, q_4 | ... | q_{d-1}, q_d ]
      \______/   \______/          \_________/
       pair 1     pair 2            pair d/2
         |           |                  |
         v           v                  v
      rotate      rotate             rotate
      by m*θ_1    by m*θ_2           by m*θ_{d/2}
      (fastest)                      (slowest)
```

The geometric frequencies decrease exponentially across pairs:
$$\theta_i = 10000^{-2(i-1)/d_k}, \quad \text{for } i = 1, 2, \dots, d_k/2$$
Low-index pairs rotate rapidly (capturing local token adjacencies), while high-index pairs rotate slowly (capturing long-range contextual themes).

### Derivation: Why Absolute Positions Cancel Out

Let $R_m$ denote the block-diagonal rotation matrix for position $m$. The attention logit between token $m$ (query) and token $n$ (key) is:
$$\text{Score} = (R_m \mathbf{q}) \cdot (R_n \mathbf{k}) = (R_m \mathbf{q})^\top (R_n \mathbf{k}) = \mathbf{q}^\top R_m^\top R_n \mathbf{k}$$

Because rotation matrices are orthogonal, $R_m^\top = R_m^{-1} = R_{-m}$. Rotating by $-m$ followed by $+n$ combines into a single net rotation of $n - m$:
$$R_m^\top R_n = R_{-m} R_n = R_{n - m}$$

Substituting back:
$$\text{Score} = \mathbf{q}^\top R_{n - m} \mathbf{k} = \mathbf{q} \cdot (R_{n - m} \mathbf{k})$$

**The absolute coordinates $m$ and $n$ vanish completely.** The attention affinity is purely a function of the relative separation distance $\Delta = n - m$.

### Concrete Numeric Proof of Translation Invariance

Let's test this invariance with concrete numbers:
Let $\mathbf{q} = [1, 0]$ and $\mathbf{k} = [1, 0]$. Let the frequency $\theta = \pi/4$ ($45^\circ$).

**Case A: Positions $m=1, n=2$ (Distance $\Delta = 1$)**
* Rotation for $m=1$ ($\theta = \pi/4$):
  $$R_1 \mathbf{q} = \begin{bmatrix} \cos(\pi/4) & -\sin(\pi/4) \\ \sin(\pi/4) & \cos(\pi/4) \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.707 \\ 0.707 \end{bmatrix}$$
* Rotation for $n=2$ ($2\theta = \pi/2$):
  $$R_2 \mathbf{k} = \begin{bmatrix} \cos(\pi/2) & -\sin(\pi/2) \\ \sin(\pi/2) & \cos(\pi/2) \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$
* Dot product score:
  $$\text{Score}_A = [0.707, 0.707] \cdot [0, 1] = (0.707 \times 0) + (0.707 \times 1) = \mathbf{0.707}$$

**Case B: Shift entire sentence forward by 1 token: $m=2, n=3$ (Distance $\Delta = 1$)**
* Rotation for $m=2$ ($2\theta = \pi/2$):
  $$R_2 \mathbf{q} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$
* Rotation for $n=3$ ($3\theta = 3\pi/4$):
  $$R_3 \mathbf{k} = \begin{bmatrix} \cos(3\pi/4) & -\sin(3\pi/4) \\ \sin(3\pi/4) & \cos(3\pi/4) \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} -0.707 \\ 0.707 \end{bmatrix}$$
* Dot product score:
  $$\text{Score}_B = [0, 1] \cdot [-0.707, 0.707] = (0 \times -0.707) + (1 \times 0.707) = \mathbf{0.707}$$

$\text{Score}_A = \text{Score}_B = 0.707$ exactly. Regardless of where the two words appear in a 100,000-token document, if their relative offset is 1 token, their positional contribution to the attention score is identical.

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
