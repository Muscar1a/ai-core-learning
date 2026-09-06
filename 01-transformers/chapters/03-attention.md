# Chapter 03 — Self-Attention

> The core innovation. Every word looks at every other word simultaneously, and the model learns which relationships matter.

---

## 3.1 The problem attention solves

Consider: "The **animal** didn't cross the street because **it** was too tired."

What does "it" refer to? The animal or the street? You know it's the animal — "tired" applies to animals, not streets. To understand "it", your brain connected it to "animal" across 6 intervening words.

An RNN processing this left-to-right would have to carry the context of "animal" through every intermediate hidden state. By the time it reaches "it", the signal has been diluted by 6 updates. With enough sentence length, it gets lost.

**Self-attention solves this directly**: when computing the representation of "it", the model computes a direct connection to every other token in the sequence — including "animal" — in one step. No intermediate states to decay through.

---

## 3.2 Precursor: Bahdanau attention (2014)

Before self-attention, there was **additive attention** (Bahdanau et al., 2014), used in encoder-decoder models for translation.

Given encoder hidden states $\{h_1, \ldots, h_n\}$ and decoder state $s_t$:

$$e_{ti} = v^\top \tanh(W_1 h_i + W_2 s_t)$$
$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_j \exp(e_{tj})}$$
$$c_t = \sum_i \alpha_{ti} h_i$$

This computes a weighted sum of encoder states, where weights $\alpha_{ti}$ reflect relevance. Two problems:
1. Sequential — decoder can only attend after the full encoder has run
2. Different-source: the query ($s_t$) comes from the decoder, keys/values from the encoder

**Self-attention** generalizes this: query, key, and value all come from the *same* sequence. Every token can attend to every other token in the same forward pass.

---

## 3.3 Query, Key, Value — the abstraction

The Q/K/V abstraction comes from database/information retrieval:

- **Query:** what you're looking for
- **Key:** labels on stored items
- **Value:** the actual content of stored items

When you search a database: your query is matched against keys, and the matching keys return their values.

In attention:
- Each token broadcasts a **Key**: "here's what I contain"
- Each token sends a **Query**: "here's what I'm looking for"
- Each token packages its information as a **Value**: "here's what I contribute if selected"

A token with a Query that matches another token's Key will receive a high weight on that token's Value — pulling in its information.

---

## 3.4 Scaled dot-product attention — derivation

Given input $\mathbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$ (sequence of $n$ token embeddings, each of dimension $d_{\text{model}}$):

**Step 1: Project to Q, K, V**

$$Q = \mathbf{X} W_Q, \quad K = \mathbf{X} W_K, \quad V = \mathbf{X} W_V$$

where $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

Result: $Q, K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$.

**Step 2: Compute attention scores**

$$S = Q K^\top \in \mathbb{R}^{n \times n}$$

Entry $S_{ij}$ = dot product between token $i$'s query and token $j$'s key. High value = token $i$ finds token $j$ relevant.

**Step 3: Scale**

$$S \leftarrow \frac{S}{\sqrt{d_k}}$$

**Step 4: Softmax (row-wise)**

$$A = \text{softmax}(S) \in \mathbb{R}^{n \times n}$$

Each row of $A$ sums to 1. Entry $A_{ij}$ = how much token $i$ attends to token $j$.

**Step 5: Weighted sum of values**

$$\text{Output} = A \cdot V \in \mathbb{R}^{n \times d_v}$$

Each output token is a weighted combination of all value vectors, where weights are the attention scores.

**Combined:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

---

## 3.5 Why scale by $\sqrt{d_k}$?

This is subtle but important. Consider two random vectors $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ with components drawn i.i.d. from $\mathcal{N}(0, 1)$.

Their dot product:
$$\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$

**Mean:** $\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$

**Variance:** $\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - 0 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$

So $\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$.

With $d_k = 128$ (as in GPT-3), raw dot products have standard deviation $\sqrt{128} \approx 11$. Feeding values of magnitude 11 into softmax:

$$\text{softmax}([11, 0, 0, \ldots]) \approx [1, 0, 0, \ldots]$$

The softmax saturates — one entry gets probability ≈ 1, all others ≈ 0. This is a nearly one-hot distribution, meaning the gradient of softmax $\approx 0$ everywhere. **Training stalls.**

Dividing by $\sqrt{d_k}$ brings the dot product variance back to 1, keeping softmax in a useful, differentiable regime.

---

## 3.6 Softmax properties

For a vector $\mathbf{z} \in \mathbb{R}^n$:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Properties:
- Output is a valid probability distribution: all values positive, sum to 1
- **Differentiable everywhere** — essential for backpropagation
- **Temperature:** scaling the input by $1/\tau$ changes sharpness. $\tau \to 0$: one-hot. $\tau \to \infty$: uniform.
- **Shift-invariant:** $\text{softmax}(\mathbf{z}) = \text{softmax}(\mathbf{z} + c)$ for any constant $c$ — used for numerical stability (subtract max before exp)

The gradient of softmax with respect to its input:

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \text{softmax}(\mathbf{z})_i \cdot (\delta_{ij} - \text{softmax}(\mathbf{z})_j)$$

where $\delta_{ij}$ is the Kronecker delta. When softmax saturates (nearly one-hot), this gradient → 0 for almost all pairs — which is why the $\sqrt{d_k}$ scaling is critical.

---

## 3.7 Causal masking for autoregression

In decoder-only models (GPT family), each token should only attend to tokens that came *before* it in the sequence. Otherwise the model could "cheat" during training by looking at future tokens.

Implementation: before the softmax, set $S_{ij} = -\infty$ for all $j > i$ (upper triangle of the score matrix).

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

$$A = \text{softmax}(S + M)$$

Since $e^{-\infty} = 0$, masked positions contribute nothing to the attention-weighted sum. This is called the **causal mask** or **autoregressive mask**.

Encoder-only models (BERT) use no mask — every token can attend to every other token bidirectionally. This is why BERT is better for understanding tasks (classification, NER) while GPT is better for generation.

---

## 3.8 Complexity analysis

The attention score matrix $S \in \mathbb{R}^{n \times n}$ has $n^2$ entries. For a sequence of length $n$ with $d_k$-dimensional keys:

| Operation | Cost |
|-----------|------|
| $Q, K, V$ projections | $\mathcal{O}(n \cdot d_{\text{model}} \cdot d_k)$ |
| Score matrix $QK^\top$ | $\mathcal{O}(n^2 \cdot d_k)$ |
| Softmax | $\mathcal{O}(n^2)$ |
| Output $AV$ | $\mathcal{O}(n^2 \cdot d_v)$ |
| **Total per head** | $\mathcal{O}(n^2 \cdot d)$ |

This $n^2$ term is the bottleneck. Double the context length → 4× the attention compute.

**GPT-4 context: 128k tokens**
- $n^2 = 1.6 \times 10^{10}$ attention values per layer
- At 32 layers: $5 \times 10^{11}$ operations just for attention
- This is why long contexts are expensive and why efficient attention is an active research area

---

## 3.9 What attention actually computes

Each output token is a **weighted average** of value vectors:

$$\text{output}_i = \sum_{j=1}^n A_{ij} \mathbf{v}_j$$

When $A_{ij}$ is high, token $i$'s output representation borrows heavily from token $j$'s value.

In the "animal...it" example:
- Token "it" has a query asking for "what animate noun can I refer to?"
- Token "animal" has a key that matches this query well (high dot product)
- So $A_{\text{it}, \text{animal}}$ is large
- Token "it"'s output representation is heavily influenced by token "animal"'s value
- The model can now use "animal"'s semantic content when deciding what to do with "it"

This is a **soft** operation — "it" doesn't binary-select "animal". It attends to every token with some weight, with "animal" receiving the highest weight. This soft differentiability is what allows backpropagation to work.

---

## 3.10 Attention as a database

A useful mental model: attention is a **soft, differentiable key-value database**.

Hard database lookup: query matches one key exactly, returns that value.
Attention: query matches all keys with varying similarity, returns a weighted mixture of all values.

The weights are determined by the dot product similarity between query and key, passed through softmax. As training proceeds, $W_Q$ and $W_K$ are adjusted so that semantically relevant query-key pairs produce high scores.

---

## Summary

| Concept | Key point |
|---------|-----------|
| Self-attention | Every token attends to every other token in one step |
| Q, K, V | Query (searching), Key (label), Value (content) |
| Score matrix | $QK^\top \in \mathbb{R}^{n \times n}$ — pairwise relevance |
| Scaling | $\div\sqrt{d_k}$ prevents softmax saturation |
| Softmax | Converts scores to probability distribution |
| Output | Weighted sum of values, weights from softmax |
| Causal mask | $-\infty$ for future positions → enables autoregression |
| Complexity | $\mathcal{O}(n^2 d)$ — quadratic in sequence length |

**Next:** [Chapter 04 — Multi-Head Attention](04-multihead-attention.md)
