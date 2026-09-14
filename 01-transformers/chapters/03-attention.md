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

## 3.5 Why scale by $\sqrt{d_k}$? (Full Mathematical Proof)

The scaling factor $1/\sqrt{d_k}$ is not an arbitrary hyperparameter—it is the mathematically exact scaling factor required to prevent the softmax function from saturating.

### The Variance Derivation Step-by-Step

Consider query vector $\mathbf{q}$ and key vector $\mathbf{k}$ in $\mathbb{R}^{d_k}$. Assume the components $q_i$ and $k_i$ are independent random variables with zero mean and unit variance:
$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1$$
$$\mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$

The dot product is the sum of $d_k$ scalar products:
$$S = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i = q_1 k_1 + q_2 k_2 + \dots + q_{d_k} k_{d_k}$$

Because each element has zero mean:
$$\mathbb{E}[S] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$

Now evaluate the variance. Because $\mathbb{E}[S] = 0$:
$$\text{Var}(S) = \mathbb{E}[S^2] - (\mathbb{E}[S])^2 = \mathbb{E}[S^2]$$

Expanding $S^2 = \left(\sum_{i=1}^{d_k} q_i k_i\right)^2$, we obtain $d_k^2$ terms which fall into two distinct groups:
1. **Squared terms ($i = j$, total $d_k$ terms):**
   $$\mathbb{E}[(q_i k_i)^2] = \mathbb{E}[q_i^2] \cdot \mathbb{E}[k_i^2] = 1 \cdot 1 = 1$$
2. **Cross terms ($i \neq j$, total $d_k^2 - d_k$ terms):**
   $$\mathbb{E}[(q_i k_i)(q_j k_j)] = \mathbb{E}[q_i]\mathbb{E}[k_i]\mathbb{E}[q_j]\mathbb{E}[k_j] = 0 \cdot 0 \cdot 0 \cdot 0 = 0$$

Summing across all terms:
$$\text{Var}(S) = \sum_{i=1}^{d_k} 1 + \sum_{i \neq j} 0 = d_k$$

**The variance of the raw dot product is exactly equal to $d_k$.**

### Scaling Resets Variance to 1

From probability theory, dividing any random variable by a constant $c$ scales its variance by $c^2$:
$$\text{Var}\left(\frac{S}{c}\right) = \frac{\text{Var}(S)}{c^2} = \frac{d_k}{c^2}$$

To normalize the variance back to 1:
$$\frac{d_k}{c^2} = 1 \implies c^2 = d_k \implies c = \sqrt{d_k}$$

Therefore, dividing by $\sqrt{d_k}$ guarantees that regardless of whether $d_k = 64$ or $d_k = 512$, the attention logits will maintain a standard deviation of 1.

### Numerical Demonstration of Softmax Saturation

What happens when we skip scaling in a real model where $d_k = 64$? Suppose three tokens produce dot products $[14.0, 10.0, 12.0]$:

* **Without Scaling ($S$):**
  $$e^{14.0} \approx 1,202,604, \quad e^{10.0} \approx 22,026, \quad e^{12.0} \approx 162,755 \quad (\Sigma \approx 1,387,386)$$
  $$\text{softmax}([14.0, 10.0, 12.0]) \approx [0.867, 0.016, 0.117]$$
  The first token captures 86.7% of all attention mass. The distribution is nearly one-hot, and the gradients of softmax with respect to inputs collapse toward zero (vanishing gradient).

* **With Scaling ($S / \sqrt{64} = S / 8$):**
  $$S_{\text{scaled}} = [1.75, 1.25, 1.50]$$
  $$e^{1.75} \approx 5.755, \quad e^{1.25} \approx 3.490, \quad e^{1.50} \approx 4.482 \quad (\Sigma \approx 13.727)$$
  $$\text{softmax}([1.75, 1.25, 1.50]) \approx [0.419, 0.254, 0.326]$$
  Attention is distributed smoothly across all three tokens. Backpropagation receives healthy, non-vanishing gradient updates across all sequence positions.

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

## 3.11 Step-by-Step Numeric Walkthrough

To anchor the theory, let's trace every tensor calculation with concrete numbers for a 2-token sequence ($n=2$) with projection dimension $d_k = d_v = 2$.

### Setup: Tokens and Projections

Suppose our two tokens have already been embedded into vectors $\mathbf{x}_1 = [1.0, 0.0]$ and $\mathbf{x}_2 = [0.0, 1.0]$.

Assume the trained projection weights are:
$$W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W_K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad W_V = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}$$

**Step 1: Compute Q, K, V matrices**
$$Q = X W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \quad (\mathbf{q}_1 = [1, 0], \mathbf{q}_2 = [0, 1])$$

$$K = X W_K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \quad (\mathbf{k}_1 = [0, 1], \mathbf{k}_2 = [1, 0])$$

$$V = X W_V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} \quad (\mathbf{v}_1 = [2, 1], \mathbf{v}_2 = [0, 3])$$

**Step 2: Compute Raw Attention Scores ($S = Q K^\top$)**
$$K^\top = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$
$$S = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$
Here, token 1's query $\mathbf{q}_1=[1,0]$ dots with token 1's key $\mathbf{k}_1=[0,1]$ to produce 0, and dots with token 2's key $\mathbf{k}_2=[1,0]$ to produce 1. Token 1 finds token 2 much more relevant than itself!

**Step 3: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$**
$$S_{\text{scaled}} = \frac{S}{\sqrt{2}} = \begin{bmatrix} 0 / 1.414 & 1 / 1.414 \\ 1 / 1.414 & 0 / 1.414 \end{bmatrix} \approx \begin{bmatrix} 0.000 & 0.707 \\ 0.707 & 0.000 \end{bmatrix}$$

**Step 4: Softmax per row ($A = \text{softmax}(S_{\text{scaled}})$)**
For Row 1:
* $e^{0.000} = 1.000, \quad e^{0.707} \approx 2.028 \quad (\text{Sum} \approx 3.028)$
* $A_{11} = 1.000 / 3.028 \approx 0.330$
* $A_{12} = 2.028 / 3.028 \approx 0.670$

By symmetry, Row 2 produces $[0.670, 0.330]$.
$$A = \begin{bmatrix} 0.330 & 0.670 \\ 0.670 & 0.330 \end{bmatrix}$$

**Step 5: Weighted Value Sum ($\text{Output} = A V$)**
$$\mathbf{y}_1 = 0.330 \cdot \mathbf{v}_1 + 0.670 \cdot \mathbf{v}_2 = 0.330 \cdot [2, 1] + 0.670 \cdot [0, 3] = [0.660, 0.330] + [0.000, 2.010] = [0.660, 2.340]$$
$$\mathbf{y}_2 = 0.670 \cdot \mathbf{v}_1 + 0.330 \cdot \mathbf{v}_2 = 0.670 \cdot [2, 1] + 0.330 \cdot [0, 3] = [1.340, 0.670] + [0.000, 0.990] = [1.340, 1.660]$$

$$\text{Output} = \begin{bmatrix} 0.660 & 2.340 \\ 1.340 & 1.660 \end{bmatrix}$$

Each output vector now combines contextual information from both tokens according to the soft affinity between their queries and keys.

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
