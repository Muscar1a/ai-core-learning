# Chapter 04 — Multi-Head Attention

> One attention head sees one kind of relationship. Running many in parallel lets the model track syntax, semantics, and position simultaneously.

---

## 4.1 The limitation of a single head

A single attention head computes one weighted combination of value vectors for each token. The weights are determined by one set of $W_Q, W_K$ matrices — one "lens" through which relationships are computed.

A sentence like "The animal didn't cross the street because it was tired" requires tracking:
- **Coreference:** "it" → "animal"
- **Syntactic structure:** subject-verb agreement, modifier-head relationships
- **Negation scope:** "didn't" modifies the entire clause
- **Semantic role:** what is the cause? what is the effect?

One set of Q/K/V projections can't specialize for all of these simultaneously. Different relationship types pull in opposite directions during optimization.

**Solution:** run $h$ independent attention heads in parallel, each with its own learned $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$. Each head can specialize freely.

---

## 4.2 Multi-head attention formulation

For head $i$:

$$\text{head}_i = \text{Attention}(X W_Q^{(i)},\, X W_K^{(i)},\, X W_V^{(i)})$$

where:
- $W_Q^{(i)}, W_K^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $d_k = d_v = d_{\text{model}} / h$ (standard choice — splits the dimension evenly)

Concatenate all heads and project:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W_O$$

where $W_O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

With $d_v = d_{\text{model}} / h$, the concatenation has dimension $h \cdot d_v = d_{\text{model}}$, so $W_O$ projects from $d_{\text{model}} \to d_{\text{model}}$.

---

## 4.3 Parameter count

For one attention layer with $h$ heads:

| Matrix | Shape | Parameters |
|--------|-------|-----------|
| $W_Q^{(1)}, \ldots, W_Q^{(h)}$ | $d_{\text{model}} \times d_k$ each | $h \cdot d_{\text{model}} \cdot d_k = d_{\text{model}}^2$ |
| $W_K^{(1)}, \ldots, W_K^{(h)}$ | $d_{\text{model}} \times d_k$ each | $d_{\text{model}}^2$ |
| $W_V^{(1)}, \ldots, W_V^{(h)}$ | $d_{\text{model}} \times d_v$ each | $d_{\text{model}}^2$ |
| $W_O$ | $h d_v \times d_{\text{model}}$ | $d_{\text{model}}^2$ |
| **Total** | | $4 d_{\text{model}}^2$ |

Note that total parameters = $4 d_{\text{model}}^2$ regardless of $h$. Changing the number of heads doesn't change the parameter count — it changes how those parameters are *organized*.

**GPT-3:** $d_{\text{model}} = 12{,}288$, $h = 96$, $d_k = d_v = 128$.  
Attention params per layer: $4 \times 12288^2 \approx 603\text{M}$. Across 96 layers: $\approx 58\text{B}$ params total in attention.

---

## 4.4 What different heads learn

Empirical analyses of trained Transformers (Michel et al., 2019; Clark et al., 2019; Voita et al., 2019) have found that different attention heads specialize in different kinds of relationships:

**Positional heads:** attend to tokens at a fixed offset (+1, -1, etc.). These handle local context without needing to compute semantic similarity.

**Syntactic heads:** track grammatical structure. One head might consistently attend from a verb to its subject; another from a pronoun to its antecedent.

**Rare/special token heads:** some heads always attend heavily to `[CLS]`, `[SEP]`, or period tokens. Thought to act as "no-op" heads — the model routes attention to these tokens when no important relationship exists, rather than distributing mass uniformly.

**Semantic heads:** attend based on semantic similarity — "cat" attends to "kitten", "dog" attends to "puppy", etc.

The specialization emerges from training, not from explicit design. Each head finds whatever pattern is most useful for reducing loss on the training data.

---

## 4.5 Head redundancy and pruning

If specialization emerges, can we remove heads that don't specialize?

Michel et al. (2019) showed that many attention heads can be **pruned** (set to zero) with minimal performance loss. In BERT, up to 70% of heads can be removed on certain tasks. The remaining heads pick up the slack.

This suggests significant redundancy in multi-head attention — the model is over-parameterized with respect to the number of heads. Why do we use so many heads then?
- Training is more stable with redundancy
- Different tasks require different specializations — a head useless for translation might be critical for coreference
- The model learns to route differently across tasks

---

## 4.6 The output projection $W_O$

After concatenating heads, $W_O$ mixes information from all heads:

$$\text{output} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W_O$$

Without $W_O$, each head's output would be independent and the concatenation would just stack them without interaction. $W_O$ allows the model to learn how to combine the different "perspectives" from different heads.

$W_O$ is also where the model can learn to *suppress* certain heads for certain inputs — by learning weights that effectively zero out a head's contribution in the output.

---

## 4.7 Efficient multi-head computation

In practice, you don't compute $h$ separate matrix multiplications. You reshape:

```python
# Instead of h separate projections:
Q_1 = X @ W_Q_1  # [n, d_k]
Q_2 = X @ W_Q_2  # [n, d_k]
# ...

# Do one big projection and reshape:
W_Q = concat([W_Q_1, W_Q_2, ..., W_Q_h], dim=1)  # [d_model, d_model]
Q = X @ W_Q        # [n, d_model]
Q = Q.reshape(n, h, d_k)  # [n, h, d_k]
Q = Q.transpose(0, 1)     # [h, n, d_k]  ← batch over heads
```

This way, all heads are computed as a single batched matrix multiplication — much more efficient on GPU.

---

## 4.8 Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)

Standard Multi-Head Attention requires that every single attention head maintain its own private set of Key ($W_K$) and Value ($W_V$) projection matrices. While this maximizes representational diversity, it creates an enormous memory bottleneck during autoregressive inference due to the **KV Cache**.

### The Three Structural Paradigms

```
MHA (8 query heads, 8 KV heads — 1:1 private pairing):
  [Q1]  [Q2]  [Q3]  [Q4]  [Q5]  [Q6]  [Q7]  [Q8]
   |     |     |     |     |     |     |     |
   v     v     v     v     v     v     v     v
  [K1]  [K2]  [K3]  [K4]  [K5]  [K6]  [K7]  [K8]
  [V1]  [V2]  [V3]  [V4]  [V5]  [V6]  [V7]  [V8]

MQA (8 query heads, 1 KV head — shared globally):
  [Q1]  [Q2]  [Q3]  [Q4]  [Q5]  [Q6]  [Q7]  [Q8]
    \     \    \    |     |    /    /     /
     +-----+----+---+-----+---+----+-----+
                       |
                       v
                  [K_shared]
                  [V_shared]

GQA (8 query heads, 2 groups — 4 queries per KV head):
  [Q1] [Q2] [Q3] [Q4]       [Q5] [Q6] [Q7] [Q8]
    \   |    |   /             \   |    |   /
     +--+----+--+                +-+----+-+
            |                          |
            v                          v
       [K_group1]                 [K_group2]
       [V_group1]                 [V_group2]
```

### Mathematical Generalization

GQA serves as a unified generalization connecting MHA and MQA:
* **When Number of Groups = Number of Heads ($G = H$):** Every head possesses its own Key and Value. This is **Multi-Head Attention (MHA)**.
* **When Number of Groups = 1 ($G = 1$):** All query heads share a single Key and Value. This is **Multi-Query Attention (MQA)**.
* **When $1 < G < H$:** Query heads are partitioned into $G$ groups where heads within a group share a Key and Value. This is **Grouped-Query Attention (GQA)**.

| Architecture | KV Heads ($G$) | KV Cache Footprint | Inference Memory Bandwidth | Quality |
|---|---|---|---|---|
| **MHA** | $H$ | Largest ($1\times$) | Highest ($1\times$) | Baseline ceiling |
| **MQA** | $1$ | Smallest ($1/H$) | Lowest ($1/H$) | Quality can degrade on complex tasks |
| **GQA** | $G$ (e.g. $H/8$) | Reduced ($G/H$, e.g. $1/8$) | Highly reduced ($G/H$) | Retains $\approx 99\%$ of MHA quality |

### Uptraining: Converting Pretrained MHA Models to GQA

A pivotal discovery in Ainslie et al. (2023) is that you do not need to pretrain a GQA model from scratch:
1. **Checkpoint Ingestion:** Take an existing, fully trained MHA checkpoint.
2. **Mean Pooling Projection Weights:** For each group $g$, calculate the arithmetic mean of all $K$ projection matrices and all $V$ projection matrices within that group:
   $$W_{K, \text{group } g} = \frac{1}{|G|} \sum_{i \in \text{group } g} W_K^{(i)}, \qquad W_{V, \text{group } g} = \frac{1}{|G|} \sum_{i \in \text{group } g} W_V^{(i)}$$
3. **Uptraining (Short Fine-Tuning):** Continue pre-training on the target corpus for only $\sim 5\%$ of the original pre-training compute. The model rapidly recovers accuracy virtually indistinguishable from the original MHA architecture while permanently running with an $8\times$ smaller KV cache.

In standard model configuration files (e.g., Hugging Face transformers), $H$ is specified as `num_attention_heads` and $G$ is specified as `num_key_value_heads`.

---

## 4.9 Cross-Attention & Sequence Bridging

In encoder-decoder architectures (the original 2017 Transformer, T5, Whisper), attention serves two distinct roles:

1. **Self-Attention:** Tokens query other tokens within the **same** sequence.
2. **Cross-Attention:** Tokens in the decoder query tokens from the **encoder's output representation**.

$$\text{CrossAttn}(X_{\text{dec}}, X_{\text{enc}}) = \text{softmax}\!\left(\frac{(X_{\text{dec}} W_Q)(X_{\text{enc}} W_K)^\top}{\sqrt{d_k}}\right) (X_{\text{enc}} W_V)$$

### Self-Attention vs. Cross-Attention

| Property | Self-Attention | Cross-Attention |
|---|---|---|
| **Source of Query ($Q$)** | Current sequence ($X_{\text{current}}$) | Decoder hidden states ($X_{\text{dec}}$) |
| **Source of Key ($K$)** | Current sequence ($X_{\text{current}}$) | Encoder output memory ($X_{\text{enc}}$) |
| **Source of Value ($V$)** | Current sequence ($X_{\text{current}}$) | Encoder output memory ($X_{\text{enc}}$) |
| **Sequence Lengths** | Square matrix: $N_{\text{current}} \times N_{\text{current}}$ | Rectangular matrix: $N_{\text{dec}} \times N_{\text{enc}}$ |
| **Masking** | Optional (Causal mask in decoder) | Typically unmasked (full access to encoder context) |
| **Role** | Intra-sequence contextualization | Cross-sequence information retrieval |

---

## Summary

| Concept | Key point |
|---------|-----------|
| Multi-head | $h$ independent attention heads in parallel |
| Specialization | Each head learns different relationship types |
| Dimensionality | $d_k = d_v = d_{\text{model}} / h$ — dimension split evenly |
| Params per layer | $4 d_{\text{model}}^2$ — same regardless of $h$ |
| Output projection | $W_O$ mixes information across heads |
| Pruning | Many heads are redundant — 70% can be removed with small loss |
| GQA/MQA | Shared K,V heads reduce KV-cache memory |
| Cross-attention | Encoder-decoder: Q from decoder, K/V from encoder |

**Next:** [Chapter 05 — Positional Encoding](05-positional-encoding.md)
