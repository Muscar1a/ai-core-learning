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

Standard multi-head attention: every head has its own $W_K$ and $W_V$. This means the KV-cache stores $h$ sets of K and V tensors per layer per token — very memory-hungry for long contexts.

**Multi-Query Attention (MQA)** (Shazeer, 2019): all heads share a *single* $W_K$ and $W_V$. Only $W_Q$ differs per head. Result: $h\times$ less KV-cache memory, slightly worse quality.

**Grouped-Query Attention (GQA)** (Ainslie et al., 2023): groups of $g$ query heads share one $K, V$ head. Standard MHA = GQA with $g=1$; MQA = GQA with $g=h$.

$$\text{GQA groups} = h / g \text{ KV heads, each shared by } g \text{ query heads}$$

Llama 3 70B uses GQA with 8 KV heads for 64 query heads ($g=8$). This reduces KV-cache by 8×, enabling longer contexts at the same memory budget. Performance is nearly identical to full MHA after sufficient training.

---

## 4.9 Cross-attention

In encoder-decoder models (T5, original Transformer), the decoder uses **cross-attention** to attend to the encoder's output:

$$\text{head}_i = \text{Attention}(\underbrace{X_{\text{dec}} W_Q^{(i)}}_{\text{from decoder}},\; \underbrace{X_{\text{enc}} W_K^{(i)}}_{\text{from encoder}},\; \underbrace{X_{\text{enc}} W_V^{(i)}}_{\text{from encoder}})$$

Queries come from the decoder's current state; keys and values come from the encoder's output. This lets the decoder selectively "read" the encoded representation of the input at each generation step.

Decoder-only models (GPT family) have no cross-attention — they don't have an encoder. The entire task (input + output) is encoded as a single sequence.

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
