# Chapter 02 — Embeddings

> Token IDs are arbitrary integers. An embedding converts them into points in a high-dimensional space where geometry encodes meaning.

---

## 2.1 The problem with integers

After tokenization, "cat" might be token ID 3797. "dog" might be 4347. These numbers have no relationship to each other — 3797 is not "closer" to 4347 than it is to 12,000. Yet clearly "cat" and "dog" are semantically related in ways that should inform every downstream computation.

We need a representation where:
- Semantically similar tokens are close together
- Semantically different tokens are far apart
- Relationships between concepts are captured geometrically

This is what embeddings do.

---

## 2.2 The embedding lookup

Define a matrix:

$$E \in \mathbb{R}^{V \times d_{\text{model}}}$$

where $V$ is the vocabulary size and $d_{\text{model}}$ is the embedding dimension (768, 1024, 4096, etc.). Every token ID maps to one row of this matrix:

$$\text{embed}(\text{id}) = E[\text{id}] \in \mathbb{R}^{d_{\text{model}}}$$

This is literally a lookup table — no computation, just indexing. But the values in $E$ are **learned parameters** that get updated during training via backpropagation. After training on enough text, the matrix encodes the semantic structure of the vocabulary.

---

## 2.3 The distributional hypothesis

> "You shall know a word by the company it keeps." — J.R. Firth (1957)

Words that appear in similar contexts will end up with similar embeddings. Consider:

- "The **cat** sat on the mat."
- "The **dog** sat on the mat."
- "I fed my **cat** tuna."
- "I fed my **dog** kibble."

Both "cat" and "dog" appear after "the", after "my", before verbs like "sat" and "ate", in subject position. The gradient updates that adjust $E[\text{cat}]$ are nearly identical to those that adjust $E[\text{dog}]$ → their embedding vectors converge to nearby points.

"Table" and "chair" are also related (furniture), but appear in different syntactic positions than animals, so they cluster separately.

---

## 2.4 Measuring similarity

The standard metric is **cosine similarity**:

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|} \in [-1, 1]$$

Cosine similarity measures the angle between two vectors, ignoring magnitude. This is preferred over Euclidean distance because embedding vectors tend to grow in magnitude with frequency — common words like "the" have large-magnitude vectors because they receive many gradient updates, not because they're semantically extreme.

| $\cos(\mathbf{a}, \mathbf{b})$ | Interpretation |
|-------------------------------|----------------|
| 1.0 | Same direction — nearly identical meaning |
| 0.8–0.95 | Very related (e.g., "cat" and "kitten") |
| 0.4–0.7 | Related but distinct (e.g., "cat" and "animal") |
| 0.0 | Orthogonal — unrelated |
| -0.5 to 0.0 | Weak opposition |

---

## 2.5 Linear structure in embedding space

One of the most striking discoveries from word2vec (Mikolov et al., 2013): semantic relationships correspond to **linear directions** in embedding space.

$$E[\text{King}] - E[\text{Man}] + E[\text{Woman}] \approx E[\text{Queen}]$$

The vector $E[\text{King}] - E[\text{Man}]$ points in the "gender direction" (masculine → feminine). Adding this to $E[\text{Woman}]$ should land near $E[\text{Queen}]$ — and it does, with cosine similarity ≈ 0.89.

Other discovered directions:
- Country → capital: $E[\text{France}] - E[\text{Paris}] \approx E[\text{Japan}] - E[\text{Tokyo}]$
- Verb tense: $E[\text{walk}] - E[\text{walked}] \approx E[\text{run}] - E[\text{ran}]$
- Comparative: $E[\text{big}] - E[\text{bigger}] \approx E[\text{slow}] - E[\text{slower}]$

**Why does this happen?** Consider "king" and "queen". They appear in nearly identical contexts except when gender is specified. The gradient updates differ only in the gender-relevant dimensions → those dimensions encode gender as a consistent direction.

This wasn't designed — it emerges from the optimization pressure to predict context.

---

## 2.6 The geometry of high-dimensional space

Human intuition about geometry breaks down at high dimensions. Some non-obvious facts about $\mathbb{R}^{768}$:

**Curse of dimensionality:** in high dimensions, almost all pairs of random vectors are nearly orthogonal. Cosine similarity → 0 for random vectors. This means the model has a lot of "room" to place things — semantic neighborhoods are sparse.

**The "hub" problem:** in practice, some vectors end up being close to many others (hubs) while most are isolated. This is called anisotropy — embeddings don't uniformly fill the space, they cluster around certain directions.

**Normalization matters:** when computing cosine similarity, you should use unit-normalized vectors. In practice, many modern systems L2-normalize embeddings before retrieval tasks.

---

## 2.7 Dimensionality: how big is $d_{\text{model}}$?

| Model | $d_{\text{model}}$ | Why |
|-------|---------|-----|
| BERT base | 768 | Standard for classification tasks |
| GPT-2 | 768–1600 | Scaled up across model sizes |
| GPT-3 | 12,288 | Larger → more expressive representations |
| Llama 3 70B | 8,192 | Large but efficient with GQA |
| PaLM | 18,432 | Google's scaling choice |

Larger $d_{\text{model}}$ means:
- More capacity to encode nuanced semantic differences
- Attention cost scales as $\mathcal{O}(n^2 \cdot d_{\text{model}})$ — not great
- More parameters in every weight matrix — more to train and store

The optimal $d_{\text{model}}$ for a given compute budget is determined empirically. Scaling laws (see Chapter 08) give guidance.

---

## 2.8 Contextual vs static embeddings

**Static embeddings** (word2vec, GloVe, fastText): each token has one fixed vector regardless of context. "bank" in "river bank" and "bank account" get the same vector. The model must disambiguate meaning through other mechanisms.

**Contextual embeddings** (what Transformers produce): after the attention layers, the representation of "bank" in "river bank" is different from "bank" in "bank account" — the context has modified the vector. This is a key advantage of Transformers over static embeddings.

The embedding lookup ($E[\text{id}]$) is the *starting* representation — static, one per token type. What makes Transformers powerful is that 32–96 layers of attention and FFN then *modify* this representation based on surrounding context.

---

## 2.9 Weight tying

A subtle but important detail: the **same matrix $E$** is used for both:
1. Input embedding (token ID → vector)
2. Output projection (final hidden state → vocabulary logits via $\mathbf{x} \cdot E^\top$)

This is called **weight tying** (Press & Wolf, 2017). It:
- Saves $V \times d_{\text{model}}$ parameters (for GPT-3: ~618M params saved)
- Improves performance: tokens that are semantically close in the output space should also be close in the input space — weight tying enforces this constraint
- Makes intuitive sense: the model uses the same "concept of 'cat'" whether reading it as input or predicting it as output

---

## 2.10 Embedding arithmetic is not magic

A common misunderstanding: "the model *knows* that King - Man + Woman = Queen."

The model doesn't know this in any symbolic sense. What happened:
1. During training on billions of sentences, "king" and "queen" appeared in similar contexts
2. Gradient descent adjusted their embedding vectors to capture those statistical regularities
3. The particular linear structure is an emergent consequence of how distributional statistics interact with the geometry of gradient descent

It's a property of the *optimization*, not of any explicit world model. This is why analogies work better for common, well-attested relationships (gender, tense, geography) and break down for rare or abstract ones.

---

## Summary

| Concept | Key point |
|---------|-----------|
| Embedding matrix | $E \in \mathbb{R}^{V \times d_{\text{model}}}$, learned during training |
| Lookup | $\text{embed}(\text{id}) = E[\text{id}]$ — one row per token |
| Cosine similarity | Angle-based metric, robust to magnitude differences |
| Linear structure | Relationships = directions in $\mathbb{R}^{d_{\text{model}}}$ |
| Distributional hypothesis | Similar context → similar vector |
| Weight tying | Input embedding = output projection (transposed) |
| Contextual vs static | Transformer layers modify the static embedding with context |

**Next:** [Chapter 03 — Self-Attention](03-attention.md)
