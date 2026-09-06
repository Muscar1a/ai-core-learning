# Chapter 09 — Inference

> Training fills the weights. Inference is when they earn their keep. Understanding KV-caching, sampling, and quantization separates someone who uses LLMs from someone who can deploy them.

---

## 9.1 Autoregressive generation

At inference, the model generates one token at a time:

1. Input prompt is tokenized: $(t_1, t_2, \ldots, t_p)$
2. Model computes logits for the next token: $\mathbf{l} = f(t_1, \ldots, t_p; \theta) \in \mathbb{R}^{V}$
3. Sample token $t_{p+1}$ from the distribution $\text{softmax}(\mathbf{l} / \tau)$
4. Append $t_{p+1}$ to the sequence; repeat from step 2 until EOS token or max length

Each step requires a full forward pass through all $L$ layers, processing the entire sequence so far. Without optimization, generating a 1000-token response from a 1000-token prompt requires processing a sequence of length 1000 to 2000 tokens, for 1000 separate forward passes. This is expensive.

---

## 9.2 KV-Cache

The core optimization for inference. In self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

At step $t$, the query is computed from the new token $t_p$. The keys and values are computed from all tokens $t_1, \ldots, t_p$.

**The insight:** $K$ and $V$ for tokens $t_1, \ldots, t_{p-1}$ were already computed at step $t-1$. They haven't changed (the model weights and previous tokens haven't changed). We can **cache** them.

**KV-cache:** store the $K$ and $V$ tensors from all previous steps. At each new step, compute only the new query, key, and value for the latest token, then concatenate with the cached K and V.

$$K_t = \text{concat}(K_{t-1}, \mathbf{k}_t), \quad V_t = \text{concat}(V_{t-1}, \mathbf{v}_t)$$

This reduces each generation step from $\mathcal{O}(t \cdot d)$ to $\mathcal{O}(d)$ for the attention computation (the new Q needs to attend to all cached K, but you're computing only one row of the attention matrix).

**KV-cache memory:** for a model with $L$ layers, $h$ heads, $d_k$-dimensional keys and values:

$$\text{KV size} = 2 \times L \times h \times d_k \times n_{\text{tokens}} \times \text{bytes per element}$$

For Llama 3 70B (80 layers, 8 KV heads, $d_k = 128$) with 128k context, FP16:
$$2 \times 80 \times 8 \times 128 \times 131072 \times 2 \approx 43\text{ GB}$$

This is why long-context inference is expensive — the KV-cache grows linearly with context length. GQA (Chapter 04) reduces it by sharing KV heads: with 8 KV heads instead of 64, cache is $8\times$ smaller ≈ 5.4 GB for the same context.

---

## 9.3 Prefill vs decode

Modern inference has two phases:

**Prefill (prompt processing):** process all prompt tokens in parallel (they're known upfront). This is highly parallelizable — GPUs can compute all $n_{\text{prompt}}$ rows of the attention matrix simultaneously. KV-cache is populated for all prompt tokens.

**Decode (generation):** generate one token at a time, using the KV-cache. GPU utilization is low — we're doing one token per step, which underutilizes the GPU's massive parallelism. This is the **memory bandwidth bottleneck** of inference.

The ratio of prefill to decode time changes with request type:
- Long prompt, short output (summarization): prefill-dominated
- Short prompt, long output (creative writing): decode-dominated

---

## 9.4 Sampling strategies

After computing logits $\mathbf{l} \in \mathbb{R}^V$, how do we pick the next token?

### Greedy decoding

$$t_{t+1} = \arg\max_i \text{softmax}(\mathbf{l})_i$$

Always pick the highest-probability token. Deterministic, fast, but poor quality — greedy decoding often gets stuck in repetitive loops ("the the the the...").

### Temperature sampling

$$t_{t+1} \sim \text{softmax}(\mathbf{l} / \tau)$$

Dividing by temperature $\tau$ before softmax changes the distribution shape:
- $\tau \to 0$: approaches greedy (one-hot distribution)
- $\tau = 1$: raw model probabilities
- $\tau > 1$: flattens distribution, more random output

Typical values: $\tau \in [0.7, 1.0]$ for general text, lower for code, higher for creative tasks.

### Top-k sampling

Keep only the top-$k$ highest probability tokens, renormalize, then sample:

$$P'(t) = \begin{cases} P(t) / Z & \text{if } t \in \text{top-k}(\mathbf{l}) \\ 0 & \text{otherwise} \end{cases}$$

where $Z$ is the normalization constant. Prevents sampling from the long tail of improbable tokens. GPT-2 used $k = 40$.

**Problem:** the right value of $k$ depends on context. Sometimes the top-1 token has 90% probability and $k = 40$ gives 39 near-zero-probability tokens. Other times, probability is spread evenly across 200 plausible continuations and $k = 40$ cuts most of them.

### Top-p (nucleus) sampling

Holtzman et al. (2019): instead of a fixed $k$, use a dynamic cut based on cumulative probability:

$$P'(t) = \begin{cases} P(t) / Z & \text{if } t \in S_p \\ 0 & \text{otherwise} \end{cases}$$

where $S_p = \{t_1, t_2, \ldots, t_m\}$ is the smallest set such that $\sum_{i=1}^m P(t_i) \geq p$.

With $p = 0.9$: include tokens until cumulative probability reaches 90%, then sample from only those. When distribution is peaked, this selects few tokens; when flat, selects many. $p = 0.9$ or $p = 0.95$ are common defaults.

**In practice:** temperature + top-p are combined. First apply temperature, then nucleus sampling.

### Min-p sampling

Phuong et al. (2024): instead of a cumulative threshold, filter tokens whose probability is below a fraction $p_{\text{min}}$ of the maximum:

$$S = \{t : P(t) \geq p_{\text{min}} \cdot \max_j P(j)\}$$

With $p_{\text{min}} = 0.05$: keep all tokens with probability at least 5% of the highest-probability token. Claimed to produce higher quality and more coherent text than top-p at the same diversity level.

---

## 9.5 Repetition penalties

Autoregressive models frequently repeat phrases without external controls. Two common fixes:

**Repetition penalty** (frequency-based): reduce logit of tokens already in the context:

$$l'_t = \frac{l_t}{\alpha} \quad \text{if token } t \text{ appears in context}$$

with $\alpha > 1$ penalizing repetition. Simple but crude — penalizes even tokens that should repeat (e.g., names in a story about a specific person).

**Presence penalty** (OpenAI API): binary — penalize any token that has appeared at least once, regardless of how many times:

$$l'_t = l_t - \beta \cdot \mathbf{1}[t \in \text{context}]$$

$\beta \in [0, 2]$, typically $\beta = 0.1$–$0.5$.

---

## 9.6 Speculative decoding

**Problem:** at each decode step, we run the full (expensive) model just to get one token. Can we do better?

**Speculative decoding** (Chen et al., 2023; Leviathan et al., 2023): use a small "draft" model to predict multiple future tokens cheaply, then verify them in parallel with the large model.

**Algorithm:**
1. Draft model generates $K$ tokens speculatively: $(\hat{t}_{p+1}, \ldots, \hat{t}_{p+K})$
2. Large model processes all $K$ tokens in parallel (one forward pass, not $K$)
3. For each speculative token, check if the large model agrees (acceptance criterion based on probability ratio)
4. Keep accepted tokens; regenerate from the first rejected one

**Speedup:** the large model runs once in parallel over $K$ tokens instead of $K$ separate times. Since verification is a prefill-mode operation (all tokens known), it fully parallelizes.

**Typical results:** 2–3× speedup with a draft model that's 7–10× smaller. Llama 3 uses speculative decoding in production.

**The acceptance criterion:** to ensure the output distribution matches the large model exactly (not just approximately), use:

$$\text{accept } \hat{t} \text{ with probability } \min\!\left(1, \frac{p_{\text{large}}(\hat{t})}{p_{\text{draft}}(\hat{t})}\right)$$

This is rejection sampling — the combined output has the exact same distribution as sampling from the large model alone.

---

## 9.7 Quantization

Model weights are stored in 32-bit floats by default. **Quantization** reduces precision to reduce memory and speed up inference.

### Post-training quantization (PTQ)

**FP16 (half precision):** 16-bit floats. Standard for GPU inference. 2× memory reduction from FP32 with negligible quality loss. Most inference frameworks default to FP16.

**INT8 (8-bit integer):** weights stored as integers in range $[-128, 127]$. 4× compression from FP32. Requires a scale factor per tensor (or per row) to map the integer range to the actual weight range.

$$w_{\text{float}} = s \cdot w_{\text{int8}}$$

where $s$ is a per-layer scale calibrated on a small dataset. Quality loss is minimal for weights > ~1B parameters.

**INT4 (4-bit integer):** 8× compression. Significant quality degradation unless using advanced methods:

- **GPTQ** (Frantar et al., 2022): quantize weights layer by layer, using second-order gradient information to minimize quantization error. Llama 3 GPTQ 4-bit works well.
- **AWQ** (Lin et al., 2023): identify weight channels with large magnitude, keep them at higher precision, quantize the rest to 4-bit. Better quality than naive INT4.

### KV-cache quantization

The KV-cache itself can be quantized separately from weights. Quantizing to INT8 halves the KV-cache memory with minimal quality impact, enabling longer contexts at the same memory budget.

### Practical memory requirements

| Model | FP16 (GB) | INT8 (GB) | INT4 (GB) |
|-------|-----------|-----------|-----------|
| Llama 3 8B | 16 | 8 | 4 |
| Llama 3 70B | 140 | 70 | 35 |
| Llama 3 405B | 810 | 405 | 202 |

INT4 Llama 3 70B fits on 2× 24GB consumer GPUs. This is why quantization matters for accessibility.

---

## 9.8 Batching and throughput

Serving multiple users simultaneously requires batching:

**Static batching:** group $B$ requests, run them as a single batch. Problem: different requests have different output lengths — short requests must wait for the longest to finish before the batch is freed.

**Continuous batching** (Orca, Yu et al., 2022): add new requests as slots open up. When one request finishes generation, immediately fill its slot with a new request. No waiting. This is the standard approach in production serving (vLLM, TGI).

**PagedAttention** (vLLM, Kwon et al., 2023): KV-cache memory is fragmented — different requests have different lengths, making contiguous allocation wasteful. PagedAttention manages KV-cache in fixed-size "pages" (like OS virtual memory), enabling memory sharing between requests and 3–4× higher throughput.

---

## Summary

| Concept | Key point |
|---------|-----------|
| Autoregressive | One token per step, conditioned on all previous |
| KV-cache | Cache K and V from all previous tokens; reuse per step |
| Prefill vs decode | Parallel processing vs sequential generation |
| Temperature | $\tau < 1$: concentrated; $\tau > 1$: diffuse |
| Top-p (nucleus) | Dynamic vocab cutoff based on cumulative probability |
| Speculative decoding | Draft model proposes; large model verifies in parallel |
| Quantization | INT8: 4× compression, negligible quality loss; INT4: 8×, needs GPTQ/AWQ |
| Continuous batching | Fill GPU slots as requests finish, not at fixed batch boundaries |

**Next:** [Chapter 10 — Modern Variants](10-modern-variants.md)
