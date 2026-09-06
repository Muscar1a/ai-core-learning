# Chapter 10 — Modern Variants

> The Transformer from "Attention Is All You Need" still runs at the core of every major LLM. But the surrounding engineering has changed substantially. This chapter covers the modifications that matter.

---

## 10.1 FlashAttention

Standard attention requires materializing the $n \times n$ attention matrix in GPU HBM (high-bandwidth memory). For $n = 128{,}000$ tokens, this matrix is $128{,}000^2 \times 2$ bytes ≈ 32 GB — larger than most GPU memory budgets.

**FlashAttention** (Dao et al., 2022) computes exact attention without ever materializing the full $n \times n$ matrix, using **tiled computation** in SRAM.

### The key insight

The bottleneck in standard attention is **memory bandwidth**, not compute. Reading and writing the large attention matrix to HBM is slower than the actual matrix multiplication.

FlashAttention processes Q, K, V in **tiles** that fit in the GPU's fast SRAM (cache), computing output incrementally without writing the intermediate attention matrix to HBM.

**Online softmax:** the trick that makes tiling possible. For softmax, you need to normalize by the row sum $Z = \sum_j e^{s_j}$. In tiled computation, you don't have all scores $s_j$ at once.

Solution: maintain running statistics. When processing tile $\ell$:
$$m_\ell = \max(m_{\ell-1}, \max_j s_j^{(\ell)}) \quad \text{(running max)}$$
$$Z_\ell = Z_{\ell-1} \cdot e^{m_{\ell-1} - m_\ell} + \sum_j e^{s_j^{(\ell)} - m_\ell}$$

The running max and sum let you compute softmax exactly across tiles without storing intermediate values.

**Results:**
- Up to 3× faster than PyTorch attention on A100 GPU
- Memory is $\mathcal{O}(n)$ instead of $\mathcal{O}(n^2)$ — enables 128k context on a single A100
- **Exact** attention (not approximate) — same numerical output as standard attention

### FlashAttention-2 and FlashAttention-3

FlashAttention-2 (Dao, 2023): better parallelism strategy, ~2× speedup over FA1. FlashAttention-3 (Shah et al., 2024): exploits H100-specific hardware (WGMMA, TMA, FP8), reaching 75% of theoretical peak GPU flops.

All major training frameworks (HuggingFace, Megatron, DeepSpeed) use FlashAttention by default.

---

## 10.2 Ring Attention

For sequences longer than fit in a single GPU's memory, **Ring Attention** (Liu et al., 2023) distributes the sequence across multiple GPUs arranged in a ring.

Each GPU holds a slice of the sequence (Q, K, V for those positions). During attention computation, each GPU:
1. Computes local attention with its own Q slice attending to its local K, V slice
2. Passes K and V to the next GPU in the ring
3. Receives K and V from the previous GPU, computes cross-attention
4. Continues until each GPU has seen all K, V slices

This achieves **linear scaling**: memory per GPU is $\mathcal{O}(n/G)$ where $G$ is the number of GPUs. Communication and computation overlap (pipelined), hiding latency.

Gemini 1.5 Pro's 1-million-token context is enabled by distributed attention (likely Ring Attention or a similar approach).

---

## 10.3 Grouped-Query Attention (GQA) — deeper

Covered in Chapter 04, but worth revisiting in the context of modern deployment.

**Why GQA matters for inference:** at decode time, the bottleneck is loading K and V from HBM for each new query. With standard MHA ($h$ KV heads), every decode step loads $h \times d_k \times n_{\text{cached}}$ elements. With GQA ($g$ KV heads), this is $g \times d_k \times n_{\text{cached}}$ — a $h/g$ reduction in bandwidth.

Since decode is bandwidth-bound, this directly translates to $h/g$ speedup on per-token latency.

**Llama 3 parameter choices:**

| Model size | Q heads | KV heads | KV ratio | d_k |
|-----------|---------|----------|----------|-----|
| 8B | 32 | 8 | 4× | 128 |
| 70B | 64 | 8 | 8× | 128 |
| 405B | 128 | 8 | 16× | 128 |

Larger models use more Q heads (more capacity) but keep KV heads fixed (bandwidth budget).

---

## 10.4 Sliding Window Attention

Standard attention: every token attends to every previous token. Cost: $\mathcal{O}(n^2)$ in time and memory.

**Sliding window attention** (SWA): each token attends only to the $W$ most recent tokens (window size $W$). Cost: $\mathcal{O}(n \cdot W)$ — linear in sequence length.

Intuition: in many tasks, the most relevant context is nearby. The word "it" most likely refers to something in the last few sentences, not 100,000 tokens ago.

**Mistral 7B** uses SWA with $W = 4096$ for a context of up to 8192 tokens. Layers alternate between SWA and full attention in some architectures.

**Problem:** with pure SWA, information from position 0 cannot reach position $n$ directly — it must propagate through $n/W$ hops, one per layer. With 32 layers and $W = 4096$, information can propagate at most $32 \times 4096 = 131{,}072$ positions. For very long contexts, early-layer information can still be lost.

**Dilated attention** (Longformer, BigBird): instead of attending to the nearest $W$ tokens, attend to every $d$-th token (dilation $d$). This gives $\mathcal{O}(n \cdot W / d)$ cost while covering a wider receptive field.

---

## 10.5 Sparse attention patterns

Beyond sliding window, several structured sparse patterns exist:

**Longformer** (Beltagy et al., 2020): combination of:
- Local window attention (all tokens, window $W$)
- Global attention (designated tokens attend everywhere — e.g., `[CLS]`, task-specific tokens)
- Random attention (random long-range connections)

**BigBird** (Zaheer et al., 2020): similar combination, proved that random sparse attention can approximate full attention theoretically.

These patterns see less use in modern decoder-only LLMs. Most production models use either full attention (with FlashAttention to manage cost) or sliding window attention, not complex hybrid patterns.

---

## 10.6 State Space Models (SSMs) and Mamba

The $\mathcal{O}(n^2)$ attention complexity has motivated research into architectures that scale linearly in sequence length.

**Linear recurrence:** an alternative to attention:

$$h_t = A h_{t-1} + B x_t, \quad y_t = C h_t$$

where $h_t$ is the hidden state, $A$ is a state transition matrix, and $B, C$ project input/output. This is a linear recurrence — can be computed in $\mathcal{O}(n)$ sequentially, or parallelized via the **parallel scan** algorithm in $\mathcal{O}(n \log n)$.

**S4** (Gu et al., 2021): parameterize $A$ using structured matrices (HiPPO theory) that provably maintain long-range information. Strong on long-sequence tasks (speech, genomics).

**Mamba** (Gu & Dao, 2023): the key innovation is making SSM parameters **input-dependent**:

$$B_t = \text{Linear}(x_t), \quad C_t = \text{Linear}(x_t), \quad \Delta_t = \text{Softplus}(\text{Linear}(x_t))$$

Unlike S4 (fixed A, B, C), Mamba's state update depends on the current input — allowing selective information retention. The model can "decide" what to keep in the hidden state.

**Tradeoff:**
- Mamba: $\mathcal{O}(n)$ training, $\mathcal{O}(1)$ per-token inference (constant-size hidden state)
- Attention: $\mathcal{O}(n^2)$ training, $\mathcal{O}(n)$ KV-cache growth

**Hybrid models** (Jamba, Zamba): alternate Mamba and attention layers. Attention layers handle tasks requiring exact retrieval from distant context; Mamba layers handle the rest cheaply. This achieves better throughput than pure attention at long contexts while matching quality on most tasks.

**Current state (2024–2025):** Mamba-scale models (~2–7B) are competitive with same-size Transformers. At 70B+ scale, pure SSMs still lag behind Transformers. Hybrid architectures are the most promising direction.

---

## 10.7 Mixture of Experts (MoE) — advanced

Covered in Chapter 06. Deeper look at the routing mechanism:

**Token-choice routing (top-k):** each token chooses the top-k experts. Problem: without auxiliary losses, all tokens route to the same 1–2 experts, leaving others unused. This is called **routing collapse**.

**Expert-choice routing** (Zhou et al., 2022): flip the relationship — each expert selects the top-$k$ tokens to process. Guarantees perfect load balance by construction, since each expert processes exactly $k$ tokens. Problem: some tokens may not be processed by any expert (dropped).

**Auxiliary load-balancing loss:**

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{e=1}^{N} f_e \cdot p_e$$

where $f_e$ is the fraction of tokens assigned to expert $e$ and $p_e$ is the average routing probability for expert $e$. This penalizes unequal load distribution. Typical $\alpha = 10^{-2}$.

**DeepSeek-V2/V3** (DeepSeek, 2024) achieves GPT-4-class performance with MoE at significantly lower inference cost. They use 256 experts with top-8 routing, effectively giving each token access to a different combination of specialists.

---

## 10.8 Efficient fine-tuning: LoRA

Full fine-tuning of a 70B model requires ~140 GB GPU memory for weights alone, plus optimizer states (~560 GB total). This is inaccessible to most.

**LoRA** (Hu et al., 2021) — Low-Rank Adaptation: freeze the pre-trained weights, add small trainable low-rank matrices:

$$W = W_0 + \Delta W = W_0 + BA$$

where $W_0 \in \mathbb{R}^{d \times k}$ is frozen, $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are trainable, and $r \ll \min(d, k)$ is the rank.

For Llama 3 70B ($d = k = 8192$), standard fine-tuning would train $8192^2 \approx 67M$ parameters per weight matrix. LoRA rank $r = 16$ adds $2 \times 8192 \times 16 \approx 262K$ parameters — a $256\times$ reduction.

**Why low rank works:** the hypothesis is that weight updates during fine-tuning have low intrinsic dimensionality. The task-specific adaptation can be captured in a low-dimensional subspace.

**QLoRA** (Dettmers et al., 2023): quantize the frozen base model to 4-bit (NF4 format), keep LoRA adapters in FP16. Fine-tune Llama 3 70B on a single 48 GB GPU. State-of-the-art open fine-tuning method.

---

## 10.9 Architecture comparison

| Architecture | Context | Training cost | Inference cost | Notes |
|-------------|---------|--------------|----------------|-------|
| Transformer (vanilla) | $\mathcal{O}(n^2)$ | $\mathcal{O}(n^2)$ | $\mathcal{O}(n)$ KV growth | FlashAttn makes this practical |
| Transformer + SWA | $\mathcal{O}(n \cdot W)$ | $\mathcal{O}(n \cdot W)$ | Fixed window KV | Loses distant info |
| MoE Transformer | $\mathcal{O}(n^2)$ | $\mathcal{O}(n^2)$ (sparse) | $\mathcal{O}(1/k)$ relative | Active params per token reduced |
| Mamba | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ | $\mathcal{O}(1)$ per step | Fixed hidden state size |
| Hybrid (Attn + SSM) | Mixed | Mixed | Mixed | Best of both at long context |

---

## Summary

| Innovation | What it solves | Used by |
|-----------|---------------|---------|
| FlashAttention | $\mathcal{O}(n^2)$ memory, slow attention | All major frameworks |
| GQA | KV-cache bandwidth | Llama 3, Mistral, Qwen |
| Sliding window attn | Long-context compute | Mistral, Mixtral |
| MoE | Parameter efficiency | Mixtral, GPT-4, DeepSeek |
| Mamba | Linear-time sequence modeling | Mamba-3, Jamba |
| LoRA / QLoRA | Fine-tuning cost | Open fine-tuning ecosystem |

---

## The big picture

As of 2025, the Transformer with Pre-LN, RMSNorm, RoPE, GQA, SwiGLU, and FlashAttention is the dominant architecture for large language models. It beats every proposed replacement at scale. But:

- **MoE** is now mainstream (DeepSeek V3, Mixtral) — not a research curiosity
- **Hybrid Transformer-SSM** is gaining traction for long contexts
- **Context length** has grown from 4k (GPT-2) → 128k (Llama 3.1) → 1M (Gemini 1.5 Pro) in three years
- **Quantization and LoRA** have democratized fine-tuning and inference

The core Transformer is stable. The surrounding engineering evolves rapidly. Understanding both layers — the mathematics of Chapters 03–07 and the systems of Chapters 08–10 — is what separates a practitioner from a user.

---

*This completes the book. Return to the [Table of Contents](../README.md).*
