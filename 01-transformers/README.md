# How Transformers Work — A Deep Guide

> **Level:** University. Assumes linear algebra, basic calculus, familiarity with neural networks.
> **Interactive:** Open [01-transformers.html](01-transformers.html) — rendered math, navigate with ← →.

### Interactive demos inside the HTML

| Page | Demo |
|------|------|
| Tokenization | Live BPE tokenizer — type any text, see tokens + IDs update |
| Embeddings | Hover SVG word map showing semantic clusters |
| Positional Encoding | Toggle position labels on/off per token |
| Self-Attention | **Full n×n attention heatmap** — click any word to highlight its query row; all 36 pairwise weights visible |
| Self-Attention | **√d_k saturation slider** — drag d_k from 1→512, watch unscaled softmax collapse to one-hot |
| Feed-Forward | Per-word semantic transformation: "before FFN" vs "after FFN" cards |
| Stacking Layers | Layer depth slider showing what abstraction each depth captures |
| Predicting Output | **Temperature slider τ** — drag from 0.1→2.5, probabilities reshape live; greedy vs nucleus regime visible |
| Predicting Output | Candidate picker + autoregressive generation animation |
| Full Architecture | Clickable pipeline accordion with per-step math |

---

## Chapters

| # | Chapter | What you'll understand |
|---|---------|------------------------|
| [01](chapters/01-tokenization.md) | Tokenization | How BPE works, vocab size trade-offs, byte-level encoding |
| [02](chapters/02-embeddings.md) | Embeddings | Vector spaces, distributional hypothesis, geometric structure |
| [03](chapters/03-attention.md) | Self-Attention | Scaled dot-product derivation, masking, complexity |
| [04](chapters/04-multihead-attention.md) | Multi-Head Attention | Why multiple heads, what they learn, parameter count |
| [05](chapters/05-positional-encoding.md) | Positional Encoding | Sinusoidal PE, RoPE derivation, ALiBi |
| [06](chapters/06-feed-forward.md) | Feed-Forward Networks | FFN as memory, GELU, SwiGLU, MoE |
| [07](chapters/07-normalization-residuals.md) | Normalization & Residuals | LayerNorm, Pre-LN vs Post-LN, gradient flow |
| [08](chapters/08-training.md) | Training | Cross-entropy, Adam, scaling laws, data |
| [09](chapters/09-inference.md) | Inference | KV-cache, sampling, speculative decoding |
| [10](chapters/10-modern-variants.md) | Modern Variants | FlashAttention, GQA, Mamba, sparse attention |

---

## Quick reference

**The core equation:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**One Transformer block:**

$$\mathbf{x} \leftarrow \mathbf{x} + \text{Attn}(\text{LN}(\mathbf{x})), \qquad \mathbf{x} \leftarrow \mathbf{x} + \text{FFN}(\text{LN}(\mathbf{x}))$$

**Parameter count per block:** $\approx 12\, d_{\text{model}}^2$ · **Total:** $\approx L \times 12\, d_{\text{model}}^2 + V \times d_{\text{model}}$

**GPT-3 check:** $96 \times 12 \times 12288^2 \approx 174\text{B}$ ✓
