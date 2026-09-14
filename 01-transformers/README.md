# How Transformers Work — A Comprehensive Guide

> **Level:** University / Professional AI Engineer. Assumes linear algebra, basic probability, and familiarity with neural networks.  
> **Interactive Laboratory:** Open [**`01-transformers.html`**](01-transformers.html) in any browser — live GPU-level math simulators, interactive heatmaps, and tensor visualizations. Navigate with `←` `→` keys or slide dots.

---

### Interactive Simulators Inside the HTML Guide

| Simulator | Focus | What You Can Test Live |
|---|---|---|
| **RNN vs. Transformer** | Parallelism & Gradients | Watch sequential BPTT decay vs. parallel $O(1)$ direct context routing across tokens |
| **Live BPE Tokenizer** | Subword Decomposition | Type arbitrary text to view byte-pair splits, token counts, and simulated hash vocabulary IDs |
| **Semantic Embedding Map** | Vector Geometry | Interactive 2D projection map with semantic clusters and the classic $\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$ vector arithmetic |
| **Positional Encoding & RoPE** | Translation Invariance | Toggle absolute sinusoidal signals vs. **Rotary Position Embedding (RoPE)** with live 2D coordinate rotation angle sliders |
| **Self-Attention & $\sqrt{d_k}$ Scaling** | Softmax Saturation | **Full $n \times n$ attention heatmap** + live $d_k$ slider ($1 \to 512$) testing unscaled softmax collapse vs. scaled variance normalization |
| **MHA vs. GQA vs. MQA** | KV-Cache Bandwidth | Interactive architecture switcher comparing private KV heads, shared KV heads, and grouped query head configurations with real-time memory calculation |
| **Feed-Forward Knowledge Transformation** | Associative Memory | Per-word semantic transformation cards showing state before and after the $d \to 4d \to d$ expansion |
| **Normalization & Residual Highways** | Gradient Stability | Step-by-step comparative calculator: **LayerNorm vs. RMSNorm** (demonstrating why skipping mean-centering saves memory bandwidth) |
| **Autoregressive Decoding & Sampling** | Inference Dynamics | Live **Temperature ($\tau$) slider** ($0.1 \to 2.5$) reshaping output probability distributions + token candidate picker and generator |
| **Modern Architecture Pipeline** | FlashAttention & ViT | Clickable end-to-end tensor pipeline accordion with per-step mathematical specifications, FlashAttention tiling mechanics, and model parameter counts |

---

## Curriculum Chapters

| # | Chapter | Key Mathematical & Engineering Insights |
|:---:|---|---|
| [**01**](chapters/01-tokenization.md) | **[Tokenization](chapters/01-tokenization.md)** | Subword tokenization, BPE merge algorithm, WordPiece, SentencePiece byte fallback, vocabulary scaling tradeoffs, and tokenization artifacts. |
| [**02**](chapters/02-embeddings.md) | **[Embeddings](chapters/02-embeddings.md)** | High-dimensional geometric representation, cosine similarity, distributional hypothesis, embedding matrices $E \in \mathbb{R}^{V \times d_{\text{model}}}$, and why permutation invariance requires positional signals. |
| [**03**](chapters/03-attention.md) | **[Self-Attention](chapters/03-attention.md)** | Query-Key-Value vector projections, scaled dot-product attention derivation, **full mathematical variance proof $\text{Var}(q \cdot k) = d_k$**, softmax saturation demonstration, and a complete 2-token step-by-step numerical walkthrough. |
| [**04**](chapters/04-multihead-attention.md) | **[Multi-Head Attention](chapters/04-multihead-attention.md)** | Parallel subspace representations, linear output projection $W_O$, Cross-Attention (decoder queries attending to encoder keys/values), **MHA vs. MQA vs. GQA** generalization formula, and the **5% uptraining technique**. |
| [**05**](chapters/05-positional-encoding.md) | **[Positional Encoding](chapters/05-positional-encoding.md)** | Absolute sinusoidal encodings, learned embeddings, **Rotary Position Embedding (RoPE)** 2D rotation matrix formulation, orthogonality cancellation proof $(R_m Q) \cdot (R_n K) = Q \cdot (R_{n-m} K)$, numeric proof of translation invariance ($0.707$), and YaRN context extension. |
| [**06**](chapters/06-feed-forward.md) | **[Feed-Forward Networks](chapters/06-feed-forward.md)** | Position-wise FFN, $d_{\text{model}} \to 4d_{\text{model}} \to d_{\text{model}}$ expansion-contraction, FFN as associative key-value factual memory, non-linear activations (ReLU, GELU, and SwiGLU). |
| [**07**](chapters/07-normalization-residuals.md) | **[Normalization & Residuals](chapters/07-normalization-residuals.md)** | Residual skip connections as gradient highways ($+1$ in chain rule), Pre-LN vs. Post-LN training stability, BatchNorm vs. LayerNorm comparison, and **RMSNorm** (formula, variance calculation without mean-centering, 10–50% kernel speedup). |
| [**08**](chapters/08-training.md) | **[Training & Optimization](chapters/08-training.md)** | Autoregressive cross-entropy loss, causal language modeling, AdamW optimizer with decoupled weight decay, learning rate warmup + cosine schedules, and Chinchilla compute-optimal scaling laws ($N \approx 20D$). |
| [**09**](chapters/09-inference.md) | **[Inference & KV-Cache](chapters/09-inference.md)** | Prefill (compute-bound) vs. Decode (memory-bandwidth-bound), KV-cache exact sizing equation, sampling strategies (Greedy, Temperature, Top-$k$, Nucleus Top-$p$, Min-$p$), Speculative Decoding, and Quantization (FP16, INT8, INT4 AWQ/GPTQ). |
| [**10**](chapters/10-modern-variants.md) | **[Modern Variants](chapters/10-modern-variants.md)** | **FlashAttention (v1–v3)** (SRAM vs. HBM memory hierarchy, tiling, online softmax algorithm), Sliding Window Attention, Mamba State Space Models, **Mixture of Experts (MoE)** routing and load balancing, and **Vision Transformers (ViT)** 6-step patchification breakdown. |

---

## Core Mathematical Reference

### 1. Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$
* **Variance normalization:** If $q_i, k_i \sim \mathcal{N}(0, 1)$, then $\text{Var}(q \cdot k) = d_k$. Dividing by $\sqrt{d_k}$ resets logit variance to $1.0$, preventing softmax saturation and vanishing gradients.

### 2. Rotary Position Embedding (RoPE)
$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}, \qquad \theta_i = 10000^{-2(i-1)/d}$$
* **Relative invariance:** $(R_m Q) \cdot (R_n K) = Q^\top (R_m^\top R_n) K = Q^\top R_{n-m} K$, meaning attention scores depend exclusively on the relative separation distance $\Delta = n - m$.

### 3. Root Mean Square Layer Normalization (RMSNorm)
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}, \qquad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \varepsilon}$$
* **Speedup:** Eliminates mean calculation and subtraction, reducing memory bandwidth transfers by up to $30\text{--}50\%$ without degrading model quality.

### 4. Modern Pre-LN Transformer Block
$$\mathbf{x}^{(l)} \leftarrow \mathbf{x}^{(l-1)} + \text{Attn}\!\left(\text{RMSNorm}(\mathbf{x}^{(l-1)})\right)$$
$$\mathbf{x}^{(l)} \leftarrow \mathbf{x}^{(l)} + \text{FFN}\!\left(\text{RMSNorm}(\mathbf{x}^{(l)})\right)$$

### 5. KV-Cache Memory Equation (with GQA)
$$\text{KV Memory} = 2 \times L \times G \times d_k \times N_{\text{tokens}} \times \text{bytes per element}$$
* where $G$ is the number of Key-Value head groups ($G = H$ for MHA, $G = 1$ for MQA, $1 < G < H$ for GQA). For LLaMA 3 70B ($G = 8, H = 64$), GQA achieves an **$8\times$ memory reduction**.

### 6. Parameter Counting Rule-of-Thumb
$$\text{Params per Block} \approx 4\,d_{\text{model}}^2 \text{ (Attention)} + 8\,d_{\text{model}}^2 \text{ (FFN)} = 12\,d_{\text{model}}^2$$
$$\text{Total Parameters} \approx L \times 12\,d_{\text{model}}^2 + V \times d_{\text{model}}$$
* **GPT-3 Check:** $96 \text{ layers} \times 12 \times 12{,}288^2 + 50{,}257 \times 12{,}288 \approx 174\text{B} + 0.6\text{B} \approx \mathbf{175\text{B}}$ ✓
