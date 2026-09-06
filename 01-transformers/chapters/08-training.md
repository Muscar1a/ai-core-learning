# Chapter 08 — Training

> A Transformer learns by predicting the next token. Everything — cross-entropy loss, Adam, the scaling laws — connects back to that single objective.

---

## 8.1 The training objective

Decoder-only LLMs are trained with **next-token prediction** (autoregressive language modeling). Given a sequence $(t_1, t_2, \ldots, t_n)$, the model predicts each token from all preceding tokens.

The loss for one sequence:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \log P(t_i \mid t_1, \ldots, t_{i-1}; \theta)$$

This is the **cross-entropy loss** averaged over $n$ positions. The model minimizes this by maximizing the log-probability of the correct next token.

**Why cross-entropy?** Given a true distribution $p$ (one-hot: $p_c = 1$ for correct class, $0$ elsewhere) and model distribution $q$:

$$H(p, q) = -\sum_c p_c \log q_c = -\log q_c$$

where $c$ is the correct token. Minimizing cross-entropy = maximizing the log-probability of the correct token. The optimal model achieves $q_c = 1$ (perfect certainty on the right answer), giving $H = 0$.

---

## 8.2 Perplexity

**Perplexity** is the standard evaluation metric for language models:

$$\text{PPL} = e^{\mathcal{L}} = \exp\!\left(-\frac{1}{n}\sum_{i=1}^n \log P(t_i \mid t_{<i})\right)$$

Interpretation: perplexity is the average number of tokens the model is "surprised by" — a model with PPL = 20 is, on average, as uncertain as if it had to choose uniformly among 20 options.

| Model | PPL (PTB) | PPL (WikiText-103) |
|-------|-----------|-------------------|
| n-gram (5-gram) | 141 | — |
| LSTM | 58 | 48 |
| GPT-2 (1.5B) | — | 18.3 |
| GPT-3 (175B) | — | ~10 |

Lower perplexity = better language model. PPL is comparable across vocabulary sizes only when the same tokenizer is used (different tokenizations make PPL incomparable).

---

## 8.3 Teacher forcing

During training, the model always receives the **true** previous tokens as input, not its own predictions. This is called **teacher forcing**.

Example: true sequence is "The cat sat on the mat".

- At position 3 ("sat"), the model gets the input "The cat" (ground truth), not whatever it predicted at position 2.
- Even if the model incorrectly predicted "cat → dog", it still sees "cat" at the next step.

**Why:** without teacher forcing, errors compound — a wrong prediction at step 2 causes the model to see the wrong context at step 3, causing another error, which causes another... Training becomes unstable.

**Consequence (exposure bias):** at inference time, the model sees its own predictions (since there's no ground truth to follow). The training distribution (always perfect context) differs from the inference distribution (its own potentially imperfect context). This mismatch is known as **exposure bias** and can cause error accumulation in long sequences.

Mitigation: **scheduled sampling** gradually substitutes model predictions for true tokens during training, closing the gap. Most modern LLMs still use teacher forcing without this fix, as very large models seem to handle it well empirically.

---

## 8.4 The Adam optimizer

Plain stochastic gradient descent (SGD) applies $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$. It works but converges slowly and is sensitive to learning rate choice.

**Adam** (Kingma & Ba, 2015) — Adaptive Moment Estimation — maintains a moving average of both gradients and squared gradients:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \qquad \text{(first moment — "momentum")}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \qquad \text{(second moment — "velocity")}$$

**Bias correction** (both moments are initialized at 0, so early estimates are biased toward 0):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter update:**

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

**Standard hyperparameters:**
- $\beta_1 = 0.9$ (momentum decay — uses last ~10 gradients)
- $\beta_2 = 0.999$ (velocity decay — uses last ~1000 gradient squares)
- $\varepsilon = 10^{-8}$ (prevents division by zero)
- $\eta$ — learning rate, scheduled separately

**Why Adam works:** the denominator $\sqrt{\hat{v}_t}$ normalizes the update by the recent gradient magnitude. Parameters that consistently receive large gradients get a smaller effective learning rate; parameters with small gradients get a larger effective rate. Adam adapts per-parameter, making training less sensitive to the initial learning rate choice.

**AdamW:** Adam + weight decay decoupled from the gradient update. Standard SGD applies weight decay as a gradient penalty ($\nabla_\theta \mathcal{L} + \lambda \theta$), which Adam then distorts through adaptive scaling. AdamW applies weight decay directly:

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \theta_{t-1}\right)$$

Most modern LLMs use AdamW. Typical $\lambda = 0.1$.

---

## 8.5 Learning rate schedule

A fixed learning rate doesn't work well for LLM training. The standard schedule:

### Warmup phase

For the first $T_{\text{warm}}$ steps, increase linearly from 0 to $\eta_{\text{max}}$:

$$\eta_t = \eta_{\text{max}} \cdot \frac{t}{T_{\text{warm}}}$$

Warmup is critical because at initialization the parameters are random, gradients are noisy, and a large learning rate causes catastrophic updates. GPT-3 uses $T_{\text{warm}} = 375$ steps; modern models often use 1000–4000 steps.

### Cosine decay phase

After warmup, decay the learning rate following a cosine curve down to a minimum $\eta_{\text{min}}$ (typically $\eta_{\text{max}}/10$):

$$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\!\left(\pi \cdot \frac{t - T_{\text{warm}}}{T_{\text{total}} - T_{\text{warm}}}\right)\right)$$

The cosine shape starts decaying slowly, then accelerates in the middle, then slows near the end — tracking the natural dynamics of gradient-based optimization well empirically.

---

## 8.6 Scaling laws

Kaplan et al. (2020) discovered that LLM performance follows **smooth power laws** in three variables: model parameters $N$, training tokens $D$, and compute budget $C$.

$$\mathcal{L}(N) \propto N^{-\alpha_N}, \quad \mathcal{L}(D) \propto D^{-\alpha_D}, \quad \mathcal{L}(C) \propto C^{-\alpha_C}$$

Key insight: **larger models are more sample-efficient**. When scaling compute, it's better to train a larger model on fewer tokens than a smaller model on more tokens (within the original Kaplan analysis).

### Chinchilla scaling laws

Hoffmann et al. (2022) — *"Training Compute-Optimal Large Language Models"* — showed that Kaplan's models were significantly undertrained. The compute-optimal relationship:

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

**Chinchilla rule:** to train compute-optimally, use roughly **20 tokens per parameter**:

$$D_{\text{opt}} \approx 20 \times N$$

| Model | Parameters | Training tokens | Tokens/param | Verdict |
|-------|-----------|----------------|--------------|---------|
| GPT-3 | 175B | 300B | 1.7 | Undertrained |
| Gopher | 280B | 300B | 1.1 | Undertrained |
| Chinchilla | 70B | 1.4T | 20 | Compute-optimal |
| Llama 2 70B | 70B | 2T | 28.6 | Overtrained (cheap inference) |
| Llama 3 70B | 70B | 15T | 214 | Heavily overtrained |

**Why overtrain intentionally?** After training, inference is cheap. A smaller, overtrained model (70B on 15T tokens) outperforms a larger, undertrained model (175B on 300B tokens) at the same quality level, but the 70B model runs 2.5× faster at inference. For deployed models, this tradeoff is worth it.

---

## 8.7 Data and tokenization at scale

Training token counts require enormous data:

| Source | Estimated tokens |
|--------|-----------------|
| Common Crawl (1 year) | ~10T |
| Books (all digitized) | ~400B |
| GitHub (all public code) | ~1T |
| Wikipedia (all languages) | ~20B |

Typical data mixture for a modern LLM: ~70% web text, ~15% code, ~10% books, ~5% other curated sources. The exact mixture is a hyperparameter — models trained with more code generalize better at reasoning, even on non-code tasks.

**Data quality matters more than quantity.** FineWeb (HuggingFace, 2024) showed that careful filtering of Common Crawl data produces a 15T-token dataset that outperforms unfiltered 100T-token datasets. Quality filters: language detection, perplexity filtering (too-easy text removed), deduplication, NSFW filters.

---

## Summary

| Concept | Key point |
|---------|-----------|
| Cross-entropy | $-\log P(\text{correct token})$ — minimize this |
| Perplexity | $e^{\mathcal{L}}$ — average "surprise factor" |
| Teacher forcing | Always use ground-truth context during training |
| Adam | Adaptive per-parameter learning rates via moment estimation |
| AdamW | Adam with decoupled weight decay — standard for LLMs |
| LR schedule | Warmup → cosine decay; warmup prevents early instability |
| Chinchilla | ~20 tokens/param for compute-optimal; practical deployments overtrain |

**Next:** [Chapter 09 — Inference](09-inference.md)
