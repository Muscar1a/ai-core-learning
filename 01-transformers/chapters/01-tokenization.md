# Chapter 01 — Tokenization

> Before any neural network can touch language, the text must become numbers. The choice of *how* to do this has far-reaching consequences for everything downstream.

---

## 1.1 Why not words?

The obvious approach: map every unique word to an integer. "cat" → 42, "running" → 1337. This is called a **word-level vocabulary**.

Problems:
- **Out-of-vocabulary (OOV):** "ChatGPT" wasn't in any 2010 corpus. A word-level model from 2010 has no token for it — it's invisible.
- **Morphological blindness:** "run", "runs", "running", "runner" are four separate, unrelated tokens. The model has to learn from scratch that they're related, requiring enormous data for each form.
- **Vocabulary explosion:** English has ~170,000 words in common use. Multiply across 50+ languages and you need millions of tokens, making the embedding matrix $E \in \mathbb{R}^{V \times d_{\text{model}}}$ enormous.

## 1.2 Why not characters?

Map every character to an integer: "a" → 1, "b" → 2, etc. Vocab size ≈ 256 (bytes). 

Problems:
- **Sequence length explodes:** "The cat sat" = 11 characters vs 3 words. Attention cost is $\mathcal{O}(n^2)$, so 3.7× longer sequences = 13.7× more compute. For GPT-3's training context of 2048 tokens, character-level would fit only ~400 words.
- **Harder to learn:** the model has to learn that "c"+"a"+"t" = cat from scratch, at every layer. Word-level starts with that already resolved.

## 1.3 Sub-word tokenization — the right trade-off

The insight: you don't need to choose between words and characters. Sub-word tokenization splits at a level between them, keeping common words whole and breaking rare words into pieces.

```
"unhappiness"  →  ["un", "happiness"]          ← rare word, known parts
"ChatGPT"      →  ["Chat", "G", "PT"]          ← unknown word, character-like fallback  
"the"          →  ["the"]                       ← common word, stays whole
"2024"         →  ["2024"]                      ← number, stays whole
```

The model can handle *any* word it's never seen, because it can always fall back to individual characters or bytes.

---

## 1.4 Byte Pair Encoding (BPE)

BPE was originally a data compression algorithm (Gage, 1994), adapted for NLP by Sennrich et al. (2016) and used by GPT-2 onwards.

### Algorithm

```
Input: corpus of text
Output: vocabulary of ~50,000 tokens

1. Initialize: vocab = { all individual characters in corpus }
2. Encode corpus using current vocab
3. Count all adjacent token pairs
4. Find the most frequent pair (e.g., "e" + "r" → "er")  
5. Merge that pair into a new token: add "er" to vocab
6. Replace all occurrences of ("e", "r") in encoded corpus with "er"
7. Repeat steps 3–6 until vocab reaches target size
```

### Worked example

Starting corpus (simplified): `"low lower lowest"`

Initial encoding: `l o w _ l o w e r _ l o w e s t`  
(where `_` = space, each character is its own token)

**Merge 1:** Most frequent pair = `(l, o)` → merge to `lo`
After: `lo w _ lo w e r _ lo w e s t`

**Merge 2:** Most frequent pair = `(lo, w)` → merge to `low`  
After: `low _ low e r _ low e s t`

**Merge 3:** Most frequent pair = `(e, r)` → merge to `er`
After: `low _ low er _ low e s t`

After 50,000 such merges on a real corpus (billions of tokens), common English words and morphemes stay intact, while rare combinations decompose gracefully.

### Byte-level BPE (GPT-2, GPT-3, GPT-4)

GPT-2 improved on standard BPE by operating at the byte level instead of the character level:

- Initialize vocab = all 256 possible bytes (not Unicode characters)
- Every possible string of bytes can be tokenized, including emoji, code, Chinese, Arabic, etc.
- No unknown characters are possible — the model can process any text in any language

This is why GPT-2/3/4 never produce `[UNK]` tokens — the fallback is always individual bytes.

---

## 1.5 WordPiece

Originally designed by Schuster & Nakajima (2012) for Japanese and Korean speech recognition, **WordPiece** was popularized in modern NLP by Devlin et al. (2018) for **BERT**.

### Merge Criterion: Maximum Likelihood
While standard BPE chooses whichever adjacent pair $(u, v)$ has the highest absolute frequency in the training corpus, WordPiece selects the pair that **maximizes the likelihood of the training data** under a statistical language model.

In practice, this is evaluated by maximizing the mutual information score:

$$\text{Score}(u, v) = \frac{\text{Count}(u, v)}{\text{Count}(u) \times \text{Count}(v)}$$

* **High score:** Tokens $u$ and $v$ appear together frequently, but appear independently relatively rarely (indicating a strong morphological or lexical bond, e.g., `"play"` + `"ing"`).
* **Low score:** Tokens occur adjacent to each other simply because one (or both) is globally ubiquitous (e.g., `"the"` + `"a"`), preventing accidental or inefficient merges.

### Continuation Tokens (`##`)
To distinguish subwords that begin a word from subwords that occur inside or at the end of a word, WordPiece prefixes continuation subwords with `##`:

$$\text{"unhappiness"} \longrightarrow [\text{"un"}, \text{"##happi"}, \text{"##ness"}]$$
$$\text{"playing"} \longrightarrow [\text{"play"}, \text{"##ing"}]$$

If an unseen character appears during inference that is completely absent from the vocabulary, WordPiece emits the special Out-Of-Vocabulary token: `[UNK]`.

---

## 1.6 SentencePiece & Byte Fallback

Developed by Taku Kudo and John Richardson at Google (2018), **SentencePiece** is an end-to-end tokenizer library and framework designed to overcome two fundamental limitations of classical BPE and WordPiece.

### 1. Language-Agnostic Raw Stream (No Pre-Tokenization)
Traditional tokenizers require language-specific *pre-tokenization* (e.g., splitting text on spaces and punctuation with regular expressions). This breaks down for non-segmented languages such as Chinese, Japanese, and Thai where words are written continuously without spaces.

SentencePiece treats the entire input as a **raw stream of characters/bytes**:
* Whitespace is treated as a regular character and replaced by the visible meta-symbol ` ` (`U+2581` Lower One Eighth Block).
* Example:
  $$\text{"Hello world"} \longrightarrow [\text{" Hello"}, \text{" world"}]$$
* **Lossless Detokenization:** Because whitespace is retained inside the token sequence, detokenization is completely reversible without language-specific reconstruction heuristics:
  $$\text{Detokenize}(T) = \text{Join}(T).\text{replace}(" ", " ").\text{strip}()$$

### 2. Byte Fallback (Zero `[UNK]` Tokens)
Earlier subword models suffered when encountering rare Unicode symbols, unseen alphabets, or novel emoji by falling back to `[UNK]`, destroying input information.

SentencePiece with **Byte Fallback** handles novel characters gracefully:
1. When a character is present in the subword vocabulary, it is encoded as a subword.
2. When a character or code point is **absent** from the vocabulary, SentencePiece decomposes the UTF-8 bytes of that character and emits raw byte tokens:
   $$\text{Emoji: } \text{"🌟"} \longrightarrow \text{UTF-8 bytes: } [0\text{xF0}, 0\text{x9F}, 0\text{x8C}, 0\text{x9F}] \longrightarrow [\text{"<0xF0>"}, \text{"<0x9F>"}, \text{"<0x8C>"}, \text{"<0x9F>"}] $$

This guarantees **100% vocabulary coverage** across all Unicode scripts without losing input fidelity.

---

## 1.7 Subword Algorithm Comparison Matrix

| Attribute | Byte Pair Encoding (BPE) | WordPiece | SentencePiece with Byte Fallback |
| :--- | :--- | :--- | :--- |
| **Origin & Paper** | Gage (1994), Sennrich et al. (2016) | Schuster & Nakajima (2012) | Kudo & Richardson (2018) |
| **Primary Criterion** | Maximum pair frequency: $\max \text{Count}(u, v)$ | Likelihood score: $\frac{\text{Count}(uv)}{\text{Count}(u)\text{Count}(v)}$ | Unigram LM (probabilistic) or BPE over raw stream |
| **Whitespace Handling** | Split by regex/whitespace first | Split by regex/whitespace first | Whitespace treated as normal symbol (` `) |
| **Continuation Marker** | Leading space (GPT) or end-of-word marker | `##` prefix (e.g. `##ing`) | Leading ` ` (e.g. ` play`) |
| **OOV Handling** | Byte-level BPE maps to 256 bytes | Emits `[UNK]` token | Falls back to `<0xXX>` UTF-8 byte tokens |
| **Prominent Models** | GPT-2/3/4, RoBERTa, LLaMA 3, Mistral | BERT, DistilBERT, Electra | T5, ALBERT, LLaMA 1 & 2, Gemma |

---

## 1.8 Vocabulary size analysis

The embedding matrix $E \in \mathbb{R}^{V \times d_{\text{model}}}$ makes vocabulary size a real engineering decision.

| Model | Vocab $V$ | $d_{\text{model}}$ | Embedding params $V \cdot d$ |
|-------|-----------|---------|---------------------------|
| GPT-2 | 50,257 | 1,600 | ~80M |
| GPT-3 | 50,257 | 12,288 | ~618M |
| Llama 3 | 128,256 | 8,192 | ~1.05B |
| GPT-4 (est.) | 100,256 | — | — |

**Larger vocab trade-offs:**

| Factor | Larger $V$ | Smaller $V$ |
|--------|-----------|------------|
| Tokens per sentence | Fewer (faster inference) | More (slower inference) |
| Embedding matrix | Larger (more memory) | Smaller |
| Multilingual coverage | Better | Worse (char-level fallback) |
| Rare word handling | Better | Worse |

**Tokens per word** varies by language. English averages ~1.3 tokens/word in BPE. Japanese and Chinese average 1.5–3 tokens/word with an English-dominated vocabulary, which means models trained primarily on English are less efficient (and effective) on other languages. Multilingual models like mT5 use larger vocabularies specifically to address this.

---

## 1.9 Tokenization artifacts and quirks

Tokenization introduces subtle artifacts that affect model behavior:

**Whitespace is part of the token:** In GPT-style tokenizers, `"cat"` and `" cat"` (with a leading space) are *different* tokens. This is why models sometimes add unexpected spaces.

**Numbers are inconsistently tokenized:** `"123"` might be one token; `"1234"` might be `["123", "4"]`. Arithmetic tasks are harder because numbers don't have a consistent sub-token structure.

**Capitalization:** `"Hello"` and `"hello"` are often different tokens, and `"HELLO"` is usually different again. This is why prompts with unusual capitalization can confuse models.

**The tokenization boundary problem:** A model generating the word "unfortunately" must predict `"un"`, then `"fort"`, then `"unately"` — three separate prediction steps. An error at step 2 produces "unfortunitely" — a valid token sequence that the model had to learn is wrong, even though each individual sub-word choice seemed plausible.

---

## 1.10 Counting tokens in practice

A rough rule: **1 token ≈ 4 characters ≈ 0.75 words** in English. But this varies significantly by content type:

- English prose: ~1.3 tokens/word
- Code (Python): ~2–4 tokens/line (identifiers tokenize cleanly, whitespace counts)
- Mathematical notation: ~3–6 tokens per symbol (LaTeX `\frac{a}{b}` → 5+ tokens)
- Chinese: ~2–2.5 tokens/character with GPT-4's tokenizer

**Context window in words:** GPT-3's 2048-token context ≈ 1,500 English words ≈ about 5 pages of text. GPT-4's 128k context ≈ ~96,000 words ≈ a short novel.

---

## Summary

| Concept | Key point |
|---------|-----------|
| Word-level | Simple but OOV and morphology problems |
| Character-level | No OOV but sequences too long |
| BPE | Merges frequent character pairs → sub-word units |
| WordPiece | Maximizes training data likelihood ratio: $\frac{\text{Count}(uv)}{\text{Count}(u)\text{Count}(v)}$ |
| SentencePiece | Language-agnostic raw stream (` `), lossless detokenization |
| Byte Fallback | Maps unseen Unicode characters to `<0xXX>` UTF-8 byte tokens (0% `[UNK]`) |
| Vocab size | Trade-off: inference speed vs memory vs coverage |

**Next:** [Chapter 02 — Embeddings](02-embeddings.md)
