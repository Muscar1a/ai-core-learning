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

## 1.5 WordPiece and SentencePiece

**WordPiece** (used by BERT, RoBERTa): similar to BPE but merges based on maximizing training data likelihood rather than frequency. Slightly different token boundaries in practice. Uses `##` prefix to mark continuation tokens: "running" → `["run", "##ning"]`.

**SentencePiece** (used by T5, Llama): treats the input as a raw byte stream — no pre-tokenization by whitespace. This makes it language-agnostic. Uses `▁` to mark word boundaries: `"Hello world"` → `["▁Hello", "▁world"]`. Llama 3's tokenizer is SentencePiece with a vocabulary of 128,256.

---

## 1.6 Vocabulary size analysis

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

## 1.7 Tokenization artifacts and quirks

Tokenization introduces subtle artifacts that affect model behavior:

**Whitespace is part of the token:** In GPT-style tokenizers, `"cat"` and `" cat"` (with a leading space) are *different* tokens. This is why models sometimes add unexpected spaces.

**Numbers are inconsistently tokenized:** `"123"` might be one token; `"1234"` might be `["123", "4"]`. Arithmetic tasks are harder because numbers don't have a consistent sub-token structure.

**Capitalization:** `"Hello"` and `"hello"` are often different tokens, and `"HELLO"` is usually different again. This is why prompts with unusual capitalization can confuse models.

**The tokenization boundary problem:** A model generating the word "unfortunately" must predict `"un"`, then `"fort"`, then `"unately"` — three separate prediction steps. An error at step 2 produces "unfortunitely" — a valid token sequence that the model had to learn is wrong, even though each individual sub-word choice seemed plausible.

---

## 1.8 Counting tokens in practice

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
| WordPiece | Like BPE but likelihood-based merges |
| SentencePiece | Language-agnostic, no whitespace pre-split |
| Vocab size | Trade-off: inference speed vs memory vs coverage |

**Next:** [Chapter 02 — Embeddings](02-embeddings.md)
