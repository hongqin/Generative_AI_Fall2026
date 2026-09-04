# Week 6, Topic 2: Tokenization, Embeddings, and Context Windows

## Topic Overview

**Plain-English summary first.** Topic 1 covered the attention operation as if it took vectors as input. Real transformers take *text* as input. This Topic covers the things on either side of attention: how text becomes tokens, how tokens become vectors (token embeddings), how the model knows about position (positional encoding, because attention itself is permutation-invariant), and how the maximum length of input a model can process (the context window) has evolved from 512 tokens in 2018 to over a million tokens in recent models. The engineering for very long context is non-trivial and worth understanding even if you do not implement it yourself.

### Tokenization: turning text into integers

A neural network takes vectors, not strings. The first step is to convert text into a sequence of integer tokens drawn from a fixed vocabulary. Three common strategies:

- **Word-level tokenization**: split on whitespace and punctuation, give each word an integer. Simple but has huge vocabularies (English has millions of distinct word forms) and cannot handle unseen words.
- **Character-level**: each character is a token. Tiny vocabulary, handles anything, but sequences are very long (1 token per character).
- **Subword tokenization**: the modern default. Common words become single tokens; rare words split into meaningful pieces. Three popular variants:
  - **Byte-Pair Encoding (BPE)**: used by GPT-2, GPT-3, GPT-4, Llama. Build the vocabulary by iteratively merging the most-frequent adjacent character pairs.
  - **WordPiece**: used by BERT. Similar to BPE but with a likelihood-based merging criterion.
  - **SentencePiece**: used by T5, Llama. Operates on raw bytes (no pre-tokenization step), supports many languages cleanly.

Subword tokenization is the right balance: vocabularies stay small (~32k to 128k tokens), every input string can be tokenized, and common words are still single tokens. The tokenizer is part of the model; you must use the same tokenizer at inference that was used at training.

### Token embeddings: lookup tables

Once a string has been turned into a sequence of integer token ids $`(x_1, x_2, \dots, x_N)`$, the model maps each integer to a learnable vector by a lookup table. If the vocabulary has $`V`$ tokens and the model has dimension $`d_{\text{model}}`$, the embedding table is a learnable $`V \times d_{\text{model}}`$ matrix. Token $`x_i`$ becomes the $`x_i`$-th row of that matrix. Embedding parameters are a non-trivial fraction of total model size (e.g., a 50k vocabulary at $`d_{\text{model}} = 768`$ is 38M parameters, before any transformer blocks are added).

### Positional encoding: telling the model where each token is

Attention is permutation-invariant: shuffling the input tokens does not change the attention computation (modulo the masking). That is a problem because word order obviously matters. The fix is to **add positional information** to the token embeddings before they enter the first attention layer. Three approaches:

#### Sinusoidal positional encoding (Vaswani et al., 2017)

For position $`\mathrm{pos} \in \{0, 1, \dots, N-1\}`$ and dimension index $`i \in \{0, 1, \dots, d_{\text{model}}/2 - 1\}`$:

``` math
\mathrm{PE}(\mathrm{pos}, 2i) \;=\; \sin\!\left( \frac{\mathrm{pos}}{10000^{\,2i / d_{\text{model}}}} \right), \qquad \mathrm{PE}(\mathrm{pos}, 2i+1) \;=\; \cos\!\left( \frac{\mathrm{pos}}{10000^{\,2i / d_{\text{model}}}} \right).
```

**Plain English**: encode position with sinusoids of geometrically-spaced frequencies. The clever property: relative positions can be recovered from linear combinations of these encodings, so the model can learn to attend by relative offset. Fixed (not learned), generalizes to longer sequences than seen during training.

#### Learned positional embeddings

An extra $`N_{\max} \times d_{\text{model}}`$ learnable lookup table, indexed by position. Used by GPT-2 and many subsequent models. Simpler than sinusoidal but does not generalize beyond the maximum training length.

#### Rotary Position Embedding (RoPE)

Used by Llama, GPT-NeoX, recent OpenAI models. Instead of adding a positional vector to the embedding, RoPE rotates the query and key vectors by an angle proportional to their position before computing the dot product. This bakes relative position into the attention scores in a way that extrapolates better and behaves nicely at long context. The standard choice in 2024-2026 models.

### Context windows and the $`O(N^2)`$ problem

The **context window** is the maximum sequence length the model can process at once. Historical evolution:

| Model                | Year | Context window (tokens) |
|----------------------|------|-------------------------|
| Original transformer | 2017 | 512                     |
| GPT-2                | 2019 | 1,024                   |
| GPT-3                | 2020 | 2,048                   |
| GPT-4 (initial)      | 2023 | 8,192                   |
| GPT-4-Turbo          | 2023 | 128,000                 |
| Claude 3             | 2024 | 200,000                 |
| Gemini 1.5 Pro       | 2024 | 1,000,000+              |

The headline obstacle to longer context is the $`O(N^2)`$ memory cost of the attention matrix. Production systems use several techniques to push past it:

- **FlashAttention**: a hardware-aware reimplementation that computes attention without materializing the full $`N \times N`$ matrix in GPU memory. Same math, much smaller memory footprint.
- **Sliding-window attention**: each position only attends to the nearest $`w`$ positions. Loses long-range information but cuts cost to $`O(N \cdot w)`$.
- **Sparse and structured attention**: attend only to a structured subset of positions (e.g., a stride pattern).
- **Linear-time alternatives**: state-space models (Mamba), linear attention variants, recurrent attention. Active research area.

### Where this Topic fits

Topic 3 stacks self-attention into the complete transformer block and walks through how to actually train and run inference on a transformer model. Topic 4 zooms out to the practical-workflow lens: most students will not train a transformer from scratch; they will use someone else’s pre-trained model.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 9, sections on tokenization and positional encoding.

Recommended:

- R. Sennrich, B. Haddow, A. Birch, “Neural Machine Translation of Rare Words with Subword Units” (BPE), ACL 2016. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909).
- J. Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embedding,” 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864). The RoPE paper.

### Consider

- If you remove positional encoding entirely from a transformer trained on text, what specifically goes wrong, and why?
- BPE tokenization makes common words single tokens and rare words multi-token sequences. What does this imply about the model’s “effective context length” in characters vs in tokens?
- Context windows have grown ~2000x in seven years. For your course project, which dimension of context (length, structure, modality) matters most, and what bound does the chosen architecture put on you?

### Resources

- T. Dao et al., “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135). The hardware-aware attention reimplementation now used almost universally.
- A. Gu and T. Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). The leading non-attention alternative.
