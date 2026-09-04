# Week 6, Topic 3: Transformer Layers, Training Intuition, and Inference Behavior

## Topic Overview

**Plain-English summary first.** A transformer model is a stack of identical building blocks called *transformer blocks*, each containing a self-attention layer and a small MLP, wrapped with layer-norm and residual connections. Stack twelve of them and you have GPT-1; stack ninety-six of them at much wider sizes and you have GPT-3. This Topic covers the block structure, the three architectural variants (encoder-only, decoder-only, encoder-decoder) and what each is good for, the training objective (next-token cross-entropy, same as Module 4 autoregressive but at much larger scale), the inference trick (KV-cache) that makes generation efficient, and the empirical scaling laws that explain why bigger transformer models keep getting better.

### The transformer block

One block takes a sequence of vectors of shape $`N \times d_{\text{model}}`$ as input and produces the same shape as output. It applies, in order:

1.  Layer normalization (pre-norm; most modern transformers use this).
2.  Multi-head self-attention (causal-masked for decoder transformers).
3.  A residual connection adding the input back.
4.  Another layer normalization.
5.  A position-wise MLP (two linear layers with a non-linearity, expanding then contracting; the inner dimension is typically $`4 \cdot d_{\text{model}}`$).
6.  Another residual connection.

In equations, given input $`x`$ (shape $`N \times d_{\text{model}}`$):

``` math
y \;=\; x \,+\, \mathrm{MultiHead}\!\left(\mathrm{LN}(x)\right), \qquad \mathrm{Block}(x) \;=\; y \,+\, \mathrm{MLP}\!\left(\mathrm{LN}(y)\right).
```

The residual connections (the $`x \,+\, \dots`$ and $`y \,+\, \dots`$ patterns) are what let you stack many blocks without the gradient vanishing. Pre-norm vs post-norm refers to whether layer-norm is applied before each sublayer (pre-norm, modern default, more stable training) or after each sublayer plus residual (post-norm, original 2017 paper).

### Three architectural variants

#### Encoder-only (BERT-style)

Self-attention is bidirectional (no causal mask): every position attends to every position. Pre-trained on *masked language modeling*: randomly mask out 15% of input tokens and train the model to predict them from context. Used for tasks that require understanding (classification, named-entity recognition, sentence embedding, retrieval).

Cannot generate sequentially; not a generative model in the way GPT is.

#### Decoder-only (GPT-style)

Self-attention is causal-masked: position $`i`$ only attends to positions $`j \leq i`$. Trained as an autoregressive language model: at each position, predict the next token. The training objective is exactly the per-token cross-entropy from Module 4:

``` math
\mathcal{L}(\theta) \;=\; -\sum_{i=1}^{N} \log p_\theta\!\left(x_i \,\middle|\, x_{\lt i}\right).
```

This is the architecture that became GPT-2, GPT-3, GPT-4, Claude, Llama, and essentially every modern large language model. The vast majority of modern transformer work happens in this variant.

#### Encoder-decoder (T5, BART, original translation transformer)

Two transformer stacks. The encoder processes the input bidirectionally; the decoder generates the output autoregressively and uses *cross-attention* to attend to the encoder’s output. Used for tasks where input and output are different (machine translation, summarization, conditional generation). Less common in modern LLM work but still important for structured tasks.

### Training intuition

Training a decoder-only transformer on a corpus is, mathematically, exactly the same as training the LSTM autoregressive model from Module 4: maximize $`\sum_i \log p_\theta(x_i \mid x_{\lt i})`$ summed over all tokens in your training set. The differences are entirely in (a) the architecture (self-attention instead of recurrence) and (b) scale (billions of parameters, trillions of tokens, weeks on thousands of GPUs).

What you can train on a single GPU in a course is a *small* transformer: 12 layers, 768 hidden dimensions, 12 attention heads, ~125M parameters, on a small text corpus, in a few hours. Foster’s `gpt` notebook does exactly this on a small Keras setup. The architecture is identical to GPT-2; only the scale is different. This is the version you can fully understand and train; the rest of the modern LLM story is “same thing, much bigger, much more data.”

### KV-cache: the inference trick that makes generation tractable

Naive autoregressive generation with a transformer would re-run the entire forward pass over the full prefix at each new token. This is wasteful because the attention computation for positions you already processed does not change when you add a new token. The **KV-cache** stores the per-layer key and value tensors from previously-processed positions and re-uses them. Each new token then only requires computing its own query, key, and value and one row of attention against the cached keys and values.

Without KV-cache, generating an $`N`$-token sequence is $`O(N^3)`$. With KV-cache, it is $`O(N^2)`$. This is the difference between “you can’t actually run inference” and “ChatGPT works.”

### Scaling laws: why bigger keeps working

The empirical observation that defined the GPT-3 era: model performance (measured as test loss) follows predictable power laws in model size, dataset size, and compute (Kaplan et al., 2020; Hoffmann et al., 2022). Concretely:

- For a fixed compute budget, there is an optimal model size and dataset size; doing more of either alone underperforms.
- The Chinchilla scaling law (Hoffmann et al., 2022) found that earlier large models (including GPT-3 itself) were severely under-trained on data; for a given parameter count, you want roughly 20 tokens of training data per parameter.
- Scaling continues to work surprisingly far. Even at the frontier (trillion-parameter models trained on tens of trillions of tokens) the laws still roughly hold; this is why the field keeps scaling up.

### Where this Topic fits

Topic 4 turns to the practical workflow: most students will not train a transformer from scratch. The modern toolkit is Hugging Face `transformers` for pre-trained models, plus fine-tuning, prompting, or RAG. The Module 6 assignment lets you pick between “train Foster’s small GPT from scratch” (close to this Topic) and “use a pre-trained model end to end” (close to Topic 4).

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 9, sections on the transformer block, encoder/decoder architectures, and the GPT-style decoder.
- A. Radford et al., “Language Models are Unsupervised Multitask Learners” (GPT-2), 2019. [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Sections 1 and 2.

Recommended for scaling laws:

- J. Hoffmann et al., “Training Compute-Optimal Large Language Models” (Chinchilla), NeurIPS 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556).

### Consider

- The decoder-only variant won. Why has bidirectional encoder-only (BERT) lost ground to causal decoder-only (GPT) for most modern tasks, even tasks like classification where BERT was originally stronger?
- The KV-cache cuts generation cost from $`O(N^3)`$ to $`O(N^2)`$. For a chat application with 100k-token context windows, what other engineering tricks are needed to keep latency tolerable?
- Scaling laws predict that bigger models trained on more data keep improving. What would it take for that prediction to break down?

### Resources

- J. Kaplan et al., “Scaling Laws for Neural Language Models,” 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361). The pre-Chinchilla scaling-laws paper.
- T. Brown et al., “Language Models are Few-Shot Learners” (GPT-3), NeurIPS 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165). The empirical demonstration that scale produces qualitatively new behavior.
