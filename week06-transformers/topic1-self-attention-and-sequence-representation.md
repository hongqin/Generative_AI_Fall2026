# Week 6, Topic 1: Self-Attention and Sequence Representation

## Topic Overview

**Plain-English summary first.** Module 4 covered RNN-style autoregressive models where each position is processed in turn, threading a hidden state forward through time. The transformer’s defining move is to drop recurrence entirely and instead let every position in a sequence directly look at every other position via a mechanism called **self-attention**. This single change unlocked deeper networks, longer contexts, and the scale that produced modern large language models. The price is computational cost that grows with the square of the sequence length, but for sequences shorter than a few tens of thousands of tokens it is the right tradeoff.

### The attention operation in one line

Given a sequence of token representations, project each position three times to produce a **query** $`q_i`$, a **key** $`k_i`$, and a **value** $`v_i`$ (each a vector of dimension $`d_k`$, $`d_k`$, and $`d_v`$ respectively; typically $`d_k = d_v = d_{\text{model}} / h`$ where $`h`$ is the number of heads). Stack them into matrices $`Q, K, V`$, each of shape $`N \times d`$. Then **scaled dot-product attention** is:

``` math
\mathrm{Attention}(Q, K, V) \;=\; \mathrm{softmax}\!\left( \frac{Q K^{\top}}{\sqrt{d_k}} \right) V.
```

#### What each piece is doing, in plain language

- $`Q K^{\top}`$ is a matrix of *similarity scores*: entry $`(i, j)`$ is the dot product of query $`q_i`$ with key $`k_j`$, measuring how relevant position $`j`$ is to position $`i`$.
- Dividing by $`\sqrt{d_k}`$ keeps those scores at a reasonable scale so that the next step does not saturate. Without this, large dot products would push the softmax into regions with vanishing gradients.
- $`\mathrm{softmax}`$ along each row turns the similarity scores into a probability distribution over positions: how much attention does position $`i`$ pay to position $`j`$?
- Multiplying by $`V`$ takes a weighted sum of values, using those attention weights. Position $`i`$’s output is a mixture of all positions’ values, weighted by how relevant they are to $`i`$.

You can think of attention as a *differentiable lookup table*: queries probe the keys; relevance scores route information from values back to the query positions.

### Multi-head attention

One attention operation gives the model one “way” of attending. Multi-head attention runs $`h`$ independent attention operations in parallel, each with its own learned projection matrices, then concatenates the results and applies a final linear projection:

``` math
\mathrm{head}_i \;=\; \mathrm{Attention}\!\left(Q W^{Q}_i,\, K W^{K}_i,\, V W^{V}_i\right), \qquad \mathrm{MultiHead}(Q, K, V) \;=\; \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W^{O}.
```

Each head can specialize on a different kind of relationship (one head might track subject-verb agreement; another might track coreference; another might track positional offset). Empirically, models with $`h = 8`$ to $`16`$ heads behave much better than single-head attention.

### Causal masking: how attention becomes autoregressive

The basic attention operation has each position attend to every other position. For autoregressive generation (like GPT), position $`i`$ must only see positions $`j \leq i`$ (otherwise the model would “cheat” by looking at the future during training). The standard trick: add a mask $`M`$ to the scores before the softmax, with $`M_{ij} = -\infty`$ for $`j \gt i`$ and $`M_{ij} = 0`$ otherwise. The softmax of $`-\infty`$ is zero, so those positions get zero attention weight:

``` math
\mathrm{Attention}^{\text{causal}}(Q, K, V) \;=\; \mathrm{softmax}\!\left( \frac{Q K^{\top}}{\sqrt{d_k}} + M \right) V, \quad M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j \gt i \end{cases}.
```

This is the only architectural difference between an encoder transformer (which uses bidirectional attention) and a decoder transformer (which uses causal attention). Topic 3 returns to this distinction.

### The computational cost

For a sequence of length $`N`$, the attention matrix $`Q K^{\top}`$ has shape $`N \times N`$. Both compute and memory are $`O(N^2 \cdot d)`$. For modest sequences (a few thousand tokens) this is fine on a single GPU. For very long contexts (32k, 128k, 1M tokens) the quadratic scaling becomes the bottleneck, and a whole subfield (FlashAttention, sliding-window attention, sparse attention, state-space models) exists to mitigate it. Topic 2 returns to this when discussing context-window evolution.

### Where this Topic fits

Topic 2 covers the things outside the attention operation itself: how text becomes tokens, how tokens become vectors (token embeddings), how the model knows about position (positional encoding), and how context length has evolved. Topic 3 stacks attention into a complete transformer block and walks through the three architectural variants. Topic 4 covers what to do when you do not want to train a transformer from scratch.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 9, intro sections through “Self-Attention.”
- A. Vaswani et al., “Attention Is All You Need,” NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Sections 1 to 3 (the canonical attention derivation).

Recommended (the line-by-line code companion):

- S. Rush, “The Annotated Transformer,” Harvard NLP blog. [nlp.seas.harvard.edu/annotated-transformer](http://nlp.seas.harvard.edu/annotated-transformer/).

### Consider

- The $`\sqrt{d_k}`$ scaling looks like a fiddly engineering detail. What goes wrong, mathematically, if you remove it for large $`d_k`$?
- Multi-head attention runs $`h`$ parallel attention operations with their own projections. Why is that strictly more expressive than one attention operation with $`h`$ times more dimensions?
- Self-attention is $`O(N^2)`$ but easily parallelizable across positions. RNN/LSTM (Module 4) is $`O(N)`$ but sequential. For training, which one wins on a modern GPU, and why?

### Resources

- J. Alammar, “The Illustrated Transformer,” 2018. [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/). The most popular accessible visual explainer.
- 3Blue1Brown, “But what is a GPT? Visual intro to transformers,” YouTube 2024. The recent video version of the same explanatory material with high-quality animations.
