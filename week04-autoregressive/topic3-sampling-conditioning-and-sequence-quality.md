# Week 4, Topic 3: Sampling, Conditioning, and Sequence Quality

## Topic Overview

**Plain-English summary first.** Training an autoregressive model is the friendliest objective in deep learning: just maximize the sum of next-token log-probabilities. Sampling from a trained model is where the engineering lives. The same model can produce dull, repetitive text (greedy decoding) or wildly incoherent text (high-temperature sampling) depending on which sampling strategy you pick. This Topic walks through the standard decoding strategies, the conditioning patterns that let you steer the generator, the quality metrics you can compute, and the asymptotic cost of autoregressive sampling that nobody can engineer their way around.

### Decoding strategies: how to pick the next token

At each step $`i`$ of generation, the model gives you a distribution $`p_\theta(x_i \mid x_{\lt i})`$ over possible next tokens (or pixels, or audio samples). You have to pick one. The choice is more consequential than it looks.

#### Greedy decoding

Always pick the highest-probability token: $`x_i = \arg\max p_\theta(\cdot \mid x_{\lt i})`$. Deterministic given the prefix. Fast. Produces repetitive, low-diversity output because the highest-probability path through the search tree is often dominated by common short tokens. Almost never the right choice for open-ended generation.

#### Temperature sampling

Sharpen or flatten the distribution before sampling:

``` math
p_T(x_i) \;\propto\; p_\theta(x_i \mid x_{\lt i})^{1/T}, \qquad T \gt 0.
```

**What temperature does, in plain language**: $`T = 1`$ leaves the distribution unchanged. $`T \lt 1`$ sharpens it (closer to greedy; more deterministic, less diverse). $`T \gt 1`$ flattens it (more random, more diverse, more incoherent). For text, $`T`$ between 0.7 and 1.0 is the typical sweet spot; for creative tasks people push higher. For factual completion, lower. The setting is task- and model-dependent and is one of the few things you tune by eye.

#### Top-k and top-p (nucleus) sampling

**Top-k**: keep only the $`k`$ highest-probability tokens, renormalize, sample from those. Cuts off the long tail of low-probability tokens that would otherwise occasionally produce nonsense. Typical $`k`$: 20 to 50 for text.

**Top-p / nucleus** (Holtzman et al., 2020): keep the smallest set of top tokens whose cumulative probability exceeds $`p`$, then sample from that set. Adapts to the distribution: when the model is confident, only a few tokens are above the threshold; when the model is uncertain, the nucleus expands. Typical $`p`$: 0.9 to 0.95. Generally preferred over top-k for natural-language generation.

#### Beam search

Maintain $`k`$ candidate prefixes at each step, expand each by all possible next tokens, keep the top-$`k`$ extended candidates. Returns the prefix of maximum total log-probability after a fixed length. Used for tasks where one best output matters more than diversity: machine translation, summarization, structured prediction. Not generally used for open-ended generation; beam search produces highly repetitive output.

### Conditioning: steering what the model produces

The vanilla autoregressive model has no knobs at inference: you give it nothing and it samples from $`p(x_1)`$ and then $`p(x_2 \mid x_1)`$ etc. To steer it, you add conditioning information that the model was trained to respect:

- **Prompt-based**: prepend a sequence $`y`$ of tokens to the model’s input. The model continues from there. This is how language models work: you give it a prompt, it samples a continuation. The model has no special “prompt” mechanism; it just sees the prompt as the start of the sequence.
- **Class-conditional**: train the model to take a class label as additional input (concatenated to each timestep’s input or used to initialize the hidden state). Foster’s class-conditional PixelCNN is the example.
- **Encoder context** (seq2seq): an encoder network compresses an input sequence into a context vector; the autoregressive decoder takes that as initialization. This is the original Sutskever et al. 2014 machine-translation setup; it is also the structure of every later encoder-decoder model.

### Quality metrics for autoregressive models

- **Perplexity**: exponentiated average negative log-likelihood per token, $`\exp(-\frac{1}{N} \sum_i \log p_\theta(x_i \mid x_{\lt i}))`$. Lower is better. Native to autoregressive models because they have exact likelihood. The standard metric for language model evaluation on held-out test data. Use with caution: perplexity measures how well the model predicts your test set, not how good its samples look to a human.
- **BLEU, ROUGE, METEOR**: reference-based metrics for translation and summarization. Useful when there is a ground-truth output to compare against; useless for open-ended generation.
- **FID (for image autoregressive)**: same as the GAN metric from Module 3. Treat the autoregressive model as a sample source and compute FID against real images.
- **Human evaluation**: still the gold standard for open-ended generation. Quantitative metrics correlate with human judgment imperfectly.

### The unavoidable cost: $`O(N)`$ sampling

To sample a length-$`N`$ sequence, you have to run the model $`N`$ times, once per token. Each step depends on all previous tokens, so the steps cannot be parallelized. This is intrinsic to the autoregressive factorization, not a software issue.

For a 100-token text response on a modest LSTM, this is invisible (a fraction of a second). For a 1024x1024 image on a PixelCNN (over one million pixels), it is brutal: hours of inference per image. The same problem dogs WaveNet on raw audio and is the main reason diffusion models (Module 5), which can parallelize their denoising steps differently, have largely displaced autoregressive image models for high-resolution work.

### Where this Topic fits

Topic 4 collects autoregressive’s strengths and limitations into one place and compares it with the VAE (Module 2) and GAN (Module 3) families on the same axes. The Module 4 coding exercise asks you to train one Foster notebook and run it with at least one decoding-strategy variation, so the abstractions on this page show up in the experiments you actually do.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 5, sections on generation and sampling. Look at the Foster `01_lstm` notebook’s sampling function to see temperature implemented in code.
- A. Holtzman et al., “The Curious Case of Neural Text Degeneration” (introduces nucleus sampling), ICLR 2020. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751). Sections 1 to 4 are the core; the rest is empirical analysis.

### Consider

- If you set $`T \to 0`$, temperature sampling becomes greedy decoding. If you set $`T \to \infty`$, what happens?
- Beam search returns the (approximately) most-likely sequence. Why does that produce repetitive, low-quality text for open-ended generation but good output for machine translation?
- Perplexity is a per-token average. If your model has very low perplexity on test data but generates incoherent samples, what does that tell you about the difference between “predicting the next token well” and “generating good sequences”?

### Resources

- I. Sutskever, O. Vinyals, Q. V. Le, “Sequence to Sequence Learning with Neural Networks,” NeurIPS 2014. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215). The original encoder-decoder paper; foundational for understanding modern conditional generation.
- D. Bahdanau, K. Cho, Y. Bengio, “Neural Machine Translation by Jointly Learning to Align and Translate,” ICLR 2015. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473). The first attention mechanism, which Topic 6 will return to as the seed of the transformer.
