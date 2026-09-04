# Week 4, Topic 4: Strengths and Limitations of Autoregressive Generation

## Topic Overview

**Plain-English summary first.** Autoregressive is the family that runs the most-deployed generative-AI systems in the world right now: every large language model, most code-completion systems, WaveNet-style audio synthesis, PixelCNN-style image generation, protein-sequence design. It earned that position by being the easiest-to-train family with the most defensible probabilistic story. But it also has hard limits: sampling is sequential and slow, the choice of ordering biases the model, and capturing very long-range structure remained difficult until transformers solved a big part of it (Module 6). This Topic puts autoregressive side by side with VAE (Module 2) and GAN (Module 3) so you can pick the right family for your project.

### Where autoregressive shines

- **Exact likelihood.** No ELBO gap, no adversarial fixed-point dependence. The training objective is a sum of per-token log-probabilities, which is well-conditioned, well-understood, and parallelizes across positions during training (even though sampling is sequential).
- **Broad applicability.** Any data with a natural ordering works: text, code, audio, DNA, time series, images via raster scan. The factorization $`p(x) = \prod p(x_i \mid x_{\lt i})`$ is universal in a way the latent-variable or adversarial setups are not.
- **Strong tooling and ecosystem.** Hugging Face Transformers, PyTorch LSTMs, JAX models, every major framework has first-class autoregressive support. Pre-trained checkpoints are everywhere.
- **Scales gracefully.** Bigger models, more data, longer context windows; performance continues to improve with scale (in a way GANs do not). The scaling-laws literature is almost entirely about autoregressive models.
- **Conditioning is easy.** Want to control the generator? Prepend the conditioning tokens. Want to generate from scratch? Use a start-of-sequence token. The same mechanism handles everything.

### Where autoregressive struggles

- **Sampling cost is $`O(N)`$ and not parallelizable across positions.** Every sample requires $`N`$ forward passes through the model. For long sequences (high-res images, long-form audio) this is the bottleneck.
- **The ordering choice biases the model.** A raster-scan PixelCNN sees the upper-left corner first and uses it to predict the rest; the resulting samples have a directional bias. There is no symmetric way to do autoregressive on intrinsically un-ordered data.
- **No bidirectional conditioning.** If you want to model $`p(x_i \mid x_{\gt i})`$ (right-to-left, useful for infilling), you need a separate model or a non-autoregressive setup. BERT-style masked language modeling exists but is not the same thing as a true probabilistic model.
- **Mode coverage and diversity require careful sampling.** The same model that produces excellent text at $`T = 0.8`$ produces nonsense at $`T = 1.5`$ or boring repetition at $`T = 0`$. Without a temperature knob, you get one operating point.
- **Long-range dependencies are hard for vanilla RNN/LSTM.** Even LSTM struggles past a few hundred timesteps. Transformers (Module 6) are the architectural fix.

### Updated VAE vs GAN vs Autoregressive table

| Dimension | VAE | GAN | Autoregressive |
|----|----|----|----|
| Training objective | ELBO (lower bound on log-likelihood) | Minimax saddle point; no likelihood | Exact log-likelihood (sum of per-token) |
| Latent variable | Yes, with encoder $`q_\phi(z|x)`$ | Implicit (noise $`z`$ to generator) | None |
| Sample sharpness | Blurry (Gaussian decoder) | Sharp when training succeeds | Sharp; sequence quality varies with sampling strategy |
| Mode coverage | Broad (KL regularizer) | Can drop modes | Broad in principle; sampling-strategy dependent |
| Training stability | Routine | Hard; many failure modes | Routine |
| Controllability | Latent arithmetic on $`z`$ | Conditional GAN, StyleGAN | Prompt-based conditioning |
| Likelihood evaluation | Lower bound only | None | Exact |
| Sampling cost | One forward pass | One forward pass | $`O(N)`$ sequential forward passes |
| Typical use cases | Representation, anomaly detection | High-quality image synthesis | Language, code, audio, DNA, anything with order |

### Forward reference: transformers and diffusion

Two architectural developments since the era of LSTM-based autoregressive models have changed the field:

- **Transformers (Module 6)** replaced recurrence with self-attention. The autoregressive factorization stays the same; the per-position function becomes self-attention over the prefix. This made it possible to train much deeper, much wider autoregressive models on much longer sequences. Every modern large language model is an autoregressive transformer.
- **Diffusion (Module 5)** sidesteps autoregressive’s $`O(N)`$ sampling cost by training a model that denoises in parallel across all positions. Diffusion has largely displaced autoregressive PixelCNN-style models for high-resolution image generation while leaving text and audio largely to autoregressive transformers.

### Where this Topic fits

You now have a working understanding of the third major generative-model family. The Module 4 coding exercise asks you to train one of three Foster notebooks and connect what you observe to the vocabulary from Topics 1 through 4. Module 5 (Diffusion) will be the next major contrast; Module 6 (Transformers) will then return to the autoregressive idea with a much more powerful architecture.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 5, concluding sections (Foster’s comparison of LSTM and PixelCNN with the families from Chapters 3 and 4).

Recommended for the scaling-laws context:

- J. Kaplan et al., “Scaling Laws for Neural Language Models,” 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361). The empirical foundation for “bigger autoregressive models work better.”

### Consider

- The $`O(N)`$ sampling cost is intrinsic to the autoregressive factorization, not a software issue. For your course project, would the cost be a deal-breaker (long sequences, real-time constraint) or invisible (short outputs, offline)?
- Looking at the three-column table, which family is the best fit for your project, and which would you want to rule out early so you do not waste time?
- Modern large language models are autoregressive transformers. Why does that combination beat (a) bigger LSTMs and (b) non-autoregressive transformer language models?

### Resources

- A. Vaswani et al., “Attention Is All You Need,” NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). The transformer paper; Module 6 covers this in depth. Worth skimming the abstract now to see the architectural shift coming.
- T. Brown et al., “Language Models are Few-Shot Learners” (GPT-3), NeurIPS 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165). The empirical demonstration that autoregressive transformers at scale produce qualitatively new behavior.
