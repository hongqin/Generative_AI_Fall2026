# Week 4, Topic 1: Sequential Factorization of Generative Distributions

## Topic Overview

**Plain-English summary first.** Module 2 (VAE) chose to introduce a hidden latent variable and train a likelihood lower bound. Module 3 (GAN) chose to skip the likelihood entirely and train an adversarial game. This module introduces the *third* major family, which makes neither of those choices. Autoregressive models keep the likelihood, drop the latent variable, and instead model a complex high-dimensional distribution as a chain of one-dimensional next-token predictions. This is the family that runs GPT-style language models, WaveNet audio, PixelCNN images, and most of the modern code-generation systems. The math is simple but the consequences are deep.

### The chain rule, applied seriously

For any joint distribution over a vector $`x = (x_1, x_2, \dots, x_n)`$, the chain rule of probability gives an exact factorization:

``` math
p(x) \;=\; p(x_1) \cdot p(x_2 \mid x_1) \cdot p(x_3 \mid x_1, x_2) \cdots p(x_n \mid x_1, \dots, x_{n-1}) \;=\; \prod_{i=1}^{n} p\!\left(x_i \mid x_{\lt i}\right).
```

This is always true; it is the chain rule of conditional probability. The autoregressive trick is to take this seriously as a modeling strategy: parameterize each conditional $`p_\theta(x_i \mid x_{\lt i})`$ with a neural network and train the whole thing by maximum likelihood. The total log-likelihood of an observation decomposes into a sum of per-position log-likelihoods,

``` math
\log p_\theta(x) \;=\; \sum_{i=1}^{n} \log p_\theta\!\left(x_i \mid x_{\lt i}\right),
```

which is exact, computable, and differentiable. No ELBO gap (VAE), no adversarial training instability (GAN). The training objective is just “maximize the sum of next-token log-probabilities,” which is the friendliest objective in deep learning.

### Choosing an ordering

The chain rule is true for *any* ordering of the dimensions, but the model only sees one ordering, so the choice matters:

- **Text**: left-to-right is the obvious choice and what every modern language model uses. The reverse ordering is also a valid model and was historically used for some translation systems.
- **Images**: PixelCNN orders the pixels in raster-scan order (top-left to bottom-right, row by row). This is somewhat arbitrary; pixels do not have a natural one-dimensional order, and the choice introduces directional artifacts (Topic 4).
- **Audio**: WaveNet orders samples in time, which is natural.
- **Code**: tokenize left-to-right, same as text.

Once chosen, the ordering is part of the model. Two PixelCNNs trained with different orderings will produce subtly different sample distributions even on the same data.

### Exact likelihood: a real benefit

Compare the three families on whether you can ask “how likely is this observation under my model?”

- **VAE**: yes, but only a lower bound (the ELBO).
- **GAN**: no, the model has no explicit likelihood.
- **Autoregressive**: yes, exactly, via the product of conditionals.

Exact likelihood is useful for compression (the optimal code length for an observation is $`-\log p(x)`$ bits), for outlier detection (low-likelihood observations are candidates), for model comparison (perplexity, which is just exponentiated negative log-likelihood), and for the theoretical guarantees that come with proper probabilistic modeling. It is the single most important practical difference between autoregressive and adversarial models.

### Where this Topic fits

Topic 2 covers how to actually parameterize the per-position conditionals; the LSTM cell is the historical answer that made long-sequence training work, and it is what Foster’s `01_lstm` notebook implements. Topic 3 covers the sampling-time decisions (temperature, top-k, nucleus, beam) that turn a trained autoregressive model into a useful generator. Topic 4 returns to the family comparison with autoregressive’s strengths and limitations spelled out.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 5, intro sections through “Autoregressive Models.”
- A. van den Oord et al., “Pixel Recurrent Neural Networks,” ICML 2016. [arXiv:1601.06759](https://arxiv.org/abs/1601.06759). Section 1 (introduction) and Section 2 (the autoregressive factorization for images) are the key reading.

### Consider

- The chain rule is true for any ordering of the dimensions, but the model only sees one. If you were modeling 3D MRI scans autoregressively, what ordering would you pick, and what artifacts would you expect from that choice?
- Exact likelihood is a real benefit. For your course project, would having a likelihood score on every example be useful (e.g., for compression, outlier detection, ranking), or is it incidental?
- Autoregressive models keep the likelihood and drop the latent variable. What capability does the VAE’s explicit latent give you that an autoregressive model does not?

### Resources

- A. van den Oord et al., “WaveNet: A Generative Model for Raw Audio,” 2016. [arXiv:1609.03499](https://arxiv.org/abs/1609.03499). The autoregressive idea applied to raw audio waveforms; the architecture (dilated causal convolutions) is now reused widely.
- I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016. Chapter 10, “Sequence Modeling: Recurrent and Recursive Nets.” Free online at [deeplearningbook.org/contents/rnn.html](https://www.deeplearningbook.org/contents/rnn.html).
