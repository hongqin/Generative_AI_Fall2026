# Week 2, Topic 1: Latent-Variable Modeling and Representation Learning

## Topic Overview

**Plain-English summary first.** Before we can talk about VAEs in Topic 2, we need to agree on what a *latent-variable* generative model is, and why a regular autoencoder is *not* one. In this Topic we introduce the latent-variable framework, examine the gap between “compresses data well” (autoencoder) and “generates new data well” (generative model), and motivate the VAE as the smallest, most principled fix to that gap.

### From observations to a hidden cause

Consider a dataset of images of handwritten digits. Each image is a long vector of pixel values, but intuitively we know that the image is determined by a much smaller set of *hidden factors*: which digit was written, how thick the strokes are, how slanted the writing is, where on the canvas the digit sits. A latent-variable generative model takes this intuition literally: it posits a low-dimensional hidden variable $`z`$ that captures these factors, and a stochastic mapping from $`z`$ to the observed $`x`$.

Formally, we specify a simple prior $`p(z)`$ (typically $`\mathcal{N}(0, I)`$) and a decoder $`p_\theta(x \mid z)`$ implemented by a neural network. The data distribution is the marginal

``` math
p_\theta(x) \;=\; \int p_\theta(x \mid z)\, p(z)\, dz.
```

Two things are now possible in principle: we can *generate* new data by sampling $`z \sim p(z)`$ and then $`x \sim p_\theta(x \mid z)`$, and we can *infer* a latent cause for an observation $`x`$ by computing the posterior $`p_\theta(z \mid x)`$. The catch (which Topic 2 addresses) is that neither the marginal nor the posterior has a closed form when the decoder is a neural network.

### Autoencoders: useful representations, not a generative model

If you have used a regular (deterministic) autoencoder, the architecture looks almost identical: an encoder network compresses $`x`$ into a low-dimensional code $`z`$, and a decoder reconstructs $`x`$ from $`z`$. Train the pair end-to-end with a reconstruction loss (e.g., mean squared error) and you get a useful representation: the bottleneck $`z`$ captures the structure the decoder needs in order to rebuild $`x`$. Foster covers this construction in Chapter 3, sections “Autoencoders” through “Analysis of the Autoencoder.”

**Why is this not a generative model?** Three concrete problems:

- There is no prior on $`z`$. The encoder is free to place each training point anywhere in the latent space. If you sample a random $`z`$ at inference and decode it, you will probably land in a region of the latent space the encoder never used; the decoder has no idea what to do, and the output is gibberish.
- There is no notion of probability over $`x`$. The decoder produces a single reconstruction, not a distribution. You cannot ask “how likely is this $`x`$?”
- The training objective is reconstruction only. There is nothing in the loss that encourages the latent space to be coherent for sampling.

A vanilla autoencoder is therefore a good *representation learner*, but not a *sampler*. The VAE will fix this by introducing a prior on $`z`$, making both the encoder and decoder probabilistic, and adding a regularizer that pulls the encoded distributions toward the prior. Those three changes turn the autoencoder into a generative model.

### Representation learning, briefly

Even setting aside generation, latent-variable models are valuable because the latent space $`z`$ is often a much better representation of $`x`$ for downstream tasks (classification, clustering, similarity search, anomaly detection) than the raw pixels. A good latent space is *smooth* (nearby $`z`$ decode to similar $`x`$), *structured* (semantic factors of variation correspond to specific directions in $`z`$), and *low-dimensional* (much smaller than $`\dim(x)`$, without losing the information you care about). Topic 3 returns to this when we look at latent-space arithmetic and interpolation.

### Where this Topic fits

Topic 2 fills in the missing pieces: how to handle the intractable marginal $`p_\theta(x)`$, how to compute an approximate posterior, and how to derive the training objective (the ELBO). Topic 3 turns the ELBO into a concrete loss function and shows what sampling looks like. Topic 4 covers the practical knobs you need for the Module 2 coding assignment.

### Read

Required for this Topic:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 3, sections *“Autoencoders,”* *“Your First Autoencoder,”* and *“Analysis of the Autoencoder.”* These cover the encoder-decoder construction at the level of intuition and code.

Optional warm-up notebook (highly recommended; takes ~30 minutes on a CPU laptop):

- Foster 2nd-edition repository: [notebooks/03_vae/01_autoencoder/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/03_vae/01_autoencoder). Train the autoencoder, then look at what happens when you sample random points in the latent space and decode them.

### Consider

- For the dataset you are considering for your course project, what would the “hidden factors” $`z`$ plausibly be? How many of them do you think there are?
- If your project data is not images, does the latent-variable picture still apply? What would a meaningful $`z`$ look like for tabular data, time series, text, or molecules?
- Run (or read) the `01_autoencoder` notebook. After training, sample a few random points in the latent space and decode them. What do you see, and why?

### Resources

- Y. Bengio, A. Courville, P. Vincent, “Representation Learning: A Review and New Perspectives,” IEEE TPAMI, 2013. [arXiv:1206.5538](https://arxiv.org/abs/1206.5538). Background on what makes a representation “good.”
- I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016. Chapter 14, “Autoencoders.” Free online at [deeplearningbook.org/contents/autoencoders.html](https://www.deeplearningbook.org/contents/autoencoders.html).
