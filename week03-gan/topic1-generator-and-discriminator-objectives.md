# Week 3, Topic 1: Generator and Discriminator Objectives

## Topic Overview

**Plain-English summary first.** Module 2 trained a generative model by maximizing a likelihood (the ELBO). This module takes a radically different approach: train a generator network by making it compete against a second network (a discriminator) whose job is to tell real data apart from generated data. No likelihood, no posterior, no latent inference. The whole bet is that the equilibrium of this two-player game produces a generator that fools the discriminator on data drawn from the true distribution. When it works, the samples are noticeably sharper than VAE samples. When it doesn't work, it fails in spectacular and weird ways. Topic 2 covers the failure modes; this Topic sets up the game itself.

### The two players

A GAN has two neural networks trained together:

- The **generator** $`G_\theta`$ maps a noise vector $`z \sim p_z`$ (typically Gaussian) to a synthetic sample $`G_\theta(z)`$ of the same shape as a real data point $`x`$.
- The **discriminator** $`D_\phi`$ takes an input (real or synthetic) and outputs a single scalar in \[0, 1\] interpretable as “probability this is real.”

Training alternates: improve $`D`$ at telling real from fake; improve $`G`$ at fooling $`D`$. The goal is a fixed point where $`G`$ generates samples indistinguishable from the data distribution and $`D`$ outputs 0.5 everywhere (no information).

### The minimax objective

Goodfellow and colleagues (2014) framed the training as a single saddle-point optimization:

``` math
\min_{\theta}\,\max_{\phi}\; V(\theta, \phi) \;=\; \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log D_\phi(x)\right] \;+\; \mathbb{E}_{z \sim p_z}\!\left[\log\!\left(1 - D_\phi\!\left(G_\theta(z)\right)\right)\right].
```

**Read this as a game.** The discriminator wants both terms large: it wants $`D_\phi(x) \to 1`$ on real samples (first term) and $`D_\phi(G_\theta(z)) \to 0`$ on generated samples (second term, since $`\log(1 - 0) = 0`$ is the maximum). The generator wants the second term small: it wants $`D_\phi(G_\theta(z)) \to 1`$, making $`\log(1 - 1) = -\infty`$. The $`\min_\theta \max_\phi`$ says: at each step, the discriminator picks its best response to the current generator; the generator then picks its best response to the new discriminator; iterate.

### Why this works in principle: connection to JS divergence

For a fixed generator $`G`$, the optimal discriminator is

``` math
D^{\star}(x) \;=\; \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}.
```

where $`p_G`$ denotes the distribution induced by passing $`z \sim p_z`$ through the generator. Plugging $`D^{\star}`$ back into $`V`$ reduces (after a one-line manipulation worked out in the tutorial PDF) to

``` math
V(\theta, \phi^{\star}) \;=\; -\log 4 \;+\; 2 \cdot \mathrm{JS}\!\left(p_{\text{data}}\,\big\|\,p_G\right),
```

where $`\mathrm{JS}`$ is the Jensen-Shannon divergence. Minimizing $`V`$ over $`\theta`$ therefore minimizes the JS divergence between the data distribution and the generated distribution. The unique global minimum is $`p_G = p_{\text{data}}`$, at which $`D^{\star} = 1/2`$ everywhere and the game reaches its theoretically guaranteed equilibrium.

**What this argument does (and does not) buy us.** It says: *if* training converges to the saddle point and *if* the optimizer has enough flexibility, the generator matches the data distribution exactly. Both ifs are doing a lot of work; Topic 2 covers what happens in practice when they fail.

### The non-saturating loss (a practical detail)

In practice, the generator loss $`\log(1 - D_\phi(G_\theta(z)))`$ saturates early in training: when $`D`$ is much better than $`G`$, $`D_\phi(G_\theta(z)) \approx 0`$, so $`\log(1 - 0) \approx 0`$, and the gradient with respect to $`\theta`$ vanishes. The fix Goodfellow proposed in the same 2014 paper: train $`G`$ to *maximize* $`\log D_\phi(G_\theta(z))`$ instead. Same fixed point, much stronger gradients early in training. Every GAN implementation since uses this trick; you will see it in Foster’s code.

### Contrast with the VAE

It is worth pausing on how different this is from Module 2:

- VAEs have an explicit likelihood objective (the ELBO). GANs have no likelihood; the game is the objective.
- VAEs learn an encoder that maps observations back to latents. GANs have no encoder; given a real $`x`$, you cannot ask “what $`z`$ would produce this?” without a separate inversion step.
- VAEs sample by encoding-decoding or by drawing from the prior. GANs sample only by drawing $`z \sim p_z`$ and decoding; sampling is one of the few things GANs do straightforwardly.
- VAEs tend to produce blurry samples (a consequence of the Gaussian decoder assumption). GANs tend to produce sharper samples but are notoriously harder to train.

### Where this Topic fits

Topic 2 walks through what goes wrong in practice when you try to train this game and the major theoretical fix (Wasserstein GAN). Topic 3 covers concrete architectures (DCGAN) and labeled-generation extensions (Conditional GAN). Topic 4 returns to the VAE-vs-GAN tradeoff with evaluation, controllability, and compute-cost lenses.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 4, sections *“Introduction to GANs”* and *“Your First GAN.”*
- I. Goodfellow et al., “Generative Adversarial Networks,” NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661). Sections 3 and 4 cover the saddle-point setup and the convergence argument.

### Consider

- The minimax framing says G and D fight each other. Why is that even a sensible objective for “generate realistic samples,” given that the game has no notion of a likelihood?
- If the optimal discriminator is $`D^{\star}(x) = p_{\text{data}}(x) / (p_{\text{data}}(x) + p_G(x))`$, what does it mean if you train $`D`$ for a long time and it converges to $`D \approx 1/2`$ everywhere?
- The non-saturating loss replaces the generator’s original objective with a different (but equivalent at the fixed point) one. What does that tell you about how brittle the theory is when the optimizer is far from the saddle point?

### Resources

- Foster 2nd-edition GAN notebooks: [notebooks/04_gan/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan) (DCGAN, WGAN-GP, Conditional GAN).
- I. Goodfellow, *NIPS 2016 Tutorial: Generative Adversarial Networks*, 2016. [arXiv:1701.00160](https://arxiv.org/abs/1701.00160). Goodfellow’s own pedagogical walk-through, recommended companion to the original paper.
