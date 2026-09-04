# Week 2, Topic 3: Reconstruction Loss, KL Divergence, and Sampling

## Topic Overview

**Plain-English summary first.** Topic 2 derived the ELBO as a single mathematical object. To actually *train* a VAE, we need to turn the ELBO into code: a concrete loss function with two easy-to-compute pieces, a way to sample from the trained model, and a few diagnostic tools to make sure training is healthy. This Topic does exactly that. By the end, you will be able to read the loss function in the Foster `02_vae_fashion` notebook and explain every line.

### From ELBO to a concrete loss

Recall the two-term ELBO from Topic 2:

``` math
\mathcal{L}(\theta, \phi; x) \;=\; \mathbb{E}_{q_\phi(z \mid x)}\!\left[\log p_\theta(x \mid z)\right] \;-\; \mathrm{KL}\!\left(q_\phi(z \mid x)\,\big\|\,p(z)\right).
```

We *maximize* the ELBO, but it is conventional in deep learning code to *minimize* a loss. So define the VAE loss as the negative ELBO:

``` math
\mathcal{L}_{\text{VAE}}(\theta, \phi; x) \;=\; \underbrace{-\,\mathbb{E}_{q_\phi(z \mid x)}\!\left[\log p_\theta(x \mid z)\right]}_{\text{reconstruction loss}} \;+\; \underbrace{\mathrm{KL}\!\left(q_\phi(z \mid x)\,\big\|\,p(z)\right)}_{\text{KL regularizer}}.
```

Both pieces have closed-form, tractable expressions for the choices we make below. That is the whole reason we set up the math the way we did.

### The reconstruction term in practice

The expectation $`\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]`$ is approximated by drawing a single sample $`z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon`$ (the reparameterization trick from Topic 2) and computing $`\log p_\theta(x \mid z)`$ on that one sample. One sample per data point is enough in practice because we average over a minibatch.

What does $`\log p_\theta(x \mid z)`$ look like? It depends on what you assume the decoder outputs.

- **Bernoulli decoder** (each pixel a Bernoulli, common for binarized image data). Then $`-\log p_\theta(x \mid z)`$ is the per-pixel binary cross-entropy summed over pixels. This is the form Foster’s Fashion-MNIST notebook uses.
- **Gaussian decoder** (each pixel a Gaussian with fixed variance). Then $`-\log p_\theta(x \mid z)`$ reduces, up to constants, to the mean squared error between $`x`$ and the decoder output. This is what most photographic-image VAEs use.

The choice matters: BCE puts the loss in the same units as a probability and is well-calibrated for \[0,1\] pixel values; MSE is faster and pairs naturally with continuous pixel intensities. The decoder architecture is the same; only the final activation and the loss change.

### The KL term in closed form

If the encoder outputs a diagonal Gaussian $`q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))`$ and the prior is $`p(z) = \mathcal{N}(0, I)`$, the KL is available analytically:

``` math
\mathrm{KL}\!\left(q_\phi(z \mid x)\,\big\|\,p(z)\right) \;=\; \tfrac{1}{2}\sum_{j=1}^{d}\!\left(\mu_j^2 \,+\, \sigma_j^2 \,-\, 1 \,-\, \log \sigma_j^2\right).
```

This is a single line of code per data point. Note that this formula is what fixes the prior to $`\mathcal{N}(0, I)`$ in most VAE implementations; if you swap the prior, you need a different (and usually less convenient) KL expression.

### The reconstruction-vs-regularization tug-of-war, made concrete

Topic 2 framed the two ELBO terms as a tug-of-war in the abstract. In code, you literally watch the two scalars compete during training:

- If the reconstruction loss is low but the KL is high, the encoder is putting each $`x`$ far from the prior to give the decoder more capacity. Your reconstructions look great; your samples look terrible.
- If the KL is low (close to zero) but the reconstruction is high, the encoder has collapsed $`q_\phi(z \mid x)`$ onto the prior and is ignoring $`x`$. This is called **posterior collapse**. Your prior is well-matched; your reconstructions are blurry blobs.

Common ways to manage the trade-off:

- **KL warmup** (also called annealing): start with the KL coefficient at zero, ramp it linearly to one over the first few epochs. This lets the encoder establish useful representations before the regularizer kicks in.
- **β-VAE** (Higgins et al., 2017): multiply the KL term by a constant $`\beta`$, giving

``` math
\mathcal{L}_{\beta}(\theta, \phi; x) \;=\; -\,\mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)] \;+\; \beta \,\mathrm{KL}\!\left(q_\phi(z \mid x)\,\big\|\,p(z)\right).
```

Larger $`\beta`$ pushes toward more disentangled latent representations at the cost of reconstruction quality; smaller $`\beta`$ does the opposite. The original Foster notebook implicitly uses $`\beta = 1`$, but it is one line to change.

### Sampling: how a trained VAE generates new data

Once training is done, generation is two lines:

1.  Draw $`z \sim \mathcal{N}(0, I)`$ from the prior.
2.  Decode: produce $`\hat{x}`$ from $`p_\theta(x \mid z)`$ (for a Bernoulli decoder, the per-pixel probability is your image; for a Gaussian decoder, the mean is your image).

The fact that this works at all is the payoff for the KL regularizer: it forces the encoder to use latent positions that are also plausible under the prior, so sampling from the prior lands in regions the decoder knows what to do with. Without the KL term (i.e., a vanilla autoencoder), step 2 would produce gibberish.

### Latent-space structure: interpolation and arithmetic

One of the most striking properties of a well-trained VAE is that the latent space is smooth and semantically organized. Foster demonstrates this in Chapter 3 with two visualizations that you can reproduce on Fashion-MNIST and on faces:

- **Linear interpolation.** Pick two real images $`x_1, x_2`$. Encode to get $`z_1, z_2`$. Walk along the line segment $`z(t) = (1-t)\,z_1 + t\,z_2`$ for $`t \in [0, 1]`$ and decode at each step. A trained VAE produces a smooth morph between the two images, not a sequence of glitches.
- **Latent arithmetic.** Average the latent codes of all training points with attribute A (e.g., “smiling”) and subtract the average code of all points without it. The resulting vector behaves like a direction for that attribute: add it to any encoded $`z`$ and the decoded image gains the attribute (Foster Ch 3, “Latent Space Arithmetic”).

Both of these are easy to run from the Foster notebooks; the assignment in this module asks you to produce at least the linear interpolation.

### Where this Topic fits

Topic 4 covers the practical training knobs (architecture, learning rate, evaluation metrics, common pitfalls) you need before opening the notebook. The Module 2 VAE Technical Assignment then asks you to run the Foster `02_vae_fashion` notebook, vary the latent dimension, sample from the prior, and visualize the latent space, then explain what you saw using the language from this Topic.

### Read

Required for this Topic:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 3, sections *“The Loss Function,”* *“Analysis of the Variational Autoencoder,”* and *“Latent Space Arithmetic.”*

Recommended for the β-VAE knob:

- I. Higgins et al., “β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework,” ICLR 2017. [openreview.net link](https://openreview.net/forum?id=Sy2fzU9gl).

### Consider

- If you saw reconstruction loss low and KL loss high during training, which knob would you turn first, and in which direction?
- The closed-form KL formula only works because the prior is $`\mathcal{N}(0, I)`$ and $`q_\phi`$ is diagonal Gaussian. What would change in your training loop if you wanted the prior to be a mixture of Gaussians?
- Why does latent-space interpolation produce a smooth morph in a VAE but not in a vanilla autoencoder?

### Resources

- Foster Chapter 3 notebooks: [notebooks/03_vae/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/03_vae). Look in particular at `02_vae_fashion/` for the loss function in code and `03_vae_faces/` for the latent-arithmetic demo.
- S. R. Bowman et al., “Generating Sentences from a Continuous Space,” CoNLL 2016. [arXiv:1511.06349](https://arxiv.org/abs/1511.06349). The classic worked example of posterior collapse and KL warmup, in a text-VAE setting.
