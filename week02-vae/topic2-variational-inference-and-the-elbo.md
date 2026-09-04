# Week 2, Topic 2: Variational Inference and the ELBO

## Topic Overview

**Plain-English summary first.** We want to fit a generative model that has a hidden (latent) variable inside it. The standard way to train such a model is to maximize the likelihood of the data it produced, but for any interesting decoder, the likelihood involves an integral we cannot actually compute. The fix has two moves: (1) introduce an *approximate* answer to "what hidden state probably produced this data point?" using a neural network called the *encoder*, and (2) replace the impossible likelihood with a quantity called the **Evidence Lower Bound (ELBO)**, which we *can* compute and which is guaranteed to sit at or below the true likelihood. Maximizing the ELBO trains the encoder and decoder together with ordinary gradient descent, with one small extra trick (reparameterization) to make random sampling differentiable. Everything below is a careful walk through these ideas.

Topic 1 set up the generative latent-variable framework: introduce a latent variable $`z`$ with a simple prior $`p(z)`$, let a decoder $`p_\theta(x \mid z)`$ describe how observations are produced from latents, and write the data marginal as

``` math
p_\theta(x) \;=\; \int p_\theta(x \mid z)\, p(z)\, dz.
```

In plain language, this integral asks: *across every possible value of the hidden state $`z`$, how likely is this observation $`x`$ under our model?* If $`z`$ were a discrete variable with a few values, we could just sum. With a continuous $`z`$ and a neural-network decoder, there is no closed form, and we cannot brute-force it because the integration domain is high-dimensional.

Worse, the true posterior

``` math
p_\theta(z \mid x) \;=\; \frac{p_\theta(x \mid z)\, p(z)}{p_\theta(x)}
```

inherits the same intractable normalizer in its denominator, so we cannot directly compute the likelihood, sample the posterior, or fit $`\theta`$ by exact maximum likelihood. The rest of this Topic builds the two ideas that make the framework workable in practice: **variational inference**, and the resulting training objective, the **Evidence Lower Bound (ELBO)**.

### Variational inference: approximate the intractable posterior

Here is the move in plain English. The true posterior $`p_\theta(z \mid x)`$ is the right answer to the question *"given that I observed this $`x`$, what hidden state $`z`$ most likely produced it?"* We cannot write that posterior down. So we pick a simpler family of distributions we *can* work with, parameterized by a neural network, and find the member of that family that comes as close as possible to the true posterior. The neural network we are about to train to do this job is the **encoder**.

Formally: pick a tractable family of distributions $`q_\phi(z \mid x)`$ (in practice, a diagonal Gaussian whose mean and standard deviation are produced by a neural network with parameters $`\phi`$), and look for the member of that family that best approximates the true posterior $`p_\theta(z \mid x)`$ in the KL-divergence sense:

``` math
q_\phi^{\star}(z \mid x) \;=\; \arg\min_{q_\phi} \; \mathrm{KL}\!\left(q_\phi(z \mid x)\,\big\|\,p_\theta(z \mid x)\right).
```

This is the central trade we make: we convert an integration problem we cannot solve into an optimization problem we can. From here on we learn $`\phi`$ (the encoder) and $`\theta`$ (the decoder) jointly by gradient descent.

### The ELBO identity

Now we connect the two moves above with a single, very useful identity. A direct calculation (worked out line by line in the tutorial PDF linked under "Read"), starting from $`\log p_\theta(x) = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x)]`$ and inserting $`p_\theta(x) = p_\theta(x, z) / p_\theta(z \mid x)`$, gives:

``` math
\log p_\theta(x) \;=\; \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\!\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right]}_{\displaystyle \mathcal{L}(\theta, \phi; x)\;=\;\mathrm{ELBO}} \;+\; \underbrace{\mathrm{KL}\!\left(q_\phi(z \mid x)\,\big\|\,p_\theta(z \mid x)\right)}_{\displaystyle \geq 0}.
```

**What does this identity buy us?** Two things, both useful.

- The right-hand side has two pieces: one we can compute (the ELBO), and one we cannot (the KL between our approximate posterior and the true posterior). Crucially, the KL is always $`\geq 0`$. So the ELBO is always $`\leq \log p_\theta(x)`$. That is why it is called a *lower bound*: it sits at or below the true log-likelihood, and equality holds exactly when our approximate posterior matches the true one.
- If we push the ELBO up by changing $`\theta`$ and $`\phi`$, the identity forces one of two things to happen: either $`\log p_\theta(x)`$ goes up (the model gets better at the data, which is the goal) or the KL gap shrinks (our approximate posterior gets closer to the true one, also good). We get a two-for-one deal: better likelihood and better posterior approximation, optimized by the same gradient step.

This is why the ELBO is a safe stand-in for the intractable likelihood. We cannot maximize the true $`\log p_\theta(x)`$, but maximizing the ELBO is guaranteed to either improve it or improve our approximation of the posterior (or both).

### Two equivalent forms of the ELBO

The ELBO above is mathematically clean but is hard to read as a loss function. Expanding $`p_\theta(x, z) = p_\theta(x \mid z)\, p(z)`$ and rearranging yields the form that appears in nearly every VAE implementation:

``` math
\mathcal{L}(\theta, \phi; x) \;=\; \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\!\left[\log p_\theta(x \mid z)\right]}_{\text{expected reconstruction log-likelihood}} \;-\; \underbrace{\mathrm{KL}\!\left(q_\phi(z \mid x)\,\big\|\,p(z)\right)}_{\text{regularizer toward the prior}}.
```

**Read this as a tug-of-war.** The first term asks: *when the encoder turns $`x`$ into a distribution over $`z`$, and the decoder turns a sampled $`z`$ back into a distribution over $`x`$, how well do we recover the original $`x`$?* Higher is better; this term wants the encoder to give the decoder enough information to rebuild $`x`$. The second term asks: *is the distribution that the encoder produced for this $`x`$ close to the prior $`p(z)`$ that we will sample from later?* Lower KL is better; this term wants the encoder to keep its output near the prior so that, after training, drawing $`z \sim p(z)`$ and decoding gives reasonable-looking samples.

These two terms genuinely fight each other. If we cared only about reconstruction, the encoder could put each training point in its own private corner of the latent space (great reconstruction, useless for sampling). If we cared only about the KL term, the encoder could collapse $`q_\phi(z \mid x)`$ onto the prior and ignore $`x`$ entirely (great regularization, no information left for the decoder). The whole engineering challenge of training a VAE is keeping both terms healthy at once. Topic 3 picks this up with the practical loss form and the β-VAE knob that lets you shift the balance.

### The reparameterization trick

One technical obstacle remains. The ELBO is an expectation $`\mathbb{E}_{q_\phi(z \mid x)}[\,\cdot\,]`$ over a distribution that depends on the encoder parameters $`\phi`$. If we just drew samples $`z \sim q_\phi(z \mid x)`$ and used them inside our loss, the gradient of the loss with respect to $`\phi`$ would have to pass through a random sampling step, which is not differentiable. In plain English: "pick a random number from this distribution" has no derivative, so backpropagation gets stuck right at the encoder.

The **reparameterization trick** (Kingma & Welling, 2014) gets around this by writing the sample as a deterministic function of $`\phi`$ and a fixed-distribution random variable:

``` math
z \;=\; \mu_\phi(x) \;+\; \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0,\, I),
```

where $`\mu_\phi`$ and $`\sigma_\phi`$ are the outputs of the encoder network and $`\odot`$ is elementwise multiplication. The randomness now lives entirely in $`\epsilon`$, which has nothing to learn. The encoder parameters $`\phi`$ appear only inside the deterministic functions $`\mu_\phi`$ and $`\sigma_\phi`$, so the gradient of any downstream loss with respect to $`\phi`$ flows through ordinary chain-rule arithmetic.

The intuition is simple: instead of saying "sample $`z`$ from a distribution whose parameters depend on $`\phi`$", we say "draw a fixed-distribution random number $`\epsilon`$ once, then transform it through learnable knobs." Same result statistically, but now it is differentiable. This single trick is what lets us train both the encoder and the decoder by ordinary stochastic gradient descent.

### What is the encoder and what is the decoder?

Once the ELBO is in place, the names line up cleanly with the two neural networks the VAE trains:

- The **encoder** implements $`q_\phi(z \mid x)`$: it maps an input $`x`$ to the parameters $`(\mu_\phi(x), \sigma_\phi(x))`$ of a diagonal Gaussian over latents.
- The **decoder** implements $`p_\theta(x \mid z)`$: it maps a latent sample $`z`$ back to a distribution over reconstructions of $`x`$.

**If you have used a regular (deterministic) autoencoder, here is the change.** A vanilla autoencoder compresses $`x`$ into a single point in a low-dimensional space and decodes that point back to $`x`$. A VAE does two extra things: the encoder outputs a *distribution* over latents instead of a single point, and the training loss includes the KL regularizer that pulls those distributions toward the prior. Those two changes are what turn a representation learner (good at compressing data) into a generative model (good at producing new data).

Sampling new data is straightforward: draw $`z \sim \mathcal{N}(0, I)`$ from the prior, then sample $`x \sim p_\theta(x \mid z)`$ from the decoder. The whole point of the ELBO is that, if training succeeds, this prior-and-decode procedure produces samples that resemble the training data.

### Where this Topic fits

This is the technical heart of the VAE. Topic 3 turns the ELBO into the concrete loss function you will minimize, addresses common failure modes (posterior collapse, blurry samples), and shows how to sample and interpolate in the latent space. Topic 4 covers the practical training and evaluation considerations you will use in this module's coding assignment.

### Read

Required for this Topic:

- D. P. Kingma and M. Welling, *An Introduction to Variational Autoencoders*, 2019. [arXiv:1906.02691](https://arxiv.org/abs/1906.02691). Sections 1 and 2 cover everything on this page in tutorial form; Section 3 anticipates Topic 3.
- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. Chapter 3, sections *"The Variational Art Exhibition"* and *"Building a Variational Autoencoder"*. These are the intuitive narrative companion to the math above.

Detailed proofs of the ELBO identity and the reparameterization trick (recommended if you want every step worked out):

- [Week 2 lecture notes (PDF)](notes/week02-notes.pdf) (full derivations).

### Consider

- The KL identity says $`\log p_\theta(x) = \mathrm{ELBO} + \mathrm{KL}(q_\phi \| p_\theta)`$. What would it mean, practically, if the KL term were exactly zero? When could that actually happen?
- The ELBO has two terms with opposite preferences (reconstruction wants to spread $`q_\phi`$ wherever the data demands; the KL regularizer wants to keep $`q_\phi`$ close to the prior). Which term do you expect to dominate when the latent dimension is very small? When it is very large?
- The reparameterization trick is what lets gradient descent reach the encoder parameters. What property of the Gaussian family makes the trick easy? Could the same idea be used if $`q_\phi`$ were a mixture of Gaussians?

### Resources

Primary papers and additional tutorials:

- D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) (the original VAE paper).
- C. Doersch, “Tutorial on Variational Autoencoders,” 2016. [arXiv:1606.05908](https://arxiv.org/abs/1606.05908) (very readable companion).
- I. Higgins et al., “*β*-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework,” ICLR 2017. [openreview.net link](https://openreview.net/forum?id=Sy2fzU9gl) (sets up the β reweighting we revisit in Topic 3).

Code starter you will use in this module’s assignment:

- [Foster 2nd-edition repository, Chapter 3 notebooks](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/03_vae). The required notebook for the Topic 4 assignment is `02_vae_fashion/`; `01_autoencoder/` is a recommended warm-up.
