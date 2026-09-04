# Week 1, Topic 1: Overview of Generative Modeling and Why It Matters

## Topic Overview

Generative modeling is the branch of machine learning concerned with learning a probability distribution over data so that we can sample new examples, evaluate likelihoods or related quantities, and learn useful representations. Given an i.i.d. sample $`x^{(1)}, x^{(2)}, \dots, x^{(N)}`$ drawn from an unknown data distribution $`p_{\text{data}}(x)`$, the generative modeling goal is to fit a parameterized distribution $`p_\theta(x)`$ such that $`p_\theta(x) \approx p_{\text{data}}(x)`$. The classical training objective is maximum likelihood,

``` math
\theta^{\star} \; = \; \arg\max_{\theta}\; \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log p_\theta(x)\right].
```

Different families of deep generative models differ in how they parameterize $`p_\theta(x)`$, what is tractable to compute, and what tradeoffs they accept between sample quality, training stability, sampling speed, and likelihood evaluation. This course covers the major families in roughly the order below.

### 1. Autoregressive models

Factor the joint distribution into a product of conditionals along an ordering of the dimensions:

\$\$p\_\theta(x) \\=\\ \prod\_{i=1}^{n} p\_\theta\\\left(x_i \mid x\_{

Each conditional is parameterized by a neural network. Modern large language models (e.g., GPT-style transformers) follow this template at scale.

### 2. Variational autoencoders (VAEs)

Introduce a latent variable $`z`$ with a simple prior $`p(z)`$, and define

\\ p\_\theta(x) \\=\\ \int p\_\theta(x \mid z)\\ p(z)\\ dz.\$\$

Because this integral is intractable, training uses an approximate posterior $`q_\phi(z \mid x)`$ and the evidence lower bound (ELBO):

``` math
\log p_\theta(x) \;\geq\; \mathbb{E}_{q_\phi(z \mid x)}\!\left[\log p_\theta(x \mid z)\right] \;-\; \mathrm{KL}\!\left(q_\phi(z \mid x)\,\big\|\, p(z)\right).
```

The VAE was introduced by Kingma and Welling (2014).

### 3. Generative adversarial networks (GANs)

Train a generator $`G_\theta`$ that maps a noise vector $`z \sim p_z`$ to a sample $`G_\theta(z)`$, and a discriminator $`D_\phi`$ that distinguishes real from generated data, via the two-player minimax game:

``` math
\min_{\theta}\,\max_{\phi}\; \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log D_\phi(x)\right] \;+\; \mathbb{E}_{z \sim p_z}\!\left[\log\!\left(1 - D_\phi\!\left(G_\theta(z)\right)\right)\right].
```

GANs were introduced by Goodfellow et al. (2014).

### 4. Normalizing flows

Parameterize an invertible map $`f_\theta : z \mapsto x`$ so that the change-of-variables formula gives an exact, tractable likelihood:

``` math
p_\theta(x) \;=\; p(z)\,\left|\det\!\left(\frac{\partial f_\theta^{-1}(x)}{\partial x}\right)\right|.
```

### 5. Energy-based models (EBMs)

Specify an unnormalized density

``` math
p_\theta(x) \;=\; \frac{\exp\!\left(-E_\theta(x)\right)}{Z(\theta)}, \quad Z(\theta) \;=\; \int \exp\!\left(-E_\theta(x)\right) dx,
```

where the partition function $`Z(\theta)`$ is generally intractable. Training relies on contrastive divergence, score matching, or related techniques.

### 6. Diffusion models

Define a fixed forward (noising) Markov chain that gradually corrupts data into Gaussian noise, and learn the reverse (denoising) chain:

``` math
q(x_t \mid x_{t-1}) \;=\; \mathcal{N}\!\left(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I\right), \qquad p_\theta(x_{t-1} \mid x_t) \;=\; \mathcal{N}\!\left(x_{t-1};\; \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t)\right).
```

Denoising diffusion probabilistic models (DDPMs) were introduced by Ho, Jain, and Abbeel (2020) and now underlie state-of-the-art image, video, and audio generation.

### 7. Transformers and modern large models

The transformer architecture (Vaswani et al., 2017) is not a separate generative family but a sequence model based on self-attention,

``` math
\mathrm{Attention}(Q, K, V) \;=\; \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V,
```

that has become the backbone of most large autoregressive language models, many diffusion backbones, and most multimodal systems.

#### 8. Multimodal, world models, and agentic systems

The same modeling principles extend to joint vision-language representations (CLIP; Radford et al., 2021), to predictive models of environments (world models), and to goal-directed agents that reason and act in sequences of decisions. These topics anchor the second half of the course.

### Why generative AI matters

Generative models are now used to design candidate drug molecules and proteins, to draft and debug code, to synthesize realistic images, video, and audio, to power research assistants and tutoring systems, and to simulate environments for science and engineering. The same models also raise serious questions about training-data provenance, evaluation, hallucination, bias, energy use, and misuse. A working understanding of how these models are formulated and trained is the foundation for using them well and for evaluating their limits.

### How this topic fits the rest of the course

Modules 2 through 8 each focus on one of the families above. Bringing them under a single probabilistic-modeling umbrella in this topic makes it easier to compare them: each family is a different answer to the same question, *how do we represent and train a model of $`p_{\text{data}}(x)`$?*

### Read

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. Chapter 1, “Generative Modeling.” (Listed as the course’s supplemental text.)
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*, MIT Press, 2016. Chapter 20, “Deep Generative Models.” Free online: [Deep Learning](https://www.deeplearningbook.org/contents/generative_models.html).

### Consider

### Resources

Foundational primary papers referenced above (open access on arXiv):

- D. P. Kingma and M. Welling. “Auto-Encoding Variational Bayes.” ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- I. Goodfellow et al. “Generative Adversarial Networks.” NIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- A. Vaswani et al. “Attention Is All You Need.” NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- J. Ho, A. Jain, P. Abbeel. “Denoising Diffusion Probabilistic Models.” NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- A. Radford et al. “Learning Transferable Visual Models From Natural Language Supervision” (CLIP). ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

Reference textbooks beyond the supplemental Foster text:

- C. M. Bishop and H. Bishop, *Deep Learning: Foundations and Concepts*, Springer, 2024. [ch1 overview of generative AI](https://www.bishopbook.com/) (free PDF).
- K. P. Murphy, *Probabilistic Machine Learning: Advanced Topics*, MIT Press, 2023. [ch1 overview of generative AI](https://probml.github.io/pml-book/book2.html) (free draft PDF).
