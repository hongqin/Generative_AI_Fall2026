# Week 3, Topic 2: Adversarial Training Dynamics and Instability

## Topic Overview

**Plain-English summary first.** Topic 1 set up the GAN game and showed that, in theory, its equilibrium recovers the data distribution. The catch is “in theory.” In practice, training a minimax game is genuinely hard: it can oscillate forever, collapse to a single output mode, give you a discriminator loss that tells you nothing useful, or just be devastatingly sensitive to hyperparameters. This Topic catalogs the four canonical failure modes, explains the mathematical reason GAN training is intrinsically harder than VAE training, and presents the Wasserstein-GAN family as the dominant fix that the community converged on.

### Four named failure modes

Foster (Chapter 4, “GAN Challenges”) groups GAN training pathologies into four:

1.  **Oscillating loss.** The discriminator and generator losses bounce around indefinitely instead of converging. The game has no monotone progress metric.
2.  **Mode collapse.** The generator finds *one* output (or a small set) that fools the discriminator and gets stuck generating only that. The other modes of the data distribution disappear from the samples.
3.  **Uninformative loss.** The discriminator and generator losses are not a useful signal of sample quality. A discriminator loss going down does not mean your samples are getting better; a generator loss going down does not mean it is winning.
4.  **Hyperparameter sensitivity.** Tiny changes to learning rate, batch size, or architecture turn a working GAN into a broken one. There is no robust autopilot configuration.

You will see at least one of these in any non-trivial GAN training run. Recognizing them by name is half the diagnosis.

### Why GAN training is intrinsically hard: the JS-divergence problem

Topic 1 showed that, for a fixed generator, the optimal discriminator turns the minimax objective into $`2 \cdot \mathrm{JS}(p_{\text{data}} \| p_G) - \log 4`$. That looked like good news. It is not. The problem is what happens when the two distributions have **disjoint or barely-overlapping supports**, which is the common case in high-dimensional data:

- If $`p_{\text{data}}`$ and $`p_G`$ put mass on different parts of the input space, the JS divergence is constant (equals $`\log 2`$) regardless of how far apart they are.
- That constant value has zero gradient with respect to the generator parameters.
- The generator gets no useful learning signal, and training stalls.

Early in training, the generator’s output is essentially noise and the data distribution lives on a low-dimensional manifold; their supports almost never overlap. So JS divergence is constant, the gradient vanishes, and the generator cannot find the data manifold. This is the deep reason the discriminator can “get too good too fast” and break training.

### Wasserstein distance: a smoother alternative

Arjovsky, Chintala, and Bottou (2017) proposed replacing JS divergence with the **Earth Mover’s Distance** (also called Wasserstein-1 distance or W1):

``` math
W(p_{\text{data}}, p_G) \;=\; \inf_{\gamma \in \Pi(p_{\text{data}}, p_G)}\; \mathbb{E}_{(x, y) \sim \gamma}\!\left[\| x - y \|\right].
```

**The intuition is exactly the name.** Imagine the two distributions as two piles of dirt on the same ground; $`W`$ is the minimum total “work” (mass times distance) you need to do to reshape one pile into the other. Unlike JS divergence, $`W`$ varies smoothly with the geometry of the distributions: if you move $`p_G`$ a little closer to $`p_{\text{data}}`$, $`W`$ decreases continuously. There is always a useful gradient.

The catch is that $`W`$ as written above is an infimum over couplings, which is intractable. The Kantorovich-Rubinstein duality saves us: $`W`$ can be rewritten as a maximum over 1-Lipschitz functions $`f`$,

``` math
W(p_{\text{data}}, p_G) \;=\; \sup_{\| f \|_L \leq 1}\; \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] \;-\; \mathbb{E}_{z \sim p_z}[f(G_\theta(z))].
```

This is now an optimization problem we can approximate with a neural network: model $`f`$ as a network (call it the “critic” rather than “discriminator” because it outputs an unbounded real value, not a probability), constrain it to be 1-Lipschitz, and the WGAN objective is

``` math
\min_\theta\, \max_{\phi:\, \| f_\phi \|_L \leq 1}\; \mathbb{E}_{x \sim p_{\text{data}}}[f_\phi(x)] \;-\; \mathbb{E}_{z \sim p_z}[f_\phi(G_\theta(z))].
```

### Enforcing the Lipschitz constraint: weight clipping then gradient penalty

Two ways to keep the critic 1-Lipschitz in practice:

- **WGAN (Arjovsky et al., 2017): weight clipping.** Clamp every weight in the critic network to a small range, e.g., $`[-0.01, 0.01]`$, after every gradient step. Crude but effective.
- **WGAN-GP (Gulrajani et al., 2017): gradient penalty.** Add a penalty term to the loss that pushes the critic’s gradient norm toward 1 on samples interpolated between real and fake:

``` math
\mathcal{L}_{\text{WGAN-GP}} \;=\; \underbrace{\mathbb{E}_{z}[f_\phi(G_\theta(z))] \;-\; \mathbb{E}_{x}[f_\phi(x)]}_{\text{critic loss for }\phi} \;+\; \lambda\, \mathbb{E}_{\hat x}\!\left[\left(\|\nabla_{\hat x} f_\phi(\hat x)\|_2 \,-\, 1\right)^2\right].
```

where $`\hat x = \epsilon\, x + (1 - \epsilon)\, G_\theta(z)`$ for $`\epsilon \sim U[0,1]`$, and $`\lambda`$ is typically 10. Gradient penalty is the default in nearly every modern GAN implementation because weight clipping is brittle (capacity wasted; can collapse to overly simple critics). Foster’s `02_wgan_gp` notebook implements this directly.

### Practical effects of switching to Wasserstein

- **The critic can be trained to optimality (or close) without breaking the generator.** This is the opposite of the JS-divergence case where a perfectly trained discriminator kills the gradient signal. Common practice: 5 critic updates per generator update.
- **The critic loss is now a meaningful sample-quality signal.** Lower critic loss correlates with better samples; you can monitor it during training. (JS-style discriminator loss has no such interpretation.)
- **Mode collapse is much rarer.** Not eliminated; still happens with bad hyperparameters or for specific datasets, but the failure mode is qualitatively less frequent.

### Where this Topic fits

Topic 3 returns to architectures: DCGAN as the standard convolutional backbone you would use under either JS or Wasserstein loss, plus Conditional GAN for controlling what the generator produces. Topic 4 contrasts GAN behavior with VAE behavior and previews diffusion (Module 5) as a third approach that sidesteps the adversarial-training problem entirely.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 4, sections *“GAN Challenges,”* *“Wasserstein GAN,”* and *“WGAN-GP.”*
- M. Arjovsky, S. Chintala, L. Bottou, “Wasserstein GAN,” ICML 2017. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875).
- I. Gulrajani et al., “Improved Training of Wasserstein GANs” (WGAN-GP), NeurIPS 2017. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028).

Detailed derivations (recommended if you want every step worked out):

- Course tutorial: [Week 3 lecture notes (PDF)](notes/week03-notes.pdf) (Principles of GANs).

### Consider

- JS divergence is constant when supports don’t overlap. In which kinds of data (images? text? tabular?) would you expect “non-overlapping supports early in training” to be most severe?
- WGAN-GP trains the critic 5 times per generator update. Why is that schedule advisable for WGAN but a disaster for vanilla JS-based GANs?
- Of the four named failure modes, which two do you think would be easiest to detect from a loss-curve plot alone, and which would require looking at sample grids?

### Resources

- L. Mescheder, A. Geiger, S. Nowozin, “Which Training Methods for GANs do actually Converge?” ICML 2018. [arXiv:1801.04406](https://arxiv.org/abs/1801.04406). A careful empirical study of GAN training stability.
- T. Salimans et al., “Improved Techniques for Training GANs,” NeurIPS 2016. [arXiv:1606.03498](https://arxiv.org/abs/1606.03498). The collection of hand-tuned tricks (feature matching, minibatch discrimination, virtual batch norm) that were SOTA before Wasserstein-GANs and are still useful intuition.
