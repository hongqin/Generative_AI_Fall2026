# Week 5, Topic 1: Forward Noising and Reverse Denoising

## Topic Overview

**Plain-English summary first.** The diffusion idea is almost cheekily simple. Instead of trying to learn how data is generated directly (the VAE, GAN, and autoregressive plays), define a fixed *destruction* process that gradually adds noise to data until it becomes pure Gaussian noise; then train a neural network to learn the inverse, step by step. Sampling generates new data by starting from pure noise and running the learned reverse process. The forward process has no learnable parameters and is trivial to write down. The reverse process is approximately Gaussian (when the per-step noise is small) and is what the neural network learns. This Topic sets up the forward and reverse Markov chains; Topic 2 derives the training objective.

### The forward process: a fixed Markov chain that destroys data

Pick a sequence of small variances $`\beta_1, \beta_2, \dots, \beta_T`$ (typically linear or cosine schedules with $`T`$ around 1000 steps). Define the forward process $`q`$ as a Markov chain that adds a little Gaussian noise at each step:

``` math
q(x_t \mid x_{t-1}) \;=\; \mathcal{N}\!\left(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t I\right).
```

**Plain English: at each step, scale the current sample down a little and add a little Gaussian noise.** By step $`T`$, almost all signal is gone and $`x_T`$ is approximately $`\mathcal{N}(0, I)`$. The forward process has no learnable parameters; the schedule $`\beta_1, \dots, \beta_T`$ is a fixed hyperparameter.

### The closed-form marginal

A useful consequence of the Gaussian-with-scaling form: you can jump directly from $`x_0`$ to any $`x_t`$ without simulating the intermediate steps. Define $`\alpha_t = 1 - \beta_t`$ and the cumulative product $`\bar\alpha_t = \prod_{s \leq t} \alpha_s`$. Then

``` math
q(x_t \mid x_0) \;=\; \mathcal{N}\!\left(x_t;\, \sqrt{\bar\alpha_t}\, x_0,\, (1 - \bar\alpha_t) I\right).
```

Equivalently, you can sample $`x_t`$ from $`x_0`$ and a single Gaussian $`\epsilon \sim \mathcal{N}(0, I)`$:

``` math
x_t \;=\; \sqrt{\bar\alpha_t}\, x_0 \;+\; \sqrt{1 - \bar\alpha_t}\, \epsilon.
```

**Why this matters in practice**: training a diffusion model means picking a random $`t`$, generating $`x_t`$ from $`x_0`$ by this one-shot formula, and learning to predict the noise that was added. No need to simulate the full forward chain for each training example, which would be catastrophically slow.

### The reverse process: what the neural network learns

The reverse Markov chain $`p_\theta(x_{t-1} \mid x_t)`$ is what generates new data. We start from $`x_T \sim \mathcal{N}(0, I)`$ and walk backward to $`x_0`$. The form of the reverse chain is the key fact:

For small $`\beta_t`$ (which is why $`T`$ is large), the true reverse conditional $`q(x_{t-1} \mid x_t)`$ is **approximately Gaussian**. This is a deep result; it means we can parameterize the reverse chain with a tractable family:

``` math
p_\theta(x_{t-1} \mid x_t) \;=\; \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\right).
```

where $`\mu_\theta`$ and $`\Sigma_\theta`$ are functions implemented by a neural network. The network takes the current noisy sample $`x_t`$ and the timestep index $`t`$ and produces the mean (and optionally the variance) of where the previous step came from. Sampling means iterating: draw $`x_T \sim \mathcal{N}(0, I)`$, then for $`t = T, T-1, \dots, 1`$, sample $`x_{t-1} \sim p_\theta(x_{t-1} \mid x_t)`$.

In practice, the variance $`\Sigma_\theta`$ is often fixed to a constant schedule (the original DDPM paper’s default), leaving only the mean to be learned. Topic 2 reformulates “learn the mean of the reverse Gaussian” into “learn the noise that was added in the forward step,” which is what the actual network does.

### The architectural choice: a time-conditional U-Net

The diffusion network has to map a noisy image $`x_t`$ (and a timestep index $`t`$) to a same-shape output (either the predicted mean or the predicted noise, depending on the parameterization). The standard architecture is a **U-Net** with timestep embeddings injected at each level. U-Nets were originally invented for biomedical image segmentation; they happen to be perfect for diffusion because they preserve spatial structure while building hierarchical features. Foster’s `01_ddm` notebook uses a small U-Net implementation.

### Where this Topic fits

Topic 2 turns the “learn the reverse mean” setup into a concrete training loss and shows why noise prediction works as a simpler equivalent. Topic 3 covers what to do at sampling time: the basic slow sampler, faster DDIM, conditioning via classifier-free guidance, and the trick (latent diffusion) that makes Stable Diffusion fit on a GPU. Topic 4 returns to the project: how to evaluate generative models in general, and how to write up your Mid-Project Report.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 8, intro sections and the “Forward Diffusion Process” and “Reverse Diffusion Process” subsections.
- J. Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” ICML 2015. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585). Sections 1 and 2; the original forward/reverse derivation.

### Consider

- The forward process has no learnable parameters. Why is that surprising for a generative model, and why does it work anyway?
- The closed-form marginal lets you jump from $`x_0`$ directly to any $`x_t`$ without simulating the chain. Why is this critical for training to be feasible at $`T \approx 1000`$?
- The reverse chain is parameterized as a Gaussian because the per-step noise is small. What goes wrong, mathematically, if you make $`\beta_t`$ large?

### Resources

- O. Ronneberger, P. Fischer, T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597). The U-Net architecture used by most diffusion models.
- L. Weng, “What are Diffusion Models?,” blog post, 2021. [lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). The most popular accessible diffusion explainer; recommended companion to the math above.
