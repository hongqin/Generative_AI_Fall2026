# Week 5, Topic 2: Training Objectives for Diffusion Models

## Topic Overview

**Plain-English summary first.** Topic 1 set up the forward and reverse chains and said “the neural network learns the reverse mean.” This Topic derives what the network actually optimizes. The variational view gives an ELBO across all $`T`$ timesteps; in principle that is what you should maximize. The Ho-Jain-Abbeel 2020 paper (DDPM) found that a vastly simpler loss, just predict the noise that was added in the forward step at a random $`t`$, works as well or better. That one insight is the reason diffusion models work in practice. This Topic walks through the derivation and the parameterization choice, then notes the connection to score matching that ties diffusion to the broader score-based generative-modeling family.

### The variational view (where the loss comes from)

The data log-likelihood under the model is

``` math
\log p_\theta(x_0) \;=\; \log \int p_\theta(x_0, x_1, \dots, x_T)\, dx_{1:T},
```

which is intractable. By the same logic as the VAE, we can derive an ELBO over the joint trajectory using the (known, fixed) forward process $`q`$ as the variational distribution:

``` math
\log p_\theta(x_0) \;\geq\; \mathbb{E}_{q}\!\left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right].
```

After algebra (worked out in the course PDF and in Ho et al. 2020), this ELBO decomposes into a sum of per-timestep KL terms, each comparing the model’s reverse step $`p_\theta(x_{t-1} \mid x_t)`$ to the true reverse conditional $`q(x_{t-1} \mid x_t, x_0)`$. The latter is Gaussian with a known mean that can be derived from $`x_t`$ and $`x_0`$, so the KL is tractable.

### The DDPM trick: parameterize the mean as noise prediction

Working out the optimal mean of the reverse step gives

``` math
\mu^{\star}_\theta(x_t, t) \;=\; \frac{1}{\sqrt{\alpha_t}}\!\left( x_t \;-\; \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon \right),
```

where $`\epsilon`$ is the noise that was added during the forward step from $`x_0`$ to $`x_t`$. Ho et al. reparameterize: instead of having the network output $`\mu_\theta`$ directly, have it predict $`\epsilon`$. Call the prediction $`\epsilon_\theta(x_t, t)`$. Then the optimal mean follows by plugging in.

Under this parameterization, the per-timestep KL term simplifies dramatically. After dropping additive constants and reweighting the timesteps (an empirical choice that improves quality), the training objective becomes:

``` math
\mathcal{L}_{\text{simple}}(\theta) \;=\; \mathbb{E}_{t,\, x_0,\, \epsilon} \!\left[\, \big\| \epsilon \;-\; \epsilon_\theta(x_t,\, t) \big\|^2 \,\right], \qquad x_t \;=\; \sqrt{\bar\alpha_t}\, x_0 \,+\, \sqrt{1 - \bar\alpha_t}\, \epsilon.
```

**This is the entire training loss for DDPM**: pick a random timestep, pick a real training image, draw a Gaussian noise, mix them with the closed-form marginal, and ask the network to predict the noise that was added. Mean-squared error against the actual noise. That is it. The network is solving a regression problem over noise, not a complicated probabilistic estimation problem.

### Why this works conceptually

Predicting the noise is equivalent to predicting where the clean signal lives in $`x_t`$, because $`x_0`$ and $`\epsilon`$ are linearly related given $`x_t`$. The network is implicitly learning the “denoising direction” at every noise level. Sampling time uses these denoising directions to walk a noisy point back to the data manifold step by step.

### The score-matching connection

Song and Ermon (2019) developed a parallel line of work called **score-based generative modeling**: instead of training a noise predictor, train a *score network* that estimates $`\nabla_{x} \log p_t(x)`$ (the gradient of the log-density of the noise-perturbed distribution at noise level $`t`$). It turns out the DDPM noise-prediction loss is an estimator of exactly this score, up to a known scaling:

``` math
\epsilon_\theta(x_t, t) \;\approx\; -\sqrt{1 - \bar\alpha_t}\, \nabla_{x_t} \log p_t(x_t).
```

Song et al. (2021) put DDPM, score matching, and the continuous-time limit into a single framework based on stochastic differential equations (SDEs). The forward process becomes an SDE; the reverse process is the time-reversed SDE; both DDPM and score-based methods are discretizations of the same continuous-time formulation. This unification is the modern theoretical view of diffusion. For practical implementation purposes, the DDPM noise-prediction loss is what you almost always implement.

### Variance: predict it or fix it?

DDPM’s simplified loss only trains the mean of the reverse Gaussian; the variance $`\Sigma_\theta`$ is typically fixed to a constant schedule (the original paper found two reasonable choices, both with similar quality). Later work (Improved DDPM, Nichol & Dhariwal 2021) found that learning the variance further improves sample quality, especially at fewer sampling steps. Foster’s notebook uses the fixed-variance choice for simplicity.

### Where this Topic fits

Topic 3 takes a trained diffusion model and covers what to do at sampling time: the basic DDPM sampler (slow), DDIM (faster, deterministic), classifier-free guidance (for conditioning), and latent diffusion (the trick that makes Stable Diffusion practical). Topic 4 is the hybrid technical-and-project Topic that closes Module 5.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 8, sections on the training objective and the noise-prediction loss.
- J. Ho, A. Jain, P. Abbeel, “Denoising Diffusion Probabilistic Models,” NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239). Sections 1 to 3 (the DDPM derivation and the simplified loss); the rest is empirical analysis.

Detailed derivations (recommended if you want every step worked out):

- Course tutorial: [Week 5 lecture notes (PDF)](notes/week05-notes.pdf) (Comments on the Foster DDM notebook).

### Consider

- The ELBO over all $`T`$ timesteps is the principled training objective. The simplified noise-prediction loss drops timestep-weighting terms but works as well or better in practice. What might that tell you about the structure of the original ELBO?
- Predicting noise is equivalent to predicting the clean signal (given $`x_t`$ and $`t`$). Why is the noise parameterization preferred over predicting $`x_0`$ directly?
- The score-matching framing says diffusion models are estimating $`\nabla \log p_t(x)`$ at each noise level. What does it mean intuitively that the gradient of the log-density is the “direction to move toward the data”?

### Resources

- Y. Song and S. Ermon, “Generative Modeling by Estimating Gradients of the Data Distribution,” NeurIPS 2019. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600). The score-matching paper that runs in parallel to DDPM.
- Y. Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456). The unified SDE view of diffusion and score-based methods.
- A. Nichol and P. Dhariwal, “Improved Denoising Diffusion Probabilistic Models,” ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672). The learn-the-variance extension that improves sample quality.
