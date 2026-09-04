# Week 5, Topic 3: Sampling Quality, Cost, and Controllability

## Topic Overview

**Plain-English summary first.** A trained diffusion model is a function that, given a noisy sample $`x_t`$ and a timestep $`t`$, predicts the noise that was added. Sampling means using that function to walk a noisy starting point back to a clean sample, one step at a time. The straightforward way (DDPM sampling) needs hundreds to thousands of steps; DDIM and other tricks reduce this dramatically. Conditioning (text-to-image, class-to-image) is done by passing the condition into the noise network and using classifier-free guidance to amplify its effect. And the trick that makes Stable Diffusion actually fit on consumer hardware is to do the entire diffusion in a learned latent space rather than in pixel space. This Topic covers all of these.

### DDPM sampling: the slow but clean version

Given a trained $`\epsilon_\theta`$, DDPM sampling is straightforward:

1.  Draw $`x_T \sim \mathcal{N}(0, I)`$.
2.  For $`t = T, T-1, \dots, 1`$: compute the predicted mean $`\mu_\theta(x_t, t)`$ from $`\epsilon_\theta(x_t, t)`$, draw $`z \sim \mathcal{N}(0, I)`$, and set $`x_{t-1} = \mu_\theta(x_t, t) + \sigma_t\, z`$ where $`\sigma_t`$ follows the fixed-variance schedule.
3.  Return $`x_0`$.

This is stochastic (each step adds new noise $`z`$) and follows the variational training derivation exactly. With $`T = 1000`$, generating one sample requires 1000 forward passes through the network. For Fashion-MNIST on a CPU this is a couple of minutes per sample; for a 512×512 image on a GPU, a minute or two per sample. Slow.

### DDIM sampling: fast and deterministic

Song et al. (2020) showed that the same trained network can be used as a deterministic sampler that takes far fewer steps:

``` math
x_{t-1} \;=\; \sqrt{\bar\alpha_{t-1}}\, \underbrace{\left(\frac{x_t - \sqrt{1 - \bar\alpha_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}\right)}_{\text{predicted } x_0} \;+\; \sqrt{1 - \bar\alpha_{t-1}}\, \epsilon_\theta(x_t, t).
```

This is the DDIM update. Practically: it predicts $`x_0`$ from $`x_t`$ using the noise network, then re-noises it to the previous timestep. The whole update is deterministic. The breakthrough is that you can use a *subset* of the original timesteps (e.g., 50 steps spaced through the original 1000) and get qualitatively similar samples to full DDPM. **20x speedup with little quality loss**; the standard choice for production diffusion sampling.

### Conditioning: text-to-image, class-to-image, image-to-image

Conditioning a diffusion model on a label, a text prompt, or an image is straightforward: pass the condition $`y`$ into the noise network as an extra input. The network becomes $`\epsilon_\theta(x_t, t, y)`$ instead of $`\epsilon_\theta(x_t, t)`$. Same training loss, same sampling procedure, just with the condition threaded through.

**Classifier-free guidance** (Ho & Salimans 2022) is the standard trick to amplify the effect of the condition. During training, occasionally drop the condition (set $`y = \emptyset`$, with probability ~10%), so the network learns to predict noise both with and without conditioning. At sampling, combine the conditional and unconditional predictions:

``` math
\tilde\epsilon_\theta(x_t, t, y) \;=\; \epsilon_\theta(x_t, t, \emptyset) \;+\; w \cdot \left( \epsilon_\theta(x_t, t, y) \;-\; \epsilon_\theta(x_t, t, \emptyset) \right).
```

The **guidance scale** $`w`$ controls how strongly samples adhere to the condition: $`w = 1`$ is plain conditional sampling; $`w \gt 1`$ (e.g., 7.5 is typical for Stable Diffusion) pushes samples to be more strongly aligned with $`y`$ at the cost of some diversity. Every modern text-to-image system uses some version of classifier-free guidance.

### Latent diffusion: the Stable Diffusion trick

Doing diffusion in pixel space is expensive: a 512×512×3 image has ~786K dimensions, and you have to run a U-Net over that for hundreds of steps. **Latent diffusion** (Rombach et al. 2022) sidesteps this:

1.  Train a VAE (Module 2!) to encode images into a much smaller latent space (e.g., 64×64×4).
2.  Run the diffusion process *in the latent space*, not on raw pixels.
3.  At sampling time, run the diffusion to get a latent, then decode with the VAE decoder to get the image.

**This is why Stable Diffusion runs on consumer GPUs.** The latent space is 48x smaller than pixel space, so the diffusion U-Net is dramatically cheaper. Image quality is preserved because the VAE captures the perceptually relevant structure; the diffusion model only has to learn the latent distribution. The encoder/decoder is trained once and frozen; the diffusion model trains on top.

### Sampling cost compared to other families

| Family | Sampling cost (per sample) | Parallelizable across positions? |
|----|----|----|
| VAE | One decoder forward pass | Yes |
| GAN | One generator forward pass | Yes |
| Autoregressive | $`O(N)`$ sequential passes | No (each position depends on prior) |
| Diffusion (DDPM) | $`O(T)`$ sequential passes, $`T \approx 1000`$ | Yes within each step (the U-Net is parallel); not across steps |
| Diffusion (DDIM) | $`O(T')`$ sequential passes, $`T' \approx 50`$ | Same as DDPM |

Diffusion is slower than VAE/GAN at sampling time but faster than autoregressive on long sequences (a 1024×1024 image has a million pixels, an autoregressive PixelCNN would need a million passes; diffusion needs 50 to 1000). This is one of the main practical reasons diffusion took over high-resolution image generation.

### Where this Topic fits

Topic 4 closes Module 5 with the hybrid of generative-model evaluation (FID and friends, plus what they don’t tell you) and explicit guidance for the Mid-Project Report formal milestone. Module 6 (Transformers) returns to the architectural lens; many modern diffusion models replace U-Nets with diffusion transformers (DiTs) for further scaling.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 8, sampling sections.
- J. Song, C. Meng, S. Ermon, “Denoising Diffusion Implicit Models” (DDIM), ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502). Sections 1 to 4 cover the deterministic sampler that became the practical default.
- J. Ho and T. Salimans, “Classifier-Free Diffusion Guidance,” NeurIPS Workshop 2021. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598). The short paper that introduces the standard conditioning trick.

### Consider

- DDIM gets ~20x speedup over DDPM with little quality loss. Given that both use the same trained network, what does that imply about the training-time choice of $`T = 1000`$ steps?
- Classifier-free guidance with high $`w`$ makes samples more strongly aligned with the prompt but less diverse. What is the intuition for this tradeoff?
- Latent diffusion trains a VAE first, then runs diffusion in the VAE’s latent space. What does this say about the role of the VAE: is it “just” a compression module, or is the latent space doing something deeper?

### Resources

- R. Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models,” CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752). The Stable Diffusion paper.
- P. Dhariwal and A. Nichol, “Diffusion Models Beat GANs on Image Synthesis,” NeurIPS 2021. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233). The paper that established diffusion as the new SOTA for image synthesis.
- W. Peebles and S. Xie, “Scalable Diffusion Models with Transformers” (DiT), ICCV 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748). Replaces the U-Net with a transformer; forward reference to Module 6.
