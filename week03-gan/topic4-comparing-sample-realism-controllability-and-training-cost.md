# Week 3, Topic 4: Comparing Sample Realism, Controllability, and Training Cost

## Topic Overview

**Plain-English summary first.** GANs are not the only generative-model family, and they are far from the obvious choice for every problem. This Topic puts GANs side by side with VAEs (the family from Module 2) along three practical dimensions: *sample realism* (how good do the outputs look?), *controllability* (can you make it produce a specific kind of output?), and *training cost* (how much compute and how much human attention does it take to train one?). It also previews how diffusion models (Module 5) sit on the same axes, so you can see why diffusion has eaten so much of the GAN application space in the last few years.

### Sample realism

VAEs trained with a Gaussian decoder produce noticeably blurry samples. The root cause is the same as the blurry-reconstruction problem we covered in Module 2 Topic 4: the mean of a multimodal posterior *is* blurry. GANs do not have this problem because they have no per-pixel reconstruction term; the generator is rewarded for producing an image the discriminator cannot distinguish from a real one, not for producing the “average” image that minimizes pixel error.

This is the headline reason to reach for a GAN: **sharpness**. A trained GAN on faces (StyleGAN, FFHQ dataset) produces images that pass for photographs at modest resolution. A VAE trained on the same data produces recognizable but blurry faces.

The asterisk: GAN samples are sharp *in the modes they cover*. If the GAN has dropped modes (Topic 2), its samples will be sharp *and* incomplete. A VAE will cover the modes (its KL regularizer pushes it toward broad coverage) at the cost of sharpness.

### Evaluation: the metrics and what they don’t tell you

Quantifying “sample quality” for an image generator is a research problem in its own right. Two common metrics:

- **Fréchet Inception Distance (FID)** (Heusel et al., 2017): pass real and generated samples through a pretrained Inception network, fit Gaussians to the activations of an intermediate layer, and compute the Fréchet distance between the two Gaussians. Lower is better. FID is the de facto standard for image-GAN comparison.
- **Inception Score (IS)** (Salimans et al., 2016): predecessor to FID; measures sample diversity and class-recognizability through the same pretrained classifier. Higher is better. Largely superseded by FID but still reported.

**What these metrics do not tell you:** how good your samples are for *your* use case. FID is calibrated for natural-image benchmarks and depends on the Inception network being a sensible feature extractor for your domain. If you are generating medical images, scientific data, or anything outside Inception’s training distribution, FID is at best a rough proxy. Plan to combine quantitative metrics with visual inspection and (where possible) downstream-task evaluation.

### Controllability

A vanilla GAN has essentially no controllability. You draw $`z \sim p_z`$ and get whatever the generator produces; there are no semantic dials. Adding controllability requires choosing an architecture that exposes one:

- **Conditional GAN**: conditioning variable $`y`$ (class label, caption, sketch) exposed as an explicit input.
- **StyleGAN**: a mapping network that gives interpretable “style” vectors at multiple resolutions; you can edit the high-level pose independently from the fine-grained texture.
- **VAE comparison**: VAEs give you the latent space $`z`$ directly (via the encoder), which means you can do latent-arithmetic editing on existing samples (Foster Ch 3 “Latent Space Arithmetic”). Vanilla GANs cannot do this without a separate inversion step.

### Training cost

GANs are notoriously expensive to train at quality, especially the modern variants:

- Foster’s `01_dcgan` notebook on the LEGO bricks dataset (about 40,000 grayscale 64x64 images) is light enough for a CPU laptop or a CPU-only Open OnDemand JupyterLab session if you cut the epoch count; on a GPU it is a matter of minutes per epoch.
- The `02_wgan_gp` and `03_cgan` notebooks on CelebA (about 200,000 RGB faces) want a GPU; expect hours per run on CPU.
- StyleGAN2 on FFHQ at 1024x1024 takes **weeks** on multi-GPU clusters. Not realistic for a course project.

If your project requires a high-quality GAN, plan your scope around what you can train on Wahab in a reasonable time. Module 1 Topic 4 has the compute guidance.

### VAE vs GAN: a summary table

| Dimension | VAE | GAN |
|----|----|----|
| Training objective | ELBO (explicit lower bound on log-likelihood) | Minimax / saddle point; no explicit likelihood |
| Encoder $`q_\phi(z|x)`$ | Yes, trained alongside the decoder | None (vanilla); separate inversion step needed |
| Sample sharpness | Blurry (Gaussian decoder) | Sharp when training succeeds |
| Mode coverage | Broad (KL regularizer pushes toward coverage) | Can drop modes (Topic 2) |
| Training stability | Routine | Hard; many named failure modes |
| Controllability | Latent arithmetic on the encoded $`z`$ | Needs Conditional GAN, StyleGAN, etc. |
| Likelihood evaluation | Yes (via ELBO) | No closed form |
| Typical use cases | Representation learning, controllable generation, anomaly detection | High-quality image synthesis, image-to-image translation, style transfer |

### Where diffusion (Module 5) will fit

Diffusion models, which we will study in two modules, are the most recent answer to “what if we could get GAN-like sharpness without the GAN training difficulty?” They have an explicit (variational) training objective like VAEs, produce samples competitive with the best GANs, support conditioning natively, and have been the dominant approach in image, video, and audio generation since 2022. The tradeoff: **sampling is slow** (many denoising steps), and training requires more compute than a comparable VAE. For now, just know that the family of choices is wider than VAE-vs-GAN.

### Where this Topic fits

You should now have a reasonable picture of the GAN landscape and how it compares to the latent-variable family we built in Module 2. The Module 3 coding exercise asks you to train one of the three Foster notebooks and explain what you saw using the vocabulary from Topics 1 through 4. The project Implementation Check-in asks you to translate this material into your own project: what generative-model family are you committing to, and why?

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 4, section *“Analysis of the WGAN-GP”* (Foster’s qualitative discussion of sample quality and failure modes).
- M. Heusel et al., “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium” (introduces FID), NeurIPS 2017. [arXiv:1706.08500](https://arxiv.org/abs/1706.08500). Skim sections 3 and 4 for FID.

### Consider

- For your course project, which of (VAE, GAN, diffusion) is the natural fit, and which would you want to rule out early so you do not waste time on it?
- If your domain is far from natural images (medical imaging, audio, scientific data, text), what would you use as your sample-quality metric instead of FID?
- The summary table above lists “controllability” as a major axis. For a real downstream user (clinician, designer, researcher), what kind of controllability would they actually need, and which architecture choice supports it?

### Resources

- L. Theis, A. van den Oord, M. Bethge, “A Note on the Evaluation of Generative Models,” ICLR 2016. [arXiv:1511.01844](https://arxiv.org/abs/1511.01844). The classic argument that no single metric captures “generative model quality.”
- A. Borji, “Pros and Cons of GAN Evaluation Measures: New Developments,” CVIU 2022. [arXiv:2103.09396](https://arxiv.org/abs/2103.09396). Modern survey of GAN evaluation methods.
