# Week 2, Topic 4: Practical Considerations When Training and Evaluating a VAE

## Topic Overview

**Plain-English summary first.** Topics 1 through 3 told you what a VAE is and what it optimizes. This Topic is the “before you open the notebook” reference: the architecture choices you will face, the training knobs that actually matter, the failure modes to recognize, the evaluation metrics to trust (and the ones to distrust), and what compute environment is appropriate for the Module 2 assignment.

### Architecture choices

The encoder and decoder are independent design problems, but they share a few rules of thumb:

- **Latent dimension $`d`$.** The single most important hyperparameter. Too small and the decoder cannot reconstruct meaningful detail; too large and the KL regularizer has trouble keeping the posterior compact. Common starting points: $`d = 2`$ for visualization, $`d = 10`$ to $`50`$ for Fashion-MNIST, $`d = 100`$ to $`200`$ for face datasets.
- **Encoder network.** For images, a few convolutional layers with stride-2 downsampling, ending in two parallel fully-connected heads that output $`\mu_\phi(x)`$ and $`\log \sigma^2_\phi(x)`$. Outputting *log*-variance keeps the value unconstrained and numerically stable.
- **Decoder network.** Mirror image of the encoder: fully-connected to a small spatial volume, then transposed convolutions (or upsampling + regular convolutions) up to the input resolution. Final activation is sigmoid for \[0,1\] pixel data (Bernoulli decoder) or linear for general-range data (Gaussian decoder).
- **Symmetry.** Encoder and decoder do not strictly need to be symmetric, but starting symmetric is a sensible default.

### Training knobs that matter

- **Optimizer:** Adam with learning rate around $`10^{-3}`$ or $`5 \cdot 10^{-4}`$. Almost never the bottleneck.
- **Batch size:** 64 to 256 is fine for Fashion-MNIST on a laptop. Larger batches give more stable KL estimates but cost memory.
- **KL warmup:** ramp the KL coefficient from 0 to 1 over the first 5 to 20 epochs. Helps avoid posterior collapse, especially when $`d`$ is large.
- **Number of epochs:** 20 to 50 for Fashion-MNIST is usually enough to see a clear pattern; longer is rarely transformative.
- **Reproducibility:** set a fixed random seed for numpy, the framework (TensorFlow / PyTorch), and the dataloader. Log the seed in your write-up.

### Failure modes to recognize

- **Posterior collapse.** KL loss collapses to (near-)zero early in training. Symptom: the encoder ignores $`x`$, so reconstructions look like a single average image regardless of input. Remedies: KL warmup, smaller $`\beta`$, more capacity in the decoder bottleneck.
- **Blurry reconstructions.** The classic VAE failure mode. Most often a consequence of the Gaussian decoder assumption (the mean of a multimodal posterior *is* blurry). Often acceptable for the assignment; not the same problem as posterior collapse.
- **Exploding variance.** If $`\log \sigma^2_\phi`$ is not bounded, training can drift into unreasonably large variances. Clipping the log-variance to, say, $`[-5, 5]`$ is a simple guard.
- **Mode dropping.** Samples cover only some of the modes of the data distribution. Less common in VAEs than in GANs, but watch for it in your prior-sample grids.

### Evaluation: how do you know it worked?

VAE evaluation is genuinely hard, because there is no single number that captures “sample quality.” In this module you will combine three kinds of evidence:

1.  **Loss curves.** Plot total loss, reconstruction loss, and KL loss separately as a function of epoch. Both terms should decrease at first; KL typically rises slightly later as the model trades off. Big gaps or single-term collapses are red flags.
2.  **Sample quality.** Draw a grid (e.g., 16 samples) from the prior and decode. Inspect by eye. For Fashion-MNIST you should recognize garment classes after a few epochs.
3.  **Latent-space structure.** Encode the test set, take the means $`\mu_\phi(x)`$, and either scatter directly (if $`d = 2`$) or project with t-SNE or UMAP. Color points by class label. A well-trained VAE shows class clusters even though it was trained without labels.

Quantitative metrics like FID (Fréchet Inception Distance) and IS (Inception Score) exist but are most useful for comparing methods on the same benchmark; they are overkill for the Module 2 assignment.

### Compute footprint and where to run

Foster’s `02_vae_fashion` notebook trains in roughly 5 to 15 minutes on a CPU laptop, depending on hardware. You do not need a GPU for the required assignment. If your machine struggles, the Wahab cluster via Open OnDemand JupyterLab is the recommended fallback (covered in Module 1 Topic 4). The `03_vae_faces` stretch notebook benefits noticeably from a GPU.

### Where this Topic fits

You are now ready to open the Module 2 VAE Technical Assignment. The assignment asks you to run the Foster `02_vae_fashion` notebook, vary the latent dimension, sample from the prior, and visualize the latent space, then write a 1- to 2-page interpretation. Topic 3 gave you the loss-function vocabulary; this Topic gave you the diagnostics and the architecture defaults.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 3, sections *“Analysis of the Variational Autoencoder”* and *“Using VAEs to Generate Faces.”* Walk through the encoder and decoder architectures and the training loop in the Foster `02_vae_fashion` notebook.

### Consider

- Open the `02_vae_fashion` notebook and find the loss function. Identify which line is the reconstruction term and which line is the KL term. Which decoder family is being assumed?
- If you only had 5 minutes to spot whether someone’s VAE was suffering from posterior collapse, which two diagnostics would you ask to see?
- For your course project dataset, what would “sample quality” even mean? Is there a sensible quantitative metric, or would you rely on visual inspection?

### Resources

- Foster 2nd-edition notebooks: [notebooks/03_vae/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/03_vae) (required `02_vae_fashion/`, stretch `03_vae_faces/`).
- D. Rezende and S. Mohamed, “Variational Inference with Normalizing Flows,” ICML 2015. [arXiv:1505.05770](https://arxiv.org/abs/1505.05770). Worth a skim if you wondered “could the approximate posterior be richer than a diagonal Gaussian?”
- D. Kingma et al., “Improved Variational Inference with Inverse Autoregressive Flow,” NeurIPS 2016. [arXiv:1606.04934](https://arxiv.org/abs/1606.04934). Same direction, more powerful posterior family.
