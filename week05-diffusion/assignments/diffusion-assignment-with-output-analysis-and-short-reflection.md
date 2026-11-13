# Diffusion Assignment with Output Analysis and Short Reflection

*Assignment, 100 points.*

## Purpose

This is the Module 5 weekly technical assignment. You will train a denoising diffusion model from Foster’s 2nd-edition repository, exercise the most important sampling-time knob (the number of denoising steps), and write a short interpretation that ties what you observed to the vocabulary in Topics 1 through 3. The point is not just to run code; it is to *read* what the sampling-step ablation showed you and explain it in your own words.

### Task

Start from Foster’s Chapter 8 denoising-diffusion notebook:

[github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/08_diffusion/01_ddm](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/08_diffusion/01_ddm)

#### Required experiments

1.  **Train the baseline.** Train the DDM end to end on the notebook’s default dataset. Log and plot the training loss (the noise-prediction MSE) as a function of training step or epoch.
2.  **Generate a sample grid at the default number of sampling steps.** Produce a grid of 16 samples drawn from the trained model.
3.  **Sampling-step ablation.** Re-run sampling at three different numbers of denoising steps: **50, 200, and 1000** (or the closest available values supported by the notebook’s sampler). Show three sample grids side by side. Report wall-clock time per sample for each setting.
4.  **Brief comparison to VAE and GAN samples** (1 paragraph). Recall the samples you produced in the Module 2 VAE assignment and the Module 3 GAN assignment. On the same data type, how do the diffusion samples compare on sharpness, diversity, and what you can say about likelihood?

#### Deliverables

Submit both of the following through this assignment:

- Your modified notebook (`.ipynb`) with *all cells run* and figures rendered, **or** a link to a public GitHub repository whose README explains how to reproduce your run.
- A **1- to 2-page PDF write-up** covering: training-loss behavior, what you observed in the sampling-step ablation (sample quality at 50 vs 200 vs 1000 steps; wall-clock cost), and your brief VAE/GAN/diffusion comparison.

#### Citation

Cite Foster correctly in your write-up: *David Foster, Generative Deep Learning (2nd ed.), O’Reilly, 2023, Chapter 8; code from github.com/davidADSP/Generative_Deep_Learning_2nd_Edition.*

#### Compute

Diffusion training is slower than VAE or GAN training because the forward pass is more expensive per step and you need more epochs. Plan for 1 to 3 hours on a CPU laptop, or 15 to 30 minutes on a GPU. Wahab via Open OnDemand JupyterLab (see Module 1 Topic 4) is the recommended path if your machine is slow.

### Evaluation Criteria (100 points)

- **30 pts:** Implementation runs correctly; training loss logged and plotted.
- **25 pts:** Default sample grid present and qualitatively discussed.
- **25 pts:** Sampling-step ablation: three settings tried, sample grids and wall-clock costs reported, quality-versus-cost tradeoff discussed.
- **10 pts:** Comparison paragraph: diffusion vs VAE vs GAN samples on the same data type; references your earlier assignments.
- **10 pts:** Code quality and reproducibility: random seed logged, environment notes present, code runs end-to-end.
