# GAN Coding Exercise with Discussion of Training Behavior

*Assignment, 100 points.*

## Purpose

This is the Module 3 weekly technical assignment. You will train one GAN from Foster’s 2nd-edition repository, diagnose its training behavior, and write a short interpretation that ties what you observed to the failure modes and design choices we discussed in Topics 1 through 4. The point is not just to run code; it is to *read* what the code did and explain it in your own words.

### Task

**Pick one** of the three Foster Chapter 4 notebooks below as your starter. Each notebook is self-contained and runs on Fashion-MNIST. Choose based on what is most interesting for your project direction; none is meant to be harder than the others.

- **Option A: DCGAN**: [notebooks/04_gan/01_dcgan](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan/01_dcgan). The canonical convolutional GAN; uses the original JS-divergence-style loss with the non-saturating generator trick.
- **Option B: WGAN-GP**: [notebooks/04_gan/02_wgan_gp](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan/02_wgan_gp). The Wasserstein-GAN-with-gradient-penalty fix; uses a critic and trains the critic multiple times per generator update.
- **Option C: Conditional GAN**: [notebooks/04_gan/03_cgan](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan/03_cgan). Adds a class-label conditioning input so you can ask the generator for a specific Fashion-MNIST class.

#### Required experiments (for whichever option you chose)

1.  **Train the baseline.** Log and plot the generator and discriminator (or critic) losses separately as a function of training step or epoch.
2.  **Generate a sample grid.** Produce a grid of 16 to 25 samples from the trained generator. For the Conditional GAN, show samples grouped by class label. For DCGAN and WGAN-GP, draw $`z \sim \mathcal{N}(0, I)`$.
3.  **Diagnose at least one specific behavior or failure mode.** Look at your loss curves and sample grid and identify at least one of the named phenomena from Topic 2: oscillating loss, mode collapse, uninformative loss, or hyperparameter sensitivity. Cite specific evidence (a screenshot of the loss curve, a sample-grid observation, a numerical detail).
4.  **Brief reflection on the other two options.** In 1 to 2 paragraphs, explain how you expect the other two options would behave on the same problem and why, based on the design differences described in Topics 1 through 3. You do not need to run them; thoughtful prediction is the point.

#### Deliverables

Submit both of the following through this assignment:

- Your modified notebook (`.ipynb`) with *all cells run* and figures rendered, **or** a link to a public GitHub repository whose README explains how to reproduce your run (random seed, environment, command).
- A **1- to 2-page PDF write-up** covering: which option you chose and why, what behaviors you observed during training, which failure mode (if any) showed up and the evidence, and your prediction for how the other two options would have behaved.

#### Citation

Cite Foster correctly in your write-up: *David Foster, Generative Deep Learning (2nd ed.), O’Reilly, 2023, Chapter 4; code from github.com/davidADSP/Generative_Deep_Learning_2nd_Edition.*

#### Compute

All three notebooks train in about 30 to 60 minutes on a CPU laptop for Fashion-MNIST; significantly faster on a GPU. If your machine struggles or you want to scale to a larger dataset, run on Wahab via Open OnDemand JupyterLab (see Module 1 Topic 4 for setup).

### Evaluation Criteria (100 points)

- **30 pts**: Implementation runs correctly; generator and discriminator (or critic) losses logged separately and plotted.
- **25 pts**: Sample grid present and discussed qualitatively (with attention to mode coverage and sharpness).
- **25 pts**: Failure-mode diagnosis with specific evidence; correctly applies the vocabulary from Topic 2.
- **10 pts**: Reflection on the other two options: predictions grounded in the architectural differences from Topics 1 through 3.
- **10 pts**: Code quality and reproducibility: random seed logged, environment notes present, code runs end-to-end.
