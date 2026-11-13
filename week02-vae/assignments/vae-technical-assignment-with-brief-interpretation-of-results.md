# VAE Technical Assignment with Brief Interpretation of Results

*Assignment, 100 points.*

## Purpose

This is the Module 2 weekly technical assignment. You will train a baseline Variational Autoencoder on Fashion-MNIST, vary one important hyperparameter, sample from the trained model, and visualize the latent space. The point is not just to run code; it is to *interpret* the results in the vocabulary you learned across Topics 1 through 4. Your write-up is graded as heavily as your code.

### Task

Start from Foster’s 2nd-edition notebook `02_vae_fashion` in the public companion repository [here](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/03_vae/02_vae_fashion)

Optional warm-up: run `notebooks/03_vae/01_autoencoder/` first to see for yourself why a vanilla autoencoder is not generative. Stretch (optional): try `notebooks/03_vae/03_vae_faces/` if you have GPU access via Wahab Open OnDemand and want to extend the analysis to a faces dataset.

#### Required experiments

1.  **Baseline run.** Train the default `02_vae_fashion` notebook. Log and plot three loss curves on the same figure: total loss, reconstruction loss, KL loss, each as a function of epoch.
2.  **Latent-dimension ablation.** Repeat training at **three latent dimensionalities** (recommended set: $`d = 2`$, $`d = 10`$, $`d = 50`$; other choices are fine if you justify them). For each $`d`$, report (a) the final loss curves, (b) a side-by-side panel of reconstructions for 8 test images.
3.  **Sample from the prior.** For your best-performing latent dim, draw 16 samples $`z \sim \mathcal{N}(0, I)`$, decode them, and show the resulting grid.
4.  **Latent-space visualization.** Encode the entire Fashion-MNIST test set, take the encoder means $`\mu_\phi(x)`$, and scatter them in 2D, colored by class label. If you used $`d = 2`$ directly, scatter the means as they are; for larger $`d`$, project with t-SNE or UMAP first.

#### Deliverables

Post both of the following through this assignment:

- Your modified notebook (`.ipynb`) with *all cells run* and figures rendered, **or** a link to a public GitHub repository whose README explains how to reproduce your runs (random seed, environment, command).
- A **1- to 2-page PDF write-up** interpreting what you observed. The write-up should answer, in your own words: how the loss curves changed across latent dims, where the model failed, what the prior samples look like for each $`d`$, and what the latent-space scatter tells you about Fashion-MNIST.

#### Citation

Cite Foster correctly in your write-up: *David Foster, Generative Deep Learning (2nd ed.), O’Reilly, 2023, Chapter 3; code from github.com/davidADSP/Generative_Deep_Learning_2nd_Edition.*

#### Compute

Fashion-MNIST trains in roughly 5 to 15 minutes on a CPU laptop. If your machine struggles, run on Wahab via Open OnDemand JupyterLab (see Module 1 Topic 4 for setup instructions).

### Evaluation Criteria (100 points)

- **30 pts:** Implementation runs correctly; loss curves logged separately for total, reconstruction, and KL.
- **25 pts:** Latent-dimension ablation: at least three values tried; results presented side-by-side with both loss curves and reconstructions.
- **20 pts:** Prior samples shown and qualitatively discussed; latent-space scatter present and colored by class.
- **15 pts:** Interpretation write-up: clear, specific, ties observations to ELBO terms and to the failure modes from Topics 3 and 4.
- **10 pts:** Code quality and reproducibility: random seed logged, environment notes present, code runs end-to-end.
