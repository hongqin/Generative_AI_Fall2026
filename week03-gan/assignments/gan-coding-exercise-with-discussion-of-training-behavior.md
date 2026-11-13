# GAN Coding Exercise with Discussion of Training Behavior

*Assignment, 100 points.*

## Purpose

This is the Module 3 weekly technical assignment. You will train one GAN from Foster’s 2nd-edition repository, diagnose its training behavior, and write a short interpretation that ties what you observed to the failure modes and design choices we discussed in Topics 1 through 4. The point is not just to run code; it is to *read* what the code did and explain it in your own words.

### Task

**Pick one** of the three Foster Chapter 4 notebooks below as your starter. Each notebook is self-contained and ships with its own dataset from Foster’s repository: the DCGAN notebook uses the **LEGO bricks** images (grayscale, 64x64), and the WGAN-GP and Conditional GAN notebooks use **CelebA faces** (RGB, 64x64). Use the dataset the notebook is built for. Both are fetched with the repository’s Kaggle helper (`bash scripts/download.sh bricks` or `bash scripts/download.sh faces`), which needs a free Kaggle account and API token; the repository README explains the setup. If you are not using Foster’s Docker setup, download the same Kaggle datasets by hand and point the `image_dataset_from_directory` path at your copy (Option C also needs `list_attr_celeba.csv` from the same CelebA download). Choose based on what is most interesting for your project direction; none is meant to be harder than the others.

- **Option A: DCGAN** ([notebooks/04_gan/01_dcgan](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan/01_dcgan)). The canonical convolutional GAN; uses the original JS-divergence-style loss with the non-saturating generator trick.
- **Option B: WGAN-GP** ([notebooks/04_gan/02_wgan_gp](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan/02_wgan_gp)). The Wasserstein-GAN-with-gradient-penalty fix; uses a critic and trains the critic multiple times per generator update.
- **Option C: Conditional GAN** ([notebooks/04_gan/03_cgan](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan/03_cgan)). Adds a class-label conditioning input so you can ask the generator for a specific class. Foster’s notebook conditions on the CelebA `Blond_Hair` attribute, so there are two classes.

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

The bricks dataset (about 40,000 small grayscale images) is the lighter option and should be workable on a CPU laptop if you cut `EPOCHS` well below Foster’s 300. The CelebA notebooks (about 200,000 RGB faces) want a GPU: Wahab via Open OnDemand JupyterLab (see Module 1 Topic 4), a personal GPU, or Google Colab; on CPU alone, expect a very long run. Note that in the WGAN-GP and Conditional GAN notebooks an “epoch” is only one or two batches by design (`steps_per_epoch`), so think in training steps rather than epochs. You do not need to match Foster’s run length; train long enough to see the loss curves and sample quality evolve, and report the number of steps or epochs, the batch size, and the hardware. If Kaggle access or compute is a real obstacle, you may substitute Fashion-MNIST for Option A or B (load it with `tf.keras.datasets.fashion_mnist`, build a `tf.data` pipeline, resize to 64x64, and keep `CHANNELS = 1`); Option C depends on the CelebA attribute labels, so keep CelebA there. Say so in your write-up if you substitute. Either dataset choice is graded the same way.

### Evaluation Criteria (100 points)

- **30 pts:** Implementation runs correctly; generator and discriminator (or critic) losses logged separately and plotted.
- **25 pts:** Sample grid present and discussed qualitatively (with attention to mode coverage and sharpness).
- **25 pts:** Failure-mode diagnosis with specific evidence; correctly applies the vocabulary from Topic 2.
- **10 pts:** Reflection on the other two options: predictions grounded in the architectural differences from Topics 1 through 3.
- **10 pts:** Code quality and reproducibility: random seed logged, environment notes present, code runs end-to-end.
