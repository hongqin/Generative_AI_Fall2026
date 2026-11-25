# Diffusion Assignment with Output Analysis and Short Reflection

*Assignment, 100 points.*

## Purpose

This is the Module 5 weekly technical assignment. You will train a denoising diffusion model from Foster’s 2nd-edition repository, exercise the most important sampling-time knob (the number of denoising steps), and write a short interpretation that ties what you observed to the vocabulary in Topics 1 through 3. The point is not just to run code; it is to *read* what the sampling-step ablation showed you and explain it in your own words.

### Task

Start from Foster’s Chapter 8 denoising-diffusion notebook:

[github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/08_diffusion/01_ddm](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/08_diffusion/01_ddm)

#### Required experiments

1.  **Train the baseline.** Train the DDM end to end on the notebook’s default dataset. Log and plot the training loss as a function of training step or epoch. Foster’s notebook trains with the mean absolute error between the true and predicted noise (logged as `n_loss`), not the mean squared error used in the DDPM paper; report the loss you actually trained with.
2.  **Generate a sample grid at the default number of sampling steps** (20 in Foster’s notebook). Produce a grid of 16 samples drawn from the trained model. The notebook’s inference cell generates 10 images and its `display` helper shows them in a single row, so call `ddm.generate(num_images=16, diffusion_steps=20).numpy()` and arrange the 16 images in a 4 by 4 grid (for example with `matplotlib` subplots).
3.  **Sampling-step ablation.** Re-run sampling at three different numbers of denoising steps: **50, 200, and 1000**. Foster’s sampler accepts any number of steps, so use these values exactly, with the same seed for all three settings so the grids are comparable. Show three sample grids side by side. Report wall-clock time per sample for each setting.
4.  **Brief comparison to VAE and GAN samples** (1 paragraph). Recall the samples you produced in the Module 2 VAE assignment and the Module 3 GAN assignment. Your Module 2 and Module 3 models were trained on different data (Fashion-MNIST, LEGO bricks, or CelebA) than Foster’s diffusion notebook (Oxford 102 Flowers, 64x64 color), so name the datasets, then compare the three model families on sharpness and diversity, and say what each family lets you claim about likelihood.

#### Deliverables

Submit both of the following through this assignment:

- Your modified notebook (`.ipynb`) with *all cells run* and figures rendered, **or** a link to a public GitHub repository whose README explains how to reproduce your run.
- A **1- to 2-page PDF write-up** covering: training-loss behavior, what you observed in the sampling-step ablation (sample quality at 50 vs 200 vs 1000 steps; wall-clock cost), and your brief VAE/GAN/diffusion comparison.

#### Citation

Cite Foster correctly in your write-up: *David Foster, Generative Deep Learning (2nd ed.), O’Reilly, 2023, Chapter 8; code from github.com/davidADSP/Generative_Deep_Learning_2nd_Edition.*

#### Compute

Diffusion training is compute-heavy: the U-Net is larger than the VAE and GAN networks you trained earlier, and Foster’s defaults (64x64 color flower images, 50 epochs, with each epoch passing over the dataset five times) add up to roughly 30,000 training steps. Plan to use a GPU; on a CPU alone a full default run takes many hours. Wahab via Open OnDemand JupyterLab on a GPU partition (see Module 1 Topic 4) is the recommended path. The notebook saves a checkpoint every epoch, so if a session runs out of time you can reload the saved weights with `LOAD_MODEL = True`. You do not need to match Foster’s run length: train for fewer epochs if you need to, and report the number of epochs, the batch size, and the hardware. The dataset comes from Kaggle (`nunenuh/pytorch-challange-flower-dataset`; a free Kaggle account and API token are required). Inside Foster’s Docker setup, fetch it with `bash scripts/download.sh flowers`. Elsewhere (for example on Wahab), download the same dataset by hand (for example `kaggle datasets download -d nunenuh/pytorch-challange-flower-dataset`, then unzip it) and point the `image_dataset_from_directory` path at the `dataset` folder of your copy. **If you are on TensorFlow 2.16 or later** (for example, current Google Colab), some lines in Foster’s notebook may fail. At minimum, replace `optimizers.experimental.AdamW` with `optimizers.AdamW`, and change the checkpoint file name from `checkpoint.ckpt` to `checkpoint.weights.h5` in both the `ModelCheckpoint` callback and the `load_weights` line. Alternatively, install the `tf-keras` version that matches your TensorFlow and set `TF_USE_LEGACY_KERAS=1` before TensorFlow is first imported (for example with `os.environ` at the top of the first cell, then restart the kernel). Foster’s repository pins TensorFlow 2.10, which avoids these issues.

### Evaluation Criteria (100 points)

- **30 pts:** Implementation runs correctly; training loss logged and plotted.
- **25 pts:** Default sample grid present and qualitatively discussed.
- **25 pts:** Sampling-step ablation: three settings tried, sample grids and wall-clock costs reported, quality-versus-cost tradeoff discussed.
- **10 pts:** Comparison paragraph: diffusion vs VAE vs GAN samples, naming the datasets; references your earlier assignments.
- **10 pts:** Code quality and reproducibility: random seed logged, environment notes present, code runs end-to-end.
