# Autoregressive or LSTM Mini-Lab with Short Comparison Memo

*Assignment, 100 points.*

## Purpose

This is the Module 4 weekly technical assignment. You will train one autoregressive model from Foster’s 2nd-edition repository, exercise at least one of the decoding-strategy knobs from Topic 3, and write a short interpretation that ties what you observed to the vocabulary in Topics 1 through 4. The point is not just to run code; it is to *read* what the code did and explain it in your own words.

### Task

**Pick one** of the three Foster Chapter 5 notebooks below as your starter. Each is self-contained; choose based on the data type that feels most relevant for your project direction.

- **Option A: LSTM text generation** ([notebooks/05_autoregressive/01_lstm](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/05_autoregressive/01_lstm)). Foster’s LSTM trained on a small text corpus; sample with different temperatures.
- **Option B: PixelCNN image generation** ([notebooks/05_autoregressive/02_pixelcnn](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/05_autoregressive/02_pixelcnn)). Autoregressive over Fashion-MNIST pixels in raster-scan order.
- **Option C: PixelCNN with Mixture Density** ([notebooks/05_autoregressive/03_pixelcnn_md](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/05_autoregressive/03_pixelcnn_md)). Same task as Option B but with a mixture-density output that gives richer per-pixel distributions; compare to plain PixelCNN.

#### Required experiments (for whichever option you chose)

1.  **Train the baseline.** Log and plot the training loss (cross-entropy or negative log-likelihood) as a function of epoch.
2.  **Generate samples.**
    - For **Option A (LSTM text)**: produce at least three sample passages at *three different temperatures* (recommended: $`T = 0.5, 1.0, 1.5`$). Show enough sample for the temperature effect to be visible.
    - For **Option B / C (PixelCNN)**: produce a grid of 16 prior samples. If you tried both Option B and Option C, compare the two grids.
3.  **Diagnose at least one specific behavior.** Identify at least one of: temperature-quality tradeoff (Option A), raster-scan directional artifact (Options B/C), repetition or mode collapse, distribution mismatch between training and sampled outputs. Cite specific evidence (a sample, a loss-curve observation, a numerical detail).
4.  **Brief reflection on the other two options.** In 1 to 2 paragraphs, explain how you expect the other two options would behave on the same problem and why, based on the design differences described in Topics 1 through 3. You do not need to run them; thoughtful prediction is the point.

#### Deliverables

Submit both of the following through this assignment:

- Your modified notebook (`.ipynb`) with *all cells run* and figures rendered, **or** a link to a public GitHub repository whose README explains how to reproduce your run.
- A **1- to 2-page PDF write-up** covering: which option you chose and why, what behaviors you observed during training and sampling, which named phenomenon from Topics 1-4 you identified with the supporting evidence, and your prediction for how the other two options would behave.

#### Citation

Cite Foster correctly in your write-up: *David Foster, Generative Deep Learning (2nd ed.), O’Reilly, 2023, Chapter 5; code from github.com/davidADSP/Generative_Deep_Learning_2nd_Edition.*

#### Compute

The LSTM and PixelCNN notebooks train in roughly 30 to 90 minutes on a CPU laptop for the small Foster default datasets; significantly faster on a GPU. If your machine struggles or you want to scale to a larger corpus or image dataset, run on Wahab via Open OnDemand JupyterLab (see Module 1 Topic 4 for setup).

### Evaluation Criteria (100 points)

- **30 pts:** Implementation runs correctly; training loss logged and plotted.
- **25 pts:** Samples present and qualitatively discussed (for text: across at least three temperatures; for image: a clear sample grid).
- **25 pts:** Behavior diagnosis with specific evidence; correctly applies the vocabulary from Topics 1-4.
- **10 pts:** Reflection on the other two options: predictions grounded in the architectural differences from Topics 1-3.
- **10 pts:** Code quality and reproducibility: random seed logged, environment notes present, code runs end-to-end.
