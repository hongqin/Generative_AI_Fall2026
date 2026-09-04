# Week 4: Autoregressive Models

## Contents

- Topic 1: [Sequential Factorization of Generative Distributions](topic1-sequential-factorization-of-generative-distributions.md)
- Topic 2: [Recurrent Models and LSTM Intuition](topic2-recurrent-models-and-lstm-intuition.md)
- Topic 3: [Sampling, Conditioning, and Sequence Quality](topic3-sampling-conditioning-and-sequence-quality.md)
- Topic 4: [Strengths and Limitations of Autoregressive Generation](topic4-strengths-and-limitations-of-autoregressive-generation.md)
- Lecture notes (LaTeX): [`notes/week04-notes.tex`](notes/week04-notes.tex), compiled [`notes/week04-notes.pdf`](notes/week04-notes.pdf)
- Assignment: [Autoregressive or LSTM Mini-Lab with Short Comparison Memo](assignments/autoregressive-or-lstm-mini-lab-with-short-comparison-memo.md), 100 pts
- Graded discussion: [Project Progress Update with Initial Implementation Notes](assignments/project-progress-update-with-initial-implementation-notes.md), 25 pts

## About

This module examines autoregressive generation through sequence modeling, with emphasis on recurrent or token-by-token approaches that estimate the next output conditioned on previous context. Students will compare autoregressive generation with previous model families and consider where sequence-based methods remain valuable in modern generative AI.

### Module Learning Objectives

After completing this module, you will be able to:

- **MLO1:** Explain the core idea of autoregressive generation and describe how sequential dependence shapes model design. (Aligned with CLO1)
- **MLO2:** Run or modify an autoregressive or LSTM-based example and interpret generated outputs. (Aligned with CLO2)
- **MLO3:** Compare autoregressive approaches with VAE and GAN methods when selecting a generative model for a project. (Aligned with CLO3)

### Required Readings & Resources

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 5, “Autoregressive Models.”** Companion code at [github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition); Chapter 5 notebooks live under [notebooks/05_autoregressive/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/05_autoregressive) (subfolders `01_lstm`, `02_pixelcnn`, `03_pixelcnn_md`).
- S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, 1997. [Original LSTM paper (free PDF)](https://Where.com).
- A. van den Oord et al., “Pixel Recurrent Neural Networks,” ICML 2016. [arXiv:1601.06759](https://arxiv.org/abs/1601.06759).
- A. van den Oord et al., “Conditional Image Generation with PixelCNN Decoders,” NeurIPS 2016. [arXiv:1606.05328](https://arxiv.org/abs/1606.05328).
- Course tutorial: [Week 4 lecture notes (PDF)](notes/week04-notes.pdf) (Principles of LSTM, with detailed derivations of the gate equations and gradient flow).

### Module To-Do List

1.  Read Topic 1 (Sequential Factorization of Generative Distributions) and skim Foster Ch 5 intro.
2.  Read Topic 2 (Recurrent Models and LSTM Intuition). Open the course tutorial PDF for the worked LSTM derivations.
3.  Read Topic 3 (Sampling, Conditioning, and Sequence Quality) and Topic 4 (Strengths and Limitations of Autoregressive Generation).
4.  Complete the **Autoregressive Mini-Lab**. Pick one of three Foster notebooks (LSTM, PixelCNN, or PixelCNN with Mixture Density) and write a 1- to 2-page interpretation.
5.  Post your **Project Progress Update** as a reply to the Module 4 project discussion (around 300 to 500 words). Reply to at least one classmate.

### Estimated Completion Time

Plan for roughly **12 to 15 hours** in this module, distributed approximately as:

- Topic pages 1-4 reading and side-reading: ~4 hours.
- Autoregressive Mini-Lab (notebook + write-up): ~5 to 6 hours.
- Project Progress Update (post + one reply): ~1 to 2 hours.


## Module Summary

This module expanded the course from image-oriented generative models to sequence-based generation and highlighted how conditioning on history changes both training and inference. Those ideas create a bridge to diffusion and transformer methods, which now dominate many modern applications.

## What You Should Have Completed in Module 4

Before moving on, use this self-check to make sure nothing in Module 4 slipped through:

- Read Topics 1 through 4 on the chain-rule factorization, the LSTM cell and its gates, sampling strategies (temperature, top-k, nucleus, beam), and the strengths and limitations of autoregressive generation.
- Picked one of three Foster notebooks (`01_lstm`, `02_pixelcnn`, or `03_pixelcnn_md`) and trained it end to end.
- Generated samples and observed at least one of: temperature / quality tradeoff, raster-scan artifact, repetition, mode coverage at different temperatures.
- Submitted the Autoregressive Mini-Lab (notebook plus 1- to 2-page interpretation).
- Posted the Project Progress Update as a reply to the Module 4 project discussion (~300 to 500 words).
- Replied to at least one classmate’s progress update.

## Next Steps

Module 5 covers **Diffusion Models**: a fourth family that takes yet another approach. Diffusion sidesteps autoregressive’s $`O(N)`$ sampling cost by training a model to denoise in parallel across all positions, and it has largely displaced GANs for high-resolution image generation since around 2022. The Mid-Project Report is also due in Module 5, so plan accordingly.

In Module 5 you will:

- Read Topics 1 through 4 on the forward noising process, the reverse denoising process, training objectives, and the relationship between diffusion and the score function.
- Complete the Diffusion Assignment with output analysis and short reflection.
- Submit the **Mid-Project Report** as a reply to the Module 5 project discussion (formal milestone, 100 pts).

Bring forward what you learned in this module’s mini-lab. If you saw the $`O(N)`$ sampling cost bite you, the diffusion sampling cost (which is $`O(T)`$ where $`T`$ is the number of denoising steps, typically 50 to 1000) is qualitatively different and worth comparing directly.
