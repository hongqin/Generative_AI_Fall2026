# Week 5: Diffusion Models

## Contents

- Topic 1: [Forward Noising and Reverse Denoising](topic1-forward-noising-and-reverse-denoising.md)
- Topic 2: [Training Objectives for Diffusion Models](topic2-training-objectives-for-diffusion-models.md)
- Topic 3: [Sampling Quality, Cost, and Controllability](topic3-sampling-quality-cost-and-controllability.md)
- Topic 4: [Midpoint Evaluation of Project Methods and Progress](topic4-midpoint-evaluation-of-project-methods-and-progress.md)
- Lecture notes (LaTeX): [`notes/week05-notes.tex`](notes/week05-notes.tex), compiled [`notes/week05-notes.pdf`](notes/week05-notes.pdf)
- Assignment: [Diffusion Assignment with Output Analysis and Short Reflection](assignments/diffusion-assignment-with-output-analysis-and-short-reflection.md), 100 pts
- Graded discussion: [Mid-Project Report](assignments/mid-project-report.md), 100 pts

## About

This module introduces diffusion models, the modern third-way alternative to VAEs (Module 2) and GANs (Module 3). Diffusion models have produced state-of-the-art image, video, and audio generation since around 2022 and now power systems like Stable Diffusion, DALL-E, Midjourney, and most contemporary video generators. The module also includes the Mid-Project Report, the first major formal milestone of your course project.

### Module Learning Objectives

After completing this module, you will be able to:

- **MLO1:** Explain the intuition behind forward noising, reverse denoising, and iterative sampling in diffusion models. (Aligned with CLO1)
- **MLO2:** Run or adapt a basic diffusion-model workflow and analyze the quality of generated outputs. (Aligned with CLO2)
- **MLO3:** Document project progress with evidence of testing, partial results, and a revised work plan. (Aligned with CLO3)

### Required Readings & Resources

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 8, “Diffusion Models.”** Companion code at [github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition); Chapter 8 notebooks live under [notebooks/08_diffusion/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/08_diffusion) (subfolder `01_ddm` is the required starter for the Module 5 assignment).
- J. Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” ICML 2015. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585) (the original diffusion-models paper).
- J. Ho, A. Jain, P. Abbeel, “Denoising Diffusion Probabilistic Models” (DDPM), NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) (the breakthrough paper; the “simplified noise-prediction loss” that made diffusion practical).
- Y. Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) (the unified SDE view connecting diffusion with score matching).
- Course tutorial: [Week 5 lecture notes (PDF)](notes/week05-notes.pdf) (Comments on the Foster DDM notebook with worked derivations).

### Module To-Do List

1.  Read Topic 1 (Forward Noising and Reverse Denoising) and skim Foster Ch 8 intro.
2.  Read Topic 2 (Training Objectives for Diffusion Models). Open the course tutorial PDF for the worked-out ELBO and noise-prediction loss derivations.
3.  Read Topic 3 (Sampling Quality, Cost, and Controllability).
4.  Read Topic 4 (Midpoint Evaluation of Project Methods and Progress). This Topic also frames the Mid-Project Report you will submit at the end of the module.
5.  Complete the **Diffusion Assignment**: train Foster’s `01_ddm` notebook, run a sampling-step ablation (50 / 200 / 1000 steps), generate samples, and write a 1- to 2-page interpretation.
6.  **Submit the Mid-Project Report** as a reply to the Module 5 project discussion (800-1500 word PDF, six required sections). Reply to at least one classmate’s report.

### Estimated Completion Time

Plan for roughly **14 to 18 hours** in this module (heavier than usual because of the Mid-Project Report):

- Topic pages 1-4 reading and side-reading: ~4 hours.
- Diffusion Assignment (notebook + write-up; diffusion training is slower than VAE/GAN): ~5 to 6 hours.
- Mid-Project Report (write the 800-1500 word PDF + generate the figures): ~4 to 6 hours.
- Peer reply on the project discussion: ~30 minutes.


## Module Summary

This module established diffusion models as the modern third-way alternative to VAEs and GANs, with comparable or better sample quality and a principled probabilistic foundation. The next module steps into the transformer architecture, which now underlies most large-scale generative systems.

## What You Should Have Completed in Module 5

Before moving on, use this self-check to make sure nothing in Module 5 slipped through:

- Read Topics 1 through 4 on the forward/reverse Markov chains, the noise-prediction training loss, sampling strategies (DDPM, DDIM, classifier-free guidance, latent diffusion), and generative-model evaluation methods.
- Trained Foster’s `01_ddm` notebook end to end.
- Completed the sampling-step ablation (recommended: 50 / 200 / 1000 steps) and observed the quality-versus-cost tradeoff.
- Submitted the Diffusion Assignment (notebook plus 1- to 2-page interpretation).
- Submitted the **Mid-Project Report** as a reply to the Module 5 project discussion (800-1500 word PDF, six required sections).
- Replied to at least one classmate’s Mid-Project Report with substantive feedback.

## Next Steps

Module 6 covers **Transformers**: the architectural change that started by replacing RNNs in machine translation (2017) and has since become the backbone of nearly every modern large generative model, including large language models, diffusion U-Net replacements, multimodal systems, and protein-structure prediction. The transformer is not a new generative family; it is an architecture used inside the autoregressive family (Module 4) at massive scale, and increasingly inside diffusion as well.

In Module 6 you will:

- Read Topics 1 through 4 on self-attention, multi-head attention, positional encoding and tokenization, and the practical use of transformer-based workflows in applied settings.
- Complete the Transformer or LLM Workflow Assignment.
- Post a brief Project Methods Refinement Note as a reply to the Module 6 project discussion.

You are now past the halfway mark. Bring forward what you committed to in the Mid-Project Report; Module 6 is the first execution-phase module, and the work you complete there should map directly onto the milestones you wrote down.
