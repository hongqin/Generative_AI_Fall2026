# Week 3: Generative Adversarial Networks (GAN)

## Contents

- Topic 1: [Generator and Discriminator Objectives](topic1-generator-and-discriminator-objectives.md)
- Topic 2: [Adversarial Training Dynamics and Instability](topic2-adversarial-training-dynamics-and-instability.md)
- Topic 3: [Classical GAN Architectures and Common Improvements](topic3-classical-gan-architectures-and-common-improvements.md)
- Topic 4: [Comparing Sample Realism, Controllability, and Training Cost](topic4-comparing-sample-realism-controllability-and-training-cost.md)
- Lecture notes (LaTeX): [`notes/week03-notes.tex`](notes/week03-notes.tex), compiled [`notes/week03-notes.pdf`](notes/week03-notes.pdf)
- Assignment: [GAN Coding Exercise with Discussion of Training Behavior](assignments/gan-coding-exercise-with-discussion-of-training-behavior.md), 100 pts
- Graded discussion: [Project Implementation Check-in Note](assignments/project-implementation-check-in-note.md), 25 pts

## About

This module focuses on adversarial learning for generative modeling. Students will study the generator-discriminator setup, training instability, evaluation challenges, and major design ideas behind classical GAN implementations. The module also extends project development from planning toward implementation.

### Module Learning Objectives

After completing this module, you will be able to:

- **MLO1:** Explain the adversarial training process and contrast GANs with VAEs. (Aligned with CLO1)
- **MLO2:** Run or adapt a GAN-based experiment and diagnose common training issues such as instability or mode collapse. (Aligned with CLO2)
- **MLO3:** Connect GAN design choices to a project implementation plan when GAN-style methods are appropriate. (Aligned with CLO3)

### Required Readings & Resources

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 4, “Generative Adversarial Networks.”** Companion code at [github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition); Chapter 4 notebooks live under [notebooks/04_gan/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan) (subfolders `01_dcgan`, `02_wgan_gp`, `03_cgan`).
- I. Goodfellow et al., “Generative Adversarial Networks,” NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) (the original GAN paper).
- M. Arjovsky, S. Chintala, L. Bottou, “Wasserstein GAN,” ICML 2017. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875).
- I. Gulrajani et al., “Improved Training of Wasserstein GANs” (WGAN-GP), NeurIPS 2017. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028).
- Course tutorial: [Week 3 lecture notes (PDF)](notes/week03-notes.pdf) (Principles of GANs, with detailed derivations).

### Module To-Do List

1.  Read Topic 1 (Generator and Discriminator Objectives) and skim Foster Ch 4 sections “Introduction to GANs” and “Your First GAN.”
2.  Read Topic 2 (Adversarial Training Dynamics and Instability). Open the course tutorial PDF for the detailed derivation of the JS-divergence theorem and the Wasserstein motivation.
3.  Read Topic 3 (Classical GAN Architectures and Common Improvements) and Topic 4 (Comparing Sample Realism, Controllability, and Training Cost).
4.  Complete the **GAN Coding Exercise**. Pick one of three Foster notebooks (DCGAN, WGAN-GP, or Conditional GAN) and write a 1- to 2-page interpretation.
5.  Post your **Project Implementation Check-in** as a reply to the Module 3 project discussion (around 300 to 500 words). Reply to at least one classmate.

### Estimated Completion Time

Plan for roughly **12 to 15 hours** in this module, distributed approximately as:

- Topic pages 1-4 reading and side-reading: ~4 hours.
- GAN Coding Exercise (notebook + write-up): ~5 to 6 hours (GANs train slower than VAEs).
- Project Implementation Check-in (post + one reply): ~1 to 2 hours.


## Module Summary

In this module, you explored adversarial learning as an alternative to latent-variable reconstruction methods and examined why GANs can produce high-quality samples while being difficult to train. The next module broadens the landscape by examining sequence-based autoregressive generation.

## What You Should Have Completed in Module 3

Before moving on, use this self-check to make sure nothing in Module 3 slipped through:

- Read Topics 1 through 4 on the minimax objective, training instability and Wasserstein-GAN, classical architectures, and the GAN-vs-VAE comparison.
- Picked one of three Foster notebooks (`01_dcgan`, `02_wgan_gp`, or `03_cgan`) and trained it end to end.
- Identified at least one named failure mode in your run (oscillating loss, mode collapse, uninformative loss, hyperparameter sensitivity) with specific evidence from your logs or sample grids.
- Submitted the GAN Coding Exercise (notebook plus 1- to 2-page interpretation).
- Posted the Project Implementation Check-in as a reply to the Module 3 project discussion (~300 to 500 words).
- Replied to at least one classmate’s check-in.

## Next Steps

Module 4 covers **Autoregressive Models**: a third family that takes yet another approach to generation, factoring the joint distribution as a product of conditionals and modeling each conditional with a neural network. This is the family that underlies modern large language models. The contrast with both VAEs (Module 2) and GANs (Module 3) is the point.

In Module 4 you will:

- Read Topics 1 through 4 on sequential factorization, RNN and LSTM intuition, sampling and conditioning, and the strengths and limitations of autoregressive generation.
- Complete the Autoregressive or LSTM Mini-Lab and write a short comparison memo.
- Post a Project Progress Update with initial implementation notes.

Bring forward what you learned in the GAN exercise. If you saw mode collapse, that is a data-coverage problem an autoregressive model handles very differently. If your samples were blurry, that is a likelihood-vs-adversarial-objective problem worth thinking about as you compare the families.
