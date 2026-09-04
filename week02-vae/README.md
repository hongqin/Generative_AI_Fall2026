# Week 2: Variational Autoencoders (VAE)

## Contents

- Topic 1: [Latent-Variable Modeling and Representation Learning](topic1-latent-variable-modeling-and-representation-learning.md)
- Topic 2: [Variational Inference and the ELBO](topic2-variational-inference-and-the-elbo.md)
- Topic 3: [Reconstruction Loss, KL Divergence, and Sampling](topic3-reconstruction-loss-kl-divergence-and-sampling.md)
- Topic 4: [Practical Considerations When Training and Evaluating a VAE](topic4-practical-considerations-when-training-and-evaluating-a-vae.md)
- Lecture notes (LaTeX): [`notes/week02-notes.tex`](notes/week02-notes.tex), compiled [`notes/week02-notes.pdf`](notes/week02-notes.pdf)
- Assignment: [VAE Technical Assignment with Brief Interpretation of Results](assignments/vae-technical-assignment-with-brief-interpretation-of-results.md), 100 pts
- Graded discussion: [Course Project Proposal](assignments/course-project-proposal.md), 50 pts

## About

This module introduces variational autoencoders as a foundational latent-variable approach to generative modeling. Students will examine the encoder-decoder framework, the ELBO objective, reconstruction quality, latent representations, and practical implementation issues. The module also asks students to begin formalizing their project plans.

### Module Learning Objectives

After completing this module, you will be able to:

- **MLO1:** Explain the role of latent variables, the ELBO objective, and the encoder-decoder structure in a VAE. (Aligned with CLO1)
- **MLO2:** Run or adapt a basic VAE implementation and interpret reconstruction and sampling results. (Aligned with CLO2)
- **MLO3:** Propose a feasible course project that includes a clear problem statement, technical approach, and supporting references. (Aligned with CLO3)

### Required Readings & Resources

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 3, “Variational Autoencoders.”** Companion code at [github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition); the Chapter 3 notebooks live under [notebooks/03_vae/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/03_vae).
- D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) (the original VAE paper).
- D. P. Kingma and M. Welling, *An Introduction to Variational Autoencoders*, 2019. [arXiv:1906.02691](https://arxiv.org/abs/1906.02691) (a tutorial-length treatment recommended for Topic 2).
- Course tutorial with detailed proofs of the ELBO and reparameterization: [Week 2 lecture notes (PDF)](notes/week02-notes.pdf).
- Optional: C. Doersch, “Tutorial on Variational Autoencoders,” 2016. [arXiv:1606.05908](https://arxiv.org/abs/1606.05908).

### Module To-Do List

1.  Read Topic 1 (Latent-Variable Modeling and Representation Learning) and skim Foster Ch 3 sections “Autoencoders” and “The Variational Art Exhibition.”
2.  Read Topic 2 (Variational Inference and the ELBO). Open the course tutorial PDF if you want every step of the ELBO derivation worked out.
3.  Read Topic 3 (Reconstruction Loss, KL Divergence, and Sampling) and Topic 4 (Practical Considerations When Training and Evaluating a VAE).
4.  Optional warm-up: run `notebooks/03_vae/01_autoencoder/` from the Foster repo to see why a vanilla autoencoder is not generative.
5.  Complete the **VAE Technical Assignment** (anchored on `notebooks/03_vae/02_vae_fashion/`). Submit a notebook plus a 1- to 2-page interpretation.
6.  Post your **Course Project Proposal** as a reply to the Module 2 project discussion (500 to 1000 words, PDF attached). Reply to two classmates' proposals.

### Estimated Completion Time

Plan for roughly **12 to 15 hours** in this module, distributed approximately as:

- Topic pages 1-4 reading and side-reading: ~4 hours.
- VAE Technical Assignment (notebook + write-up): ~4 to 6 hours.
- Course Project Proposal (research, write, post, two replies): ~3 to 4 hours.


## Module Summary

This module established the first core generative modeling framework of the course and introduced students to the tradeoffs between faithful reconstruction and meaningful latent structure. Those ideas prepare us for GANs, which target data realism through a very different training objective.

## What You Should Have Completed in Module 2

Before moving on, use this self-check to make sure nothing in Module 2 slipped through:

- Read Topics 1 through 4 on latent-variable modeling, the ELBO derivation, the loss function and sampling, and the practical training knobs.
- Ran the Foster `02_vae_fashion` notebook end to end with the default latent dimension.
- Repeated the run with at least two more latent-dimension settings (recommended: 2, 10, 50).
- Generated a grid of prior samples and a latent-space scatter colored by class label.
- Submitted the VAE Technical Assignment (notebook plus 1- to 2-page interpretation).
- Posted your Course Project Proposal as a reply to the Module 2 project discussion (500 to 1000 words, PDF attached).
- Replied to at least two classmates’ project proposals with substantive feedback.

## Next Steps

Module 3 covers **Generative Adversarial Networks (GANs)**. GANs share VAEs’ goal (a generative model that produces realistic samples) but use a very different training objective: a two-player game between a generator and a discriminator, with no explicit likelihood term at all. The contrast with VAEs is the point.

In Module 3 you will:

- Read Topics 1 through 4 on the generator-discriminator setup, training instability, classical and Wasserstein variants, and how to evaluate GAN samples.
- Complete the GAN Coding Exercise (run or adapt a baseline GAN and diagnose its training behavior).
- Post a brief project Implementation Check-in showing initial progress on the proposal you committed to in Module 2.

Bring forward your project proposal. Module 3 is when most students start writing or adapting real code; the project discussion is the place to share what worked and what did not.
