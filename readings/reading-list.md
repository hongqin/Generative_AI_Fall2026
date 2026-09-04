# Reading List

Readings are cited by author, title, and venue with links to open-access versions (arXiv, project pages, GitHub). No copyrighted PDFs are distributed in this repository. The primary reference text is David Foster, *Generative Deep Learning* (2nd ed.), O'Reilly, 2023, with companion code at https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition.

## Week 1: Overview and Course Launch

- Course orientation materials and Canvas overview pages.
- Recorded lecture: overview of generative AI and the course roadmap.
- Selected introductory materials from Generative Deep Learning or equivalent introductory notes posted in Canvas.
- Setup instructions for Python, notebooks, and approved compute platforms.

## Week 2: Variational Autoencoders

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 3, “Variational Autoencoders.”** Companion code at [github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition); the Chapter 3 notebooks live under [notebooks/03_vae/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/03_vae).
- D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) (the original VAE paper).
- D. P. Kingma and M. Welling, *An Introduction to Variational Autoencoders*, 2019. [arXiv:1906.02691](https://arxiv.org/abs/1906.02691) (a tutorial-length treatment recommended for Topic 2).
- Course tutorial with detailed proofs of the ELBO and reparameterization: [Week 2 lecture notes (PDF)](../week02-vae/notes/week02-notes.pdf).
- Optional: C. Doersch, “Tutorial on Variational Autoencoders,” 2016. [arXiv:1606.05908](https://arxiv.org/abs/1606.05908).

## Week 3: Generative Adversarial Networks

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 4, “Generative Adversarial Networks.”** Companion code at [github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition); Chapter 4 notebooks live under [notebooks/04_gan/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/04_gan) (subfolders `01_dcgan`, `02_wgan_gp`, `03_cgan`).
- I. Goodfellow et al., “Generative Adversarial Networks,” NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) (the original GAN paper).
- M. Arjovsky, S. Chintala, L. Bottou, “Wasserstein GAN,” ICML 2017. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875).
- I. Gulrajani et al., “Improved Training of Wasserstein GANs” (WGAN-GP), NeurIPS 2017. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028).
- Course tutorial: [Week 3 lecture notes (PDF)](../week03-gan/notes/week03-notes.pdf) (Principles of GANs, with detailed derivations).

## Week 4: Autoregressive Models

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 5, “Autoregressive Models.”** Companion code at [github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition); Chapter 5 notebooks live under [notebooks/05_autoregressive/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/05_autoregressive) (subfolders `01_lstm`, `02_pixelcnn`, `03_pixelcnn_md`).
- S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, 1997. [Original LSTM paper (free PDF)](https://Where.com).
- A. van den Oord et al., “Pixel Recurrent Neural Networks,” ICML 2016. [arXiv:1601.06759](https://arxiv.org/abs/1601.06759).
- A. van den Oord et al., “Conditional Image Generation with PixelCNN Decoders,” NeurIPS 2016. [arXiv:1606.05328](https://arxiv.org/abs/1606.05328).
- Course tutorial: [Week 4 lecture notes (PDF)](../week04-autoregressive/notes/week04-notes.pdf) (Principles of LSTM, with detailed derivations of the gate equations and gradient flow).

## Week 5: Diffusion Models

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 8, “Diffusion Models.”** Companion code at [github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition); Chapter 8 notebooks live under [notebooks/08_diffusion/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/08_diffusion) (subfolder `01_ddm` is the required starter for the Module 5 assignment).
- J. Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” ICML 2015. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585) (the original diffusion-models paper).
- J. Ho, A. Jain, P. Abbeel, “Denoising Diffusion Probabilistic Models” (DDPM), NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) (the breakthrough paper; the “simplified noise-prediction loss” that made diffusion practical).
- Y. Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) (the unified SDE view connecting diffusion with score matching).
- Course tutorial: [Week 5 lecture notes (PDF)](../week05-diffusion/notes/week05-notes.pdf) (Comments on the Foster DDM notebook with worked derivations).

## Week 6: Transformers

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 9, “Transformers.”** Companion code at [github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition); the Chapter 9 GPT-from-scratch notebook is at [notebooks/09_transformer/gpt/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/09_transformer/gpt).
- A. Vaswani et al., “Attention Is All You Need,” NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) (the original transformer paper; read sections 1 to 3).
- A. Radford et al., “Language Models are Unsupervised Multitask Learners” (GPT-2), 2019. [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). The decoder-only autoregressive architecture that became the modern LLM default.
- S. Rush, “The Annotated Transformer,” Harvard NLP blog, 2018 / updated 2022. [nlp.seas.harvard.edu/annotated-transformer](http://nlp.seas.harvard.edu/annotated-transformer/). Line-by-line annotated PyTorch implementation of the original paper; the canonical accessible reading-the-code companion.

## Week 7: World Models

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 12, “World Models.”** Note: the 2nd-edition GitHub repository does not include companion code for Chapter 12; use the book and the original Ha & Schmidhuber materials below.
- D. Ha and J. Schmidhuber, “World Models,” NeurIPS 2018. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122). The seminal modern world-models paper.
- Companion interactive blog: [worldmodels.github.io](https://worldmodels.github.io/). One of the best-designed AI interactive explainers ever published; recommended over the PDF for first reading.
- D. Hafner et al., “Mastering Diverse Domains through World Models” (DreamerV3), 2023. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104). The modern model-based RL benchmark that uses a world model.

## Week 8: Multimodal Models

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 13, “Multimodal Models.”** Note: like Chapter 12, the 2nd-edition GitHub repository does not include companion code for Chapter 13; use the book and the primary papers below.
- A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision” (CLIP), ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). The seminal modern multimodal paper; read sections 1 to 3.
- A. Ramesh et al., “Zero-Shot Text-to-Image Generation” (DALL-E), 2021. [arXiv:2102.12092](https://arxiv.org/abs/2102.12092).
- A. Ramesh et al., “Hierarchical Text-Conditional Image Generation with CLIP Latents” (DALL-E 2), 2022. [arXiv:2204.06125](https://arxiv.org/abs/2204.06125).
- OpenAI CLIP code: [github.com/openai/CLIP](https://github.com/openai/CLIP). The reference implementation, useful for the Module 8 Case Analysis option.
