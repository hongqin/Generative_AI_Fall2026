# Week 3, Topic 3: Classical GAN Architectures and Common Improvements

## Topic Overview

**Plain-English summary first.** Topics 1 and 2 set up the GAN game and the Wasserstein fix at the level of objectives. This Topic moves to the concrete architectures the community has actually used: the convolutional backbone (DCGAN) that made GANs work on images at all, the label-conditioning extension (Conditional GAN) that gives you a knob to control what the generator produces, and a brief tour of the modern high-end variants (StyleGAN and friends). The goal is for you to recognize the family tree well enough to choose the right starting architecture for your project.

### DCGAN: the convolutional backbone

Radford, Metz, and Chintala (2016) found a set of architecture choices that consistently produced trainable convolutional GANs:

- **Replace pooling layers with strided convolutions** (in the discriminator) and **fractionally-strided convolutions** (in the generator). Lets the network learn its own spatial down- and up-sampling.
- **Use batch normalization** in both the generator and the discriminator (except the generator’s output layer and the discriminator’s input layer).
- **Remove fully-connected hidden layers** for deeper architectures.
- **Use ReLU activations** in the generator (except the output, which uses tanh) and **LeakyReLU** in the discriminator.

These choices read like a recipe because they were arrived at empirically. The DCGAN paper’s real contribution is not any single architectural innovation but the discovery that this particular combination is stable enough to train. Foster’s `01_dcgan` notebook implements DCGAN almost verbatim.

### Conditional GAN: telling the generator what to make

The vanilla GAN has no knobs at inference: you draw a random $`z`$ and get whatever the generator produces. **Conditional GAN** (Mirza and Osindero, 2014) adds a conditioning variable $`y`$ to both networks. The generator now takes $`(z, y)`$ and produces $`G_\theta(z, y)`$; the discriminator takes $`(x, y)`$ and outputs whether $`x`$ is a real sample *for that condition*:

``` math
\min_\theta\, \max_\phi\; \mathbb{E}_{(x, y) \sim p_{\text{data}}}\!\left[\log D_\phi(x, y)\right] \;+\; \mathbb{E}_{z \sim p_z,\, y \sim p_y}\!\left[\log\!\left(1 - D_\phi\!\left(G_\theta(z, y), y\right)\right)\right].
```

For class-conditional image generation, $`y`$ is a one-hot class label that is concatenated to the noise vector $`z`$ (and to the intermediate feature maps of the discriminator). At inference, you choose $`y`$ to pick the class you want. Foster’s `03_cgan` notebook implements this on CelebA faces, conditioning on a single binary attribute (`Blond_Hair`), so $`y`$ has two classes.

The conditioning variable does not have to be a class label. It can be text (an early text-to-image GAN), a sketch, another image (Pix2Pix, CycleGAN), a segmentation map, or any structured side information. The architectural pattern is the same: inject $`y`$ into both networks.

### A brief tour of the modern GAN zoo

Beyond DCGAN and Conditional GAN, a few extensions are worth knowing about by name even if you do not implement them this term:

- **Pix2Pix** (Isola et al., 2017): image-to-image translation with a conditional GAN where $`y`$ is itself an image (sketch, edge map, segmentation). Used for sketch-to-photo, day-to-night, etc.
- **CycleGAN** (Zhu et al., 2017): unpaired image-to-image translation. Trains two GANs (A→B and B→A) with a cycle-consistency loss $`\| F(G(x)) - x \|`$ so you do not need matched pairs.
- **Progressive Growing GAN** (Karras et al., 2018): start training at low resolution (4x4), progressively add layers to reach high resolution (1024x1024). Stabilizes large-scale training.
- **StyleGAN** and **StyleGAN2** (Karras et al., 2019, 2020): redesigned generator with a mapping network and per-layer style modulation. State-of-the-art face generation for several years; the source of the famous “person who does not exist” photos.

The unifying theme: every one of these is a particular architectural and loss-function combination layered on top of the basic GAN game. None of them changes the underlying minimax setup; they change the network architecture and the conditioning.

### Where this Topic fits

Topic 4 zooms out: how do GANs actually compare to VAEs (Module 2) and to diffusion (Module 5) on sample quality, controllability, and compute cost, and how should that inform your project choice? The Module 3 coding exercise gives you a concrete experience training one of the three Foster notebooks (DCGAN, WGAN-GP, or Conditional GAN), which is a good way to make the abstractions in this Topic stick.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 4 (Foster’s DCGAN treatment). Skim Chapter 9 sections on the modern variants (StyleGAN, BigGAN) for context.
- A. Radford, L. Metz, S. Chintala, “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks” (DCGAN), ICLR 2016. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434).
- M. Mirza, S. Osindero, “Conditional Generative Adversarial Nets,” 2014. [arXiv:1411.1784](https://arxiv.org/abs/1411.1784).

### Consider

- The DCGAN paper is unusual in that its contribution is a particular combination of standard ideas, not a new theoretical insight. Why do you think that combination is so much more stable than the alternatives?
- For your course project, what conditioning variable $`y`$ would be useful if you used a Conditional GAN? (Class label? Text caption? Sensor reading? Time index?)
- StyleGAN works by separating “what to generate” from “how to generate it.” What would that separation buy you in a non-image domain?

### Resources

- P. Isola et al., “Image-to-Image Translation with Conditional Adversarial Networks” (Pix2Pix), CVPR 2017. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004).
- J.-Y. Zhu et al., “Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks” (CycleGAN), ICCV 2017. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593).
- T. Karras, S. Laine, T. Aila, “A Style-Based Generator Architecture for Generative Adversarial Networks” (StyleGAN), CVPR 2019. [arXiv:1812.04948](https://arxiv.org/abs/1812.04948).
