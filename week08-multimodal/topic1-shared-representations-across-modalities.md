# Week 8, Topic 1: Shared Representations Across Modalities

## Topic Overview

**Plain-English summary first.** The single most consequential idea in modern multimodal AI is the joint embedding space: train modality-specific encoders that map their inputs into a shared vector space. Once a 64-dimensional vector represents either an image or its caption, you can compare image to image, caption to caption, or image to caption with the same dot-product operation. Retrieval becomes vector search. Classification becomes argmax over candidate-class embeddings. Conditional generation becomes “decode something whose embedding matches this target embedding.” This Topic sets up the joint-embedding framework; Topic 2 covers the contrastive training procedure that makes it work; Topic 3 surveys the systems built on top.

### The setup

Consider a dataset of image-caption pairs $`\{(x_i, y_i)\}_{i=1}^{N}`$, where $`x_i`$ is an image and $`y_i`$ is the caption that describes it. The standard approach has three pieces:

- An **image encoder** $`f_{\text{img}}`$ that maps an image to a vector in $`\mathbb{R}^d`$. A vision transformer (ViT) or ResNet-50 are typical choices.
- A **text encoder** $`f_{\text{txt}}`$ that maps a caption to a vector in $`\mathbb{R}^d`$. A transformer (Module 6) is the standard choice.
- A **training objective** that pushes matched image-caption pairs to map to nearby vectors and unmatched pairs to far-apart vectors. Topic 2 covers this in detail.

After training, the two encoders independently produce $`d`$-dimensional vectors in a shared space. The bet is that the shared structure (“cat-ness,” “sunset-ness,” etc.) is captured along consistent directions in this space, regardless of which encoder produced the vector.

### Cross-modal retrieval as the canonical use

Once you have a joint embedding space, the natural first application is **cross-modal retrieval**: search images by text, or search text by images. The procedure is simple:

1.  At indexing time, embed all candidate items in the gallery (e.g., all images in a dataset). Store the vectors.
2.  At query time, embed the query (in the other modality). Compute cosine similarity between the query vector and all gallery vectors. Return the top-k.

The cosine similarity between two vectors $`u`$ and $`v`$ is

``` math
\mathrm{cos}(u, v) \;=\; \frac{u \cdot v}{\|u\|_2 \, \|v\|_2},
```

which ranges from $`-1`$ (opposite direction) to $`+1`$ (same direction). The standard practice is to *L2-normalize* all embeddings (project onto the unit sphere) so that cosine similarity reduces to a plain dot product. This is how systems like Google Images and modern visual search engines work.

### Aligned vs grounded representations

Two terms you will see in the literature, with different meanings:

- **Aligned** representations: an image and its caption map to nearby vectors. The model knows which images and which captions correspond, but not necessarily what objects within the image map to what words within the caption.
- **Grounded** representations: the model can localize *which region* of an image corresponds to *which word* in the caption. Stronger than alignment; requires either supervision on bounding boxes or attention-based methods.

CLIP-style models are *aligned* but not *grounded*. Grounded models like GLIP and the captioning side of Flamingo do extra work. Both are useful; aligned models are vastly more common because the training data is much easier to come by.

### Why this generalizes beyond text and image

The same joint-embedding idea extends to any pair of modalities with naturally-paired data:

- **Audio + text**: train on transcribed speech. The result is Whisper-style audio-text models.
- **Video + text**: train on subtitled or captioned video. The result is VideoCLIP-style models and the conditioning module inside text-to-video generators.
- **Image + image**: train on different views of the same scene. The result is self-supervised vision models like SimCLR and DINO.
- **Code + natural language**: train on code with docstrings or commit messages. The result is code-search systems.
- **Protein sequence + protein structure**: train on paired sequence-structure data. The result is the embedding spaces underlying AlphaFold and ESM.

The recipe is always the same: two encoders, a contrastive objective, a shared latent space. Topic 2 covers the objective.

### Connection to earlier modules

You have seen all the pieces. The image encoder is a CNN or transformer (architectural ideas from Module 6). The text encoder is a transformer (Module 6). The shared latent space generalizes the Module 2 latent-variable setup, except now there are two encoders feeding into the same space rather than one encoder per observation. The decoder of choice is usually a diffusion model conditioned on the multimodal embedding (Module 5).

### Where this Topic fits

Topic 2 covers the training procedure (contrastive learning, the InfoNCE loss, the CLIP architecture in detail) that makes the joint-embedding setup actually work. Topic 3 surveys the systems built on top: DALL-E, Stable Diffusion, multimodal LLMs (GPT-4V, Claude with vision, Gemini, LLaVA, Flamingo). Topic 4 is the course-wrap-up.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 13, intro sections on shared representations and contrastive learning.
- A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision” (CLIP), ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). Sections 1 and 2 (the framework).

### Consider

- The shared embedding space is just a vector space; nothing about its training inherently requires modality structure. Why does the resulting space “know” that “cat-ness” should be a consistent direction regardless of whether the query is an image or a caption?
- For your course project domain, what would the natural “paired modalities” be? If there are no obvious natural pairings, can you still get a useful joint embedding (e.g., via paired-data augmentation or weak supervision)?
- L2-normalizing embeddings projects them onto the unit sphere, and cosine similarity becomes a dot product. What information about the embedding is being thrown away by this normalization, and why might it not matter?

### Resources

- OpenAI CLIP repository: [github.com/openai/CLIP](https://github.com/openai/CLIP). Reference implementation; useful for the Module 8 Case Analysis option.
- Hugging Face transformers *CLIPModel*: [huggingface.co/docs/transformers/model_doc/clip](https://huggingface.co/docs/transformers/model_doc/clip). Drop-in API for embedding images and text with pre-trained CLIP weights.
