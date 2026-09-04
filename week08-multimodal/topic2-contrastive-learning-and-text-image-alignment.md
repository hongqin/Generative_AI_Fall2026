# Week 8, Topic 2: Contrastive Learning and Text-Image Alignment

## Topic Overview

**Plain-English summary first.** Training a joint embedding space requires telling the encoders “this image and this caption belong together; that image and that other caption do not.” The brilliant trick of contrastive learning is that you do not need explicit negative labels: within a batch, every pair other than the matched one is treated as a negative. Combined with massive scale (CLIP trained on 400 million image-text pairs scraped from the web), this produces a remarkable joint embedding that supports zero-shot classification. This Topic walks through the contrastive loss, the CLIP architecture, the zero-shot trick, and the modern variants.

### The contrastive idea

Take a batch of $`N`$ image-caption pairs. Embed all $`N`$ images and all $`N`$ captions, then compute the $`N \times N`$ matrix of cosine similarities between every image and every caption in the batch. The training objective is: *make the diagonal entries large and the off-diagonal entries small.* The diagonal corresponds to the matched pairs (the image in row $`i`$ goes with the caption in column $`i`$); the off-diagonals correspond to mismatched (negative) pairs.

This is what makes contrastive learning data-efficient: every batch of $`N`$ matched pairs gives you $`N(N-1)`$ *free* negative pairs without any extra labeling. Scale the batch to thousands of examples and you have millions of training signals per step.

### The InfoNCE loss

The standard contrastive loss is symmetric across image-to-text and text-to-image lookups. For image $`i`$, the loss pushes its caption $`y_i`$ to be the most-similar caption in the batch:

``` math
\mathcal{L}_{\text{i2t}} \;=\; -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\!\left(\mathrm{sim}(x_i, y_i) / \tau\right)}{\sum_{j=1}^{N} \exp\!\left(\mathrm{sim}(x_i, y_j) / \tau\right)}.
```

where $`\mathrm{sim}(x, y) = f_{\text{img}}(x) \cdot f_{\text{txt}}(y)`$ (or the cosine version of the same, if embeddings are L2-normalized), and $`\tau`$ is a learnable temperature parameter (CLIP initializes it to ~0.07). The text-to-image loss $`\mathcal{L}_{\text{t2i}}`$ is symmetric: swap rows and columns of the similarity matrix. The final loss is

``` math
\mathcal{L}_{\text{CLIP}} \;=\; \tfrac{1}{2}\!\left( \mathcal{L}_{\text{i2t}} \,+\, \mathcal{L}_{\text{t2i}} \right).
```

#### What this loss is doing, in plain language

- Each row of the similarity matrix is treated as a classification problem: “which of the $`N`$ captions in the batch matches this image?” The correct answer is the diagonal entry.
- The softmax + log turns this into a multi-class cross-entropy. The model is rewarded for putting probability mass on the correct caption.
- Symmetric application of the same logic to columns ensures the embedding is good both directions: images find their captions, and captions find their images.

### CLIP in detail

CLIP (Contrastive Language-Image Pretraining; Radford et al., 2021) is the canonical implementation:

- **Image encoder**: either a ResNet-50 (or larger) or a Vision Transformer (ViT-B/32 or ViT-L/14). The ViT variants ended up being the standard.
- **Text encoder**: a small transformer (12 layers, 512 dim).
- **Both encoders project to a shared 512-dim space** (via a final linear layer); embeddings are L2-normalized before similarity is computed.
- **Training data**: 400 million image-text pairs scraped from the web (the “WIT” dataset). Notably, this data is mostly noisy and unstructured; the diversity is what drives CLIP’s generality.
- **Batch size**: huge (32,768 in the original paper). The contrastive loss benefits from many negatives, so big batches help a lot. This is one reason CLIP-style training is hardware-intensive.

### The zero-shot classification trick

The result that made CLIP famous: a pre-trained CLIP can classify images into *arbitrary* categories without any task-specific training, just by phrasing the categories as text:

1.  Write a list of candidate class names, e.g., “a photo of a dog,” “a photo of a cat,” “a photo of a car.”
2.  Embed each candidate with the text encoder.
3.  Embed the image with the image encoder.
4.  Predict the class whose text embedding has highest cosine similarity to the image embedding.

This works because the joint embedding space encodes semantic similarity, not task-specific features. Adding “a photo of a” as a prompt template (“prompt engineering”) improves accuracy because it matches the training distribution of captions. CLIP zero-shot accuracy on standard benchmarks like ImageNet rivals supervised models from a few years earlier. This was a genuine surprise in 2021 and is still impressive.

### Modern variants and successors

- **ALIGN** (Jia et al., 2021): same idea, even bigger and noisier dataset (1.8B pairs). Demonstrated that scale matters more than data cleaning.
- **SigLIP** (Zhai et al., 2023): replaces the softmax-cross-entropy with a sigmoid binary classification for each pair. Removes the need for a large batch; works better at smaller scales.
- **OpenCLIP**: open-source reproductions of CLIP at various scales, available on Hugging Face. Usually the practical choice when you want CLIP-style embeddings.
- **EVA-CLIP, SigLIP 2**: ongoing improvements to scaling and training data curation.

### Where this Topic fits

Topic 3 surveys the multimodal generative systems built on top of CLIP-style joint embeddings: DALL-E (uses CLIP as a text-conditioning module), Stable Diffusion (uses a CLIP text encoder), and the multimodal LLMs that bolt vision encoders onto language models. Topic 4 is the course-wrap-up.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 13, sections on contrastive learning and CLIP.
- A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision” (CLIP), ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). Sections 1 to 3 cover the loss and architecture; Section 3.1 is the zero-shot procedure.

Recommended:

- A. van den Oord, Y. Li, O. Vinyals, “Representation Learning with Contrastive Predictive Coding” (introduces InfoNCE), 2018. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748). The earlier paper that established the InfoNCE loss CLIP uses.

### Consider

- The contrastive loss treats every image other than the matched one as a negative. Some of those off-diagonal pairs might actually be semantically similar (two different photos of a dog, say). What is the right response to this issue?
- CLIP’s zero-shot classification accuracy on ImageNet rivals supervised models from a few years earlier. What does that comparison tell you about supervised training in general?
- The temperature parameter $`\tau`$ is learnable. What happens, intuitively, when $`\tau`$ is very small? What about very large?

### Resources

- X. Zhai et al., “Sigmoid Loss for Language Image Pre-training” (SigLIP), 2023. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343). The successor objective that removed the need for huge batch sizes.
- OpenCLIP repository: [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip). Open-source CLIP reproductions and pre-trained weights.
- T. Chen et al., “A Simple Framework for Contrastive Learning of Visual Representations” (SimCLR), ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709). The image-image contrastive approach; useful background to the CLIP idea.
