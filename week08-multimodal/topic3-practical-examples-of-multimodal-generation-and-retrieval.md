# Week 8, Topic 3: Practical Examples of Multimodal Generation and Retrieval

## Topic Overview

**Plain-English summary first.** Once you have a joint embedding space across modalities, you can build remarkable systems on top: text-to-image generators that take a prompt and produce a matching image, multimodal language models that can describe an image or answer questions about a video, retrieval systems that search massive video corpora by description, and audio-text models that transcribe speech or generate it from text. This Topic surveys the landscape, showing how the CLIP-style joint embedding (Topic 2), the diffusion decoder (Module 5), and the transformer language model (Module 6) combine to produce the modern multimodal-foundation-model toolkit.

### Text-to-image generation

#### DALL-E (2021): autoregressive text + image tokens

The original DALL-E (Ramesh et al., 2021) combined a discrete image tokenizer (a VQ-VAE) with an autoregressive transformer (the Module 4 / Module 6 family) that generated both text and image tokens in a single sequence. Conceptually, it was an autoregressive language model over a mixed text-and-image vocabulary. Cool, but expensive at inference time (autoregressive sampling is $`O(N)`$).

#### DALL-E 2 / unCLIP (2022): CLIP latents + diffusion decoder

DALL-E 2 took a different approach. Given a text prompt, generate the corresponding image in three steps:

1.  Embed the text with a CLIP text encoder to get $`z_{\text{text}}`$.
2.  Generate a CLIP image embedding $`z_{\text{img}}`$ conditioned on $`z_{\text{text}}`$, using a separate “prior” diffusion or autoregressive model.
3.  Decode $`z_{\text{img}}`$ into pixel space using a diffusion-based decoder.

The architecture leverages CLIP’s joint embedding twice: once as the text encoder, and once as the target representation that the decoder is trained to invert. This is the “unCLIP” name: it inverts the CLIP embedding back into pixels.

#### Stable Diffusion (2022): latent diffusion + CLIP text conditioning

Covered in Module 5 Topic 3, but worth restating here. Stable Diffusion (Rombach et al., 2022) runs the diffusion process in a VAE’s latent space rather than pixel space, conditioned on a CLIP text encoder’s output. This is what makes Stable Diffusion fit on consumer hardware. The multimodal piece is exactly the CLIP text encoder: it converts a prompt into the conditioning signal that drives the diffusion U-Net.

#### Imagen, Midjourney, Flux, and the modern landscape

The text-to-image space has continued to evolve rapidly. Imagen (Google, 2022) replaced CLIP with a frozen T5 text encoder, demonstrating that large language models work as well or better than CLIP for text conditioning. Midjourney and Flux (Black Forest Labs) are commercially-trained diffusion models with proprietary architectures. The unifying pattern remains: **text encoder for conditioning + diffusion decoder for pixels**.

### Multimodal LLMs (vision-language models)

The other major direction. Take a strong pre-trained large language model and add the ability to consume images as input. The architectural pattern shared by GPT-4V, Claude with vision, Gemini, LLaVA, and Flamingo:

1.  A **vision encoder** (typically a CLIP-style ViT) encodes the input image into a sequence of patch embeddings.
2.  A **projection layer** (one or a few linear layers, or sometimes a small cross-attention module) maps the vision embeddings into the LLM’s token-embedding space.
3.  The **LLM** processes the projected vision tokens just like any other token sequence, producing a text response.

#### Examples

- **LLaVA** (Liu et al., 2023): the canonical accessible open multimodal LLM. CLIP visual encoder + a small MLP projection + Vicuna or Llama LLM. Trained on synthetic image-instruction-response data generated with GPT-4. Reproducible in academic settings.
- **Flamingo** (Alayrac et al., 2022): introduced “Perceiver Resampler” cross-attention between vision and LLM. Sophisticated; less commonly reproduced than LLaVA.
- **GPT-4V, Claude 3.5/4 with vision, Gemini**: closed-source production systems. The architectural details are not public but the broad pattern is the same.

From the user’s perspective, these systems are simply “chat models that can see.” You paste an image and ask a question; the model responds in text. The implementation is exactly the vision-encoder + projection + LLM pattern above.

### Audio-text and other modalities

- **Whisper** (Radford et al., 2022): an encoder-decoder transformer (Module 6, encoder-decoder variant) trained on 680k hours of multilingual audio + transcripts. State-of-the-art automatic speech recognition; runs even on a laptop.
- **Text-to-speech**: VITS, Tortoise, ElevenLabs systems combine text encoders with audio generators (often diffusion-based).
- **Music generation**: MusicLM (Google, 2023), MusicGen (Meta, 2023), Suno; combine text encoders with autoregressive or diffusion audio models.
- **Video generation**: Sora (OpenAI, 2024), Veo (Google), Movie Gen (Meta), Hunyuan Video (Tencent). Text-conditioned diffusion models over video latents.

### Retrieval-augmented multimodal applications

The most under-discussed practical use of multimodal embeddings is retrieval-augmented systems: build a vector database of embedded items, retrieve by text query, then process the retrieved items with a multimodal LLM. Applications:

- “Find me all the photos of beaches in my Google Photos library” (text-to-image retrieval).
- Visual search inside e-commerce catalogs (image-to-image retrieval).
- Document understanding across PDFs, screenshots, and natural text (multimodal document retrieval).
- Video archive search by natural-language description (text-to-video retrieval, then LLM summarization of the retrieved clips).

These are not flashy systems but they are where most of the real-world business value of multimodal AI currently sits.

### Where this Topic fits

Topic 4 is the three-part course-wrap-up: cross-family synthesis, evaluation, and responsible use. The Module 8 Final Project Report and Recorded Final Project Presentation pull everything from the term together.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 13, sections on text-to-image generation and multimodal large models.
- A. Ramesh et al., “Hierarchical Text-Conditional Image Generation with CLIP Latents” (DALL-E 2 / unCLIP), 2022. [arXiv:2204.06125](https://arxiv.org/abs/2204.06125). Sections 1 to 3.

Recommended (pick ONE of the multimodal LLM papers based on what is closest to your project):

- H. Liu et al., “Visual Instruction Tuning” (LLaVA), NeurIPS 2023. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485). The most accessible open multimodal LLM.
- J.-B. Alayrac et al., “Flamingo: a Visual Language Model for Few-Shot Learning,” NeurIPS 2022. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198).

### Consider

- DALL-E 2 generates a CLIP image embedding first, then decodes it. Imagen skips this step and conditions directly on a T5 text encoder. What does this comparison tell you about whether the joint CLIP space is intrinsically necessary, or just one workable option?
- For multimodal LLMs, the vision encoder is usually a frozen pre-trained CLIP-style ViT. Why freeze it? When would you NOT freeze it?
- Many of the most-deployed multimodal systems in industry are retrieval-augmented applications (visual search, document understanding, photo libraries). Why does the popular discourse focus so much on text-to-image generation rather than these less flashy use cases?

### Resources

- A. Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision” (Whisper), 2022. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356).
- OpenAI Sora technical report: [openai.com/research/video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators). The current frontier for text-to-video.
- LLaVA implementation: [github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA). The most accessible end-to-end multimodal LLM you can actually run yourself.
