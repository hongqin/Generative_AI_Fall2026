# Week 8, Topic 4: Final Synthesis of Methods, Evaluation, and Responsible Use

## Topic Overview

**Plain-English summary first.** This is the course-wrap Topic. We have covered seven model families across eight weeks; the work now is to pull them together so you finish the course with a coherent map of the field, an evaluation framework that you can apply to anything new, and an honest accounting of the responsibilities that come with deploying these systems. Three parts: cross-family synthesis (one table summarizing what we have learned), evaluation across families (the metrics and their limits), and responsible use (the non-negotiable conversation about real-world deployment).

### Part 1: The seven families, in one table

This is the master comparison table that subsumes the smaller tables in Module 3 Topic 4, Module 4 Topic 4, and Module 5 Topic 3.

| Dimension | VAE (M2) | GAN (M3) | Autoregressive (M4) | Diffusion (M5) | Transformer (M6) | World Model (M7) | Multimodal (M8) |
|----|----|----|----|----|----|----|----|
| Training objective | ELBO lower bound | Minimax saddle point | Per-token log-likelihood | Noise-prediction MSE | Same as AR | Reconstruction + dynamics | Contrastive (InfoNCE) |
| Latent variable | Yes, with encoder | Implicit (noise input) | None | None (in pixel space) or VAE latent (in latent diffusion) | None | Yes (V module) | Yes (joint embedding) |
| Likelihood evaluation | Lower bound | None | Exact | ELBO; difficult in practice | Exact (as AR) | Not directly applicable | Not directly applicable |
| Sample quality | Blurry | Sharp; can drop modes | Sharp; sampling-strategy dependent | Sharp; near-SOTA on images | Depends on objective | As good as components | Depends on components |
| Training stability | Routine | Hard | Routine | Routine | Routine at scale | Hard (multi-stage) | Routine |
| Sampling cost | O(1) | O(1) | O(N) sequential | O(T) sequential, T~50-1000 | As underlying generator | Dream rollouts | As underlying generator |
| Controllability | Latent arithmetic | Conditional GAN, StyleGAN | Prompt-based | Classifier-free guidance | Prompt + fine-tune + RAG | Through controller | Modality conditioning |
| Typical use | Representation, anomaly | Image synthesis | Language, code, audio | Image/video/audio gen | LLMs, multimodal backbones | Model-based RL, agents | Retrieval, conditional gen |

#### How to use this table

The table is decision support, not a leaderboard. The right family depends on what you want to do:

- Need *exact likelihood*? Autoregressive (or normalizing flows, which we touched briefly).
- Need *controllable sharp images*? Diffusion with classifier-free guidance.
- Need a *useful latent representation*? VAE or CLIP-style contrastive.
- Need to *train an agent that plans ahead*? World model.
- Need *cross-modal retrieval*? CLIP-style multimodal.
- Need *any general-purpose intelligence* with text I/O? Autoregressive transformer (large language model).

### Part 2: Evaluating across families

Across the course you have seen several evaluation methods. The summary you should leave with:

#### Family-native metrics

- **Likelihood-based** (autoregressive, VAE, flows): perplexity (text), bits-per-dim (images), held-out NLL. Easy to compute when available; not always correlated with sample quality.
- **Sample-distribution metrics**: FID (image), Inception Score (image), CLIP score (text-to-image alignment), audio-based equivalents. Compare generated samples to real distribution.
- **Pairwise / referenced metrics**: BLEU, ROUGE, METEOR (text generation against references). Useful when there is a ground truth.
- **Precision / recall for generative models** (Kynkäänniemi et al., 2019): explicit quality-vs-diversity tradeoff.

#### Cross-family principles

- No single metric captures generative-model quality. Always combine quantitative + qualitative + downstream-task evidence.
- For practical projects, downstream-task accuracy (or the moral equivalent in your domain) usually beats abstract sample-quality metrics. Ground evaluations in actual use.
- Human evaluation remains the gold standard for open-ended generation, despite being slow and expensive.
- Be specific about your evaluation set, sampling procedure, and metric implementation. “FID” alone is ambiguous; “FID computed against 10,000 real images from the validation set using the standard InceptionV3 feature extractor” is what you want in a paper.

### Part 3: Responsible use

You are now in a position to deploy generative AI systems in the real world. Eight things to keep on your radar:

1.  **Training-data provenance.** Most large generative models are trained on web-scraped data with unclear copyright and consent status. This is the subject of ongoing litigation (NYT v OpenAI, Getty v Stability AI, Andersen v Stability AI / Midjourney / DeviantArt). If you deploy a model trained on questionable data, the legal exposure is real.
2.  **Hallucination and confident wrongness.** LLMs and multimodal models routinely produce plausible-sounding incorrect outputs. Any deployed system needs guardrails: retrieval grounding, citation requirements, output verification, human review for consequential decisions.
3.  **Bias amplification.** Models trained on internet data reproduce the biases in that data, often more strongly than the source data. Demographic skews in image generators, stereotypes in language models, accent and dialect bias in speech recognition.
4.  **Deepfake and misuse risk.** Text-to-image and text-to-video generators can produce realistic depictions of real people doing things they did not do. Voice cloning is now consumer-grade. Defenses (watermarking, provenance standards like C2PA, detection classifiers) lag the offense.
5.  **Environmental cost.** Training and inference of large generative models consumes substantial energy and water. The marginal cost per query is small; the cumulative cost across millions of users is not. Several research groups now report carbon costs of training runs as part of their methodology.
6.  **Privacy.** Generative models can memorize and emit training data verbatim, including personal information. Deployed systems with user data need explicit handling: differential privacy, user opt-out, deletion policies.
7.  **Academic integrity in this course and beyond.** AI tools are allowed in this course (per the Meet Your Instructor page); you remain responsible for the correctness, originality, and integrity of everything you submit. The same principle scales to professional work: AI-generated content needs human review and attribution.
8.  **Evaluation in your own deployment context.** Whatever you build, write down what success looks like, what failure looks like, what the worst-case failure looks like, and how you will detect each. Generic “the model is great” reports are insufficient for any real-world deployment.

None of this is intended to be discouraging. The point is that responsible use is a real, ongoing piece of the job, not a one-time checkbox.

### Where to go next

The course covered the foundations; the field moves fast. A few directions to track:

- **Agent frameworks** (LangChain, LlamaIndex, OpenAI Assistants, Anthropic’s Claude Agent SDK, AutoGPT). The natural extension of multimodal LLMs into multi-step task execution; closely related to the world-models material in Module 7.
- **Open-weights vs closed-weights ecosystems**. Llama, Mistral, Qwen, Gemma on the open side; GPT, Claude, Gemini on the closed side. Track both.
- **Scientific applications**: AlphaFold, ESM, molecular diffusion models, materials-discovery generators. The Module 7 / 8 multimodal-and-foundation-models tools are now being applied across scientific domains.
- **Mechanistic interpretability**: understanding what large generative models actually compute internally. Active research area; relevant to safety and to responsible deployment.
- **Local sources**: ODU’s Research & Cloud Computing group, the ODU CS department, and (more broadly) the AI for Science community publish current work that connects to many of the families we covered.

### Where this Topic fits

Module 8 closes the course. The Final Project Report and the Recorded Final Project Presentation pull together everything you have done across the term. The Multimodal Case Analysis or Final Reflective Discussion lets you choose between technical depth (a case analysis) and synthesis (a personal reflection on the course as a whole).

### Read

Required (skim, do not read in full):

- P. Mitchell et al., “Model Cards for Model Reporting,” FAT\* 2019. [arXiv:1810.03993](https://arxiv.org/abs/1810.03993). The proposal for standardized documentation of trained models. Pervasive in modern practice and worth knowing about.
- R. Bommasani et al., “On the Opportunities and Risks of Foundation Models,” Stanford CRFM, 2021. [arXiv:2108.07258](https://arxiv.org/abs/2108.07258). The 200-page survey that named “foundation models” as a category. Skim the executive summary and the responsible-AI sections.

### Consider

- For your course project, which family of the seven did you spend the most time with? Which one do you wish you had spent more time on?
- Of the eight responsible-use items above, which is most relevant to your project (or to a hypothetical deployment of it)? What is your plan for handling it?
- Looking back across the eight modules: which idea or technique do you expect to use most often in your career after this course?

### Resources

- C2PA (Coalition for Content Provenance and Authenticity): [c2pa.org](https://c2pa.org/). Industry standard for content provenance metadata.
- Stanford HAI policy briefs: [hai.stanford.edu/policy](https://hai.stanford.edu/policy). Ongoing analysis of policy and ethical questions around AI deployment.
- ODU Research & Cloud Computing: [wiki.hpc.odu.edu](https://wiki.hpc.odu.edu/). Local infrastructure for follow-on work.
