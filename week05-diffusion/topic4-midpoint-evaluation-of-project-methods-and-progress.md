# Week 5, Topic 4: Midpoint Evaluation of Project Methods and Progress

## Topic Overview

**Plain-English summary first.** This Topic does double duty. The first half is about generative-model evaluation in general: the mQetrics people use (FID, IS, perplexity, sample diversity), what they actually tell you, and where they mislead. The second half is about the **Mid-Project Report**, the formal milestone due at the end of this module. The report is the first major project deliverable since the Proposal in Module 2; it asks you to honestly assess what you have built so far, present concrete evaluation results, identify gaps, and revise your plan for the second half of the term.

### Part 1: Evaluating generative models

This is genuinely hard. Generative models do not have an obvious single-number quality metric (unlike classifiers, which have accuracy). The state of the art is to combine three kinds of evidence: quantitative metrics, qualitative inspection, and downstream-task evaluation.

#### Quantitative metrics

- **Fréchet Inception Distance (FID)** (Module 3 Topic 4 in detail). The standard for image generators. Pass real and generated samples through a pretrained Inception network, fit Gaussians to the intermediate activations, compute the Fréchet distance. Lower is better. Reasonable on natural images; less reliable on out-of-distribution data (medical, scientific, synthetic).
- **Inception Score (IS)**. Older, less reliable than FID. Still reported for legacy comparisons.
- **Perplexity**. For autoregressive models (Module 4 Topic 3). Exponentiated negative log-likelihood per token. Lower is better. Measures how well the model predicts test data, not how good its samples look to a human.
- **CLIP score**. For text-to-image: measure cosine similarity between the CLIP embedding of the prompt and the CLIP embedding of the generated image. Higher is better. The standard alignment metric for text-conditional generation.
- **Bits per dimension**. The likelihood-based metric for image generative models with exact likelihood (autoregressive, normalizing flows, some diffusion variants). $`-\log p(x) / (\text{num dims} \cdot \log 2)`$. Lower is better.
- **Reconstruction error**. For VAEs and autoencoders. Useful for representation quality, less useful for generative quality.

#### Sample diversity and mode coverage

A model can score well on FID and still drop modes (the GAN mode-collapse problem from Module 3). Diversity checks:

- **Visual inspection of large sample grids**. Generate 100+ samples; look for repetition, missing classes, suspicious uniformity. Cheap and surprisingly effective.
- **Precision and recall metrics for generative models** (Kynkäänniemi et al., 2019). Precision = quality (do generated samples look like real data); Recall = diversity (do generated samples cover the data distribution). Standard for analyzing GAN-style failure modes.
- **Per-class FID**. If your data has labels, compute FID per class. Imbalanced FID across classes flags mode coverage problems.

#### Downstream-task evaluation

Often the best evaluation: use the generative model for a real task and measure end-to-end performance. Examples:

- Synthetic data augmentation: train a classifier on real + generated data, measure classifier accuracy.
- Imputation: use a generative model to fill in missing data; measure downstream utility.
- Anomaly detection: use likelihood (or reconstruction error) as an anomaly score; measure AUC against labeled anomalies.

Downstream-task metrics ground the abstract “is this a good generative model” question in a concrete decision-relevant outcome.

#### Human evaluation

Still the gold standard for open-ended generation. Side-by-side comparisons, MOS (mean opinion score), pairwise preference judgments. Expensive, slow, and labor-intensive; but unavoidable for honest evaluation of text or image quality where quantitative metrics are known to misjudge.

### Part 2: The Mid-Project Report

The Mid-Project Report is your first major formal project milestone (100 pts) since the Proposal in Module 2. It is graded heavily because it is the gate: after this point, the second half of the term should execute the refined plan you commit to here. Honest, specific, evidence-backed reports earn full credit. Generic “going well” reports do not.

#### Required structure

Write an **800 to 1500 word PDF** (roughly 3 to 5 pages including figures) organized into the following six sections in this order:

1.  **Title and team.** Recap from the Proposal. If the team or title has changed since Module 2, note it here.
2.  **What you have built.** Concrete: data loaded, models trained (which checkpoint, what hyperparameters), evaluation metrics computed (which ones, on which split). Be specific enough that a classmate could reproduce your runs.
3.  **Results so far.** Quantitative metrics (from the list in Part 1 above; pick the ones appropriate for your model family and data) AND qualitative evidence (sample grids, learned representations, downstream-task plots). Include 2 to 4 figures or tables.
4.  **Gap analysis.** What is NOT yet done? What is harder than you expected? Which Proposal assumptions turned out to be wrong? Honest gap analyses get full credit; generic “running into issues” statements do not.
5.  **Revised plan for Modules 6, 7, and 8.** Explicit milestones for each remaining module. The Module 7 Draft Report and the Module 8 Final Report are fixed; what will be in them?
6.  **Updated risks and mitigations.** Carry forward the risks from your Proposal; update them based on what you have learned. Add 1 to 2 new risks discovered during implementation. Each risk gets a one-sentence mitigation.

#### Tips for a strong Mid-Project Report

- **Show your work.** A figure beats a sentence about a figure. A table beats a paragraph claiming “results were promising.”
- **Be specific about what is broken.** “The encoder loss plateaus around epoch 12 and the latent space scatter shows two clusters instead of the expected ten” is a great gap-analysis sentence; “the model needs more tuning” is not.
- **If you have pivoted, say so.** Project pivots are normal and not penalized. Surprises that get hidden until the Final Report are penalized.
- **Match the evaluation metric to your model family and data.** Use FID for image GANs/diffusion; perplexity for autoregressive language models; downstream-task accuracy for representation-learning projects. Picking the wrong metric loses credibility.

#### Engagement

Reply to **at least one classmate’s Mid-Project Report** with a substantive comment: a methodological suggestion, a related paper, an evaluation idea, or a question about their gap analysis.

### Where this Topic fits

The Mid-Project Report is the gate between exploration (Modules 1-5) and execution (Modules 6-8). Module 6 introduces transformers, the architectural backbone of most modern generative systems. Module 7 carries the Paper Presentation milestone and the Draft Report. Module 8 wraps up with the Final Project Report and Recorded Presentation.

### Read

Required:

- L. Theis, A. van den Oord, M. Bethge, “A Note on the Evaluation of Generative Models,” ICLR 2016. [arXiv:1511.01844](https://arxiv.org/abs/1511.01844). The classic argument that no single metric captures “generative model quality.” Sections 1 and 5 are the key reading.

Recommended:

- T. Kynkäänniemi et al., “Improved Precision and Recall Metric for Assessing Generative Models,” NeurIPS 2019. [arXiv:1904.06991](https://arxiv.org/abs/1904.06991). The precision/recall framework for diagnosing quality vs diversity tradeoffs.

### Consider

- For your specific project data, which quantitative metric makes most sense, and what is its biggest blind spot? Plan to address that blind spot in the Mid-Project Report.
- What single “negative result” or surprising finding from your implementation could you highlight in the gap analysis? Surprises usually carry more information than confirmations.
- Looking forward to Modules 6, 7, 8: what is the riskiest thing remaining in your plan, and what is the earliest milestone at which you could confirm or rule out that risk?

### Resources

- A. Borji, “Pros and Cons of GAN Evaluation Measures: New Developments,” CVIU 2022. [arXiv:2103.09396](https://arxiv.org/abs/2103.09396). Modern survey of generative-model evaluation.
- P. Sajjadi et al., “Assessing Generative Models via Precision and Recall,” NeurIPS 2018. [arXiv:1806.00035](https://arxiv.org/abs/1806.00035). Earlier precision/recall framework; companion to Kynkäänniemi et al.
