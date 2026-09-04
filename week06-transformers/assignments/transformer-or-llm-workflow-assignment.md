# Transformer or LLM Workflow Assignment

*Assignment, 100 points.*

## Purpose

This is the Module 6 weekly technical assignment. You will exercise either the “train a small transformer from scratch” workflow or the “use a pre-trained large model” workflow, then write a short interpretation that ties what you observed to the vocabulary from Topics 1 through 4. The two options map to the two ends of the modern transformer landscape; pick whichever is more relevant to your project direction.

### Task

**Pick one** of the two options below.

#### Option A: Train a small GPT from scratch (Foster Ch 9)

Start from Foster’s 2nd-edition GPT notebook: [notebooks/09_transformer/gpt](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/09_transformer/gpt). This is a small Keras GPT-style decoder-only transformer trained on Foster’s default text corpus.

Required experiments:

1.  Train the baseline GPT end to end. Log the training loss (next-token cross-entropy).
2.  Sample three generated passages at **three different temperatures** (recommended: $`T = 0.5, 1.0, 1.5`$). Show enough text for the temperature effect to be visible.
3.  Visualize the attention. Pick one input sequence and one or two attention heads in one layer; show the attention-weight heatmap (this should be supported by the notebook or trivial to add). Describe what the head appears to attend to.
4.  Brief reflection (1 paragraph): how did the transformer differ qualitatively from the LSTM you trained (or could have trained) in Module 4?

#### Option B: Pre-trained LLM workflow (Hugging Face)

Use the Hugging Face `transformers` library to load a small pre-trained model (recommended: `distilgpt2`, `gpt2`, or `microsoft/Phi-3-mini-4k-instruct` if you have enough memory) and exercise the practical workflow.

Required experiments:

1.  Run inference with the pre-trained model on at least three prompts of your choice. Use **three different decoding strategies**: greedy, temperature sampling (e.g., $`T = 0.8`$), and nucleus (top-p) sampling. Show all nine outputs (3 prompts × 3 strategies).
2.  Demonstrate prompt sensitivity: write two prompts that are very similar in wording but lead to qualitatively different outputs. Briefly explain why.
3.  Either: (a) fine-tune the model on a small custom text dataset using either standard fine-tuning or LoRA via Hugging Face `peft`; show the loss curve and a few before/after generation samples. OR (b) Build a minimal retrieval-augmented prompting setup: embed 5 to 10 short documents, retrieve the top-2 by query similarity, prepend to the prompt; show how the answers change vs without retrieval.
4.  Brief reflection (1 paragraph): which of the four adaptation strategies from Topic 4 (prompting, RAG, LoRA, full fine-tuning) is the natural next step for your project?

#### Deliverables (any option)

- Your modified notebook (`.ipynb`) with *all cells run* and figures rendered, **or** a link to a public GitHub repository whose README explains how to reproduce your run.
- A **1- to 2-page PDF write-up** covering: which option you chose, what behaviors you observed, the named phenomena from Topics 1-4 that showed up, and your brief reflection.

#### Citation

For Option A: cite Foster as *David Foster, Generative Deep Learning (2nd ed.), O’Reilly, 2023, Chapter 9; code from github.com/davidADSP/Generative_Deep_Learning_2nd_Edition.* For Option B: cite the specific pre-trained model card (URL on Hugging Face Hub) and the relevant primary paper for the model architecture.

#### Compute

Option A trains in 1 to 3 hours on a CPU laptop, faster on a GPU. Option B runs in minutes on CPU for inference; fine-tuning a small model with LoRA takes 30 to 90 minutes on CPU or 5 to 15 minutes on a GPU. Wahab via Open OnDemand JupyterLab (see Module 1 Topic 4) is recommended for both options.

### Evaluation Criteria (100 points)

- **30 pts**: Implementation runs correctly; required artifacts produced (loss curves, generation samples, attention visualization OR fine-tuning curves, etc.).
- **25 pts**: Decoding-strategy / temperature variation covered with comparable evidence across settings.
- **25 pts**: Behavior diagnosis with specific evidence (attention-head interpretation, prompt-sensitivity examples, fine-tuning effect, RAG impact); applies the vocabulary from Topics 1-4.
- **10 pts**: Reflection: ties to Module 4 LSTM (Option A) or to project planning (Option B).
- **10 pts**: Code quality and reproducibility: random seed logged, environment notes, runs end-to-end.
