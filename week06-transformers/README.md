# Week 6: Transformers

## Contents

- Topic 1: [Self-Attention and Sequence Representation](topic1-self-attention-and-sequence-representation.md)
- Topic 2: [Tokenization, Embeddings, and Context Windows](topic2-tokenization-embeddings-and-context-windows.md)
- Topic 3: [Transformer Layers, Training Intuition, and Inference Behavior](topic3-transformer-layers-training-intuition-and-inference-behavior.md)
- Topic 4: [Practical Use of Large-Model Workflows in Applied Settings](topic4-practical-use-of-large-model-workflows-in-applied-settings.md)
- Lecture notes (LaTeX): [`notes/week06-notes.tex`](notes/week06-notes.tex), compiled [`notes/week06-notes.pdf`](notes/week06-notes.pdf)
- Assignment: [Transformer or LLM Workflow Assignment](assignments/transformer-or-llm-workflow-assignment.md), 100 pts
- Graded discussion: [Project Methods Refinement Note](assignments/project-methods-refinement-note.md), 25 pts

## About

This module covers the transformer, the architectural backbone of nearly every modern large generative model. Introduced by Vaswani et al. in 2017 for machine translation, the transformer dropped recurrence in favor of self-attention and made it possible to train much deeper, much wider sequence models on much longer contexts than RNNs ever could. Today the transformer underlies large language models (GPT, Claude, Llama), text-to-image diffusion backbones (DiT), multimodal systems (CLIP), and protein-structure prediction (AlphaFold). This module focuses both on the mathematics of self-attention and on the practical workflows of using pre-trained large models in applied settings.

### Module Learning Objectives

After completing this module, you will be able to:

- **MLO1:** Explain self-attention, transformer blocks, and the role of tokenization and embeddings in transformer-based generation. (Aligned with CLO1)
- **MLO2:** Use a transformer-focused notebook, worksheet, or cloud workflow to study model behavior and outputs. (Aligned with CLO2)
- **MLO3:** Evaluate whether transformer or LLM-style methods are appropriate for the current project scope. (Aligned with CLO3)

### Required Readings & Resources

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 9, “Transformers.”** Companion code at [github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition); the Chapter 9 GPT-from-scratch notebook is at [notebooks/09_transformer/gpt/](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/tree/main/notebooks/09_transformer/gpt).
- A. Vaswani et al., “Attention Is All You Need,” NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) (the original transformer paper; read sections 1 to 3).
- A. Radford et al., “Language Models are Unsupervised Multitask Learners” (GPT-2), 2019. [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). The decoder-only autoregressive architecture that became the modern LLM default.
- S. Rush, “The Annotated Transformer,” Harvard NLP blog, 2018 / updated 2022. [nlp.seas.harvard.edu/annotated-transformer](http://nlp.seas.harvard.edu/annotated-transformer/). Line-by-line annotated PyTorch implementation of the original paper; the canonical accessible reading-the-code companion.

### Module To-Do List

1.  Read Topic 1 (Self-Attention and Sequence Representation) and skim Foster Ch 9 intro through “Self-Attention.”
2.  Read Topic 2 (Tokenization, Embeddings, and Context Windows).
3.  Read Topic 3 (Transformer Layers, Training Intuition, and Inference Behavior).
4.  Read Topic 4 (Practical Use of Large-Model Workflows in Applied Settings).
5.  Complete the **Transformer or LLM Workflow Assignment**. Pick one of two options (Foster GPT from scratch, or a Hugging Face pre-trained workflow) and write a 1- to 2-page interpretation.
6.  Post your **Project Methods Refinement Note** as a reply to the Module 6 project discussion (around 300 to 500 words). Reply to at least one classmate.

### Estimated Completion Time

Plan for roughly **12 to 15 hours** in this module, distributed approximately as:

- Topic pages 1-4 reading and side-reading: ~4 hours.
- Transformer / LLM Workflow Assignment (notebook + write-up): ~5 to 6 hours.
- Project Methods Refinement Note (post + one reply): ~1 to 2 hours.


## Module Summary

This module covered the architectural backbone of modern large generative models. The transformer is the foundation under nearly every system you have heard of, from large language models to multimodal generators to protein-structure prediction. The next module turns to world models, a recent direction that combines generative modeling with planning and reasoning, and brings the Recorded Paper Presentation milestone you set up in Module 1.

## What You Should Have Completed in Module 6

Before moving on, use this self-check to make sure nothing in Module 6 slipped through:

- Read Topics 1 through 4 on self-attention, tokenization and positional encoding, the transformer block and three architectural variants, and practical large-model workflows.
- Picked one of two Mini-Lab options (Foster GPT from scratch, or a Hugging Face pre-trained workflow) and run it end to end.
- Submitted the Transformer / LLM Workflow Assignment (notebook plus 1- to 2-page interpretation).
- Posted the Project Methods Refinement Note as a reply to the Module 6 project discussion (~300 to 500 words).
- Replied to at least one classmate’s methods refinement note.

## Next Steps

Module 7 covers **World Models**, a generative-modeling direction that combines latent-variable models (Module 2) with recurrent prediction (Module 4) to learn compact predictive models of environments. World models are how a generative AI can be used to train an agent inside its own “imagined” rollouts, which is the bridge between generative modeling and reinforcement learning. Module 7 also brings the **Recorded Paper Presentation** milestone (10% of the course grade), where you present the primary research paper you proposed in the Module 1 Setup Discussion. Finally, the **Draft Report** formal milestone (50 pts) is due in Module 7 as well.

In Module 7 you will:

- Read Topics 1 through 4 on predictive world models, planning and imagination, the connection to current research directions, and technical communication.
- Complete the World-Model Analysis assignment and your **Recorded Paper Presentation** (12 to 15 minutes).
- Submit the **Draft Report** as a reply to the Module 7 project discussion.

Module 7 is heavy: a regular assignment, a major recorded presentation, and a formal project milestone. Start the paper presentation early in the week.
