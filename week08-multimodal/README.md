# Week 8: Multimodal Models

## Contents

- Topic 1: [Shared Representations Across Modalities](topic1-shared-representations-across-modalities.md)
- Topic 2: [Contrastive Learning and Text-Image Alignment](topic2-contrastive-learning-and-text-image-alignment.md)
- Topic 3: [Practical Examples of Multimodal Generation and Retrieval](topic3-practical-examples-of-multimodal-generation-and-retrieval.md)
- Topic 4: [Final Synthesis of Methods, Evaluation, and Responsible Use](topic4-final-synthesis-of-methods-evaluation-and-responsible-use.md)
- Lecture notes (LaTeX): [`notes/week08-notes.tex`](notes/week08-notes.tex), compiled [`notes/week08-notes.pdf`](notes/week08-notes.pdf)
- Assignment: [Multimodal Case Analysis or Final Reflective Discussion](assignments/multimodal-case-analysis-or-final-reflective-discussion.md), 100 pts
- Graded discussion: [Final Project Report](assignments/final-project-report.md), 100 pts
- Graded discussion: [Recorded Final Project Presentation](assignments/recorded-final-project-presentation.md), 100 pts

## About

This final module covers multimodal models, systems that learn joint representations across modalities (text + image, text + audio, text + video, action + observation) and the foundation models built on top of those joint representations. CLIP, DALL-E, Flamingo, GPT-4V, Claude with vision, Gemini, Whisper, and Sora are the canonical examples. Module 8 also concludes the course project: the Final Project Report and the Recorded Final Project Presentation are both due, in addition to the weekly Multimodal Case Analysis or Final Reflective Discussion. This is the heaviest module of the term in terms of deliverable load; **start the Final Report and the Recorded Presentation early in the week.**

### Module Learning Objectives

After completing this module, you will be able to:

- **MLO1:** Explain the basic ideas behind multimodal representation learning and text-image or vision-language modeling. (Aligned with CLO1)
- **MLO2:** Analyze how multimodal generative systems extend concepts from earlier modules such as latent spaces, sequence modeling, and transformers. (Aligned with CLO2)
- **MLO3:** Present a complete course project with appropriate technical framing, evidence, and reflection on limitations and future directions. (Aligned with CLO3)

### Required Readings & Resources

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 13, “Multimodal Models.”** Note: like Chapter 12, the 2nd-edition GitHub repository does not include companion code for Chapter 13; use the book and the primary papers below.
- A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision” (CLIP), ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). The seminal modern multimodal paper; read sections 1 to 3.
- A. Ramesh et al., “Zero-Shot Text-to-Image Generation” (DALL-E), 2021. [arXiv:2102.12092](https://arxiv.org/abs/2102.12092).
- A. Ramesh et al., “Hierarchical Text-Conditional Image Generation with CLIP Latents” (DALL-E 2), 2022. [arXiv:2204.06125](https://arxiv.org/abs/2204.06125).
- OpenAI CLIP code: [github.com/openai/CLIP](https://github.com/openai/CLIP). The reference implementation, useful for the Module 8 Case Analysis option.

### Module To-Do List

1.  Read Topic 1 (Shared Representations Across Modalities).
2.  Read Topic 2 (Contrastive Learning and Text-Image Alignment).
3.  Read Topic 3 (Practical Examples of Multimodal Generation and Retrieval).
4.  Read Topic 4 (Final Synthesis of Methods, Evaluation, and Responsible Use). This is the course-wrap topic; do not skip it.
5.  Complete the **Multimodal Case Analysis or Final Reflective Discussion** (pick one of two options).
6.  Submit the **Final Project Report** as a reply to the Module 8 project discussion (1500-3000 word PDF, full structure).
7.  Submit the **Recorded Final Project Presentation** as a reply to its discussion (12-15 minute video about your own project). Comment substantively on at least one classmate’s presentation.

### Estimated Completion Time

Plan for roughly **18 to 22 hours** in this module (the heaviest of the term):

- Topic pages 1-4: ~3 hours.
- Multimodal Case Analysis or Final Reflective Discussion: ~3 to 4 hours.
- Final Project Report (writing + figures): ~5 to 7 hours.
- Recorded Final Project Presentation (slides + recording): ~4 to 5 hours.
- Peer engagement (replies on Final Report and Final Presentation discussions): ~1 hour.


## Module Summary

This module covered the multimodal turn in generative AI: joint embedding spaces across modalities, contrastive learning, and the foundation models built on top (CLIP, DALL-E, Stable Diffusion, multimodal LLMs, Whisper, Sora). The course-wrap Topic synthesized the seven families covered across Modules 2 through 8, the cross-family evaluation framework, and the responsible-use expectations that come with deploying these systems in the real world.

## What You Should Have Completed in Module 8

Before the term ends, use this self-check:

- Read Topics 1 through 4 on shared representations, contrastive learning, practical multimodal generation and retrieval, and the course-wrap synthesis of methods, evaluation, and responsible use.
- Completed the Multimodal Case Analysis or Final Reflective Discussion (one of two options; 100 pts).
- Submitted the **Final Project Report** as a reply to the Module 8 project discussion (1500-3000 word PDF, full structure; 100 pts formal milestone).
- Submitted the **Recorded Final Project Presentation** as a reply to its graded discussion (12-15 minute video about your own project; 100 pts).
- Commented substantively on at least one classmate’s Final Project Report and at least one classmate’s Recorded Final Project Presentation.

## Course Complete

Across eight weeks, you have built or analyzed implementations across seven distinct generative-model families, written four formal project deliverables, recorded two video presentations, and (for many of you) deployed a working prototype of a real-world generative-AI application. That is a substantial body of work. The model families we covered are the foundation under almost every modern generative-AI system; the engineering practices we exercised (suggest before pushing, build small first then scale, evaluate quantitatively and qualitatively, write down what success looks like) generalize well beyond this course.

If you continue with generative AI after this course, three areas to track in 2026 and beyond:

- **Agentic AI**: large models combined with tools, planning, and multi-step task execution. The natural extension of the multimodal foundation-model material in this module.
- **Scientific applications**: protein design, molecular generation, materials discovery, hypothesis-driven model-building. The same techniques from this course applied to scientific data.
- **Open-weights ecosystem**: Llama, Qwen, Mistral, Gemma, and successors. The ability to run state-of-the-art generative models on your own hardware (or on Wahab) keeps expanding.

Thank you for taking the course. Best of luck with what you build next.
