# Week 7: World Models

## Contents

- Topic 1: [Predictive Modeling of Latent States or Environments](topic1-predictive-modeling-of-latent-states-or-environments.md)
- Topic 2: [Planning, Imagination, and Structured Reasoning](topic2-planning-imagination-and-structured-reasoning.md)
- Topic 3: [Linking World Models to Current Research Directions](topic3-linking-world-models-to-current-research-directions.md)
- Topic 4: [Technical Communication Through Paper Presentation](topic4-technical-communication-through-paper-presentation.md)
- Lecture notes (LaTeX): [`notes/week07-notes.tex`](notes/week07-notes.tex), compiled [`notes/week07-notes.pdf`](notes/week07-notes.pdf)
- Assignment: [Recorded Research Paper Presentation](assignments/recorded-research-paper-presentation.md), 100 pts
- Graded discussion: [Project Draft Report or Progress Report](assignments/project-draft-report-or-progress-report.md), 50 pts

## About

This module introduces world models, a generative-modeling direction that combines a learned latent representation (Module 2) and a recurrent next-step predictor (Module 4) into a compact predictive model of an environment. An agent can then plan or be trained inside the world model’s “imagined” rollouts rather than the real world, which is the bridge between generative modeling and reinforcement learning. Module 7 is also the heaviest non-final module: it carries the **Recorded Research Paper Presentation** (the paper you proposed in the Module 1 Setup Discussion, 100 points in the Paper Presentation group) and the **Draft Report** formal milestone (50 points). Start early.

### Module Learning Objectives

After completing this module, you will be able to:

- **MLO1:** Explain the main intuition behind world models and their relationship to prediction, planning, and reasoning. (Aligned with CLO1)
- **MLO2:** Analyze a world-model example, paper, or lecture and connect it to broader generative AI trends. (Aligned with CLO2)
- **MLO3:** Present and critique a primary research paper in a clear and technically grounded way while refining the course project draft. (Aligned with CLO3)

David Ha’s NeurIPS 2018 presentation of the original World Models paper. The arc closely matches Topics 1 and 2 of this module: the V/M/C architecture, in-dream training, and the agent-controller results on car-racing and Doom.

### Required Readings & Resources

- David Foster, *Generative Deep Learning* (2nd ed.), O’Reilly, 2023. **Chapter 12, “World Models.”** Note: the 2nd-edition GitHub repository does not include companion code for Chapter 12; use the book and the original Ha & Schmidhuber materials below.
- D. Ha and J. Schmidhuber, “World Models,” NeurIPS 2018. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122). The seminal modern world-models paper.
- Companion interactive blog: [worldmodels.github.io](https://worldmodels.github.io/). One of the best-designed AI interactive explainers ever published; recommended over the PDF for first reading.
- D. Hafner et al., “Mastering Diverse Domains through World Models” (DreamerV3), 2023. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104). The modern model-based RL benchmark that uses a world model.

### Module To-Do List

1.  Read Topic 1 (Predictive Modeling of Latent States or Environments) and the [worldmodels.github.io](https://worldmodels.github.io/) interactive.
2.  Read Topic 2 (Planning, Imagination, and Structured Reasoning).
3.  Read Topic 3 (Linking World Models to Current Research Directions).
4.  Read Topic 4 (Technical Communication Through Paper Presentation). This Topic frames the Recorded Paper Presentation deliverable.
5.  Record and submit your **Recorded Research Paper Presentation** (12 to 15 minutes) on the primary paper you proposed in the Module 1 Setup Discussion.
6.  Submit your **Draft Report** as a reply to the Module 7 project discussion (700-1200 word PDF, six required sections). Reply to at least one classmate.

### Estimated Completion Time

Plan for roughly **15 to 20 hours** in this module (the heaviest of the term because of the two major deliverables):

- Topic pages 1-4: ~3 hours.
- Recorded Research Paper Presentation (re-reading the paper, building slides, recording, editing): ~6 to 8 hours.
- Draft Report (writing + figures): ~3 to 5 hours.
- Peer reply on the project discussion: ~30 minutes.


## Module Summary

This module introduced world models as the framework that combines a learned latent representation, a recurrent next-step predictor, and a controller into an agent that can plan or train inside its own predictions. The Recorded Research Paper Presentation and the Draft Report are both due this module; together they are the heaviest deliverable load of the term. The next module closes the course with multimodal models, the Final Project Report, and the Recorded Final Project Presentation.

## What You Should Have Completed in Module 7

Before moving on, use this self-check to make sure nothing in Module 7 slipped through:

- Read Topics 1 through 4: the V/M/C architecture, in-dream training and the Dreamer line of work, world models in modern research (video generation, robotics, LLM-implicit), and the technical-communication topic on giving a paper presentation.
- Recorded and submitted the **Recorded Research Paper Presentation** (12 to 15 minutes; captions or transcript provided).
- Submitted the **Draft Report** as a reply to the Module 7 project discussion (700-1200 word PDF, six required sections).
- Replied to at least one classmate’s Draft Report with substantive feedback.

## Next Steps

Module 8 covers **Multimodal Models**: systems that learn joint representations across modalities (text + image, text + audio, text + video, action + observation, etc.) and the foundation models built on top of those joint representations. CLIP (text-image), Flamingo (text-vision), and modern systems like GPT-4o, Gemini, and Claude with vision are the canonical examples.

Module 8 also concludes the course project. The **Final Project Report** (100 pts formal milestone) and the **Recorded Final Project Presentation** (100 pts in the Course Project Presentation 10% group) are both due. Together with the multimodal weekly tech assignment, this is a heavy module. Start the final report and the recording early in the week.

In Module 8 you will:

- Read Topics 1 through 4 on shared representations across modalities, contrastive learning and text-image alignment, practical examples of multimodal generation and retrieval, and final synthesis of methods and responsible use.
- Complete the Multimodal Case Analysis or Final Reflective Discussion.
- Submit the **Final Project Report** as a reply to the Module 8 project discussion.
- Submit the **Recorded Final Project Presentation** (12 to 15 minutes; uses the same presentation skills you exercised in this module’s Paper Presentation).
