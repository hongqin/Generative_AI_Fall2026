# Week 7, Topic 3: Linking World Models to Current Research Directions

## Topic Overview

**Plain-English summary first.** World models have had a renaissance since 2023. The same ideas you learned in Topics 1 and 2 (predict the future, plan inside the predictor, train a policy from imagined rollouts) now show up in several places that do not call themselves “world models” explicitly: foundation video generators, robot foundation models, and the implicit world-modeling that large language models do when they reason. This Topic gives you a tour so you can recognize the framework when you see it in modern papers and so you can place your project against the current research landscape if it is relevant.

### Video generation as world modeling

Modern text-to-video models (Sora from OpenAI, Veo from Google, Movie Gen from Meta) train on enormous video datasets and learn to predict future video frames conditioned on a text prompt. The result is a generative model that *implicitly* learns physics, object permanence, and basic causal reasoning. OpenAI explicitly framed Sora as a step toward “general-purpose simulators of the physical world,” which is the same framing as a world model.

Whether these video models are “real” world models is contested. They predict pixel evolution beautifully but have no explicit notion of objects, no inverse model for actions, and no inherent planning loop. The pragmatic answer is that they are world models for the narrow task they were trained on (predicting plausible future frames) but not yet general-purpose world models suitable for arbitrary agent control.

### Genie and the playable world model

Google DeepMind’s Genie (2024) and Genie 2 (2024) trained on Internet video learn to generate playable 2D and (in Genie 2) interactive 3D environments from a single image prompt, with an implicit action model that lets you control a character within the generated world. This is the most direct realization of a Ha-and-Schmidhuber-style world model at internet scale: the generative model predicts the next frame conditioned on a discrete action, and you can drive it interactively.

### Robot foundation models

Recent large robot policy models like RT-1 (2022), RT-2 (2023), and the RT-X dataset (2023) are trained on giant datasets of robot trajectories across many embodiments. They are not labeled “world models” in the V/M/C sense, but they have a closely related structure: a perception network maps observations to a representation; an autoregressive transformer predicts action sequences conditioned on a goal. The dynamics model is implicit (the robot embodies it), and the policy is integrated, but the family resemblance to the Ha & Schmidhuber framework is clear.

The follow-on work (Octo, OpenVLA, the various Pi-Zero variants) makes the world-modeling component more explicit and lets the same base model drive many different robot platforms.

### World models inside LLM reasoning

Recent interpretability work (Li et al., Othello; many others) has found that large language models trained to predict next tokens learn internal representations that function like world models: they track entities, infer hidden state, and roll forward simulations. This is one mechanistic explanation for why chain-of-thought prompting works: the model is using its implicit world model to imagine consequences before committing to a response.

This is the bridge between Module 4 (autoregressive modeling), Module 6 (transformers), and Module 7 (world models). Modern large generative models are increasingly understood as containing implicit world models, and the practical advances in agent frameworks build on top of that.

### Where world models will likely matter for your project

Even if your course project is not explicitly about reinforcement learning, the world-models framework gives you vocabulary for:

- Any project where the generative model is being used to *simulate* something (synthetic data generation, what-if analysis, scenario planning) rather than just sample-from-data.
- Any project where an agent or pipeline takes multiple steps that depend on each other (a tool-use agent, a multi-step analysis pipeline, a robot-action sequence).
- Any project that uses video, time-series, or sequence data where future-prediction is part of the task.

If your project does not naturally fit any of these, the world-models material is still a useful piece of the modern AI vocabulary; it will show up in papers you read after the course ends.

### Where this Topic fits

Topic 4 is the practical “how to give a good recorded paper presentation” Topic. Module 8 covers multimodal models, which are tightly connected to several of the topics above (Sora, video generation, robotics).

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 12 closing sections on the modern landscape.
- Pick ONE of the following recent papers to skim, based on what is closest to your project:
  - OpenAI, “Video generation models as world simulators” (Sora technical report), 2024. [openai.com/research/video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators).
  - J. Bruce et al., “Genie: Generative Interactive Environments,” ICML 2024. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391).
  - Open X-Embodiment Collaboration, “Open X-Embodiment: Robotic Learning Datasets and RT-X Models,” 2023. [arXiv:2310.08864](https://arxiv.org/abs/2310.08864).

### Consider

- Sora and other video models predict pixel evolution conditioned on text. Are they world models in any useful sense, or only “next-frame predictors”? What is the right standard?
- For your course project, is there an implicit or explicit world model component? If yes, what is being predicted; if no, would adding one make the project more powerful?
- Large language models contain implicit world models over text. Is that the same kind of thing as a Ha-and-Schmidhuber world model over a physical environment, or is it a different category that just shares a name?

### Resources

- J. Bruce et al., “Genie 2: A Large-Scale Foundation World Model,” Google DeepMind, 2024. [DeepMind blog post](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/). Interactive 3D environments generated from a single image.
- K. Brohan et al., “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control,” CoRL 2023. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818).
- K. Li et al., “Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task” (OthelloGPT), ICLR 2023. [arXiv:2210.13382](https://arxiv.org/abs/2210.13382). The interpretability paper that found explicit world-modeling structure inside an autoregressive model.
