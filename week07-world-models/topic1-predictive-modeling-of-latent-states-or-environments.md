# Week 7, Topic 1: Predictive Modeling of Latent States or Environments

## Topic Overview

**Plain-English summary first.** A “world model” is a learned compact predictive model of an environment. Once you have one, an agent can use it to plan ahead, train inside imagined rollouts instead of the real world, or simply build better representations for downstream tasks. The most influential modern formulation is Ha & Schmidhuber (2018), which factors the world model into three pieces that you have already met in Modules 2 and 4: a vision module that encodes observations into a compact latent (a VAE), a memory module that predicts the next latent given the current latent and action (an MDN-RNN), and a tiny controller that maps the latent state to actions. This Topic introduces the framework and shows why splitting the agent into these three parts is a useful decomposition.

### The agent-environment setup

Background framing from reinforcement learning. At each timestep $`t`$, the agent receives an observation $`o_t`$ from the environment, picks an action $`a_t`$, and receives a reward $`r_t`$ and a new observation $`o_{t+1}`$. The objective is to maximize the expected discounted sum of rewards $`\mathbb{E}\!\left[\sum_t \gamma^t r_t\right]`$ for some discount $`\gamma \in (0, 1)`$.

Two big approaches:

- **Model-free RL**: learn the policy (or a value function) directly from interaction with the environment. Examples: REINFORCE, DQN, PPO. Sample-inefficient (you need a lot of real environment steps).
- **Model-based RL**: learn a model of the environment (the dynamics and reward), then use the model to plan or train a policy. Sample-efficient (one real step provides a lot of imagined experience), but only as good as the model is.

World models are the centerpiece of modern model-based RL.

### Ha & Schmidhuber’s V/M/C architecture

The 2018 paper factors the agent into three modules:

#### V: Vision module

A **variational autoencoder** (Module 2!) that encodes the high-dimensional observation $`o_t`$ (e.g., a 64×64×3 image from a CarRacing environment) into a compact latent $`z_t`$ of dimension ~32. The VAE is trained on a buffer of observations collected by a random policy; the encoder and decoder are then frozen. This is the same VAE you built in Module 2, applied to environment observations.

#### M: Memory module (MDN-RNN)

A **mixture-density-network LSTM** (an MDN-RNN, mixing Module 4’s LSTM with Module 5’s mixture-density output) that, given the current latent $`z_t`$, action $`a_t`$, and recurrent state $`h_t`$, predicts a distribution over the next latent $`z_{t+1}`$:

``` math
p\!\left(z_{t+1} \,\middle|\, z_t,\, a_t,\, h_t\right) \;=\; \mathrm{MDN}\!\left(\mathrm{RNN}(z_t, a_t, h_t)\right).
```

The mixture-density output captures multimodal futures (the environment can branch). The recurrent state $`h_t`$ carries summary information about past observations. M is trained on rollouts from the agent collected through environment interaction; once trained, the RNN gives the controller a compact memory of the agent’s situation.

#### C: Controller

A **tiny linear policy** that maps the concatenation $`[z_t, h_t]`$ to an action $`a_t`$. For the CarRacing environment in the paper, the controller has ~1000 parameters total. It is trained with the **CMA-ES evolutionary algorithm** (not backprop), because the controller is so small that population-based search is fast.

The split is doing real work: the big networks (V and M) are trained *generatively* (reconstruct observations; predict next latents) without any reward signal. The reward signal goes only into the tiny controller. This separation is the reason the agent learns to play CarRacing with surprisingly few real-environment interactions.

### Why factor the agent this way?

- The generative pieces (V and M) are **reusable**. Once you have a vision module trained on a domain, you can train multiple controllers for different tasks in that domain without retraining V.
- Training is **more sample-efficient**. Generative training does not require reward labels and can use any collected experience (including random-policy rollouts).
- The controller is **tiny enough to evolve**. Backprop through a chain of stochastic RNN steps is hard; CMA-ES sidesteps that entirely by treating C as a black box.
- The system is **interpretable**: you can inspect the decoder reconstructions to debug V, and you can sample MDN-RNN rollouts to debug M, independently of the agent’s performance.

### Where this Topic fits

Topic 2 covers the most important practical use of a world model: training the agent *inside the model* (“dreaming”) rather than in the real environment. Topic 3 then surveys what has happened since 2018, including the Dreamer line of work and the current research landscape. Topic 4 is the “how to give a good paper presentation” Topic that prepares you for the Recorded Paper Presentation due at the end of this module.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 12, sections introducing the V/M/C architecture.
- **The [worldmodels.github.io](https://worldmodels.github.io/) interactive blog**. Read this in addition to (or instead of) the arXiv PDF. The interactive diagrams of the V module’s reconstructions, the M module’s dreaming, and the C module’s evolved policy are the single best way to internalize the framework.
- D. Ha and J. Schmidhuber, “World Models,” NeurIPS 2018. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122). Sections 1 to 3 cover everything on this page.

### Consider

- The vision (V) and memory (M) modules are trained without any reward signal. Why is that a feature rather than a bug, even though the ultimate goal is to maximize reward?
- The controller (C) has ~1000 parameters and is trained with evolutionary search, not backprop. What would go wrong if you tried to train the entire V/M/C pipeline end-to-end with backprop on the reward signal?
- The MDN-RNN outputs a *mixture* distribution rather than a single Gaussian for the next latent. What is the environment property that makes mixtures essential?

### Resources

- Original Ha & Schmidhuber code (TensorFlow 1, archived; mostly of historical interest): [github.com/hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments). The full pipeline that produced the paper’s results.
- J. Schmidhuber, “On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models,” 2015. [arXiv:1511.09249](https://arxiv.org/abs/1511.09249). The conceptual predecessor that frames the C/M split.
