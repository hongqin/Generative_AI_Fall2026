# Week 7, Topic 2: Planning, Imagination, and Structured Reasoning

## Topic Overview

**Plain-English summary first.** Once you have a fast, differentiable predictive model of an environment, you can train an agent *inside* the model, rolling out imagined trajectories rather than wasting expensive real-environment interactions. This is the central use of world models in reinforcement learning, and it is why model-based RL is so much more sample-efficient than model-free RL when the model is good enough. This Topic walks through how “dreaming” or “imagination” training actually works in the Ha & Schmidhuber setup, then summarizes the Dreamer line of work that became the modern model-based-RL benchmark.

### In-dream training: the core idea

In Ha & Schmidhuber (2018), the agent can be trained entirely inside the world model:

1.  Initialize: encode a starting observation $`o_0`$ with V to get $`z_0`$. Initialize the RNN state $`h_0`$.
2.  Roll out in imagination: at each step, the controller takes $`[z_t, h_t]`$ and outputs an action $`a_t`$. The MDN-RNN then samples the next latent $`z_{t+1}`$ and updates $`h_{t+1}`$. Repeat.
3.  Define a reward (or use one learned alongside the latent prediction; the original paper used a separate small reward model).
4.  Train the controller to maximize the imagined reward via CMA-ES.

The agent never sees a real environment step during this phase. Real interaction is only used to collect data for training V and M; the policy is trained inside the resulting world model. When this works, it is dramatically sample-efficient compared to model-free RL.

### Why this works (and when it does not)

The agent’s policy is only as good as the world model is accurate. Two failure modes:

- **Model exploitation**: the controller finds states the world model predicts incorrectly (typically states far from the training distribution) and learns a policy that gets high imagined reward by visiting those states. Performance in the real environment is then terrible.
- **Compounding error**: long rollouts in the world model accumulate prediction errors at each step. Short rollouts work better; long rollouts drift off the data manifold.

Mitigations: ensemble world models (Janner et al., 2019), short imagined rollouts mixed with real experience (the original Ha paper does this), conservative policy improvement objectives that bound the gap between imagined and real performance. The Dreamer line of work below tackles these systematically.

### The Dreamer line of work

Hafner et al. extended the Ha & Schmidhuber framework into a series of increasingly capable systems, all built on the same V/M/C-style decomposition with much better engineering:

#### PlaNet (Hafner et al., 2018)

Latent-space planning with a recurrent state-space model. The world model is a sequential VAE-like architecture (deterministic and stochastic latents both); planning uses cross-entropy method directly in latent space. **No separate controller learning**; planning is done online at each timestep.

#### Dreamer / DreamerV1 (Hafner et al., 2020)

Replaces online planning with a learned actor-critic policy trained inside the world model. The key technical contribution: backprop through imagined trajectories all the way through the dynamics model into the actor (made possible by the differentiable continuous-latent dynamics). Sample-efficient and works on a wide range of continuous-control benchmarks.

#### DreamerV2 (Hafner et al., 2021)

Discrete latents (a key change), KL-balancing in the world-model objective. First model-based agent to **beat the strongest model-free baselines on Atari**, a benchmark long dominated by model-free methods.

#### DreamerV3 (Hafner et al., 2023)

The current state of the art for general-purpose model-based RL. Same fixed hyperparameters work across **over 150 tasks** spanning Atari, continuous control, Minecraft, robotic manipulation. First system to solve the Minecraft diamond-collection challenge from scratch without curriculum or human data.

### Connection to LLM-based agents

Large language models can be viewed as implicit world models over the space of language. When you prompt an LLM to “think step by step” or run a chain-of-thought, you are effectively rolling out an imagined trajectory in its world model and using it to plan. The agent frameworks (LangChain, OpenAI Assistants, Claude Agent SDK) covered briefly in Module 6 Topic 4 are doing the same thing: using the LLM’s implicit world model to plan multi-step tool-use sequences.

The Yann LeCun JEPA proposal (2022) is an explicit attempt to build dedicated world models for non-language data using the same generative-modeling tools you have studied in this course. Whether explicit (Dreamer) or implicit (LLMs), the idea that intelligence requires a world model is now mainstream.

### Where this Topic fits

Topic 3 surveys the modern landscape of world-model work beyond Dreamer (Genie, video-as-world-model, robotics applications). Topic 4 is the practical “how to give a good recorded paper presentation” Topic that prepares the M7 Recorded Paper Presentation.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 12, in-dream training and CMA-ES sections.
- D. Hafner et al., “Mastering Diverse Domains through World Models” (DreamerV3), 2023. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104). Sections 1 to 3 cover the architecture; the rest is empirical.

Recommended:

- D. Hafner et al., “Dream to Control: Learning Behaviors by Latent Imagination” (DreamerV1), ICLR 2020. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603). The actor-critic-through-imagination training that defines the Dreamer family.

### Consider

- If the world model is slightly inaccurate but the controller exploits the inaccuracy to maximize imagined reward, the real-environment performance can be catastrophic. What does this say about how you should evaluate a world-model-based agent?
- Dreamer trains an actor-critic by backprop through the differentiable world-model dynamics. Why does this not work in the Ha & Schmidhuber setup, which uses CMA-ES on the controller instead?
- If large language models implicitly contain a world model over text, what does “in-dream training” look like for them? (Hint: think about what is happening when you fine-tune on synthetic data.)

### Resources

- D. Hafner’s DreamerV3 implementation: [github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3). Modern, well-engineered PyTorch (and JAX) code. Use this if you want to actually run a state-of-the-art model-based RL agent. Substantially more complex than the Foster notebooks from earlier modules.
- Y. LeCun, “A Path Towards Autonomous Machine Intelligence” (the JEPA proposal), 2022. [OpenReview PDF](https://openreview.net/pdf?id=BZ5a1r-kVsf). Influential proposal for a non-RL world-modeling agenda using self-supervised learning.
