# Week 4, Topic 2: Recurrent Models and LSTM Intuition

## Topic Overview

**Plain-English summary first.** Topic 1 said the autoregressive trick is to parameterize each per-position conditional $`p_\theta(x_i \mid x_{\lt i})`$ with a neural network. This Topic is about *how* you build a network that takes a variable-length sequence as input. The original answer is the recurrent neural network (RNN): a small block of weights that you apply over and over, threading a hidden state through time. Vanilla RNNs train poorly for long sequences because of the vanishing-gradient problem. The LSTM cell, with its three gates and a separate cell state, is the fix that made long-sequence training tractable and dominated the field for two decades before transformers took over (Module 6).

### The recurrent computation

An RNN processes a sequence $`(x_1, x_2, \dots, x_T)`$ by maintaining a hidden state $`h_t`$ that gets updated at each timestep:

``` math
h_t \;=\; \tanh\!\left(W_{xh}\, x_t \,+\, W_{hh}\, h_{t-1} \,+\, b_h\right), \qquad y_t \;=\; W_{hy}\, h_t \,+\, b_y.
```

The crucial point is that the same weights $`W_{xh}, W_{hh}, W_{hy}, b_h, b_y`$ are applied at every timestep. The model has a fixed number of parameters regardless of sequence length, which is the only reason RNNs can handle variable-length inputs at all. The hidden state $`h_t`$ is the model’s “memory” of everything it has seen so far.

For autoregressive generation, we set up the RNN to output the parameters of $`p_\theta(x_i \mid x_{\lt i})`$ at each timestep. For a categorical output (text), that means a softmax over the vocabulary at each position; for a continuous output, a parametric distribution (e.g., a Gaussian mean and variance, or a mixture).

### Why vanilla RNNs fail at long sequences

Training an RNN means backpropagating gradients through the recurrence. Each backward step multiplies by the Jacobian of the recurrence relation, which involves the recurrent weight matrix $`W_{hh}`$. Over many timesteps, the gradient is a product of many such matrices. Two failure modes:

- **Vanishing gradient**: if the eigenvalues of $`W_{hh}`$ are smaller than 1, the gradient shrinks exponentially with sequence length. The network cannot learn long-range dependencies because the gradient signal does not reach the early timesteps.
- **Exploding gradient**: if the eigenvalues are larger than 1, the gradient grows exponentially. Training becomes numerically unstable.

Exploding gradients can be controlled by gradient clipping. Vanishing gradients are the harder problem: the network has no useful signal to learn from. This is why vanilla RNNs struggle on dependencies more than ~10 timesteps apart, while real text or audio has dependencies thousands of timesteps long.

### The LSTM cell

Hochreiter and Schmidhuber (1997) proposed the long short-term memory (LSTM) cell, which adds three gates and a separate cell state alongside the hidden state. The gates are sigmoid-valued functions of the input and previous hidden state; they decide what to forget, what to write, and what to read:

``` math
\begin{aligned} f_t &= \sigma\!\left(W_f \cdot [h_{t-1}, x_t] + b_f\right) \quad \text{(forget gate)} \\ i_t &= \sigma\!\left(W_i \cdot [h_{t-1}, x_t] + b_i\right) \quad \text{(input gate)} \\ o_t &= \sigma\!\left(W_o \cdot [h_{t-1}, x_t] + b_o\right) \quad \text{(output gate)} \\ \tilde{C}_t &= \tanh\!\left(W_C \cdot [h_{t-1}, x_t] + b_C\right) \\ C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\ h_t &= o_t \odot \tanh(C_t) \end{aligned}
```

The key piece is the cell-state update $`C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t`$. The forget gate decides how much of the old cell state to keep; the input gate decides how much new information to write. Crucially, when the forget gate is close to 1 and the input gate is close to 0, information in the cell state passes through unchanged. **Gradients can flow back through this multiplicative-identity path without vanishing.** That is what makes LSTM training work on long sequences.

You do not need to memorize the equations. The intuition you do need: *each gate is a learned switch that decides whether information flows through; the cell state is a highway for information to travel long distances without being squashed by repeated tanh operations.* Foster’s `01_lstm` notebook implements all of this in Keras with a few lines.

### Variants you will hear about

- **GRU (Gated Recurrent Unit)**: a simpler gating scheme with two gates instead of three. Often comparable to LSTM in practice, faster to train.
- **Bidirectional RNN/LSTM**: run one RNN forward and one backward over the sequence, concatenate the hidden states. Useful for tasks where you have the whole sequence available at training time (sequence labeling, machine translation encoder). NOT useful for autoregressive generation, where you only have the past.
- **Stacked / deep RNN**: stack multiple RNN layers vertically. The first layer takes the input; each subsequent layer takes the previous layer’s hidden state. Common for text generation.

### Where this Topic fits

Topic 3 covers what to do once you have a trained autoregressive model: how to sample, the temperature/top-k/nucleus knobs, conditioning patterns, and quality metrics. Topic 4 returns to the family comparison. Module 6 (Transformers) is the architectural successor to RNNs: it replaces recurrence with self-attention and is what powers modern large language models.

### Read

Required:

- David Foster, *Generative Deep Learning* (2nd ed.), Chapter 5, sections on RNN and LSTM architectures. Walk through the Foster `01_lstm` notebook to see the gates in code.
- Course tutorial: [Week 4 lecture notes (PDF)](notes/week04-notes.pdf) (Principles of LSTM, with worked-out gate equations and gradient-flow derivation).

Recommended (the original LSTM paper, very readable):

- S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, 1997. [Free PDF](https://www.bioinf.jku.at/publications/older/2604.pdf).

### Consider

- The cell-state update $`C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t`$ is what fixes the vanishing-gradient problem. What does it mean intuitively if the forget gate is consistently close to 0 during training? What about consistently close to 1?
- Bidirectional LSTMs are useful for sequence labeling but not for autoregressive generation. Why?
- For your course project, would you choose an RNN/LSTM, a transformer (Module 6), or a non-sequence model entirely? Whatever the answer, what makes the sequence model a good or bad fit for your data?

### Resources

- C. Olah, “Understanding LSTM Networks,” 2015. [colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). The canonical accessible LSTM explainer with hand-drawn diagrams; recommended companion to the equations above.
- K. Cho et al., “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation” (the GRU paper), EMNLP 2014. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078).
