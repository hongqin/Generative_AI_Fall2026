# Python environment

The course uses Python 3.10 or later with PyTorch or TensorFlow, Jupyter, and the Hugging Face `transformers` library. Most weekly assignments start from the companion notebooks for David Foster, *Generative Deep Learning* (2nd ed.), which use TensorFlow/Keras; the transformer and multimodal weeks use PyTorch through Hugging Face. Either framework is acceptable for the course project.

## Three tiers of compute

1. **Laptop or desktop (CPU only).** Fine for reading code, debugging on tiny inputs, and small VAE or LSTM experiments (Weeks 2 and 4). Not realistic for training GANs, diffusion models, or transformers beyond toy scale.
2. **ODU HPC (recommended).** Free for ODU students, real GPUs, and the right home for most coursework and the course project. See [hpc-access.md](hpc-access.md).
3. **Google Colab.** Zero-install notebooks with a free GPU tier. Useful for prototypes and quick exploration; free-tier sessions disconnect and long runs get killed, so treat Colab as supplemental.

## Local setup (conda)

```
conda create -n genai python=3.11
conda activate genai
pip install jupyterlab numpy matplotlib scikit-learn pandas
# choose one framework, or both:
pip install torch torchvision torchaudio        # see pytorch.org for the CUDA build that matches your GPU
pip install tensorflow                           # Foster's notebooks
pip install transformers datasets accelerate    # Weeks 6 and 8
```

Foster's companion repository has its own environment instructions (a Dockerfile and `requirements.txt`): https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition

## Where each week's work runs

| Week | Typical hardware |
|---|---|
| 2 (VAE), 4 (autoregressive) | CPU laptop or Colab is enough; HPC is more comfortable |
| 3 (GAN), 5 (diffusion), 6 (transformers), 7 (world models), 8 (multimodal) | Plan on a real GPU: ODU HPC via Open OnDemand |
| Course project | Scope it to the GPU budget you actually have; request HPC access in Week 1 |

## Recommended accounts

- GitHub (for submitting code and hosting your project repository)
- Hugging Face (model and dataset hub; a free account is enough)
- Google account (for Colab)
