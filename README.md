# Generative AI (CS 782/882), Fall 2026

Public course materials for **Generative AI**, an 8-week accelerated, asynchronous online graduate course at Old Dominion University (CS 782/882, cross-listed as DASC 782/882), taught by Hong Qin in Fall 2026.

This repository is the open mirror of the course: lecture notes, topic pages, assignments, exercises, and project templates. It contains no student work, no recordings, and no copyrighted third-party material. Enrolled students should treat Canvas as the version of record for due dates, submissions, and grades.

## What the course covers

Eight weekly modules, one model family per week, plus a term-long course project and a recorded research-paper presentation.

| Week | Module | Topics | Weekly technical activity | Project track |
|---|---|---|---|---|
| 1 | [Overview and Course Launch](week01-overview/) | What generative modeling is, the model families, course workflow, compute setup | Orientation and self-introduction | Setup confirmation and project-interest note |
| 2 | [Variational Autoencoders](week02-vae/) | Latent variables, variational inference and the ELBO, reconstruction and KL, training a VAE | VAE technical assignment | **Course project proposal** |
| 3 | [Generative Adversarial Networks](week03-gan/) | Minimax objective, training instability and Wasserstein GANs, DCGAN and conditional GANs, comparing families | GAN coding exercise | Implementation check-in |
| 4 | [Autoregressive Models](week04-autoregressive/) | Sequential factorization, RNN and LSTM, sampling and conditioning, strengths and limits | Autoregressive or LSTM mini-lab | Progress update |
| 5 | [Diffusion Models](week05-diffusion/) | Forward noising and reverse denoising, training objectives, sampling cost and guidance, project midpoint | Diffusion assignment | **Mid-project report** |
| 6 | [Transformers](week06-transformers/) | Self-attention, tokenization and embeddings, transformer layers and inference, practical LLM workflows | Transformer or LLM workflow assignment | Methods refinement note |
| 7 | [World Models](week07-world-models/) | Predictive latent-state models, planning and imagination, current research directions, giving a paper presentation | **Recorded research paper presentation** | **Draft report** |
| 8 | [Multimodal Models](week08-multimodal/) | Shared representations, contrastive learning and CLIP, multimodal generation and retrieval, synthesis and responsible use | Multimodal case analysis or reflective discussion | **Final report and recorded presentation** |

## Course learning objectives

- **CLO1.** Explain the theoretical foundations, assumptions, strengths, and limitations of major generative AI models, including VAEs, GANs, autoregressive models, diffusion models, transformers, world models, and multimodal systems.
- **CLO2.** Implement, adapt, and evaluate representative generative AI workflows using Python and modern AI tools.
- **CLO3.** Design, develop, document, and present a generative AI project applied to a real-world domain or research problem.

## Grading

| Category | Weight |
|---|---:|
| Weekly technical assignments and discussions | 40% |
| Research paper presentation | 10% |
| Course project (proposal, check-ins, mid-project report, draft, final report) | 40% |
| Course project presentation | 10% |

## Repository layout

```
syllabus/                 public version of the approved syllabus
setup/                    Python environment and ODU HPC access notes
weekNN-<topic>/           one folder per week
  README.md               module overview, learning objectives, readings, to-do list, summary
  topicK-*.md             the four topic pages (plain-English summary, math, payoff)
  notes/weekNN-notes.tex  LaTeX lecture notes, one section per topic, with compiled PDF
  notes/supplementary/    optional notes on topics from the 16-week version (weeks 5 and 6)
  assignments/            the weekly technical assignment and the project-track item
  demos/                  instructor notebooks (week 6)
  paper-presentation/     guidelines, rubric, and suggested papers (week 7)
project/                  project overview, proposal and report templates, rubrics
readings/                 consolidated reading list with open-access links
supplementary/            optional notes not in the 8-week core (agentic AI, conformal prediction, group SHAP)
notes-build/              shared LaTeX preamble and a Makefile that compiles every .tex
```

## Lecture notes

Each week has a LaTeX document with one section per topic; the compiled PDF sits next to the source. To rebuild all of them:

```
cd notes-build && make
```

This needs a TeX distribution with `pdflatex` (TeX Live or MacTeX). Each `.tex` file is self-contained and also compiles on Overleaf.

## Hands-on code

Coding assignments anchor on the open-source companion notebooks for David Foster, *Generative Deep Learning* (2nd ed., O'Reilly, 2023): https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition. The notebooks are not copied here; each assignment links to the notebook directory it uses. Week 6 also uses the Hugging Face `transformers` library.

## License

- Course text, notes, and assignments: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See `LICENSE-CONTENT.md`.
- Code (scripts, notebooks, Makefile): MIT. See `LICENSE`.

Third-party papers and books are cited, not redistributed.

## Citation

If you reuse these materials, please cite:

> Hong Qin. *Generative AI (CS 782/882), Fall 2026: Course Materials.* Old Dominion University. https://github.com/hongqin/Generative_AI_Fall2026
