# Week 1, Topic 4: Introductory Computing Setup for Local or Cloud Execution

## Overview

Most coursework in this course (training small generative models, running notebooks, fine-tuning transformers, agent experiments) is much easier with a GPU than on a CPU-only laptop. This topic walks through your three realistic options, recommends a default, and tells you exactly how to get started. **Do not skip Week 1 here**; account approval can take several days, and the Module 1 Setup Discussion asks you to confirm ODU HPC access by the end of the week.

### Three tiers of compute

1.  **Your laptop or desktop (CPU only).** Fine for reading code, debugging on tiny inputs, and small VAE experiments. Not realistic for training transformers, diffusion models, or anything multimodal at non-toy scale.
2.  **ODU HPC (recommended).** Free for ODU students, real GPUs, professionally administered, and the right home for most coursework and your course project. Two clusters are available to this course: **Wahab** (default) and **Waterfield** (fallback).
3.  **Google Colab.** Convenient zero-install notebook with a free GPU tier. Useful for prototypes, small demos, and quick exploration. Heavy GPU usage is a real issue here: free-tier sessions disconnect, GPU availability is throttled, and longer training runs frequently get killed. Treat Colab as supplemental rather than primary.

### The ODU HPC story for this course

The instructor recommends **Wahab** as the default cluster for this course. It is well-documented, has a stable Open OnDemand web portal, and meets the GPU needs of every weekly module on this syllabus. Use **Waterfield** (ODU's newer Google-Cloud HPC cluster) as a fallback when Wahab is congested, when a job needs a newer GPU type, or when you want to try a more recent software stack. Both clusters are accessed with the same MIDAS credentials and the same one-time access request form.

### Step-by-step setup

#### 1. Request an HPC account (do this on Day 1)

Submit the ODU Research & Cloud Computing (RCC) access request form: [Request HPC Resources](https://forms.odu.edu/view.php?id=93440). This is a **one-time** form that covers all ODU HPC systems (Wahab, Turing, Waterfield). You will receive an email when your account is provisioned. Approval can take several business days, so do not wait until you need it.

#### 2. Pick how you want to connect

Most students new to HPC should start with the **browser portal (Open OnDemand)**. It hides SLURM and the Linux shell behind a friendly web interface, and lets you launch a JupyterLab session with a GPU in a few clicks. Use SSH only if you are already comfortable with the command line and want the speed of a terminal.

- **Wahab via browser (recommended starting point):** [Wahab via browser](https://ondemand.wahab.hpc.odu.edu)
- **Wahab via SSH:** SSH is supported, but the current login hostname is not assumed here. Please confirm the active Wahab SSH hostname with ODU Research & Cloud Computing (<rcc@odu.edu> or Tuesday office hours), or check the ODU HPC wiki's "Getting Started" page, before you set up SSH keys or your `~/.ssh/config`.
- **Waterfield via browser (fallback):** [Waterfield via browser](https://ondemand.waterfield.hpc.odu.edu)
- **Waterfield via SSH (fallback):** `ssh <midas_username>@waterfield.hpc.odu.edu` (this hostname is taken directly from the Waterfield wiki page).

All of the above require your **ODU MIDAS** credentials and a **Duo two-factor authentication** approval.

#### 3. Launch a JupyterLab session with a GPU (Open OnDemand)

1.  Log in to the Wahab Open OnDemand portal (link above).
2.  Open the *Interactive Apps* menu and select *JupyterLab*.
3.  Pick a partition that includes a GPU, set a reasonable walltime (start with 1 to 2 hours), and click *Launch*.
4.  Once the session starts, click *Connect to JupyterLab*. You now have a browser notebook running on a real GPU.

If a partition is busy, retry with a shorter walltime, or fall back to Waterfield and repeat the same steps in its Open OnDemand portal.

#### 4. Run Python from the shell (optional, for SSH users)

ODU HPC manages software through the **LMOD module system**. The common pattern:

```
module avail              # see what is available
module load python3       # make Python available
crun python3 --version    # run Python through the standard wrapper
```

On Waterfield, application launches use a wrapper named `crun` (each module is backed by a container image). For PyTorch with GPU, the pattern is typically `module load pytorch-gpu` then `crun.pytorch-gpu python my_train.py`. For project-specific Python dependencies, use a Python virtual environment rather than asking RCC for a system-wide install. The ODU guide is at [here](https://wiki.hpc.odu.edu/en/Software/Python).

#### 5. Know your storage

- **`/home/<midas>`**: persistent home directory. Put code, configuration, and small datasets here. Back up anything you cannot reproduce.
- **`/scratch/<midas>`**: high-throughput scratch (1 TB quota on Waterfield). Fast, but **not backed up**; treat it as temporary job space and clean it up.
- **`/cm/shared`**: shared read-only software and data, common across clusters.

Move data between clusters with `scp` or `rsync` over SSH, e.g. from Waterfield: `rsync -av <midas>@wahab.hpc.odu.edu:~/mydir .`

### Mapping compute choices to the course timeline

A rough guide so you know where to run each module's work:

- **Module 2 (VAE)** and **Module 4 (Autoregressive / LSTM)**: small enough to fit on a laptop CPU or Colab for the assignment, but Wahab is more comfortable.
- **Module 3 (GAN)**, **Module 5 (Diffusion)**, **Module 6 (Transformer)**, **Module 7 (World Models)**, **Module 8 (Multimodal)**: plan to run on a real GPU. Wahab via Open OnDemand is the default path; switch to Waterfield if Wahab is congested or if your project needs the newer GPU types.
- **Course project**: scope it so it fits the GPU budget you actually have. If you plan to train a large transformer or diffusion model from scratch, plan early for Waterfield rather than Colab.

### If you need help

ODU Research & Cloud Computing offers **Tuesday office hours** for HPC users and can be reached at <rcc@odu.edu>. For course-related help (assignments, project planning), use Canvas Inbox to the instructor. For class-wide HPC questions whose answers might help your peers (a flag you ran into, a successful module-load incantation), post to the Q&A Café in the Start Here module.

### Read

Before launching your first job:

- ODU RCC, [Introduction to HPC](https://wiki.hpc.odu.edu/Training/HPC-Intro): recorded 2026 workshop covering Wahab, SLURM job submission, the LMOD module system, data transfer, and resource etiquette. Workshop recordings are on Panopto and the slides are linked from the page.
- ODU RCC, [Open OnDemand on Wahab](https://wiki.hpc.odu.edu/en/open-ondemand): the official walkthrough of the browser portal you will use for most of your course coding.
- If you have never used a UNIX shell, work through Software Carpentry's [UNIX Shell lesson](http://swcarpentry.github.io/shell-novice/) (about 2 hours, free). You can also use [DeapSECURE's shorter Wahab-tailored shell tutorial](https://deapsecure.gitlab.io/deapsecure-lesson01-hpc/11-shell/).

### Consider

- Which tier of compute (laptop / Wahab via browser / Wahab via SSH / Waterfield / Colab) fits your current setup, and what is the next tier up you should learn before you need it?
- For your candidate course project, what is the biggest experiment you can imagine wanting to run? Will it fit in a 1- to 2-hour Open OnDemand JupyterLab session, or will it need a multi-hour SLURM batch job?
- If your account approval is delayed, what is your interim plan so you can still complete Week 1 and Week 2 work?

### Resources

Bookmark these:

- **RCC access request form** (one form covers Wahab, Turing, and Waterfield): [RCC access request form](https://forms.odu.edu/view.php?id=93440)
- **Wahab Open OnDemand portal**: [Wahab Open OnDemand portal](https://ondemand.wahab.hpc.odu.edu)
- **Waterfield Open OnDemand portal** (fallback): [Waterfield Open OnDemand portal](https://ondemand.waterfield.hpc.odu.edu)
- **Waterfield wiki**: [Waterfield wiki](https://wiki.hpc.odu.edu/en/gcp/hpc)
- **Open OnDemand wiki**: [Open OnDemand wiki](https://wiki.hpc.odu.edu/en/open-ondemand)
- **Intro to HPC workshop**: [Intro to HPC workshop](https://wiki.hpc.odu.edu/Training/HPC-Intro)
- **Python on ODU HPC**: [Python on ODU HPC](https://wiki.hpc.odu.edu/en/Software/Python)
- **ODU RCC contact**: <rcc@odu.edu> (Tuesday office hours)
- **Google Colab** (supplemental, light use): [Google Colab](https://colab.research.google.com/)
