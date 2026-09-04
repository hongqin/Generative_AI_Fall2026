# ODU HPC access

Old Dominion University's Research and Cloud Computing (RCC) group operates the clusters used in this course. Access is free for enrolled students.

## Clusters

- **Wahab** (default). Well documented, stable Open OnDemand web portal, GPUs sufficient for every weekly module.
- **Waterfield** (fallback). ODU's newer Google-Cloud-based HPC cluster; use it when Wahab is congested or a job needs a newer GPU type.
- **Turing** is covered by the same access form but is not needed for this course.

## Step 1: request an account (do this in Week 1)

Submit the one-time RCC access request form: https://forms.odu.edu/view.php?id=93440. It covers Wahab, Turing, and Waterfield. Approval can take several business days.

## Step 2: connect

Most students should start with the browser portal (Open OnDemand), which hides SLURM and the shell behind a web interface and lets you launch JupyterLab with a GPU in a few clicks.

- Wahab via browser: https://ondemand.wahab.hpc.odu.edu
- Waterfield via browser: https://ondemand.waterfield.hpc.odu.edu
- Waterfield via SSH: `ssh <midas_username>@waterfield.hpc.odu.edu`
- Wahab via SSH: confirm the current login hostname with RCC (rcc@odu.edu) or the ODU HPC wiki before configuring SSH keys.

All access uses ODU MIDAS credentials with Duo two-factor authentication.

## Step 3: launch JupyterLab with a GPU

1. Log in to the Open OnDemand portal.
2. Open *Interactive Apps* and choose *JupyterLab*.
3. Pick a GPU partition, set a walltime of 1 to 2 hours to start, and click *Launch*.
4. When the session starts, click *Connect to JupyterLab*.

## Step 4: shell usage (optional)

Software is managed with the LMOD module system:

```
module avail
module load python3
crun python3 --version
```

On Waterfield each module is container-backed and launched through `crun` (for example `module load pytorch-gpu` then `crun.pytorch-gpu python train.py`). Use a Python virtual environment for project-specific packages. Python guide: https://wiki.hpc.odu.edu/en/Software/Python

## Storage

- `/home/<midas>`: persistent, for code and small data; back up anything you cannot reproduce.
- `/scratch/<midas>`: fast, large, not backed up; temporary job space.
- `/cm/shared`: shared read-only software and data.

## Help and references

- RCC contact: rcc@odu.edu (weekly office hours)
- Introduction to HPC workshop: https://wiki.hpc.odu.edu/Training/HPC-Intro
- Open OnDemand guide: https://wiki.hpc.odu.edu/en/open-ondemand
- Waterfield wiki: https://wiki.hpc.odu.edu/en/gcp/hpc
- UNIX shell primer (Software Carpentry): http://swcarpentry.github.io/shell-novice/
