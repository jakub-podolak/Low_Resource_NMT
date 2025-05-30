# Low Resource Neural Machine Translation
## Project for NLP2 course at UVA (2025)

The goal of this project is to revisit domain adaptation of small NMT systems and try to apply SoTA methods to improve the translation quality when the number of in-domain parallel samples is limited.


# 1. ⚙️ Setup

## 1.1 Install environment

```
# install env (done already)
sbatch install_environment.job
```

## 1.2 Run interactive shell job
To use Python scripts or Jupyter notebooks. It was easier for us to run some experiments in an interactive shell than in sbatch jobs.

```
# interactive job, 2/4th of a node
srun --partition=gpu_a100 --gpus=1 --ntasks=1 --cpus-per-task=18 --time=02:00:00 --pty bash -i
module purge
module load 2022
module load Anaconda3/2022.05
source activate nmt
```

## 1.3 Start jupyter notebook

```
# start jupyter notebook
jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888

# Install all necessary VSCode Jupyter extensions  

# select in vscode: select kernel, existing Jupyter server, paste the link containing "gcn"
```
