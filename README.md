# Low_Resource_NMT
Project for NLP2 course at UVA (2025)

## Useful commands to run on snellius

```
# install env (done already)
sbatch install_environment.job

# interactive job, 2/4th of a node
srun --partition=gpu_a100 --gpus=1 --ntasks=1 --cpus-per-task=18 --time=02:00:00 --pty bash -i
module purge
module load 2022
module load Anaconda3/2022.05
source activate nmt

# start jupyter notebook
jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888

# Install all neccessary vscode Jupyter extensions  

# select in vscode: select kernel, existing jupyter server, paste the link containing "gcn"
```