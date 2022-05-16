# Cluster advice for python users:

## Setting up a conda env:

Load anaconda

```bash
module load miniconda3-4.5.4-gcc-5.4.0-hivczbz
```

Create an anaconda virtual environment:

```bash
conda create -n myenvname python=3.9.0 -p /home/nt409/software/conda_envs/hrhr_env --copy

# or
conda create -n poly2_env python=3.8.12
conda create -n polygenic_env python=3.9.0
```

NB: to install a specific version of a package, do package_name=<version_number> e.g. iris=2.0.0

Install any python modules required inside the environment:

```bash
conda install -y -c conda-forge iris=2.4.0 -p /home/nt409/software/conda_envs/hrhr_env --copy
```

To clone/copy an existing conda environment

```bash
 conda create --name poly_env2 --clone polygenic_env
```

Or create new env from environment.yml file (NB had issues last time with this
on cluster)

```bash
conda env create -f environment.yml
```

<!-- TODO -->
<!-- conda install -c conda-forge pickle -->

<!-- conda install -c anaconda pillow -->

<!-- conda install -c conda-forge pygam -->
<!-- conda install -c plotly plotly -->

<!--  -->

See `https://shandou.medium.com/export-and-create-conda-environment-with-yml-5de619fe5a2`.

To see list of conda envs:

```bash
conda info --envs
```

## Using a conda env:

To activate:

```bash
source activate /home/nt409/software/conda_envs/hrhr_env
```

To deactivate:

- open a new login, or

```bash
conda deactivate
```

## Script for job:

Set up with boilerplate from wiki.

Removed:

```bash
module purge                               # Removes all modules still loaded
module load rhel7/default-peta4            # REQUIRED - loads the basic environment
module load use.own                        # This line loads the own module list
```

NOW USING STUFF AS SENT BY DG - see `param_scan/failed.submit`

```bash
# module load miniconda3-4.5.4-gcc-5.4.0-hivczbz
source activate /home/nt409/software/conda_envs/hrhr_env
```

## To test a job:

```bash
bash foldername/myjobname.submit
# e.g.
bash param_scan/scan.submit
```

or just run individual python file locally:

```bash
python -m foldername.script_name PARAM_0
# e.g.
python -m param_scan.r4_re_run_failures_cluster 0
```

## To submit a job:

```bash
sbatch foldername/myjobname.submit
# e.g.
sbatch param_scan/scan.submit
# OR
sbatch param_scan/failed.submit
# OR
sbatch alternation_scan/scan.submit
```

## To check progress:

```bash
squeue -u nt409
```

## Est start time:

```bash
squeue -u nt409 --start
```

## To cancel a job:

```bash
scancel [JOBID]
```

e.g:

```bash
scancel 47650567
```

## To delete files of a certain type (command line)

Find files - run first:

```bash
find . -name "*.err" -type f
find . -name "*.out" -type f
```

If happy to delete these, then go ahead with

```bash
find . -name "*.err" -type f -delete
find . -name "*.out" -type f -delete
```

## Git stuff

```bash
git clone https://github.com/nt409/polygenic2
```
