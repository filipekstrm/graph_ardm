# Install
These are the instructions to install the packages for which this code was tested. To install the required packages, we recommend using Anaconda (see instructions [here](https://docs.anaconda.com/free/anaconda/install/index.html) if you do not yet have used Anaconda)

1. Start by creating an environment with Python 3.9 and rdkit

``conda create --name ardm -c conda-forge python=3.9 rdkit=2023.03.1``

2. Activate environment

``conda activate ardm``

3. Install Pytorch 1.10.0 (here with cuda 11.3, see [here](https://pytorch.org/get-started/previous-versions/) for other options)

``conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge``

4. Install Pytorch Geometric 2.0.4

``conda install pyg=2.0.4 -c pyg``

5. Install wandb

``conda install wandb torchmetrics pandas=1.5 -c conda-forge``

6. Install molsets for MOSES metrics

``pip install molsets``