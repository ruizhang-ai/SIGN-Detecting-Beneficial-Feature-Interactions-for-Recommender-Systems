
# Instructions for reproducing the experiments 

## Environmental Requirement

0. You need to set up the environment for running the experiments (Anaconda 3-2020.02 or above and Python 3.7 or above). First, you can install Anaconda by following the instruction from  the [website](https://docs.anaconda.com/anaconda/install/).
   
   Next, you can create a virtual environment using the following commands:
   <pre><code>$ conda create -n GMCF python=3.7 anaconda
   $ conda activate GMCF</code></pre>

1. Install Pytorch with version 1.6.0 or later.

   For example (with CPU only version):
   ```
   $ pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. Install ```torch-geometric``` package with version 1.4.3.

   *(Note that it may need to appropriatly install the package ```torch-geometric``` based on the CUDA version (or CPU version if GPU is not avaliable). Please refer to the official website https://pytorch-geometric.readthedocs.io/en/1.4.3/notes/installation.html for more information of installing prerequisites of ```torch-geometric```)*

   For example (CPU only version):
   ```
   $ CUDA=cpu
   $ TORCH=1.6.0
   $ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
   $ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
   $ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
   $ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
   $ pip install torch-geometric==1.4.3
   ```
3. Download the code and data.

   clone this repository:
   ```
   $ git clone https://github.com/suyixin12123/L0-SIGN.git
   ```

   and go to the folder of the repository:
   ```
   $ cd GMCF_Neural_Graph_Matching_based_Collaborative_Filtering
   ```

   Now, our source codes are in the folder ```code/``` and the datasets are in the folder ```data/```.

4. Install other packages listed in requirements.txt.
   ```
   $ pip install -r requirements.txt
   ```

## Run the code


Go to the ```code/``` folder and run the ```main.py``` file:
   ```
   $ cd code
   $ python main.py --dataset=frappe --predict_edge=1 --lr=0.05
   ```
   Main arguments:
   ```
   --dataset [frappe, ml-tag, twitter, DBPL_v1]: the dataset to run.
   --dim: the embedding dimension of each attribute.
   --hidden_layer: the MLP hidden layer for the inner interaction modeling function.
   --l0_weight: weight of the L0 activation regularization term.
   --l2_weight: weight of the L2 activation regularization term.
   --lr: the learning rate.
   --l0_para: l0 parameters, which are beta (temprature), zeta (interval_min) and gama (interval_max).
   --pred_edges: 0 for using edges in dataset (may need *{dataset}.edge* file), 1 for predicting edges using L_0.
   ```
   For more argument options, please refer to ```SIGN_main.py```
