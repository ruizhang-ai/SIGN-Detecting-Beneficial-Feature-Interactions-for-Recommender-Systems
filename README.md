# Detecting Beneficial Feature Interactions for Recommender Systems (L0-SIGN)

This is our implementation for the paper:

Su, Y., Zhang, R., Erfani, S., & Xu, Z. (2021). *Detecting Beneficial Feature Interactions for Recommender Systems*. In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI). [Link](https://arxiv.org/abs/2008.00404)

## Description

Feature interactions are essential for achieving high accuracy in recommender systems. Many studies take into account the interaction between every pair of features. However, this is suboptimal because some feature interactions may not be that relevant to the recommendation result, and taking them into account may introduce noise and decrease recommendation accuracy. To make the best out of feature interactions, we propose a graph neural network approach to effectively model them, together with a novel technique to automatically detect those feature interactions that are beneficial in terms of recommendation accuracy. The automatic feature interaction detection is achieved via edge prediction with an L0 activation regularization. Our proposed model is proved to be effective through the information bottleneck principle and statistical interaction theory.

<p align="center">
  <img src="https://github.com/suyixin12123/L0-SIGN/blob/main/img/SIGN_frame.png", alt="Model Structure" width="800">
  <p align="center"><em>Figure2: An Overview of the L0-SIGN Model.</em></p>
</p>


## What are in this Repository
This repository contains the following contents:

```
/
├── code/                   --> (The folder containing the source code)
|   ├── dataloader.py       --> (The code to proceed the data into code-usable format)
|   ├── SIGN_main.py             --> (The main code file. The code is run through this file)
|   ├── SIGN_model.py            --> (Contains the function of our GMCF model.)
|   ├── SIGN_train.py            --> (Contains the code to train and evaluate our GMCF model.)
├── data/                   --> (The folder containing three used datasets)   
|   ├── frappe/             --> (The frappe dataset to evaluate recommendation.)
|   ├── ml-tag/             --> (The MovieLens Tag dataset to evaluate recommendation.)
|   ├── twitter/            --> (The Twitter dataset to evaluate graph classification.)
|   ├── DBLP_v1/            --> (The DBLP dataset to evaluate graph classification.)
├── img/                    --> (The images for README (not used for the code))   
|   ├── SIGN_frame.png      --> (The overall structure of our L0-SIGN model)
├── LICENCE                 --> (The licence file)
```

## Run our code

To run our code, please follow the instructions in our [code/](code/) folder.

## Cite our paper

Please credit our work by citing the following paper:

```
@inproceedings{su2021detecting,
  title={Detecting Beneficial Feature Interactions for Recommender Systems},
  author={Su, Yixin and Zhang, Rui and Erfani, Sarah and Xu, Zhenghua},
  booktitle={Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI)},
  year={2021}
}
```
