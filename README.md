# Factorized Contrastive Learning

This repository contains the source code to our [paper](https://arxiv.org/abs/2306.05268)

Factorized Contrastive Learning (FactorCL) is a new multimodal representation learning method to go beyond multi-view redundancy. It factorizes task-relevant information into shared and unique representations and captures task-relevant information via maximizing MI lower bounds and removing task-irrelevant information via minimizing MI upper bounds.

## MI Estimation with the NCE-CLUB Estimator
We first compare our proposed NCE-CLUB estimator to the [InfoNCE](https://arxiv.org/pdf/1807.03748.pdf) and [CLUB](https://arxiv.org/abs/2006.12013) estimators on a toy Gaussian dataset for MI estimation. 

All objectives (InfoNCE, CLUB, NCE-CLUB) and the critic models are implemented in ```critic_objectives.py```.

Please follow the steps in the notebook ```Gaussian_MI_Est/NCE_CLUB_Gaussian.ipynb``` to get a demonstration of the estimation quality of each estimator. 

## Controlled Experiments on Synthetic Dataset
We perform experiments on data with controllable ratios of task-relevant shared and unique information. The synthetic data allows us to investigate the performance of each objective under different conditions of shared information. 

The synthetic dataset and generation process are implemented in ```Synthetic/dataset.py```. 

Follow steps in the notebook ```Synthetic/synthetic_example.ipynb``` to generate synthetic data with customized shared information and run FactorCL/[SimCLR](https://arxiv.org/abs/2002.05709)/[SupCon](https://arxiv.org/abs/2004.11362) on the generated data. The implementations for SimCLR and SupCon are adapted from [here](https://github.com/HobbitLong/SupContrast).

## Multibench Experiments
We evaluate the performance of our proposed FactorCL objective on a suite of datasets from [Multibench](https://arxiv.org/abs/2107.07502) with different shared and unique information. 

You can find examples of running the model in the notebook ```Multibench/multibench_example.ipynb```. 

We used encoders and preprocessed features provided by the Multibench [repository](https://github.com/pliang279/MultiBench). You can also train the model using raw data and other encoders designs.

## IRFL Experiments
[IRFL](https://arxiv.org/abs/2303.15445) (Image Recognition of Figurative Language) is a dataset for examining vision and language models' abilities on figurative languages. In our experiment the model is evaluated for the task of predicting the type of figurative language, including idiom, metaphor, and simile. 

For this dataset we compare performances of different objectives on pretrained [CLIP](https://arxiv.org/abs/2103.00020) image and text encoders. The encoder weights are continually pretrained using each objective.

Follow steps in the notebook ```IRFL/IRFL_example.ipynb``` to process the IRFL data and train using CLIP models.
