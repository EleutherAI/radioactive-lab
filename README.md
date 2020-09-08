# Radioactive data

This is EleutherAI's reimplementation of the paper ["Radioactive data: tracing through training"](https://arxiv.org/abs/2002.00937). 

Their GitHub repo can be found [here](https://github.com/facebookresearch/radioactive_data). 
**Warning:** The official open source implementation has a fairly complicated framework, as well as miscellaneous issues like hard-coded paths that will prevent you from running it. 

We spent a fair amount of time getting the original implementation working with CIFAR10, then chose to do a full refactor after understanding the core requirements.

Our implementation is a refined version designed to demonstrate the marking, training and detection steps using the CIFAR10 dataset, with each stage self contained within its own python module. We have added logging to TensorBoard for your visualization enjoyment.

Prior to implementing the training stage we created a full working example of a resnet18 classifer trained on CIFAR10. This can be used to benchmark your particular hardware and choose a good optimizer prior to running the main code. It's also a good starting point for ML beginners.

## Install

First setup pytorch. Example for GPU enabled using conda:

```bash
conda create --name radioactive_lab python=3.8
conda activate radioactive_lab
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

After that use pip to install other requirements:

```bash
pip install -r requirements.txt
```

If anything is missing, simply pip install [missing requirement].

## Running

Please follow the basic_example.ipynb example. 

For any other questions please visit our Discord!

## License

This repository is licensed under the CC BY-NC 4.0.
