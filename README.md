# Radioactive data

This is EleutherAI's reimplementation of the paper ["Radioactive data: tracing through training"](https://arxiv.org/abs/2002.00937). Their GitHub repo can be found [here](https://github.com/facebookresearch/radioactive_data). **Warning:** the official open source implementation has some bugs, as well as miscellaneous issues like hard-coded paths that will prevent you from running it. To run the code, use the installation instructions below and then open `notebook.ipynb`

**Note:** @researcher2 is almost done with a full refactor so don't go getting too comfortable with the current code. 

# Original README

At some point this will be rewritten, but for now we are preserving the original README.

## Install

We have generated a requirements.txt for you to use after conda install pytorch.
```python
conda install numpy
# See http://pytorch.org for details
conda install pytorch -c pytorch
```

## Creating radioactive data

Marking the data is very easy.
First, specify a marking model
```python
import torch
from torchvision import models

resnet18 = models.resnet18(pretrained=True)
torch.save({
    "model": resnet18.state_dict(),
    "params": {
      "architecture": "resnet18",
      "num_classes" : ????
    }
  }, "pretrained_resnet18.pth")

```

This seems like extra work for nothing. We do the sampling here across both dimensions and it ends up just getting sliced out inside make_data_radioactive.py with final dimension as (1, dim) as expected??

Then sample random (normalized) directions as carriers:
```python
import torch

n_classes, dim = 1000, 512
carriers = torch.randn(n_classes, dim)
carriers /= torch.norm(carriers, dim=1, keepdim=True)
torch.save(carriers, "carriers.pth")
```

The `make_data_radioactive.py` script does the actual marking.
For example, to mark images `img1.jpeg` and `img2.jpeg` with the carrier #0 (which should be the same as the class id), do:
```
python make_data_radioactive.py \
--carrier_id 0 \
--carrier_path carriers.pth \
--data_augmentation random \
--epochs 90 \
--img_paths img1.jpeg,img2.jpeg \
--lambda_ft_l2 0.01 \
--lambda_l2_img 0.0005 \
--marking_network pretrained_resnet18.pth \
--dump_path /path/to/images \
--optimizer sgd,lr=1.0
```

This takes about 1 minute and the output images are stored in /path/to/images.


## Training a model

The training is not controlled by the adversary. 
However, to simulate it, we perform a standard training as follows.

To train a model, we need to create a file that lists all the images that have been replaced with marked versions:

```python
import torch

torch.save({
  'type': 'per_sample',
  'content': {
    988: 'img1_radio.npy',
  }
}, "radioactive_data.pth")
```

We can then launch training:
```
python train-classif.py \
--architecture resnet18 \
--dataset imagenet \
--epochs 90 \
--train_path radioactive_data.pth \
--train_transform random
```

## Detecting if a model is radioactive

So now we use the carriers and the trained network to detect the radioactive marks.

```
python detect_radioactivity.py \
--carrier_path carriers.pth \
--marking_network pretrained_resnet18.pth \
--tested_network checkpoint-0.pth
```

On the output, you should obtain a line with "log10(p)=...", which gives the (log of the) p-value of radioactivity detection.

## License

This repository is licensed under the CC BY-NC 4.0.
