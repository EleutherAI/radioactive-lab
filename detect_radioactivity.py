# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torch.nn as nn
import argparse
from src.model import build_model
from src.stats import cosine_pvalue
from scipy.stats import combine_pvalues
import time
import toml
import torchvision

from src.dataset import NORMALIZE_CIFAR
from src.net import extractFeatures

#parser = argparse.ArgumentParser()
#parser.add_argument("--batch_size", default=256)
#parser.add_argument("--carrier_path", type=str, required=True)
#parser.add_argument("--crop_size", default=224)
#parser.add_argument("--dataset", default="imagenet")
#parser.add_argument("--img_size", default=256)
#parser.add_argument("--marking_network", type=str, required=True)
#parser.add_argument("--nb_workers", default=20)
#parser.add_argument("--num_classes", default=1000)
#parser.add_argument("--seed", default=1)
#parser.add_argument("--target_network", type=str, required=True)
#params = parser.parse_args()

class Params:
    def __init__(self):
        self.dataset = "cifar10"
        self.dataset_root = ""
        
        self.crop_size = 224
        self.img_size = 256

        self.carrier_path = ""
        self.marking_network = ""
        self.target_network = ""

        self.batch_size = 64
        self.nb_workers = 20
        self.seed = 1

def loadParams():
    #with open("config_detect_radioactivity.toml", "w") as fh:
    #    toml.dump(params.__dict__, fh)
    #exit(0)

    params = Params()
    with open("config_detect_radioactivity.toml", "r") as fh:
        loaded = toml.load(fh)
        for k, v in loaded.items():
            params.__dict__[k] = v

    assert(params.carrier_path != "")
    assert(params.marking_network != "")
    assert(params.target_network != "")

    return params

if __name__ == "__main__":

    params = loadParams()

    use_cuda = torch.cuda.is_available()
    print(f"CUDA Available? {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")

    # Setup Dataloader
    valid_data_loader = None
    if params.dataset.lower() == "cifar10":
        transforms = torchvision.transforms.transforms.Compose([
            torchvision.transforms.transforms.Resize(params.img_size),
            torchvision.transforms.transforms.CenterCrop(params.crop_size),
            torchvision.transforms.transforms.ToTensor(),
            NORMALIZE_CIFAR,
            ])

        valid_dataset = torchvision.datasets.CIFAR10(root=params.dataset_root, train=False, download=True, transform=transforms)

        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=params.nb_workers,
            pin_memory=True,
            sampler=None
        )
    else:
        raise ValueError(f"{params.dataset} dataloader not implemented, please change dataset config.")

    # Load Carrier
    carrier = torch.load(params.carrier_path).numpy()

    # Load marking network & remove fully connected layer
    print(f"Loading marking network '{params.marking_network}'")
    marking_ckpt = torch.load(params.marking_network)
    marking_arch = marking_ckpt['params']['architecture']
    marking_classes = marking_ckpt['params']['num_classes']
    print(f"Marking network architecture: {marking_arch}")
    print(f"Marking network original classes: {marking_classes}")   
    marking_network = build_model(marking_arch, marking_classes).eval().to(device)

    print("Removing marking network fully connected layer - save weights for later first")
    marking_network.fc = nn.Sequential()
    marking_state = marking_ckpt['model']
    W_old = marking_state['fc.weight'].cpu().numpy()
    del marking_state['fc.weight']
    del marking_state['fc.bias']

    print(marking_network.load_state_dict(marking_state, strict=False))
    
    # Start Timer
    start_all = time.time()

    # Load Target Network and remove fully connected layer
    # when running train-classif.py, trainer.py dumps the entire global params into the model save
    print(f"Loading target network '{params.target_network}'")
    target_ckpt = torch.load(params.target_network)
    target_arch = target_ckpt['params']['architecture'] 
    target_classes = target_ckpt['params']['num_classes']
    print(f"Target network architecture: {target_arch}")
    print(f"Target network classes: {target_classes}")
    target_network = build_model(target_arch, target_classes).eval().to(device)

    print("Removing target network fully connected layer")
    tested_state = {k.replace("module.", ""): v for k, v in target_ckpt['model'].items()}

    if target_arch.startswith("resnet"):
        target_network.fc = nn.Sequential()
        del tested_state['fc.weight']
        del tested_state['fc.bias']
    elif target_arch.startswith("vgg"):
        target_network.classifier[6] = nn.Sequential()
        del tested_state['classifier.6.weight']
        del tested_state['classifier.6.bias']
    elif target_arch.startswith("densenet"):
        target_network.classifier = nn.Sequential()
        del tested_state['classifier.weight']
        del tested_state['classifier.bias']

    print(target_network.load_state_dict(tested_state, strict=False))

    # Extract features
    start = time.time()
    print("Commence feature extraction on marking network...", end="")
    features_marker, _ = extractFeatures(valid_data_loader, marking_network, device, verbose=False)
    print("done")
    print("Commence feature extraction on target network...", end="")
    features_tested, _  = extractFeatures(valid_data_loader, target_network, device, verbose=False)
    print("done")
    print("Extracting features took %.2f" % (time.time() - start))
    features_marker = features_marker.numpy()
    features_tested = features_tested.numpy()

    # Align spaces
    X, residuals, rank, s = np.linalg.lstsq(features_marker, features_tested)
    print("Norm of residual: %.4e" % np.linalg.norm(np.dot(features_marker, X) - features_tested)**2)

    # Put classification vectors into aligned space
    if target_arch.startswith("resnet"):
        key = 'fc.weight'
        if key not in target_ckpt['model']:
            key = 'module.fc.weight'
    elif target_arch == "vgg16":
        key = 'classifier.6.weight'
        if key not in target_ckpt['model']:
            key = 'module.classifier.6.weight'
    elif target_arch.startswith("densenet"):
        key = 'classifier.weight'
        if key not in target_ckpt['model']:
            key = 'module.classifier.weight'

    W = target_ckpt['model'][key].cpu().numpy()
    W = np.dot(W, X.T)
    W /= np.linalg.norm(W, axis=1, keepdims=True)

    # Computing scores
    scores = np.sum(W * carrier, axis=1)

    print("Mean p-value is at %d times sigma" % int(scores.mean() * np.sqrt(W.shape[0] * carrier.shape[1])))
    print("Epoch of the model: %d" % target_ckpt["epoch"])

    p_vals = [cosine_pvalue(c, d=carrier.shape[1]) for c in list(scores)]
    print(f"log10(p)={np.log10(combine_pvalues(p_vals)[1])}")

    print("Total took %.2f" % (time.time() - start_all))
