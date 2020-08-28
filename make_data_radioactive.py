# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
from os.path import basename, join
import toml

import sys

from src.data_augmentations import RandomResizedCropFlip, CenterCrop
from src.dataset import getImagenetTransform, NORMALIZE_IMAGENET
from src.datasets.folder import default_loader
from src.model import build_model
from src.utils import initialize_exp, bool_flag, get_optimizer, repeat_to


image_mean = torch.Tensor(NORMALIZE_IMAGENET.mean).view(-1, 1, 1)
image_std = torch.Tensor(NORMALIZE_IMAGENET.std).view(-1, 1, 1)

def numpyTranspose(x):
    return np.transpose(x.numpy(), (1, 2, 0))


def numpyPixel(x):
    pixel_image = torch.clamp(255 * ((x * image_std) + image_mean), 0, 255)
    return np.transpose(pixel_image.numpy().astype(np.uint8), (1, 2, 0))


def roundPixel(x):
    x_pixel = 255 * ((x * image_std) + image_mean)
    y = torch.round(x_pixel).clamp(0, 255)
    y = ((y / 255.0) - image_mean) / image_std
    return y


def project_linf(x, y, radius):
    delta = x - y
    delta = 255 * (delta * image_std)
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / image_std
    return y + delta


def psnr(delta):
    return 20 * np.log10(255) - 10 * np.log10(np.mean(delta**2))


#def get_parser():
#    parser = argparse.ArgumentParser()

#    # main parameters
#    parser.add_argument("--dump_path", type=str, default="")
#    parser.add_argument("--exp_name", type=str, default="bypass")
#    parser.add_argument("--exp_id", type=str, default="")

#    parser.add_argument("--img_size", type=int, default=256)
#    parser.add_argument("--crop_size", type=int, default=224)
#    parser.add_argument("--data_augmentation", type=str, default="random", choices=["center", "random"])

#    parser.add_argument("--radius", type=int, default=10)
#    parser.add_argument("--epochs", type=int, default=300)
#    parser.add_argument("--lambda_ft_l2", type=float, default=0.5)
#    parser.add_argument("--lambda_l2_img", type=float, default=0.05)
#    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1-0.01-0.001,momentum=0.9,weight_decay=0.0001")
#    parser.add_argument("--carrier_path", type=str, default="", help="Direction in which to move features")
#    parser.add_argument("--carrier_id", type=int, default=0, help="Id of direction in direction array")

#    parser.add_argument("--angle", type=float, default=None, help="Angle (if cone)")
#    parser.add_argument("--half_cone", type=bool_flag, default=True)

#    parser.add_argument("--img_list", type=str, default=None,
#                        help="File that contains list of all images")
#    parser.add_argument("--img_paths", type=str, default='',
#                        help="Path to image to which apply adversarial pattern")

#    parser.add_argument("--marking_network", type=str, required=True)
#    # parser.add_argument("--image_sizes")

#    # debug
#    parser.add_argument("--debug_train", type=bool_flag, default=False,
#                        help="Use valid sets for train sets (faster loading)")
#    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
#                        help="Debug from a SLURM job")
#    parser.add_argument("--debug", help="Enable all debug flags",
#                        action="store_true")

#    return parser

# I hate programs with too many arguments
class Params:
    def __init__(self):
        self.dump_path = "data/dump"
        self.exp_name = "bypass" # setting to bypass doesn't do any chronos or slurm stuff?!?! (utils.py)
        self.exp_id = "" # ignored if exp_name is "bypass"
        self.img_size = 256
        self.crop_size = 224    
        self.data_augmentation = "random" # center or random
        self.radius = 10
        self.epochs = 90
        self.lambda_ft_l2 = 0.01
        self.lambda_l2_img = 0.0005
        self.optimizer = "sgd,lr=1.0"    
        self.carrier_path = "data/carriers.pth" # Pre-generated random array containing u for each class    
        self.carrier_id = 0 # Id of direction in direction array (image class)
        self.angle = None # Angle (if cone)
        self.half_cone = True    
        self.img_list = None # File that contains list of all images (requires setting range in "img_paths")  
        self.img_paths = ":" # Comma separted list of images to which apply adversarial pattern (or list slicing for img_list)
        self.marking_network = "data/pretrained_resnet18.pth"    
        self.batch_size = 50
        self.debug_train = False # Use valid sets for train sets (faster loading)    
        self.debug_slurm = False # Debug from a SLURM job    
        self.debug = False # Enable all debug flags


def main(params):
    # Initialize experiment - starts logger and dumps experiment params
    # We are passing "bypass" for the experiment so it doesn't anything with clusters or job ids
    logger = initialize_exp(params)

    # CPU Seems to be fast enough to do an image every 15 seconds or so.
    use_cuda = torch.cuda.is_available()
    logger.info(f"CUDA Available? {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")

    if params.img_list is None:
        params.img_paths = [s.strip() for s in params.img_paths.split(",")]
    else:
        # img_paths is used with dual functionality in this case to provide
        # a start and end index range of the images to be loaded from the list file.        
        img_list = torch.load(params.img_list)
        assert ":" in params.img_paths
        if params.img_paths == ":":
            params.img_paths = img_list
        else:
            chunks = params.img_paths.split(":")
            assert len(chunks) == 2
            n_start, n_end = int(chunks[0]), int(chunks[1])

            params.img_paths = [img_list[i] for i in range(n_start, n_end)]


    logger.info(f"Image paths {params.img_paths}")

    # Load Previously Created Pretrained network - see readme or notebook
    logger.info(f"Loading network '{params.marking_network}' to device")
    ckpt = torch.load(params.marking_network)
    params.num_classes = ckpt["params"]["num_classes"]
    params.architecture = ckpt['params']['architecture']
    logger.info(f"Marking network architecture: {params.architecture}")
    logger.info(f"Marking network original classes: {params.num_classes}")    
    model = build_model(params)
    model.to(device)
    
    # No idea why we're stripping "module", as our network has no module keys but could be something like this
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt['model'].items()}, strict=False)
    model = model.eval()

    # We remove the fully connected layer at the end as we don't need the classifier
    logger.info("Removing fully connected layer from marking network.")
    model.fc = nn.Sequential()    

    # Transforms our images (resize and crop to use with network)
    # This just ends up performing: ToTensor, NORMALIZE_IMAGENET
    # Don't worry too much about the distribution of our particular data, just use imagenet mean and std
    logger.info(f"Loading Images to Tensor and running NORMALIZE_IMAGENET")
    loader = default_loader
    transform = getImagenetTransform("none", img_size=params.img_size, crop_size=params.crop_size)
    img_orig = [transform(loader(p)).unsqueeze(0) for p in params.img_paths]

    # Loading carriers
    # Multi-class training is currently NOT supported in the loop below, so we just slice out the relevant u vector
    logger.info(f"Loading randomised carrier from '{params.carrier_path}' to device and slicing")
    direction = torch.load(params.carrier_path).to(device)
    assert direction.dim() == 2
    direction = direction[params.carrier_id:params.carrier_id + 1]
    logger.info(f"Carrier Shape: {direction.shape}")

    # No idea what angle does - we don't appear to be using it
    rho = -1
    if params.angle is not None:
        rho = 1 + np.tan(params.angle)**2

    img = [x.clone() for x in img_orig]

    # Load differentiable data augmentations, allowing propagation to original size image
    center_da = CenterCrop(params.img_size, params.crop_size)
    random_da = RandomResizedCropFlip(params.crop_size)
    if params.data_augmentation == "center":
        data_augmentation = center_da
    elif params.data_augmentation == "random":
        logger.info(f"Chosen image augmentation for training: RandomResizedCropFlip")
        data_augmentation = random_da

    # Turn on backpropagation for the images to allow marking
    logger.info(f"Enabling gradient on the images")
    for i in range(len(img)):
        img[i].requires_grad = True

    # Program blows up with too many images at once, break into batches
    for i_batch in range(0, len(img), params.batch_size):
        img_slice = img[i_batch:min(i_batch + params.batch_size, len(img))]
        img_orig_slice = img_orig[i_batch:min(i_batch + params.batch_size, len(img_orig))]

        optimizer, schedule = get_optimizer(img_slice, params.optimizer)
        if schedule is not None:
            schedule = repeat_to(schedule, params.epochs)

        # Center images and send to device    
        logger.info(f"Center Cropping Images and sending to device")
        img_center = torch.cat([center_da(x, 0).to(device, non_blocking=True) for x in img_orig_slice], dim=0)
        # ft_orig = model(center_da(img_orig, 0).cuda(non_blocking=True)).detach()

        # Save original features for use in loss function
        logger.info(f"Running model on center cropped images to obtain original features")
        ft_orig = model(img_center).detach()

        if params.angle is not None:
            ft_orig = torch.load("/checkpoint/asablayrolles/radioactive_data/imagenet_ckpt_2/features/valid_resnet18_center.pth").cuda()

        logger.info(f"Commence training (marking image)")
        for iteration in range(params.epochs):
            if schedule is not None:
                lr = schedule[iteration]
                logger.info("New learning rate for %f" % lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Differentially augment images
            logger.info("Augmenting images")
            batch = []
            for x in img_slice:
                aug_params = data_augmentation.sample_params(x)
                aug_img = data_augmentation(x, aug_params)
                batch.append(aug_img.to(device, non_blocking=True))
            batch = torch.cat(batch, dim=0)

            # Forward augmented images
            ft = model(batch)

            # Loss - See section 3.3 in the paper. 
            if params.angle is None:
                loss_ft = - torch.sum((ft - ft_orig) * direction) # First Term (Encourage u alignment)
                loss_ft_l2 = params.lambda_ft_l2 * torch.norm(ft - ft_orig, dim=1).sum() # Third term (min feature diff)
            else:
                dot_product = torch.sum((ft - ft_orig) * direction)
                logger.info("Dot product: {dot_product.item()}")
                if params.half_cone:
                    loss_ft = - rho * dot_product * torch.abs(dot_product)
                else:
                    loss_ft = - rho * (dot_product ** 2)
                loss_ft_l2 = torch.norm(ft - ft_orig)**2

            # Second term (Minimize image diff)
            loss_norm = 0
            for i in range(len(img_slice)):
                loss_norm += params.lambda_l2_img * torch.norm(img_slice[i].to(device, non_blocking=True) - 
                                                               img_orig_slice[i].to(device, non_blocking=True))**2
            loss = loss_ft + loss_norm + loss_ft_l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs = {
                "keyword": "iteration",
                "loss": loss.item(),
                "loss_ft": loss_ft.item(),
                "loss_norm": loss_norm.item(),
                "loss_ft_l2": loss_ft_l2.item(),
            }
            if params.angle is not None:
                logs["R"] = - (loss_ft + loss_ft_l2).item()
            if schedule is not None:
                logs["lr"] = schedule[iteration]
            logger.info("__log__:%s" % json.dumps(logs))

            # ??????
            for i in range(len(img_slice)):
                img_slice[i].data[0] = project_linf(img_slice[i].data[0], img_orig_slice[i][0], params.radius)
                if iteration % 10 == 0:
                    img_slice[i].data[0] = roundPixel(img_slice[i].data[0])

        img_new = [numpyPixel(x.data[0]).astype(np.float32) for x in img_slice]
        img_old = [numpyPixel(x[0]).astype(np.float32) for x in img_orig_slice]

        img_totest = torch.cat([center_da(x, 0).to(device,non_blocking=True) for x in img_slice])
        with torch.no_grad():
            ft_new = model(img_totest)

        logger.info("__log__:%s" % json.dumps({
            "keyword": "final",
            "psnr": np.mean([psnr(x_new - x_old) for x_new, x_old in zip(img_new, img_old)]),
            "ft_direction": torch.mv(ft_new - ft_orig, direction[0]).mean().item(),
            "ft_norm": torch.norm(ft_new - ft_orig, dim=1).mean().item(),
            "rho": rho,
            "R": (rho * torch.dot(ft_new[0] - ft_orig[0], direction[0])**2 - torch.norm(ft_new - ft_orig)**2).item(),
        }))

        for i in range(i_batch, min(i_batch + params.batch_size, len(img))):
            img_name = basename(params.img_paths[i])

            extension = ".%s" % (img_name.split(".")[-1])
            np.save(join(params.dump_path, img_name).replace(extension, ".npy"), img_new[i].astype(np.uint8))

if __name__ == '__main__':
    ## generate parser / parse parameters
    #parser = get_parser()
    #params = parser.parse_args()

    params = Params()
    #with open("config_make_radioactive.toml", "w") as fh:
    #    toml.dump(params.__dict__, fh)

    with open("config_make_radioactive.toml", "r") as fh:
        loaded = toml.load(fh)
        for k, v in loaded.items():
            params.__dict__[k] = v

    # debug mode
    if params.debug is True:
        params.exp_name = 'debug'
        params.debug_slurm = True
        params.debug_train = True

    # run experiment
    main(params)
