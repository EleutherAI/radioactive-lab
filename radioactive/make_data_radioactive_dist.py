# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import json
import glob
import shutil
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.transforms as transforms
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from utils.utils import NORMALIZE_CIFAR10
import radioactive.differentiable_augmentations as differentiable_augmentations

import logging
from utils.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)


def numpyTranspose(x):
    "Convert Sqeeuzed Torch Tensor C,H,W to numpy H,W,C"
    return np.transpose(x.numpy(), (1, 2, 0))

# The optimization process results in the image data becoming continuous and possibly
# outside the 0-255 range after denormalization. The following two functions are used 
# to clamp & discretize.

def numpy_pixel(x, mean, std):
    """ 
    De-normalizes the torch tensor, clamps in case of drift beyond 0-255,
    rounds to whole with unsafe cast and returns a numpy pixel (H,W,C) 
    """
    pixel_image = torch.clamp(255 * ((x * std) + mean), 0, 255)
    return np.transpose(pixel_image.numpy().astype(np.uint8), (1, 2, 0))

def round_pixel(x, mean, std):
    """ 
    De-normalizes the torch tensor, clamps in case of drift beyond 0-255,
    rounds to whole and re-normalizes, returning a torch tensor (C,H,W)
    """
    x_pixel = 255 * ((x * std) + mean)
    y = torch.round(x_pixel).clamp(0, 255)
    y = ((y / 255.0) - mean) / std
    return y


def project_linf(x, y, radius, std):
    delta = x - y
    delta = 255 * (delta * std)
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / std
    return y + delta


def get_psnr(delta):
    return 20 * np.log10(255) - 10 * np.log10(np.mean(delta**2))


def main(mp_args, images, original_indexes, *args, **kwargs):
    images_per_gpu = len(images) / mp_args.gpu
    image_slices = [images[x:x+images_per_gpu] for x in range(0, len(images), images_per_gpu)]
    index_slices = [original_indexes[x:x+images_per_gpu] for x in range(0, len(original_indexes), images_per_gpu)]

    for i, image_slice in enumerate(image_slices):
        torch.save(image_slice, f"images_{i}.pth")

    for i, index_slice in enumerate(index_slices):
        torch.save(index_slice, f"indexes_{i}.pth")

    annoying_args = (args, kwargs)
    mp.spawn(annoying_wrapper, annoying_args, mp_args.gpus)

def annoying_wrapper(device, args, kwargs):
    image_slice = torch.load(f"images_{device}.pth")
    index_slice = torch.load(f"indexes_{device}.pth")

    main_dist(image_slice, index_slice, *args, **kwargs)

def main_dist(images, original_indexes, output_directory, marking_network, carriers, class_id, normalizer,
         optimizer_fn, tensorboard_log_directory_base, batch_size=32, epochs=90, lambda_1=0.0005, lambda_2=0.01, 
         angle=None, half_cone=True, radius=10, overwrite=False, augmentation=None):
    if os.path.isdir(output_directory) and overwrite:
        shutil.rmtree(output_directory)
        #else:                
        #    raise FileExistsError(f"Image output directory {output_directory} already exists." \
        #       "Set overwrite=True if this is what you desire.")

    # Save the images in their respective class - We use torchvision.datasets.datasetfolder in training classifier
    output_directory = os.path.join(output_directory, str(class_id))
    os.makedirs(output_directory, exist_ok=True)

    # Reshape the mean and std for later use
    mean = torch.Tensor(normalizer.mean).view(-1, 1, 1)
    std = torch.Tensor(normalizer.std).view(-1, 1, 1)

    # Setup Device
    use_cuda = torch.cuda.is_available()
    logger.info(f"CUDA Available? {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")

    # Configure Marking Network
    logger.info("Configuring marking network")    
    marking_network.fc = nn.Sequential()    
    marking_network.to(device)
    marking_network.eval()   

    # Loading carriers - Here we slice the full carrier array to get the vector u for our class
    logger.info(f"Slicing carrier for class id {class_id}")
    direction = carriers.to(device)
    assert direction.dim() == 2
    direction = direction[class_id, :].view(1,-1)

    # No idea what angle does - we don't appear to be using it
    rho = -1
    #if params.angle is not None:
    #    rho = 1 + np.tan(params.angle)**2

    img_orig = images
    img = [x.clone() for x in img_orig]

    # Turn on backpropagation for the images to allow marking
    logger.info(f"Enabling gradient on the images")
    for i in range(len(img)):
        img[i].requires_grad = True

    # Unlike a classification task we are training the images.
    # We perform a certain number of epochs on each fixed batch of images.
    # We save one tensorboard run for each image batch.
    logger.info("=========== Commencing Training ===========")
    logger.info(f"Batch Size: {batch_size}")
    batch_number = 0
    images_for_tensorboard = []
    for i_batch in range(0, len(img), batch_size):
        i_batch_end = min(i_batch + batch_size, len(img))
        img_slice = img[i_batch:i_batch_end]
        img_orig_slice = img_orig[i_batch:i_batch_end]
        batch_number +=1

        # Setup optimizer on images
        optimizer = optimizer_fn(img_slice)

        # Get original features
        logger.info("Getting original features")        
        if augmentation:
            # Center crop required here - for imagenette
            center_da = differentiable_augmentations.CenterCrop(256, 224)
            img_center = torch.cat([center_da(x, 0).cuda(non_blocking=True) for x in img_orig_slice], dim=0)
            ft_orig = marking_network(img_center).detach()
        else:
            # CIFAR10
            img_orig_slice_tensor = torch.cat(img_orig_slice, dim=0).to(device)        
            ft_orig = marking_network(img_orig_slice_tensor).detach() # Remove from graph

        #if params.angle is not None:
        #    ft_orig = torch.load("/checkpoint/asablayrolles/radioactive_data/imagenet_ckpt_2/features/valid_resnet18_center.pth").cuda()

        tensorboard_log_directory = os.path.join(f"{tensorboard_log_directory_base}", f"batch{batch_number}")
        tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)

        for iteration in tqdm(range(epochs)):
            batch = []
            for x in img_slice:
                if augmentation:
                    # Differentiable augment
                    aug_params = augmentation.sample_params(x)
                    aug_img = augmentation(x, aug_params)
                    batch.append(aug_img)
                else:
                    batch.append(x)
            batch = torch.cat(batch).to(device)

            # Forward images
            ft = marking_network(batch)

            # Loss - See section 3.3 in the paper. 
            if angle is None:
                loss_ft = - torch.sum((ft - ft_orig) * direction) # First Term (Encourage u alignment)
                loss_ft_l2 = lambda_2 * torch.norm(ft - ft_orig, dim=1).sum() # Third term (min feature diff)
            #else:
            #    dot_product = torch.sum((ft - ft_orig) * direction)
            #    logger.info("Dot product: {dot_product.item()}")
            #    if params.half_cone:
            #        loss_ft = - rho * dot_product * torch.abs(dot_product)
            #    else:
            #        loss_ft = - rho * (dot_product ** 2)
            #    loss_ft_l2 = torch.norm(ft - ft_orig)**2

            # Second term (Minimize image diff)
            loss_norm = 0
            for i in range(len(img_slice)):
                loss_norm += lambda_1 * torch.norm(img_slice[i].to(device, non_blocking=True) - 
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

            tensorboard_summary_writer.add_scalar("train_loss", loss.item(), iteration)
            tensorboard_summary_writer.add_scalar("alignment_loss", loss_ft.item(), iteration)
            tensorboard_summary_writer.add_scalar("image_diff_loss", loss_norm.item(), iteration)
            tensorboard_summary_writer.add_scalar("feature_diff_loss", loss_ft_l2.item(), iteration)

            #if angle is not None:
            #    logs["R"] = - (loss_ft + loss_ft_l2).item()

            #logger.info("__log__:%s" % json.dumps(logs))

            for i in range(len(img_slice)):
                img_slice[i].data[0] = project_linf(img_slice[i].data[0], img_orig_slice[i][0], radius, std)
                if iteration % 10 == 0:
                    img_slice[i].data[0] = round_pixel(img_slice[i].data[0], mean, std)

        img_new = [numpy_pixel(x.data[0], mean, std).astype(np.float32) for x in img_slice]
        img_old = [numpy_pixel(x[0], mean, std).astype(np.float32) for x in img_orig_slice]

        if augmentation:
            img_totest = torch.cat([center_da(x, 0).cuda(non_blocking=True) for x in img_slice], dim=0).to(device)
        else:
            img_totest = torch.cat(img_slice).to(device)

        with torch.no_grad():
            ft_new = marking_network(img_totest)

        # Batch Summary Info
        psnr = np.mean([get_psnr(x_new - x_old) for x_new, x_old in zip(img_new, img_old)])
        ft_direction = torch.mv(ft_new - ft_orig, direction[0]).mean().item()
        ft_norm = torch.norm(ft_new - ft_orig, dim=1).mean().item()
        R = (rho * torch.dot(ft_new[0] - ft_orig[0], direction[0])**2 - torch.norm(ft_new - ft_orig)**2).item()
        logger.info(f"psnr: {psnr}")
        logger.info(f"ft_direction: {ft_direction}")
        logger.info(f"ft_norm: {ft_norm}")
        logger.info(f"rho: {rho}")
        logger.info(f"R: {R}")

        for i, img_to_save in enumerate(img_new):
            img_to_save = img_to_save.astype(np.uint8)
            images_for_tensorboard.append(img_to_save)

            marked_images_index = i_batch + i
            original_train_set_index = original_indexes[marked_images_index]
            output_file_name = f"train_{original_train_set_index}.npy"   
            output_path = os.path.join(output_directory, output_file_name)

            np.save(output_path, img_to_save)

    return images_for_tensorboard
