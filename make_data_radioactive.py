# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import json
import numpy as np
import os
import shutil
import torchvision
import torchvision.transforms.transforms as transforms
import random
import logging
import glob

from torch.utils.tensorboard import SummaryWriter

from utils import NORMALIZE_CIFAR
from differentiable_augmentations import RandomResizedCropFlip
from logger import setup_logger

# Get an unconfigured logger, will propagate to root we configure in main program
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


def psnr(delta):
    return 20 * np.log10(255) - 10 * np.log10(np.mean(delta**2))


# Supports one class at a time - just run multiple times
def main(output_directory, marking_network, images, original_indexes, carriers, class_id,
         optimizer_fn, tensorboard_log_directory_base, batch_size=32, epochs=90, lambda_1=0.0005, lambda_2=0.01, 
         angle=None, half_cone=True, radius=10, overwrite=False):

    if os.path.isdir(output_directory) and overwrite:
        shutil.rmtree(output_directory)
        #else:                
        #    raise FileExistsError(f"Image output directory {output_directory} already exists." \
        #       "Set overwrite=True if this is what you desire.")

    # Save the images in their respective class - We use torchvision.datasets.datasetfolder in training classifier
    output_directory = os.path.join(output_directory, str(class_id))
    os.makedirs(output_directory, exist_ok=True)

    # Reshape the mean and std for later use
    mean = torch.Tensor(NORMALIZE_CIFAR.mean).view(-1, 1, 1)
    std = torch.Tensor(NORMALIZE_CIFAR.std).view(-1, 1, 1)

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

    # Load differentiable data augmentations, allowing gradients to backprop
    #center_da = CenterCrop(scaled_image_size, crop_size) # Unused for CIFAR10
    #random_da = RandomResizedCropFlip(crop_size)
    data_augmentation = RandomResizedCropFlip(28) # Crop from 32 to 28

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

        # Get original features - Facebook code had differentiable rescale we don't need for CIFAR10
        logger.info("Getting original features")
        img_orig_slice_tensor = torch.cat(img_orig_slice, dim=0).to(device)        
        ft_orig = marking_network(img_orig_slice_tensor).detach() # Remove from graph

        #if params.angle is not None:
        #    ft_orig = torch.load("/checkpoint/asablayrolles/radioactive_data/imagenet_ckpt_2/features/valid_resnet18_center.pth").cuda()

        tensorboard_log_directory = f"{tensorboard_log_directory_base}_batch{batch_number}"
        tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)

        for iteration in range(epochs):
            # Differentially augment images
            logger.info("Augmenting images")
            batch = []
            for x in img_slice:
                aug_params = data_augmentation.sample_params(x)
                aug_img = data_augmentation(x, aug_params)
                batch.append(aug_img)
            batch = torch.cat(batch).to(device)

            # Forward augmented images
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

            logger.info("__log__:%s" % json.dumps(logs))

            for i in range(len(img_slice)):
                img_slice[i].data[0] = project_linf(img_slice[i].data[0], img_orig_slice[i][0], radius, std)
                if iteration % 10 == 0:
                    img_slice[i].data[0] = round_pixel(img_slice[i].data[0], mean, std)

        img_new = [numpy_pixel(x.data[0], mean, std).astype(np.float32) for x in img_slice]
        img_old = [numpy_pixel(x[0], mean, std).astype(np.float32) for x in img_orig_slice]

        img_totest = torch.cat(img_slice).to(device)
        with torch.no_grad():
            ft_new = marking_network(img_totest)

        logger.info("__log__:%s" % json.dumps({
            "keyword": "final",
            "psnr": np.mean([psnr(x_new - x_old) for x_new, x_old in zip(img_new, img_old)]),
            "ft_direction": torch.mv(ft_new - ft_orig, direction[0]).mean().item(),
            "ft_norm": torch.norm(ft_new - ft_orig, dim=1).mean().item(),
            "rho": rho,
            "R": (rho * torch.dot(ft_new[0] - ft_orig[0], direction[0])**2 - torch.norm(ft_new - ft_orig)**2).item(),
        }))

        for i, img_to_save in enumerate(img_new):
            img_to_save = img_to_save.astype(np.uint8)
            images_for_tensorboard.append(img_to_save)

            marked_images_index = i_batch + i
            original_train_set_index = original_indexes[marked_images_index]
            output_file_name = f"train_{original_train_set_index}.npy"   
            output_path = os.path.join(output_directory, output_file_name)

            np.save(output_path, img_to_save)

    return images_for_tensorboard


def get_images_for_marking(training_set, tensorboard_log_directory, class_marking_percentage):

    # Index images by class
    images_by_class = [[] for x in training_set.classes]
    for index, (img, label) in enumerate(training_set):
        images_by_class[label].append(index)

    # Randomly choose an image class
    chosen_image_class = random.choice(list(range(0, len(training_set.classes))))
    logger.info(f"Randomly selected image class {chosen_image_class} ({training_set.classes[chosen_image_class]})")

    # Randomly sample images from that class
    total_marked_in_class = int(len(images_by_class[chosen_image_class]) * (class_marking_percentage / 100))
    train_marked_indexes = random.sample(images_by_class[chosen_image_class], total_marked_in_class)

    # Save to tensorboard - never use pyplot for grids, so slow....
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)
    images = []
    for index in train_marked_indexes:
        images.append(transforms.ToTensor()(training_set.data[index]))
    img_grid = torchvision.utils.make_grid(images, nrow=16)
    tensorboard_summary_writer.add_image('images_for_marking', img_grid)

    # Transform and add to list
    transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_CIFAR])
    images_for_marking = []
    for index in train_marked_indexes:
        image , _ = training_set[index]
        images_for_marking.append(transform(image).unsqueeze(0))

    return chosen_image_class, images_for_marking, train_marked_indexes

def get_images_for_marking_multiclass(training_set, tensorboard_log_directory, overall_marking_percentage):

    # Randomly sample images
    total_marked_images = int(len(training_set) * (overall_marking_percentage / 100))
    train_marked_indexes = random.sample(range(0, len(training_set)), total_marked_images)

    # Sort images into classes, rewriting the marking code for multi-class is messy
    # Transform and add to dictionary of lists, dictionary index is class id
    # { 0 : [(image1, original_index1),(image2, original_index2)...], 1 : [....] }
    transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_CIFAR])
    image_data = {class_id:[] for class_id in range(0, len(training_set.classes))}    
    for index in train_marked_indexes:
        image, label = training_set[index]
        image_data[label].append((transform(image).unsqueeze(0), index))

    # Save to tensorboard - sorted by class, original_index
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)
    images = []
    for class_id, image_list in image_data.items():
        _, original_indexes = zip(*image_list)
        original_indexes = list(original_indexes)
        for index in original_indexes:
            images.append(transforms.ToTensor()(training_set.data[index]))
    img_grid = torchvision.utils.make_grid(images, nrow=16)
    tensorboard_summary_writer.add_image('images_for_marking', img_grid)

    return image_data

if __name__ == '__main__':

    # Setup experiment directory, logging
    experiment_directory = "experiments/radioactive"
    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)

    logfile_path = os.path.join(experiment_directory, 'marking.log')
    setup_logger(filepath=logfile_path)

    # Clear old tensorboard logs
    our_tensorboard_logs = glob.glob('runs/radioactive*') # main creates extra log dirs
    for tensorboard_log in our_tensorboard_logs:
        shutil.rmtree(tensorboard_log)
    tensorboard_log_directory="runs/radioactive"

    # Load randomly sampled images from random class along with list of original indexes 
    training_set = torchvision.datasets.CIFAR10(root="experiments/datasets", download=True)
    class_marking_percentage = 10
    class_id, images, original_indexes = get_images_for_marking(training_set, tensorboard_log_directory, 
                                                                class_marking_percentage)

    # Marking network is a pretrained resnet18
    marking_network = torchvision.models.resnet18(pretrained=True)

    # Carriers
    marking_network_fc_feature_size = 512
    carriers = torch.randn(len(training_set.classes), marking_network_fc_feature_size)
    carriers /= torch.norm(carriers, dim=1, keepdim=True)
    torch.save(carriers, os.path.join(experiment_directory, "carriers.pth"))

    # Run!
    #optimizer = lambda x : torch.optim.SGD(x, lr=1) # Doesn't produce good loss
    optimizer = lambda x : torch.optim.Adam(x, lr=0.1)
    epochs = 100
    batch_size = 32
    output_directory = os.path.join(experiment_directory, "marked_images")
    marked_images = main(output_directory, marking_network, images, original_indexes, carriers, 
                         class_id, optimizer, tensorboard_log_directory, epochs=epochs, 
                         batch_size=batch_size, overwrite=True)

    # Show marked images in Tensorboard
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)
    images_for_tensorboard = [transforms.ToTensor()(x) for x in marked_images]
    img_grid = torchvision.utils.make_grid(images_for_tensorboard, nrow=16)
    tensorboard_summary_writer.add_image('marked_images', img_grid)

