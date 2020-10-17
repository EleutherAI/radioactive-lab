"""
Table 1
-------
1. Download resnet18 pretrained on Imagenet ILSVRC 2012.
2. Generate 1, 2, 5, 10% markings using network from step 1.
3. Retrain the logistic regression layer on a copy of the pretrained resnet18 using the marked
   images merged with vanilla images.
   Train it using the radioactive data from step 2.
4. Perform the radioactive detection p-tests on the network trained in step 3. Compare the top-1 accuracy
   of this network with the network downloaded in step 1.
5. Generate Table 1.

NOTES: We use differentiable center crop for marking.
       We use random crop for training the target network.
"""

import random
import os
import glob
import shutil
from functools import partial
import argparse

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.transforms as transforms
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cutie

from radioactive.make_data_radioactive_dist import main as do_marking_dist
import radioactive.train_marked_classifier_dist as train_marked_classifier_dist
from radioactive.detect_radioactivity import main as detect_radioactivity
import radioactive.differentiable_augmentations as differentiable_augmentations
from utils.utils import NORMALIZE_IMAGENET

import logging
from utils.logger import setup_logger_tqdm
logger = logging.getLogger(__name__)

def get_images_for_marking_multiclass(training_set, tensorboard_log_directory, overall_marking_percentage):

    # Randomly sample images
    total_marked_images = int(len(training_set) * (overall_marking_percentage / 100))
    train_marked_indexes = random.sample(range(0, len(training_set)), total_marked_images)

    # Sort images into classes
    # { 0 : [(image1, original_index1),(image2, original_index2)...], 1 : [....] }
    transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])
    image_data = {class_id:[] for class_id in range(0, len(training_set.classes))}    
    for index in train_marked_indexes:
        image, label = training_set[index]
        image_data[label].append((transform(image).unsqueeze(0), index))

    # Save to tensorboard - sorted by class, original_index
    tensorboard_summary_writer = SummaryWriter(log_dir=tensorboard_log_directory)
    images = []
    transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
    for class_id, image_list in image_data.items():
        if image_list:
            _, original_indexes = map(list, zip(*image_list))
            for index in original_indexes:
                image, label = training_set[index]
                images.append(transform(image))
    img_grid = torchvision.utils.make_grid(images, nrow=3)
    tensorboard_summary_writer.add_image('images_for_marking', img_grid)

    return image_data

# Doesn't work on multi-pc
def do_marking_run_multiclass(overall_marking_percentage, experiment_directory, tensorboard_log_directory,
                              marking_network, training_set, mp_args):

    # Setup experiment directory
    if os.path.isdir(experiment_directory):
        error_message = f"Directory {experiment_directory} already exists. By default we assume you don't want to "\
                        "repeat the marking stage."
        logger.info(error_message)
        return

    os.makedirs(experiment_directory)

    logfile_path = os.path.join(experiment_directory, 'marking.log')
    setup_logger_tqdm(filepath=logfile_path)

    # Carriers
    marking_network_fc_feature_size = 512
    carriers = torch.randn(len(training_set.classes), marking_network_fc_feature_size)
    carriers /= torch.norm(carriers, dim=1, keepdim=True)
    torch.save(carriers, os.path.join(experiment_directory, "carriers.pth"))


    # { 0 : [(image1, original_index1),(image2, original_index2)...], 1 : [....] }
    image_data = get_images_for_marking_multiclass(training_set,
                                                   tensorboard_log_directory,
                                                   overall_marking_percentage)


    for class_id, image_list in image_data.items():
        if image_list:
            images, original_indexes = map(list, zip(*image_list))            
            epochs = 250
            batch_size = 8
            output_directory = os.path.join(experiment_directory, "marked_images")
            #augmentation = differentiable_augmentations.CenterCrop(256, 224)
            augmentation = differentiable_augmentations.RandomResizedCropFlip(256)

            tensorboard_class_log = os.path.join(tensorboard_log_directory, f"class_{class_id}")
            do_marking_dist(mp_args, images, original_indexes, output_directory, marking_network, carriers, 
                            class_id, NORMALIZE_IMAGENET, tensorboard_class_log, epochs=epochs, 
                            batch_size=batch_size, overwrite=False, augmentation=augmentation)

    # Record marking completion
    with open(os.path.join(experiment_directory, "marking.complete"),"w") as fh:
        fh.write("1")

def calculate_p_values(marking_percentages, batch_size):
    logfile_path = f"experiments/table1_imagenet/detect_radioactivity.log"
    setup_logger_tqdm(logfile_path)

    p_values = []

    for run in marking_percentages:
        run_name = f"{run}_percent"
        carrier_path = f"experiments/table1_imagenet/{run_name}/carriers.pth"

        target_checkpoint_path = f"experiments/table1_imagenet/{run_name}/marked_classifier/rank_0/checkpoint.pth"
        target_checkpoint = torch.load(target_checkpoint_path)
        target_checkpoint['model_state_dict'] = {k.replace("module.", ""): v for k, v in target_checkpoint['model_state_dict'].items()}  
  
        (scores, p_vals, combined_pval) = detect_radioactivity(carrier_path, None, 
                                                               None, target_checkpoint,
                                                               batch_size=batch_size,
                                                               align=False, test_set_loader=None)
        p_values.append(combined_pval)

    return p_values

def generate_table_1(marking_percentages, p_values, vanilla_accuracy):

    accuracies = [vanilla_accuracy]
    for run in marking_percentages:
        run_name = f"{run}_percent"
        target_checkpoint_path = f"experiments/table1_imagenet/{run_name}/marked_classifier/rank_0/checkpoint.pth"
        target_checkpoint = torch.load(target_checkpoint_path)
        accuracies.append(target_checkpoint["test_accuracy"])

    formatted_accuracies = list(map(lambda x: f"{float(x):0.2f}", accuracies))

    # Create the table!
    column_labels = tuple([0] + marking_percentages)
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(column_labels)))
    row_labels = ["log10(p)", "Top-1 %"]
    formatted_pvalues = ["n/a"]
    formatted_pvalues += [f"{p:0.4f}" for p in np.log10(p_values)]

    cell_text = [formatted_pvalues, formatted_accuracies]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axis('off')
    table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colColours=colors,
                     colLabels=column_labels,
                     loc='center')
    plt.savefig("experiments/table1_imagenet/table1.png")
    plt.show()

def main(imagenet_path, step_3_batch_size, mp_args):
    logger.info("Table 1 Preparation Commencing")
    logger.info("=============================")
    marking_percentages = [1, 2, 5, 10]

    train_images_path = os.path.join(imagenet_path, "train")
    test_images_path = os.path.join(imagenet_path, "val")
    p_values_file = "experiments/table1_imagenet/p_values.pth"

    logger.info("")
    logger.info("Step 1 - Download Marking Network")
    logger.info("------------------------------")
    marking_network = torchvision.models.resnet18(pretrained=True)

    logger.info("")
    logger.info("Step 2 - Image Marking")
    logger.info("----------------------")
    # Parallelized separately
    training_set = torchvision.datasets.ImageFolder(train_images_path)
    for marking_percentage in marking_percentages:
        experiment_directory = os.path.join("experiments", "table1_imagenet", f"{marking_percentage}_percent")
        if os.path.exists(os.path.join(experiment_directory, "marking.complete")):
            message = f"Marking step already completed for {marking_percentage}%. Do you want to restart this part of " \
                      "the experiment?"
            #if not cutie.prompt_yes_or_no(message, yes_text="Restart", no_text="Skip marking step"):
            #    continue

            logger.info("SKIPPING MARKING - Fix this up before sending to DGX")
            continue

        tensorboard_log_directory = os.path.join("runs", "table1_imagenet", f"{marking_percentage}_percent", "marking")
        shutil.rmtree(experiment_directory, ignore_errors=True)
        shutil.rmtree(tensorboard_log_directory, ignore_errors=True)
        do_marking_run_multiclass(marking_percentage, experiment_directory, tensorboard_log_directory,
                                  marking_network, training_set, mp_args)

    logger.info("")
    logger.info("Step 3 - Training Target Networks")
    logger.info("---------------------------------")
    # Parallelized with DDP
    for marking_percentage in marking_percentages:
        marked_images_directory = os.path.join("experiments", "table1_imagenet", f"{marking_percentage}_percent", "marked_images")
        output_directory = os.path.join("experiments", "table1_imagenet", f"{marking_percentage}_percent", "marked_classifier")
        tensorboard_log_directory = os.path.join("runs", "table1_imagenet", f"{marking_percentage}_percent", "target")

        # Load a new pretrained resnet18
        model = torchvision.models.resnet18(pretrained=True)

        # Retrain the fully connected layer only
        for param in model.parameters():
            param.requires_grad = False
        num_classes = 10
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        optimizer = train_marked_classifier_dist.adamw_logistic

        epochs = 20
        dataloader_func = partial(train_marked_classifier_dist.get_data_loaders_imagenet, 
                                  train_images_path, test_images_path, marked_images_directory, step_3_batch_size, 1)

        train_args = (mp_args, dataloader_func, model, optimizer, output_directory, tensorboard_log_directory, epochs)
        mp.spawn(train_marked_classifier_dist.main, nprocs=mp_args.gpus, args=train_args)

        #(dataloader_func, model, optimizer, output_directory, tensorboard_log_directory, 
        #                             epochs=epochs)

    logger.info("")
    logger.info("Step 4 - Calculating p-values")
    logger.info("-----------------------------")
    test_set_loader = train_marked_classifier_dist.get_imagenet_test_loader(test_images_path, NORMALIZE_IMAGENET, 
                                                                               batch_size=step_3_batch_size)
    p_values = calculate_p_values(marking_percentages, step_3_batch_size)
    torch.save(p_values, p_values_file)
    p_values = torch.load(p_values_file)

    logger.info("")
    logger.info("Step 5 - Generating Table 1")
    logger.info("---------------------------")
    # Get Vanilla Accuracy
    marking_network = torchvision.models.resnet18(pretrained=True)
    marking_network.to("cuda")
    vanilla_accuracy = train_marked_classifier_dist.test_model("cuda", marking_network, test_set_loader)

    # Finish Up
    generate_table_1(marking_percentages, p_values, vanilla_accuracy)

parser_description = 'Perform experiments and generate Table 1 and 2 for imagenet.'
parser = argparse.ArgumentParser(description=parser_description)
parser.add_argument("-dir", "--imagenet_path", default="E:/imagenet2/")
parser.add_argument("-bs", "--batch_size_step_3", type=int, default=16)
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')

class AnonObject(object):
    def __init__(self):
        pass

if __name__ == '__main__':
    setup_logger_tqdm() # Commence logging to console

    assert(torch.cuda.is_available())
    assert(torch.distributed.is_available())
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'

    mp_args = AnonObject()
    mp_args.nr = args.nr
    mp_args.gpus = args.gpus
    mp_args.world_size = args.gpus * args.nodes

    main(args.imagenet_path, args.batch_size_step_3, mp_args)

